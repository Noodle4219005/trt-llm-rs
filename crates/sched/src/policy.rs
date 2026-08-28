//! Queue ordering policies.

use trtllm_core::config::PrefillPolicy;
use trtllm_core::{Millis, RequestId};

/// One unit of prefill work as the ordering policy sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Job {
    pub id: RequestId,
    pub arrival_ms: Millis,
    /// Absolute first-token deadline.
    pub deadline_ms: Millis,
    /// Estimated time on the prefill pool, milliseconds.
    pub service_ms: f64,
}

/// The result of ordering a queue.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct JobOrdering {
    /// Requests to run, in the order they should run.
    pub on_time: Vec<RequestId>,
    /// Requests the policy has given up on meeting the deadline for. They are
    /// still served - this is a closed-loop benchmark and dropping a request
    /// just stalls a client thread - but they run behind everything that can
    /// still be good, so they cost one bad request instead of several.
    pub demoted: Vec<RequestId>,
}

impl JobOrdering {
    pub fn all(&self) -> impl Iterator<Item = &RequestId> {
        self.on_time.iter().chain(self.demoted.iter())
    }
}

/// Order a queue under `policy`.
pub fn order_jobs(jobs: &[Job], now: Millis, policy: PrefillPolicy) -> JobOrdering {
    match policy {
        PrefillPolicy::Fcfs => {
            let mut v = jobs.to_vec();
            v.sort_by(|a, b| a.arrival_ms.total_cmp(&b.arrival_ms).then(a.id.cmp(&b.id)));
            JobOrdering {
                on_time: v.into_iter().map(|j| j.id).collect(),
                demoted: Vec::new(),
            }
        }
        PrefillPolicy::Edf => {
            let mut v = jobs.to_vec();
            v.sort_by(|a, b| {
                a.deadline_ms
                    .total_cmp(&b.deadline_ms)
                    .then(a.id.cmp(&b.id))
            });
            JobOrdering {
                on_time: v.into_iter().map(|j| j.id).collect(),
                demoted: Vec::new(),
            }
        }
        PrefillPolicy::MooreHodgson => moore_hodgson(jobs, now),
    }
}

/// Moore-Hodgson, the optimal algorithm for `1 || sum U_j`: maximise the number
/// of jobs that finish by their deadline on a single machine.
///
/// Walk the queue in deadline order accumulating completion time. The moment
/// the schedule goes late, evict the *longest* job accepted so far - it is the
/// one buying the least deadline relief per unit of machine time. Evicting is
/// what makes this better than plain EDF: EDF minimises the worst lateness,
/// which is the wrong objective when the score counts requests, not milliseconds.
///
/// The prefill pool is several workers, not one machine, so `service_ms` must
/// already be expressed against the *aggregate* prefill rate. That makes this
/// an approximation, and it is exact only when the pool is saturated - which is
/// precisely the regime where the ordering matters.
///
/// One honest caveat for the scored workload: it fixes ISL at 4000 with stddev
/// 0, so every `service_ms` is equal and deadlines are `arrival + 3000`
/// (agreeable). Under those conditions Moore-Hodgson degenerates towards FCFS
/// and buys little. It earns its place on mixed traffic, on partial prefix
/// hits, and whenever chunking has left different amounts of work outstanding.
fn moore_hodgson(jobs: &[Job], now: Millis) -> JobOrdering {
    let mut by_deadline = jobs.to_vec();
    by_deadline.sort_by(|a, b| {
        a.deadline_ms
            .total_cmp(&b.deadline_ms)
            .then(a.id.cmp(&b.id))
    });

    let mut accepted: Vec<Job> = Vec::with_capacity(by_deadline.len());
    let mut demoted: Vec<Job> = Vec::new();
    let mut clock = now;

    for job in by_deadline {
        accepted.push(job);
        clock += job.service_ms;
        if clock > job.deadline_ms {
            // Evict the longest accepted job; ties go to the later deadline so
            // the decision is deterministic across runs.
            let victim = accepted
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.service_ms
                        .total_cmp(&b.service_ms)
                        .then(a.deadline_ms.total_cmp(&b.deadline_ms))
                        .then(a.id.cmp(&b.id))
                })
                .map(|(i, _)| i)
                .expect("just pushed at least one job");
            let removed = accepted.remove(victim);
            clock -= removed.service_ms;
            demoted.push(removed);
        }
    }

    // Demoted work still has to run; order it by deadline so the least hopeless
    // goes first and may still squeak in if the queue drains.
    demoted.sort_by(|a, b| {
        a.deadline_ms
            .total_cmp(&b.deadline_ms)
            .then(a.id.cmp(&b.id))
    });

    JobOrdering {
        on_time: accepted.into_iter().map(|j| j.id).collect(),
        demoted: demoted.into_iter().map(|j| j.id).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: u64, arrival: f64, deadline: f64, service: f64) -> Job {
        Job {
            id: RequestId(id),
            arrival_ms: arrival,
            deadline_ms: deadline,
            service_ms: service,
        }
    }

    #[test]
    fn fcfs_preserves_arrival_order() {
        let jobs = [job(2, 20.0, 3020.0, 100.0), job(1, 10.0, 3010.0, 100.0)];
        let o = order_jobs(&jobs, 0.0, PrefillPolicy::Fcfs);
        assert_eq!(o.on_time, vec![RequestId(1), RequestId(2)]);
        assert!(o.demoted.is_empty());
    }

    /// The textbook case: one long job at the head starves three short ones.
    /// EDF still runs it first and loses all three; Moore-Hodgson demotes it
    /// and keeps three of four requests good.
    #[test]
    fn moore_hodgson_sacrifices_the_hog() {
        let jobs = [
            job(0, 0.0, 100.0, 100.0),
            job(1, 0.0, 150.0, 100.0),
            job(2, 0.0, 160.0, 10.0),
            job(3, 0.0, 170.0, 10.0),
        ];
        let o = order_jobs(&jobs, 0.0, PrefillPolicy::MooreHodgson);
        assert_eq!(o.on_time, vec![RequestId(0), RequestId(2), RequestId(3)]);
        assert_eq!(o.demoted, vec![RequestId(1)]);

        // EDF keeps the hog at the head and everything behind it goes late.
        let edf = order_jobs(&jobs, 0.0, PrefillPolicy::Edf);
        let mut clock = 0.0;
        let mut edf_good = 0;
        for id in &edf.on_time {
            let j = jobs.iter().find(|j| j.id == *id).expect("job");
            clock += j.service_ms;
            if clock <= j.deadline_ms {
                edf_good += 1;
            }
        }
        assert_eq!(
            edf_good, 1,
            "EDF keeps fewer requests on time than Moore-Hodgson's 3"
        );
    }

    #[test]
    fn nothing_is_demoted_when_everything_fits() {
        let jobs = [job(0, 0.0, 5000.0, 100.0), job(1, 0.0, 5000.0, 100.0)];
        let o = order_jobs(&jobs, 0.0, PrefillPolicy::MooreHodgson);
        assert_eq!(o.on_time.len(), 2);
        assert!(o.demoted.is_empty());
    }

    #[test]
    fn every_job_comes_back_exactly_once() {
        let jobs: Vec<Job> = (0..20)
            .map(|i| job(i, 0.0, 500.0 + i as f64, 200.0))
            .collect();
        let o = order_jobs(&jobs, 0.0, PrefillPolicy::MooreHodgson);
        let mut ids: Vec<u64> = o.all().map(|r| r.0).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..20).collect::<Vec<_>>());
    }

    /// With equal service times and agreeable deadlines - the scored workload -
    /// Moore-Hodgson must not reorder anything gratuitously.
    #[test]
    fn uniform_workload_degenerates_to_arrival_order() {
        let jobs: Vec<Job> = (0..10)
            .map(|i| job(i, i as f64 * 50.0, i as f64 * 50.0 + 3000.0, 400.0))
            .collect();
        let o = order_jobs(&jobs, 0.0, PrefillPolicy::MooreHodgson);
        let head: Vec<u64> = o.on_time.iter().take(5).map(|r| r.0).collect();
        assert_eq!(head, vec![0, 1, 2, 3, 4]);
    }
}
