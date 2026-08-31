//! The launcher and the model must describe the same deployment.
//!
//! They drifted, and it was not caught by anything: `Config::default()` said
//! 4P1D with TP2 prefill and a TP8 decode worker long after the sbatch script
//! had moved to 2P2D on TP4 -- and TP2 had by then been measured unrunnable.
//! Every prediction made from the default was about a topology nobody would
//! run, and the two files agreed only by whoever last edited both.
//!
//! Reading the launcher is not elegant. It is cheaper than the alternative,
//! which is a 163 SU run whose numbers turn out to describe something else.

use std::fs;
use std::path::PathBuf;

use trtllm_core::config::Config;

fn launcher() -> String {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/stage-d-235b-disagg.sbatch");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// Pull `: "${NAME:=value}"` out of the script.
fn default_of(script: &str, name: &str) -> String {
    let needle = format!(": \"${{{name}:=");
    let start = script
        .find(&needle)
        .unwrap_or_else(|| panic!("{name} has no default in the launcher"))
        + needle.len();
    let rest = &script[start..];
    let end = rest
        .find("}\"")
        .unwrap_or_else(|| panic!("{name}'s default is not terminated"));
    rest[..end].to_string()
}

fn number(script: &str, name: &str) -> f64 {
    default_of(script, name)
        .parse()
        .unwrap_or_else(|e| panic!("{name} is not a number: {e}"))
}

#[test]
fn the_launcher_and_the_default_config_describe_one_deployment() {
    let s = launcher();
    let c = Config::default();

    let pairs: [(&str, f64, f64); 6] = [
        (
            "PREFILL_WORKERS",
            number(&s, "PREFILL_WORKERS"),
            c.topology.prefill_workers.into(),
        ),
        (
            "PREFILL_TP",
            number(&s, "PREFILL_TP"),
            c.topology.prefill_tp.into(),
        ),
        (
            "DECODE_WORKERS",
            number(&s, "DECODE_WORKERS"),
            c.topology.decode_workers.into(),
        ),
        (
            "DECODE_TP",
            number(&s, "DECODE_TP"),
            c.topology.decode_tp.into(),
        ),
        (
            "KV_XFER_CONCURRENCY",
            number(&s, "KV_XFER_CONCURRENCY"),
            c.kv.xfer_concurrency.into(),
        ),
        ("N", number(&s, "N"), c.workload.concurrency.into()),
    ];
    for (name, launcher_value, config_value) in pairs {
        assert_eq!(
            launcher_value, config_value,
            "{name}: the launcher runs {launcher_value} and the model predicts \
             for {config_value}. Whichever is right, the other one is making \
             claims about a deployment that does not exist."
        );
    }
}

#[test]
fn the_deployment_the_launcher_runs_is_one_the_model_would_recommend() {
    let s = launcher();
    let c = Config::default();
    let m = c.capacity_model();

    let ptp = number(&s, "PREFILL_TP") as u32;
    let dtp = number(&s, "DECODE_TP") as u32;
    assert!(
        m.fits_in_memory(ptp),
        "the launcher runs TP{ptp} prefill workers, which leave {:.1} GiB per \
         rank against a {:.1} GiB requirement",
        m.free_gib_per_rank(ptp),
        m.min_free_gib_per_rank
    );
    assert!(m.fits_in_memory(dtp), "TP{dtp} decode does not fit");

    // And it should be on the shortlist, not merely feasible.
    let best = m
        .search(c.topology.total_gpus, &[2, 4, 8], &[2, 4, 8])
        .into_iter()
        .next()
        .expect("the search found no topology at all");
    assert_eq!(
        (best.prefill_tp, best.decode_tp),
        (ptp, dtp),
        "the model's best split is TP{}/TP{} and the launcher runs TP{ptp}/TP{dtp}",
        best.prefill_tp,
        best.decode_tp
    );
}
