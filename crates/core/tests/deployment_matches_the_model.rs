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

use trtllm_core::capacity::Role;
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
    // Each role against its own requirement. They differ by more than 2x,
    // and charging prefill for decode's residency is what made the model
    // reject the TP2 prefill that reached goodput 22.419 elsewhere.
    assert!(
        m.fits(ptp, Role::Prefill),
        "the launcher runs TP{ptp} prefill workers: {:.1} GiB free per rank \
         against {:.1} GiB needed",
        m.free_gib_per_rank(ptp),
        m.needed_gib_per_rank(ptp, Role::Prefill)
    );
    assert!(
        m.fits(dtp, Role::Decode),
        "the launcher runs TP{dtp} decode workers: {:.1} GiB free per rank \
         against {:.1} GiB needed",
        m.free_gib_per_rank(dtp),
        m.needed_gib_per_rank(dtp, Role::Decode)
    );

    // The launcher follows the measurement, and the model disagrees with it.
    //
    // A sweep of 18 configurations over 88 points on this model and hardware
    // (~/TODO_LLM_Wiki/problems/hpcai26-qwen/notes.md) put p2x4_d2x4 at N=80
    // first with goodput 12.42, against p4x2_d2x4's 9.54 at its own best N.
    // This model prefers narrower prefill workers, because
    // tok_s_per_gpu_at_tp(2) exceeds tok_s_per_gpu_at_tp(4) -- fewer ranks in
    // the all-reduce -- and that preference is not borne out end to end.
    //
    // The test asserts the launcher matches the measurement, and separately
    // that the disagreement is still there, so nobody can quietly tune the
    // model into agreement and call it a fix. Whatever the model is missing --
    // most likely the cost of four prefill workers' KV handoffs against two --
    // it is still missing it until someone finds and models the term.
    let best = m
        .search(c.topology.total_gpus, &[2, 4, 8], &[2, 4, 8])
        .into_iter()
        .next()
        .expect("the search found no topology at all");
    // 4 x TP2 prefill: what THIS stack measured. The team's SGLang sweep
    // prefers 2 x TP4 and we followed it once, which returned 0.66 req/s
    // against 13.70 and 13.74 for 4 x TP2 on the same nodes. A cross-stack
    // measurement does not outrank a same-stack one.
    assert_eq!((ptp, dtp), (2, 4), "the launcher must run 4 x TP2 prefill");

    // The model now prefers TP2 decode as well, and the launcher does not
    // follow it there. Dropping the concurrency from 128 to 80 shrank the
    // decode residency enough that TP2 fits -- 80/2 sequences at max_seq_len
    // 4608 is 8.3 GiB of fp8 KV per rank against 20.9 free -- so the memory
    // filter stopped excluding it and the narrow-worker preference took over.
    //
    // That preference has been right about prefill on this stack and is
    // untested for decode, and the one decode width we have measured is TP8,
    // which was bad for a reason specific to width: Qwen3-235B has four KV
    // heads, so TP8 duplicates them. TP2 would not have that problem, but
    // "would not have that problem" is not a measurement, and the last time I
    // changed a topology on reasoning rather than a run it cost 196 SU and
    // returned 0.66 req/s.
    //
    // So: assert what runs, and assert the disagreement is still open, so it
    // is not quietly closed by tuning either side.
    assert_ne!(
        (best.prefill_tp, best.decode_tp),
        (ptp, dtp),
        "the model now agrees about decode width. If TP2 decode was measured, \
         update the launcher and restore the equality assertion."
    );
}

/// Every knob the launcher reads must come from the config.
///
/// This replaces a spot check. The old test compared six values by name and
/// passed while twenty-two others drifted freely, which is the shape of a test
/// that makes a reviewer feel covered. A user tuning the TOML would find that
/// most of what they changed never reached the engine.
///
/// The launcher declares its knobs as `: "${X:=default}"`. Every one of those
/// must appear in `Config::to_env`, and every variable `to_env` emits must be
/// one the launcher actually reads -- an emitted variable nobody consumes is a
/// setting a user can change with no effect, which is worse than not offering
/// it.
#[test]
fn launcher_knobs_all_come_from_the_config() {
    let script = launcher();
    let env = Config::default().to_env();

    let declared: std::collections::BTreeSet<&str> = script
        .lines()
        .filter_map(|l| l.strip_prefix(": \"${"))
        .filter_map(|r| r.split(":=").next())
        .collect();
    let emitted: std::collections::BTreeSet<&str> = env
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split('=').next())
        .collect();

    let missing: Vec<_> = declared.difference(&emitted).collect();
    assert!(
        missing.is_empty(),
        "the launcher reads {:?}, which Config::to_env does not emit. A knob \
         on one side only is a setting a user cannot reach from the file they \
         were told to edit.",
        missing
    );

    let unread: Vec<_> = emitted.difference(&declared).collect();
    assert!(
        unread.is_empty(),
        "Config::to_env emits {:?}, which the launcher never declares. An \
         emitted variable nobody consumes is a setting that silently does \
         nothing.",
        unread
    );

    assert!(
        declared.len() >= 25,
        "only {} knobs found -- the parser probably stopped matching the \
         script's idiom rather than the script losing 25 settings",
        declared.len()
    );
}

/// The emitted file must let an explicit override win.
///
/// `X="${X:-value}"` keeps the layering a person expects: a variable set on the
/// sbatch command line beats the file, which beats the built-in default. A
/// plain `X=value` would silently reverse that and make every documented
/// override a no-op.
#[test]
fn an_explicit_override_still_beats_the_emitted_file() {
    let env = Config::default().to_env();
    for line in env.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
        let (name, rest) = line.split_once('=').expect("KEY=value");
        assert!(
            rest.starts_with(&format!("\"${{{name}:-")),
            "{line} does not preserve an existing value; it should read \
             {name}=\"${{{name}:-...}}\""
        );
    }
}

/// The committed deployment.env must be what the config produces today.
///
/// It is generated, sourced by the launcher, and checked in so a reader can see
/// the settings without running anything. All three of those are only true
/// while it is current: a stale generated file that is still being sourced is
/// worse than no file, because it looks authoritative.
#[test]
fn the_committed_deployment_env_is_current() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../deployment.env");
    let on_disk = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("{path}: {e}. Run `trt-llm-rs config --emit-env > deployment.env`")
    });
    assert_eq!(
        on_disk,
        Config::default().to_env(),
        "deployment.env is stale. Regenerate it: \
         `trt-llm-rs config --emit-env > deployment.env`"
    );
}

/// Turning speculation on in the config must change what the model predicts.
///
/// The launcher and the config agree by construction now, but the capacity
/// model is a third party and was reading its own `calibration.decode`. A
/// deployment running EAGLE3 while `plan` predicted without it would print a
/// number for a configuration nobody was running -- which is the failure the
/// launcher's own prediction banner exists to prevent.
#[test]
fn enabling_speculation_changes_the_prediction() {
    let plain = Config::default();
    assert!(!plain.engine.speculation.enabled, "the default is off");
    assert!(plain.capacity_model().decode.speculation.is_none());

    let mut spec = Config::default();
    spec.engine.speculation.enabled = true;
    let m = spec.capacity_model();
    let s = m
        .decode
        .speculation
        .expect("the model must follow the engine");
    assert_eq!(s.draft_tokens, spec.engine.speculation.draft_tokens);

    // And it must be worth something, or the plumbing is decorative.
    let before = plain.capacity_model().evaluate(8, 8, 2, 4).decode_req_s;
    let after = m.evaluate(8, 8, 2, 4).decode_req_s;
    assert!(
        after > before * 1.3,
        "speculation should lift decode capacity: {before:.2} -> {after:.2}"
    );
}
