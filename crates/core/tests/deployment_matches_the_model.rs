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
