//! Parameter tuning, with NVIDIA AIConfigurator in the loop.
//!
//! AIConfigurator is an analytical performance model over a database of
//! measured kernel timings. Given a model, a GPU system, a backend and an SLA
//! it searches parallelism layouts and replica counts and ranks them. That is a
//! genuinely good *generator*: it knows things about TensorRT-LLM kernel
//! performance that this repository does not, and it explores a space that is
//! tedious to sweep by hand.
//!
//! It is not a good *judge* for this deployment, for three specific reasons:
//!
//! 1. **It filters on TTFT and TPOT targets, not on the metric we are scored
//!    on.** Our score is the fraction of requests where TTFT <= 3000 ms *and*
//!    mean ITL <= 20 ms, and a layout that meets the mean can still fail the
//!    fraction. Nothing in an analytical throughput model sees that.
//! 2. **It does not model the scheduler.** Its own documentation says so: no
//!    request-by-request scheduling, no KV-cache behaviour, no queueing. Every
//!    lever this repository adds lives in exactly that gap.
//! 3. **Its estimates are only reliable inside its support matrix.** Outside
//!    it, it silently falls back to weaker modelling. Check with
//!    `aiconfigurator cli support` before believing a row.
//!
//! So the flow here is generator-then-judge:
//!
//! ```text
//! aiconfigurator  ->  candidate layouts (TP/PP, replica counts, GPU split)
//!        |
//!        v
//! trtllm-tuner    ->  cross-check each against our measured capacity model
//!        |             (disagreement is information, not an error)
//!        v
//! trtllm-sim      ->  score each under the real per-request goodput metric
//!        |
//!        v
//! one real run    ->  confirm the winner on hardware
//! ```
//!
//! Nothing here runs AIConfigurator for you by default; [`AicRun`] builds the
//! command line so it can be reviewed, and [`load_candidates`] reads a
//! `--save-dir` that already exists.

pub mod aic;
pub mod csv;
pub mod plan;

pub use aic::{AicCandidate, AicRun, DeploymentMode, ParallelSpec};
pub use plan::{
    CandidateEvaluator, CrossCheck, MeasuredRunEvaluator, SimulationEvaluator, TuningPlan,
    TuningRow,
};

use std::path::Path;

use trtllm_core::Result;

/// Read the candidate tables AIConfigurator wrote under `save_dir`.
///
/// The documented layout is `<save-dir>/{agg,disagg}/...`, but what 0.7.0
/// actually writes is nested two levels deeper, under the model id and a
/// generated run name:
///
/// ```text
/// <save-dir>/Qwen/Qwen3-235B-A22B-FP8_h200_sxm_trtllm_isl4000_osl200_ttft3000_tpot20_92874/
///     agg/{pareto.csv, best_config_topn.csv, exp_config.yaml, top1/…}
///     disagg/{pareto.csv, best_config_topn.csv, exp_config.yaml, top1/…}
/// ```
///
/// So this walks for `agg`/`disagg` directories rather than assuming where they
/// sit. Assuming cost a run: the search succeeded, the files were written, and
/// the loader reported "no candidate rows found".
///
/// Column names are matched by pattern rather than by position, because they
/// are AIConfigurator's to change - see [`csv::Table::column`].
pub fn load_candidates(save_dir: &Path) -> Result<Vec<AicCandidate>> {
    let mut out = Vec::new();
    for (dir, mode) in find_result_dirs(save_dir, 0) {
        for name in ["best_config_topn.csv", "pareto.csv"] {
            let path = dir.join(name);
            if !path.exists() {
                continue;
            }
            let text = std::fs::read_to_string(&path)
                .map_err(|e| trtllm_core::Error::Config(format!("{}: {e}", path.display())))?;
            let table = csv::Table::parse(&text)?;
            out.extend(aic::candidates_from_table(
                &table,
                mode,
                &path.display().to_string(),
            ));
            break;
        }
    }
    Ok(out)
}

/// Depth-limited search for `agg` / `disagg` result directories.
fn find_result_dirs(root: &Path, depth: usize) -> Vec<(std::path::PathBuf, DeploymentMode)> {
    const MAX_DEPTH: usize = 5;
    let mut found = Vec::new();
    if depth > MAX_DEPTH {
        return found;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return found;
    };
    let mut children: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    // Deterministic order so two runs over the same tree rank identically.
    children.sort();
    for child in children {
        match child.file_name().and_then(|n| n.to_str()) {
            Some("agg") => found.push((child, DeploymentMode::Agg)),
            Some("disagg") => found.push((child, DeploymentMode::Disagg)),
            // `top1/` holds the generated deployment YAML, not a candidate table.
            Some("top1") => {}
            _ => found.extend(find_result_dirs(&child, depth + 1)),
        }
    }
    found
}
