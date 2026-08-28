//! `trt-llm-rs` - the command line.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use trtllm_core::config::{Config, PrefillPolicy};
use trtllm_engine::cost::{DecodeCurve, PrefillCurve};
use trtllm_engine::mock::{mock_decode_worker, mock_prefill_worker, TimeMode};
use trtllm_engine::{CostModel, Engine};
use trtllm_frontend::{serve, server::AppState};
use trtllm_sim::{SimSetup, Simulator};
use trtllm_tuner::{AicRun, TuningPlan};
use trtllm_worker::Deployment;

#[derive(Parser)]
#[command(
    name = "trt-llm-rs",
    about = "A Rust control plane for disaggregated LLM serving",
    version
)]
struct Cli {
    /// Deployment config; defaults are the 4P1D Qwen3 setup.
    #[arg(long, short, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the default configuration as TOML.
    Config,
    /// Enumerate prefill/decode topologies and rank them by modelled goodput.
    Plan {
        #[arg(long, default_value_t = 16)]
        total_gpus: u32,
        #[arg(long, value_delimiter = ',', default_value = "2,4,8")]
        prefill_tp: Vec<u32>,
        #[arg(long, value_delimiter = ',', default_value = "4,8")]
        decode_tp: Vec<u32>,
        #[arg(long, default_value_t = 12)]
        top: usize,
    },
    /// Run the deployment in simulation and print the scored result.
    Sim {
        #[arg(long)]
        concurrency: Option<u32>,
        /// Override the prefill tensor parallel degree.
        #[arg(long)]
        prefill_tp: Option<u32>,
        /// Override the number of prefill workers.
        #[arg(long)]
        prefill_workers: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// Sweep one dimension in simulation.
    Sweep {
        #[arg(long, value_delimiter = ',', default_value = "32,48,64,80,96,128")]
        concurrency: Vec<u32>,
        #[arg(long, value_delimiter = ',', default_value = "moore-hodgson,fcfs,edf")]
        policy: Vec<String>,
    },
    /// AIConfigurator in the loop: print the commands, or score a save-dir.
    Tune {
        /// An existing `aiconfigurator --save-dir`. Without it, only the
        /// commands to produce one are printed.
        #[arg(long)]
        save_dir: Option<PathBuf>,
        #[arg(long, default_value = "h200_sxm")]
        system: String,
        #[arg(long, default_value = "trtllm")]
        backend: String,
        #[arg(long)]
        json: bool,
    },
    /// Serve on HTTP with mock engines - the whole control plane, no GPUs.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8000")]
        addr: String,
    },
}

fn load_config(path: Option<&PathBuf>) -> Result<Config> {
    match path {
        Some(p) => Config::load(p).with_context(|| format!("loading {}", p.display())),
        None => Ok(Config::default()),
    }
}

fn parse_policy(s: &str) -> Result<PrefillPolicy> {
    match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "fcfs" => Ok(PrefillPolicy::Fcfs),
        "edf" => Ok(PrefillPolicy::Edf),
        "moore-hodgson" | "mh" => Ok(PrefillPolicy::MooreHodgson),
        other => anyhow::bail!("unknown prefill policy {other:?}"),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let cfg = load_config(cli.config.as_ref())?;

    match cli.command {
        Command::Config => {
            println!("{}", toml::to_string_pretty(&cfg)?);
        }
        Command::Plan {
            total_gpus,
            prefill_tp,
            decode_tp,
            top,
        } => cmd_plan(&cfg, total_gpus, &prefill_tp, &decode_tp, top),
        Command::Sim {
            concurrency,
            prefill_tp,
            prefill_workers,
            json,
        } => cmd_sim(cfg, concurrency, prefill_tp, prefill_workers, json)?,
        Command::Sweep {
            concurrency,
            policy,
        } => cmd_sweep(cfg, &concurrency, &policy)?,
        Command::Tune {
            save_dir,
            system,
            backend,
            json,
        } => cmd_tune(cfg, save_dir.as_deref(), &system, &backend, json)?,
        Command::Serve { addr } => cmd_serve(cfg, &addr)?,
    }
    Ok(())
}

fn cmd_plan(cfg: &Config, total_gpus: u32, prefill_tp: &[u32], decode_tp: &[u32], top: usize) {
    let model = cfg.capacity_model();
    println!(
        "Qwen-shape capacity model  ISL {}  OSL {}  TTFT<={:.0}ms  ITL<={:.0}ms  assumed good_frac {:.2}",
        model.isl, model.osl, model.slo.ttft_ms, model.slo.itl_ms, model.good_frac
    );
    println!(
        "prefill calibration {:.0} tok/s/GPU at TP{}, {:.0}% of GPU time in TP all-reduce",
        model.prefill.tok_s_per_gpu,
        model.prefill.tp_ref,
        model.prefill.tp_allreduce_frac * 100.0
    );
    println!();
    println!(
        "{:>5} {:>5} {:>7} {:>7} {:>9} {:>12} {:>11} {:>10}",
        "P gpu", "D gpu", "P tp", "D tp", "P workers", "prefill r/s", "decode r/s", "goodput"
    );
    for s in model
        .search(total_gpus, prefill_tp, decode_tp)
        .into_iter()
        .take(top)
    {
        println!(
            "{:>5} {:>5} {:>7} {:>7} {:>9} {:>12.2} {:>11.2} {:>10.2}",
            s.prefill_gpus,
            s.decode_gpus,
            s.prefill_tp,
            s.decode_tp,
            s.prefill_workers,
            s.prefill_req_s,
            s.decode_req_s,
            s.goodput_req_s
        );
    }
    println!();
    println!(
        "Note: the decode column extrapolates an ITL-versus-concurrency curve fitted to ONE \n\
         saturated measurement. Treat it as a shortlist, not a prediction - run `sim` next."
    );
}

fn cmd_sim(
    mut cfg: Config,
    concurrency: Option<u32>,
    prefill_tp: Option<u32>,
    prefill_workers: Option<u32>,
    json: bool,
) -> Result<()> {
    if let Some(n) = concurrency {
        cfg.workload.concurrency = n;
    }
    if let Some(tp) = prefill_tp {
        cfg.topology.prefill_tp = tp;
    }
    if let Some(w) = prefill_workers {
        cfg.topology.prefill_workers = w;
    }
    cfg.validate()?;
    let t = cfg.topology;
    let report = Simulator::new(SimSetup { config: cfg }).run();
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "topology: {}x prefill TP{} + {}x decode TP{}  ({} GPUs)",
            t.prefill_workers,
            t.prefill_tp,
            t.decode_workers,
            t.decode_tp,
            t.prefill_workers * t.prefill_tp + t.decode_workers * t.decode_tp
        );
        println!("{}", report.summary());
        println!();
        println!("{:#?}", report.diagnostics);
        for c in report.caveats() {
            println!("\nCAVEAT: {c}");
        }
    }
    Ok(())
}

fn cmd_sweep(cfg: Config, concurrency: &[u32], policies: &[String]) -> Result<()> {
    println!(
        "{:<16} {:>5} {:>10} {:>8} {:>10} {:>10} {:>8}",
        "policy", "N", "goodput", "good%", "TTFT p99", "ITL mean", "batch"
    );
    for p in policies {
        let policy = parse_policy(p)?;
        for &n in concurrency {
            let mut c = cfg.clone();
            c.workload.concurrency = n;
            c.scheduler.prefill_policy = policy;
            c.validate()?;
            let r = Simulator::new(SimSetup { config: c }).run();
            println!(
                "{:<16} {:>5} {:>10.2} {:>7.1}% {:>9.0}ms {:>9.2}ms {:>8.2}",
                p,
                n,
                r.goodput.goodput_req_s,
                r.goodput.good_frac * 100.0,
                r.goodput.ttft.p99,
                r.goodput.itl.mean,
                r.diagnostics.mean_prefill_batch_seqs
            );
        }
    }
    Ok(())
}

fn cmd_tune(
    cfg: Config,
    save_dir: Option<&std::path::Path>,
    system: &str,
    backend: &str,
    json: bool,
) -> Result<()> {
    let out_dir = save_dir
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "./aic-results".to_string());
    let run = AicRun::from_config(&cfg, system, backend, &out_dir);

    println!("# 1. confirm the combination is inside AIConfigurator's support matrix.");
    println!("#    Outside it, AIConfigurator falls back to weaker modelling *silently*.");
    println!("{}", run.shell(&run.support_command()));
    println!();
    println!("# 2. search the layout space.");
    println!("{}", run.shell(&run.search_command()));
    println!();

    let Some(dir) = save_dir else {
        println!("# 3. re-run this command with --save-dir {out_dir} to score the candidates");
        println!("#    under the metric we are actually judged on (per-request good_frac),");
        println!("#    which AIConfigurator does not model.");
        return Ok(());
    };

    let candidates = trtllm_tuner::load_candidates(dir)?;
    if candidates.is_empty() {
        anyhow::bail!(
            "no candidate rows found under {} - expected {{agg,disagg}}/best_config_topn.csv or pareto.csv",
            dir.display()
        );
    }
    println!(
        "# 3. {} candidate layouts, scored in simulation:",
        candidates.len()
    );
    let plan = TuningPlan::evaluate(&candidates, &cfg);
    if json {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else {
        println!("{}", plan.table());
        if let Some(best) = plan.best() {
            println!("best: {}", best.label);
            println!("  {}", best.sim.summary());
            println!(
                "  cross-check vs AIConfigurator: {} - {}",
                best.cross_check.verdict, best.cross_check.note
            );
            for c in best.sim.caveats() {
                println!("  CAVEAT: {c}");
            }
        }
    }
    Ok(())
}

fn cmd_serve(cfg: Config, addr: &str) -> Result<()> {
    cfg.validate()?;
    let addr: std::net::SocketAddr = addr.parse().context("parsing --addr")?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let t = cfg.topology;
        let prefill_curve = PrefillCurve::for_worker(&cfg.calibration.prefill, t.prefill_tp, t.prefill_tp);
        let decode_curve = DecodeCurve::from_calibration(&cfg.calibration.decode, t.decode_tp);
        let cost = CostModel::new(prefill_curve, decode_curve, 0.0);

        let prefill: Vec<Arc<dyn Engine>> = (0..t.prefill_workers)
            .map(|i| {
                let gpus = (0..t.prefill_tp).map(|g| i * t.prefill_tp + g).collect();
                Arc::new(mock_prefill_worker(&cfg.model.name, gpus, t.prefill_tp, cost, TimeMode::Wall))
                    as Arc<dyn Engine>
            })
            .collect();
        let decode: Vec<Arc<dyn Engine>> = (0..t.decode_workers)
            .map(|j| {
                let base = t.prefill_workers * t.prefill_tp + j * t.decode_tp;
                let gpus = (0..t.decode_tp).map(|g| base + g).collect();
                Arc::new(mock_decode_worker(
                    &cfg.model.name,
                    gpus,
                    t.decode_tp,
                    cfg.kv.num_blocks,
                    cost,
                    TimeMode::Wall,
                )) as Arc<dyn Engine>
            })
            .collect();

        let handle = Deployment::spawn(cfg, prefill, decode, None)?;
        tracing::warn!(
            "serving with MOCK engines: latencies come from the calibrated cost model, not from a GPU"
        );
        let state = Arc::new(AppState::new(handle.deployment.clone()));
        serve(state, addr).await?;
        anyhow::Ok(())
    })
}
