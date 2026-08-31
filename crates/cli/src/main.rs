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
use trtllm_router::RouterTuning;
use trtllm_sim::{SimSetup, Simulator};
use trtllm_tuner::{AicRun, TuningPlan};
use trtllm_worker::serving::ServingDeployment;
use trtllm_worker::tokenizer::{HfTokenizer, Tokenizer};
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
    /// Compare a finished run's AIPerf export with what the model predicted
    /// for it. Six 16-GPU jobs were read against a model that was never
    /// printed beside a result.
    Verdict {
        /// Path to profile_export_aiperf.json.
        #[arg(long)]
        aiperf: std::path::PathBuf,
        /// Predicted goodput. Defaults to what the capacity model says for the
        /// configured topology.
        #[arg(long)]
        expect: Option<f64>,
        /// Fraction the measurement may fall short and still count as agreement.
        #[arg(long, default_value_t = 0.15)]
        tolerance: f64,
    },
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
    /// Serve on HTTP. With `--worker` this is the real control plane in front of
    /// real engine processes; without it, mock engines and no GPUs.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8000")]
        addr: String,
        /// Engine endpoint, repeatable: `--worker http://host:8200`. Each is one
        /// `trtllm-serve`-style process exposing `POST /generate` with SSE.
        /// Given any, the mock engines are not used at all.
        #[arg(long = "worker", value_delimiter = ',')]
        workers: Vec<String>,
        /// Path to the model's `tokenizer.json`. Required with `--worker`:
        /// AIPerf sends text, the frontend tokenizes it, and the byte-level
        /// stand-in would quadruple ISL.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Model name to advertise on `/v1/models`.
        #[arg(long)]
        served_model_name: Option<String>,
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
        Command::Verdict {
            aiperf,
            expect,
            tolerance,
        } => {
            let text = std::fs::read_to_string(&aiperf)
                .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", aiperf.display()))?;
            let doc: serde_json::Value = serde_json::from_str(&text)?;
            let measured = trtllm_core::verdict::MeasuredRun::from_aiperf_json(&doc)
                .map_err(|e| anyhow::anyhow!("{}: {e}", aiperf.display()))?;

            let model = cfg.capacity_model();
            let split = model.evaluate(
                cfg.topology.prefill_workers * cfg.topology.prefill_tp,
                cfg.topology.decode_workers * cfg.topology.decode_tp,
                cfg.topology.prefill_tp,
                cfg.topology.decode_tp,
            );
            let predicted = expect.unwrap_or(split.goodput_req_s);

            let v = trtllm_core::verdict::Verdict::assess(
                measured,
                predicted,
                &cfg.slo,
                split.bottleneck,
                tolerance,
            );
            println!("{}", v.summary());
            println!();
            println!(
                "  goodput      {:>8.2} measured   {:>8.2} predicted",
                v.measured.goodput_req_s, v.predicted_goodput_req_s
            );
            println!(
                "  throughput   {:>8.2} req/s      {:>8.0} tok/s output",
                v.measured.request_throughput_req_s, v.measured.output_token_throughput
            );
            println!(
                "  TTFT         {:>8.0} avg ms     {:>8.0} p90 ms   budget {:.0}",
                v.measured.ttft_avg_ms, v.measured.ttft_p90_ms, cfg.slo.ttft_ms
            );
            println!(
                "  ITL          {:>8.1} avg ms                  budget {:.0}",
                v.measured.itl_avg_ms, cfg.slo.itl_ms
            );
            println!("  requests     {:>8.0}", v.measured.request_count);
        }
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
        Command::Serve {
            addr,
            workers,
            tokenizer,
            served_model_name,
        } => cmd_serve(
            cfg,
            &addr,
            &workers,
            tokenizer.as_deref(),
            served_model_name,
        )?,
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
    println!(
        "                    {:.0} TFLOP/s of {:.0} peak = MFU {:.1}%. \
         Reaching 35% would be {:.2}x this prefill throughput,",
        model.prefill.achieved_tflops_per_gpu(),
        model.prefill.peak_tflops_per_gpu,
        model.prefill.mfu() * 100.0,
        0.35 / model.prefill.mfu().max(1e-9)
    );
    let b = model.prefill.mfu_breakdown();
    let (lever, worth) = b.largest_lever(0.50);
    println!(
        "                    of which {:.0}% of wall time is outside the forward pass \
         ({:.2}x), {:.0}% of kernel",
        (1.0 - model.prefill.duty_cycle) * 100.0,
        b.duty_cycle_worth,
        model.prefill.tp_allreduce_frac * 100.0
    );
    println!(
        "                    time is TP all-reduce ({:.2}x), and the remaining compute \
         runs at {:.1}% MFU ({:.2}x to 50%).",
        b.allreduce_worth,
        b.compute_mfu * 100.0,
        b.compute_mfu_worth(0.50)
    );
    println!(
        "                    Largest lever: {lever} at {worth:.2}x. This is a kernel \
         question, not a topology one -"
    );
    println!("                    no P/D split changes MFU.");
    println!();
    println!(
        "{:>5} {:>5} {:>7} {:>7} {:>9} {:>12} {:>11} {:>10} {:>10} {:<11} {:>8}",
        "P gpu",
        "D gpu",
        "P tp",
        "D tp",
        "P workers",
        "prefill r/s",
        "decode r/s",
        "xfer r/s",
        "goodput",
        "binds on",
        "headroom"
    );
    for s in model
        .search(total_gpus, prefill_tp, decode_tp)
        .into_iter()
        .take(top)
    {
        println!(
            "{:>5} {:>5} {:>7} {:>7} {:>9} {:>12.2} {:>11.2} {:>10.2} {:>10.2} {:<11} {:>7.1}x",
            s.prefill_gpus,
            s.decode_gpus,
            s.prefill_tp,
            s.decode_tp,
            s.prefill_workers,
            s.prefill_req_s,
            s.decode_req_s,
            s.transfer_req_s,
            s.goodput_req_s,
            format!("{:?}", s.bottleneck),
            s.headroom_ratio
        );
    }
    // What to do about the constraint the best split reports. Naming a
    // bottleneck without naming a knob leaves the reader to guess, and the
    // guesses in this project have gone to the wrong lever more than once.
    if let Some(best) = model
        .search(total_gpus, prefill_tp, decode_tp)
        .into_iter()
        .next()
    {
        println!();
        let remedies = model.remedies(&best);
        if remedies.is_empty() {
            println!(
                "Best split binds on {:?}, and every remedy this model knows \
                 is already applied.",
                best.bottleneck
            );
        } else {
            println!(
                "Best split binds on {:?}. Ordered by how well each is backed, \
                 not by size:",
                best.bottleneck
            );
        }
        for r in remedies {
            let mult = if (r.multiplier - 1.0).abs() < 1e-9 {
                "unblocks".to_string()
            } else {
                format!("{:.2}x", r.multiplier)
            };
            println!(
                "  {:<12} {:>9}  {}={}",
                r.evidence.label(),
                mult,
                r.knob,
                r.setting
            );
            println!("               {}", r.because);
            println!(
                "               {}",
                r.command("scripts/stage-d-235b-disagg.sbatch")
            );
        }
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
        println!("#    ranked by official score: good_output_tok_s (good output tokens from good requests per benchmark window);");
        println!("#    good_frac is only the passing-request fraction, not the ranked metric.");
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
            println!(
                "  official score: {:.2} good output tokens/s",
                best.score.good_output_tok_s
            );
            println!(
                "  cross-check vs AIConfigurator: {} - {}",
                best.cross_check.verdict, best.cross_check.note
            );
        }
    }
    Ok(())
}

fn cmd_serve(
    cfg: Config,
    addr: &str,
    workers: &[String],
    tokenizer: Option<&std::path::Path>,
    served_model_name: Option<String>,
) -> Result<()> {
    cfg.validate()?;
    let addr: std::net::SocketAddr = addr.parse().context("parsing --addr")?;
    if !workers.is_empty() {
        return cmd_serve_real(cfg, addr, workers, tokenizer, served_model_name);
    }
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

/// The real control plane: Rust frontend and router in front of engine
/// processes. No mock engines are constructed on this path, deliberately -- a
/// fallback to mocks when a worker is unreachable would produce a plausible
/// number from no GPU at all.
fn cmd_serve_real(
    cfg: Config,
    addr: std::net::SocketAddr,
    workers: &[String],
    tokenizer: Option<&std::path::Path>,
    served_model_name: Option<String>,
) -> Result<()> {
    let tokenizer_path = tokenizer.context(
        "--tokenizer is required with --worker: AIPerf sends text, the frontend \
         tokenizes it, and the byte-level stand-in would quadruple ISL",
    )?;
    let tok = Arc::new(HfTokenizer::from_file(tokenizer_path)?) as Arc<dyn Tokenizer>;
    let model_name = served_model_name.unwrap_or_else(|| cfg.model.name.clone());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let deployment = Arc::new(ServingDeployment::aggregated(
            workers,
            model_name,
            tok,
            // Aggregated: prefill and decode are the same process, so the
            // default 10 ms KV handoff term would price a transfer that never
            // happens. It biases every prediction identically and so does not
            // change the ranking, but a predicted TTFT that is knowably 10 ms
            // wrong is not worth keeping.
            RouterTuning {
                kv_transfer_ms: 0.0,
                ..RouterTuning::default()
            },
            cfg.slo.ttft_ms,
            f64::from(cfg.kv.num_blocks.min(4096)),
            STALE_AFTER_MS,
        )?);
        tracing::info!(
            workers = workers.len(),
            tokenizer = %tokenizer_path.display(),
            "serving with REAL engines"
        );
        let state = Arc::new(AppState::new(deployment));
        serve(state, addr).await?;
        anyhow::Ok(())
    })
}

/// A worker that has not been heard from for this long is not routed to. Sized
/// well above one decode step so a busy worker is never mistaken for a dead one.
const STALE_AFTER_MS: f64 = 5_000.0;
