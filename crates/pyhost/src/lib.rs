//! Rust <-> embedded CPython control-plane boundary.
//!
//! This is a feasibility spike, not production code: can a Rust control plane
//! drive TensorRT-LLM's PyTorch backend through an embedded CPython? Everything
//! here is about the *boundary*, not about model quality, and [`crates/engine`]
//! deliberately does not depend on this crate -- the mock engine, the
//! simulator, and CI all stay Python-free and GPU-free.
//!
//! # One thread owns Python
//!
//! CPython has a single process-wide GIL. `libtensorrt_llm.so` links
//! `libpython3.12.so.1.0` directly (it is in the shared object's `NEEDED`
//! list) and torch's own C extensions do the same, so every real engine needs
//! a thread that holds the GIL for the duration of a Python call. [`PyHost`]
//! dedicates exactly one OS thread to this: it is the only thread that ever
//! calls into Python, so there is never contention for the GIL, and no
//! `tokio` task ever blocks a runtime worker on it -- an async caller only
//! ever awaits a `oneshot` channel.
//!
//! # Never `dlopen` `libtensorrt_llm.so` directly from Rust
//!
//! `libtensorrt_llm.so`'s `RUNPATH` points at a nonexistent build-machine
//! path, so it cannot find `libtorch.so` on its own. The only reason
//! `import tensorrt_llm` resolves at all is that `import torch` has *already*
//! mapped `libtorch.so` (and its siblings) into the process's address space
//! by the time `tensorrt_llm`'s own extension modules are loaded. See
//! [`initialize`] for the load-bearing import order this implies.

use std::ffi::CString;
use std::sync::mpsc as std_mpsc;
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::oneshot;

/// Commands the dedicated Python thread understands.
///
/// Each variant carries a `tokio::sync::oneshot::Sender` for its reply.
/// Commands travel over a plain `std::sync::mpsc::Sender` because the
/// Python thread is not async and never touches a `tokio` runtime; the
/// `oneshot` half is what lets an async caller `.await` a reply without
/// blocking a runtime worker.
enum Cmd {
    Eval {
        code: String,
        reply: oneshot::Sender<Result<i64>>,
    },
    /// Like `Eval`, but extracts the result as a `String` instead of an
    /// `i64`. Added for gates that read text (e.g. generated output) back
    /// across the boundary.
    EvalStr {
        code: String,
        reply: oneshot::Sender<Result<String>>,
    },
    /// Runs a Python *statement* (not just an expression) against the
    /// persistent module-level namespace, for effects rather than a value:
    /// constructing an `LLM`, launching CUDA work on a stream, stashing an
    /// object for a later crossing to see. See [`run_python_thread`] for why
    /// a later `Eval`/`Exec`/`EvalStr` on the same `PyHost` can still see
    /// what an earlier `Exec` assigned.
    Exec {
        code: String,
        reply: oneshot::Sender<Result<()>>,
    },
    Call {
        module: String,
        func: String,
        arg: i64,
        reply: oneshot::Sender<Result<i64>>,
    },
    Ping {
        reply: oneshot::Sender<Result<()>>,
    },
    Shutdown,
}

/// A live handle onto TensorRT-LLM's PyTorch backend, reached through one
/// dedicated OS thread that owns the interpreter for the lifetime of the
/// host. See the module docs for why it must be exactly one thread.
pub struct PyHost {
    cmd_tx: std_mpsc::Sender<Cmd>,
    join_handle: Option<JoinHandle<()>>,
    trtllm_version: Option<String>,
}

impl PyHost {
    /// Spawns the dedicated Python thread and blocks the caller until
    /// initialisation reports success or failure.
    ///
    /// Deliberately not `async`: starting the host needs no `tokio` runtime,
    /// and the wait is bounded by Python import time, not by anything that
    /// should contend with a runtime worker.
    ///
    /// `rank` selects the CUDA device (see [`initialize`]); `site_packages`,
    /// when given, is prepended to `sys.path` before any import, for
    /// environments where `tensorrt_llm` is not already importable.
    pub fn start(rank: usize, site_packages: Option<String>) -> Result<PyHost> {
        let (cmd_tx, cmd_rx) = std_mpsc::channel::<Cmd>();
        let (init_tx, init_rx) = std_mpsc::channel::<Result<Option<String>>>();

        let join_handle = std::thread::Builder::new()
            .name("pyhost".to_string())
            .spawn(move || run_python_thread(rank, site_packages, cmd_rx, init_tx))
            .context("failed to spawn the pyhost thread")?;

        let trtllm_version = match init_rx.recv() {
            Ok(result) => result?,
            Err(_) => {
                // The thread vanished (most likely panicked) before it could
                // report anything. Reap it so the default panic hook's
                // stderr message is not left dangling, then report clearly.
                let _ = join_handle.join();
                return Err(anyhow!(
                    "pyhost thread exited before completing initialisation \
                     (see stderr above for a panic message, if any)"
                ));
            }
        };

        Ok(PyHost {
            cmd_tx,
            join_handle: Some(join_handle),
            trtllm_version,
        })
    }

    /// The `tensorrt_llm.__version__` string read during initialisation.
    ///
    /// Always `Some` for a `PyHost` that started successfully: `start`
    /// returns `Err` outright if this step fails.
    pub fn trtllm_version(&self) -> Option<&str> {
        self.trtllm_version.as_deref()
    }

    /// Sends `cmd` to the Python thread and awaits its reply without ever
    /// blocking the calling task on the GIL -- only this channel is awaited.
    async fn call<T>(&self, make_cmd: impl FnOnce(oneshot::Sender<Result<T>>) -> Cmd) -> Result<T> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .send(make_cmd(reply_tx))
            .map_err(|_| anyhow!("pyhost worker thread has exited"))?;
        reply_rx
            .await
            .map_err(|_| anyhow!("pyhost worker thread dropped the reply channel"))?
    }

    /// Round-trips to the Python thread and back, acquiring and releasing
    /// the GIL but touching no Python object. Used both as a liveness check
    /// and, via [`crossing_overhead_ns`](Self::crossing_overhead_ns), as the
    /// unit of the crossing-cost benchmark.
    pub async fn ping(&self) -> Result<()> {
        self.call(|reply| Cmd::Ping { reply }).await
    }

    /// Evaluates a Python expression and extracts the result as an `i64`.
    pub async fn eval_int(&self, code: &str) -> Result<i64> {
        self.call(|reply| Cmd::Eval {
            code: code.to_string(),
            reply,
        })
        .await
    }

    /// Calls `module.func(arg)` and extracts the result as an `i64`.
    pub async fn call_int(&self, module: &str, func: &str, arg: i64) -> Result<i64> {
        self.call(|reply| Cmd::Call {
            module: module.to_string(),
            func: func.to_string(),
            arg,
            reply,
        })
        .await
    }

    /// Evaluates a Python expression and extracts the result as a `String`.
    pub async fn eval_str(&self, code: &str) -> Result<String> {
        self.call(|reply| Cmd::EvalStr {
            code: code.to_string(),
            reply,
        })
        .await
    }

    /// Runs a Python statement (or block of statements) for its side
    /// effects against the host's persistent module-level namespace.
    ///
    /// Unlike [`eval_int`](Self::eval_int)/[`eval_str`](Self::eval_str),
    /// which evaluate a single expression, `exec` accepts arbitrary
    /// statements (assignments, `import`, multi-line blocks) via Python's
    /// `exec()` semantics. A name bound here (e.g. `llm = tensorrt_llm.LLM(...)`)
    /// remains visible to a later `exec`/`eval_int`/`eval_str` call on the
    /// same `PyHost`, even though each call is a separate crossing -- see
    /// [`run_python_thread`] for why.
    pub async fn exec(&self, code: &str) -> Result<()> {
        self.call(|reply| Cmd::Exec {
            code: code.to_string(),
            reply,
        })
        .await
    }

    /// Benchmarks the Rust <-> Python crossing cost: times 10,000
    /// back-to-back [`ping`](Self::ping) round trips and returns the mean
    /// nanoseconds per crossing.
    ///
    /// This is a *measurement*, not a correctness check -- it is a gate
    /// value meant to be printed and read by a human, so this method
    /// intentionally carries no pass/fail threshold of its own.
    pub async fn crossing_overhead_ns(&self) -> Result<f64> {
        const ITERATIONS: u32 = 10_000;
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            self.ping().await?;
        }
        Ok(start.elapsed().as_nanos() as f64 / f64::from(ITERATIONS))
    }
}

impl Drop for PyHost {
    fn drop(&mut self) {
        // Best-effort: if the thread already exited, `send` fails and there
        // is nothing left to signal.
        let _ = self.cmd_tx.send(Cmd::Shutdown);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Runs on the dedicated OS thread for the lifetime of the [`PyHost`].
///
/// Holds the GIL only for the duration of each command (via
/// [`Python::attach`]); `cmd_rx.recv()` between commands holds no GIL, so
/// there is nothing here for the GIL to contend with even though this is
/// the only thread that ever touches it.
///
/// # Why state set in one crossing survives to a later one
///
/// `globals` is a single `PyDict` created once, right after `initialize`,
/// and held for the lifetime of this thread -- every `Eval`/`EvalStr`/`Exec`
/// below runs against this same dict as both globals and locals. The
/// *interpreter* itself is also process-global and never torn down between
/// commands. So a name an `Exec` binds (`llm = tensorrt_llm.LLM(...)`, or a
/// `torch.cuda.Stream` stashed in a module-level dict) is not copied
/// anywhere -- it is the same Python object, reachable from `globals`, that
/// a later command on this thread will find still bound. This is what lets
/// gate 4 create a stream/event in one crossing and observe it, correctly
/// ordered, from a later one.
fn run_python_thread(
    rank: usize,
    site_packages: Option<String>,
    cmd_rx: std_mpsc::Receiver<Cmd>,
    init_tx: std_mpsc::Sender<Result<Option<String>>>,
) {
    let trtllm_version = match initialize(rank, site_packages) {
        Ok(version) => version,
        Err(err) => {
            let _ = init_tx.send(Err(err));
            return;
        }
    };

    if init_tx.send(Ok(trtllm_version)).is_err() {
        // `PyHost::start` gave up waiting (its receiver was dropped);
        // nothing left to serve.
        return;
    }

    let globals: Py<PyDict> = Python::attach(|py| PyDict::new(py).unbind());

    for cmd in cmd_rx.iter() {
        match cmd {
            Cmd::Shutdown => break,
            Cmd::Ping { reply } => {
                let _ = reply.send(Ok(Python::attach(|_py| ())));
            }
            Cmd::Eval { code, reply } => {
                let _ = reply.send(Python::attach(|py| eval_expr(py, globals.bind(py), &code)));
            }
            Cmd::EvalStr { code, reply } => {
                let _ =
                    reply.send(Python::attach(|py| eval_str_expr(py, globals.bind(py), &code)));
            }
            Cmd::Exec { code, reply } => {
                let _ = reply.send(Python::attach(|py| exec_stmt(py, globals.bind(py), &code)));
            }
            Cmd::Call {
                module,
                func,
                arg,
                reply,
            } => {
                let _ = reply.send(Python::attach(|py| call_func(py, &module, &func, arg)));
            }
        }
    }
}

/// Initialises the embedded interpreter, in an order that is load-bearing:
///
/// 1. if `site_packages` is `Some`, prepend it to `sys.path`
/// 2. `import torch` -- this is what maps `libtorch.so` into the process
///    before anything downstream needs it (see the module docs)
/// 3. `torch.cuda.set_device(rank)`, but only if `torch.cuda.is_available()`;
///    when CUDA is absent this logs and skips rather than failing, so this
///    crate is testable on a machine with no GPU
/// 4. `import tensorrt_llm` -- resolves only because step 2 already happened
/// 5. read and return `tensorrt_llm.__version__`
///
/// Reordering steps 2 and 4 turns a working import into an `ImportError`
/// about a missing shared object that looks nothing like the actual cause,
/// because `libtensorrt_llm.so`'s `RUNPATH` points at a nonexistent
/// build-machine path and depends on `libtorch.so` already being mapped in.
fn initialize(rank: usize, site_packages: Option<String>) -> Result<Option<String>> {
    Python::attach(|py| {
        if let Some(site_packages) = site_packages {
            let sys = py.import("sys").context("import sys")?;
            sys.getattr("path")
                .context("sys.path")?
                .call_method1("insert", (0usize, site_packages))
                .context("sys.path.insert(0, site_packages)")?;
        }

        let torch = py.import("torch").context("import torch")?;

        let cuda = torch.getattr("cuda").context("torch.cuda")?;
        let cuda_available: bool = cuda
            .call_method0("is_available")
            .context("torch.cuda.is_available()")?
            .extract()
            .context("torch.cuda.is_available() did not return a bool")?;

        if cuda_available {
            cuda.call_method1("set_device", (rank,))
                .with_context(|| format!("torch.cuda.set_device({rank})"))?;
        } else {
            eprintln!(
                "pyhost: CUDA not available; skipping torch.cuda.set_device({rank}) \
                 (expected on a machine with no GPU)"
            );
        }

        let tensorrt_llm = py.import("tensorrt_llm").context("import tensorrt_llm")?;

        let version: String = tensorrt_llm
            .getattr("__version__")
            .context("tensorrt_llm.__version__")?
            .extract()
            .context("tensorrt_llm.__version__ was not a string")?;

        Ok(Some(version))
    })
}

/// Evaluates a Python expression and extracts the result as an `i64`.
///
/// Any `PyErr` -- including one raised by the expression itself, such as
/// `1/0` -- is converted to an `anyhow::Error` here, inside the `Python`
/// token's scope, and returned as `Err`. It never panics and never poisons
/// the worker thread: the `for cmd in cmd_rx.iter()` loop in
/// [`run_python_thread`] moves straight on to the next command regardless.
///
/// `globals` is the host's persistent namespace (see [`run_python_thread`]);
/// passing it as both globals and locals is what lets this expression see a
/// name an earlier `Exec` bound.
fn eval_expr(py: Python<'_>, globals: &Bound<'_, PyDict>, code: &str) -> Result<i64> {
    let c_code = CString::new(code)
        .with_context(|| format!("eval code contained an embedded NUL byte: {code:?}"))?;
    py.eval(&c_code, Some(globals), None)
        .context("Python::eval")?
        .extract()
        .context("eval result did not extract as i64")
}

/// Like [`eval_expr`], but extracts the result as a `String`.
fn eval_str_expr(py: Python<'_>, globals: &Bound<'_, PyDict>, code: &str) -> Result<String> {
    let c_code = CString::new(code)
        .with_context(|| format!("eval code contained an embedded NUL byte: {code:?}"))?;
    py.eval(&c_code, Some(globals), None)
        .context("Python::eval")?
        .extract()
        .context("eval result did not extract as String")
}

/// Runs a Python statement (or block of statements) against `globals` for
/// its side effects, via the same `PyErr`-to-`anyhow::Error` contract as
/// [`eval_expr`]. Passing `globals` as both globals and locals means a name
/// bound here (e.g. `llm = tensorrt_llm.LLM(...)`) lands in the same
/// persistent namespace a later `Eval`/`EvalStr`/`Exec` reads from.
fn exec_stmt(py: Python<'_>, globals: &Bound<'_, PyDict>, code: &str) -> Result<()> {
    let c_code = CString::new(code)
        .with_context(|| format!("exec code contained an embedded NUL byte: {code:?}"))?;
    py.run(&c_code, Some(globals), None).context("Python::run")?;
    Ok(())
}

/// Calls `module.func(arg)` and extracts the result as an `i64`. See
/// [`eval_expr`] for the `PyErr` handling contract this shares.
fn call_func(py: Python<'_>, module: &str, func: &str, arg: i64) -> Result<i64> {
    py.import(module)
        .with_context(|| format!("import {module}"))?
        .getattr(func)
        .with_context(|| format!("{module}.{func}"))?
        .call1((arg,))
        .with_context(|| format!("{module}.{func}({arg})"))?
        .extract()
        .with_context(|| format!("{module}.{func}({arg}) did not return an int"))
}
