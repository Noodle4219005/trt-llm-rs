//! Fails the build loudly if PyO3 resolved a *static* libpython.
//!
//! A static libpython here would put a second, independent CPython
//! interpreter into this process's address space alongside the one
//! `libtensorrt_llm.so` already links dynamically -- `libpython3.12.so.1.0`
//! is in its `NEEDED` list, so at runtime the process would end up with two
//! CPython instances that know nothing about each other, contending for
//! process-global interpreter state (the GIL, `sys.modules`, refcounts on
//! objects each side thinks it owns). That is undefined behaviour, not a
//! slow path, so this check runs at build time instead of being discovered
//! from a crash hours into a run.
//!
//! The fix, when this fires: build `trtllm-pyhost` INSIDE the
//! `tensorrtllm-runtime` container (its `libpython3.12.so.1.0` is
//! `Py_ENABLE_SHARED=1`), not against a pixi/conda static Python on the host.
//!
//! This guard is cheap insurance, not an expected failure: the container is
//! currently `shared=true`, so in normal operation it must never fire.

fn main() {
    let config = pyo3_build_config::get();
    if !config.shared() {
        panic!(
            "trtllm-pyhost: PyO3 resolved a STATIC libpython (interpreter: {:?}, \
             Python {}). A static libpython here would put a second CPython \
             interpreter in this process's address space alongside the one \
             `libtensorrt_llm.so` already links dynamically (`libpython3.12.so.1.0` is \
             in its NEEDED list) -- two interpreters sharing one process is undefined \
             behaviour, not a slow path. Fix: build this crate INSIDE the \
             tensorrtllm-runtime container (its libpython3.12 is Py_ENABLE_SHARED=1), \
             not against a pixi/conda static Python on the host.",
            config.executable(),
            config.version(),
        );
    }
}
