//! Feature-gated Dynamo Worker entrypoint for the pinned Python TRT-LLM worker.
//!
//! Required environment: TRTLLM_WORKER_URL and TRTLLM_MODEL. Optional values
//! are TRTLLM_SERVED_MODEL and TRTLLM_DEFAULT_MAX_TOKENS.

#[cfg(feature = "dynamo-v1")]
fn main() -> anyhow::Result<()> {
    use std::env;

    let worker_url = required_env("TRTLLM_WORKER_URL")?;
    let model = required_env("TRTLLM_MODEL")?;
    let default_max_tokens = env::var("TRTLLM_DEFAULT_MAX_TOKENS")
        .unwrap_or_else(|_| "200".into())
        .parse::<u32>()
        .map_err(|error| anyhow::anyhow!("TRTLLM_DEFAULT_MAX_TOKENS must be a u32: {error}"))?;
    let config = trtllm_dynamo::DynamoEngineConfig {
        model,
        served_model_name: env::var("TRTLLM_SERVED_MODEL").ok(),
        default_max_tokens,
    };

    trtllm_dynamo::run_with_factory(trtllm_dynamo::HttpTransportFactory::new(worker_url), config)
}

#[cfg(feature = "dynamo-v1")]
fn required_env(name: &str) -> anyhow::Result<String> {
    std::env::var(name).map_err(|_| anyhow::anyhow!("{name} is required"))
}

#[cfg(not(feature = "dynamo-v1"))]
fn main() {
    eprintln!("enable --features dynamo-v1 to run the Dynamo Worker entrypoint");
}
