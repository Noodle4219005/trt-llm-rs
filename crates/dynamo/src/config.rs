//! Dynamo Worker configuration.

use dynamo_backend_common::{ModelInput, WorkerConfig};

#[derive(Clone, Debug)]
pub struct DynamoEngineConfig {
    pub model: String,
    pub served_model_name: Option<String>,
    pub default_max_tokens: u32,
}

impl DynamoEngineConfig {
    pub fn worker_config(&self) -> WorkerConfig {
        WorkerConfig {
            model_name: self.model.clone(),
            served_model_name: self.served_model_name.clone(),
            model_input: ModelInput::Tokens,
            ..Default::default()
        }
    }
}

impl Default for DynamoEngineConfig {
    fn default() -> Self {
        Self {
            model: "trtllm".into(),
            served_model_name: None,
            default_max_tokens: 200,
        }
    }
}
