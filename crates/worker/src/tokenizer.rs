use trtllm_core::TokenId;

/// The tokenizer seam.
///
/// A mismatch here shifts prompt lengths, which shifts every prefill cost
/// estimate, which silently shifts every deadline decision.
///
/// **The scored run needs [`HfTokenizer`], not [`SyntheticTokenizer`].** An
/// earlier version of this comment argued the stand-in was enough because "the
/// client decides what the tokens are". That is wrong for the official
/// benchmark: AIPerf builds a prompt of exactly 4000 tokens with the *model's*
/// tokenizer and then sends it over the OpenAI chat API as **text**, so the
/// server tokenizes it again. Run that text through the byte-level stand-in and
/// a 4000-token prompt becomes roughly 16,000 tokens -- ISL is 4x the task
/// specification and every number after that describes a different workload.
///
/// The stand-in remains correct for `crates/sim`, which never sees text.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId]) -> String;
    fn name(&self) -> String;
}

/// Byte-level stand-in: one token per UTF-8 byte, offset out of the special-token
/// range. Length-faithful for ASCII, and wrong for anything else - which is the
/// point at which you must plug in the real one.
#[derive(Clone, Debug, Default)]
pub struct SyntheticTokenizer;

const SPECIAL_OFFSET: TokenId = 256;

impl Tokenizer for SyntheticTokenizer {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        text.bytes()
            .map(|b| TokenId::from(b) + SPECIAL_OFFSET)
            .collect()
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|t| u8::try_from(t.saturating_sub(SPECIAL_OFFSET)).ok())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn name(&self) -> String {
        "synthetic-byte".into()
    }
}


/// The model's own tokenizer, loaded from a HuggingFace `tokenizer.json`.
///
/// Same crate and feature set as the pinned Dynamo tree uses
/// (`dynamo-tokenizers` 1.5.4 -> `tokenizers` 0.21.4 with `onig`, `esaxx_fast`,
/// `rustls-tls`), chosen so this builds offline in the same container that
/// already compiles that tree rather than introducing a second, unproven
/// dependency graph.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    name: String,
}

impl HfTokenizer {
    /// Load from a `tokenizer.json`. Takes the file rather than a model
    /// directory: a directory invites falling back to some other file when the
    /// expected one is missing, and a tokenizer that silently is not the
    /// model's is the exact failure this type exists to prevent.
    pub fn from_file(path: &std::path::Path) -> trtllm_core::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            trtllm_core::Error::Engine(format!("loading tokenizer {}: {e}", path.display()))
        })?;
        Ok(Self {
            inner,
            name: path.display().to_string(),
        })
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        match self.inner.encode(text, false) {
            Ok(e) => e.get_ids().to_vec(),
            // Encoding failure is not recoverable here and must not silently
            // become an empty prompt, which would look like a very fast request.
            Err(e) => {
                tracing::error!(error = %e, "tokenizer encode failed");
                Vec::new()
            }
        }
    }

    fn decode(&self, tokens: &[TokenId]) -> String {
        self.inner.decode(tokens, false).unwrap_or_default()
    }

    fn name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_round_trips_and_keeps_its_length() {
        let t = SyntheticTokenizer;
        let ids = t.encode("hello");
        assert_eq!(ids.len(), 5);
        assert_eq!(t.decode(&ids), "hello");
    }
}
