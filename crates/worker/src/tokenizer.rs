use trtllm_core::TokenId;

/// The tokenizer seam.
///
/// **Nothing in this crate ships a real tokenizer.** The scored benchmark
/// generates synthetic prompts of an exact token count, and the client - not
/// the server - decides what those tokens are, so a length-faithful stand-in is
/// enough to exercise every scheduling path. Serving real traffic needs a
/// tokenizer that matches the model's exactly: a mismatch shifts prompt lengths,
/// which shifts every prefill cost estimate, which silently shifts every
/// deadline decision.
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
