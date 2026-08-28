//! A small RFC 4180 reader.
//!
//! Deliberately dependency-free and header-addressed. AIConfigurator's column
//! names are not part of any contract we control, and this tree has never seen
//! a live install, so every lookup is a *pattern* match with an explicit
//! failure rather than an index into a row.

use std::collections::BTreeMap;

use trtllm_core::{Error, Result};

#[derive(Clone, Debug, Default)]
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn parse(text: &str) -> Result<Self> {
        let mut records = parse_records(text);
        if records.is_empty() {
            return Err(Error::Config("empty CSV".into()));
        }
        let header: Vec<String> = records
            .remove(0)
            .into_iter()
            .map(|c| c.trim().to_string())
            .collect();
        let rows = records
            .into_iter()
            .filter(|r| r.iter().any(|c| !c.trim().is_empty()))
            .collect();
        Ok(Self { header, rows })
    }

    /// Index of the first column whose lower-cased name contains every one of
    /// `needles`. Returns `None` rather than guessing.
    pub fn column(&self, needles: &[&str]) -> Option<usize> {
        self.header.iter().position(|h| {
            let h = h.to_ascii_lowercase();
            needles.iter().all(|n| h.contains(&n.to_ascii_lowercase()))
        })
    }

    pub fn cell<'a>(&self, row: &'a [String], needles: &[&str]) -> Option<&'a str> {
        self.column(needles)
            .and_then(|i| row.get(i))
            .map(|s| s.trim())
    }

    pub fn number(&self, row: &[String], needles: &[&str]) -> Option<f64> {
        self.cell(row, needles)
            .and_then(|s| s.replace(',', "").parse::<f64>().ok())
    }

    pub fn row_map(&self, row: &[String]) -> BTreeMap<String, String> {
        self.header
            .iter()
            .cloned()
            .zip(row.iter().cloned())
            .collect()
    }
}

fn parse_records(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            ',' if !in_quotes => record.push(std::mem::take(&mut field)),
            '\r' if !in_quotes => {}
            '\n' if !in_quotes => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_fields_and_embedded_commas_survive() {
        let t = Table::parse("a,b\n1,\"x,y\"\n").expect("parse");
        assert_eq!(t.header, vec!["a", "b"]);
        assert_eq!(t.rows[0][1], "x,y");
    }

    #[test]
    fn columns_are_found_by_pattern_not_position() {
        let t = Table::parse("(p)parallel,(d)parallel,tokens/s/gpu\ntp2pp1,tp8pp1,446.85\n")
            .expect("parse");
        assert_eq!(t.cell(&t.rows[0], &["(p)", "parallel"]), Some("tp2pp1"));
        assert_eq!(t.cell(&t.rows[0], &["(d)", "parallel"]), Some("tp8pp1"));
        assert_eq!(t.number(&t.rows[0], &["tokens/s/gpu"]), Some(446.85));
    }

    #[test]
    fn a_missing_column_is_none_not_a_wrong_answer() {
        let t = Table::parse("a,b\n1,2\n").expect("parse");
        assert!(t.column(&["ttft"]).is_none());
        assert!(t.number(&t.rows[0], &["ttft"]).is_none());
    }

    #[test]
    fn a_trailing_newline_does_not_produce_a_phantom_row() {
        let t = Table::parse("a\n1\n2\n").expect("parse");
        assert_eq!(t.rows.len(), 2);
    }
}
