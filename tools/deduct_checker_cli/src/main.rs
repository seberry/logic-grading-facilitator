use std::io::{self, Read};
use std::process;

use deduct::{
    normalize_ops, Checker, Proof, SYSTEM_K, SYSTEM_S4, SYSTEM_S5, SYSTEM_T, TFL_BASIC,
    TFL_DERIVED,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct RequestLine {
    depth: u16,
    sentence: String,
    citation: String,
}

#[derive(Deserialize)]
struct CheckRequest {
    system: String,
    #[serde(default)]
    include_derived_rules: bool,
    conclusion: Option<String>,
    lines: Vec<RequestLine>,
}

#[derive(Serialize)]
struct IssueResponse {
    line: u16,
    code: String,
    message: String,
}

#[derive(Serialize)]
struct NormalizedLineResponse {
    depth: u16,
    sentence: String,
    citation: String,
}

#[derive(Serialize)]
struct ResponseMetadata {
    system: String,
    include_derived_rules: bool,
    checker_version: String,
}

#[derive(Serialize)]
struct CheckResponse {
    ok: bool,
    parse_errors: Vec<IssueResponse>,
    check_errors: Vec<IssueResponse>,
    conclusion_reached: Option<bool>,
    normalized_lines: Vec<NormalizedLineResponse>,
    metadata: ResponseMetadata,
}

fn build_checker(system: &str, include_derived_rules: bool) -> Result<Checker, String> {
    let mut checker = Checker::new();
    checker.add_ruleset(TFL_BASIC);
    if include_derived_rules {
        checker.add_ruleset(TFL_DERIVED);
    }

    match system {
        "K" => {
            checker.add_ruleset(SYSTEM_K);
        }
        "T" => {
            checker.add_ruleset(SYSTEM_K);
            checker.add_ruleset(SYSTEM_T);
        }
        "S4" => {
            checker.add_ruleset(SYSTEM_K);
            checker.add_ruleset(SYSTEM_T);
            checker.add_ruleset(SYSTEM_S4);
        }
        "S5" => {
            checker.add_ruleset(SYSTEM_K);
            checker.add_ruleset(SYSTEM_T);
            checker.add_ruleset(SYSTEM_S5);
        }
        _ => {
            return Err(format!("unsupported modal system '{system}'"));
        }
    }

    Ok(checker)
}

fn run() -> Result<(), String> {
    let mut stdin = String::new();
    io::stdin()
        .read_to_string(&mut stdin)
        .map_err(|err| format!("failed to read stdin: {err}"))?;

    let request: CheckRequest =
        serde_json::from_str(&stdin).map_err(|err| format!("invalid request JSON: {err}"))?;

    let checker = build_checker(&request.system, request.include_derived_rules)?;

    let tuples: Vec<(u16, &str, &str)> = request
        .lines
        .iter()
        .map(|line| (line.depth, line.sentence.as_str(), line.citation.as_str()))
        .collect();

    let normalized_lines: Vec<NormalizedLineResponse> = request
        .lines
        .iter()
        .map(|line| NormalizedLineResponse {
            depth: line.depth,
            sentence: normalize_ops(&line.sentence),
            citation: line.citation.clone(),
        })
        .collect();

    let metadata = ResponseMetadata {
        system: request.system.clone(),
        include_derived_rules: request.include_derived_rules,
        checker_version: format!("deduct_checker_cli/{}", env!("CARGO_PKG_VERSION")),
    };

    let proof = match Proof::parse(tuples.as_slice()) {
        Ok(proof) => proof,
        Err(errors) => {
            let response = CheckResponse {
                ok: true,
                parse_errors: errors
                    .into_iter()
                    .map(|(line, error)| IssueResponse {
                        line,
                        code: format!("{error:?}"),
                        message: error.to_string(),
                    })
                    .collect(),
                check_errors: vec![],
                conclusion_reached: None,
                normalized_lines,
                metadata,
            };
            let json = serde_json::to_string(&response)
                .map_err(|err| format!("failed to serialize parse error response: {err}"))?;
            println!("{json}");
            return Ok(());
        }
    };

    let check_errors = match checker.check_proof(&proof) {
        Ok(()) => vec![],
        Err(errors) => errors
            .into_iter()
            .map(|(line, error)| IssueResponse {
                line,
                code: format!("{error:?}"),
                message: error.to_string(),
            })
            .collect(),
    };

    let conclusion_reached = request
        .conclusion
        .as_deref()
        .map(|conclusion| proof.reached_conclusion(conclusion));

    let response = CheckResponse {
        ok: true,
        parse_errors: vec![],
        check_errors,
        conclusion_reached,
        normalized_lines,
        metadata,
    };

    let json = serde_json::to_string(&response)
        .map_err(|err| format!("failed to serialize success response: {err}"))?;
    println!("{json}");
    Ok(())
}

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        process::exit(1);
    }
}
