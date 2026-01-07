use super::compile::{RuntimeFlavor, compile_program_to_elf};
use miette::{IntoDiagnostic, miette};
use serde::Deserialize;
use tokio::fs;
use uplc::ast::{DeBruijn, Program};

/// Build validators from a `plutus.json` file
#[derive(clap::Args)]
pub struct Args {
    /// Path to the `plutus.json` bundle
    #[clap(short, long, default_value = "plutus.json")]
    file: String,

    /// Validator title to compile when multiple are present
    #[clap(long)]
    validator: Option<String>,

    /// Output ELF path
    #[clap(short, long, default_value = "program.elf")]
    output: String,

    #[clap(short, long, default_value = "false")]
    no_input: bool,
}

#[derive(Deserialize)]
struct PlutusJson {
    validators: Vec<Validator>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Validator {
    title: String,
    compiled_code: String,
}

impl Args {
    pub async fn exec(self) -> miette::Result<()> {
        let json = fs::read_to_string(&self.file).await.into_diagnostic()?;
        let plutus: PlutusJson = serde_json::from_str(&json).into_diagnostic()?;

        if plutus.validators.is_empty() {
            return Err(miette!("no validators found in {}", self.file));
        }

        let available = plutus
            .validators
            .iter()
            .map(|validator| validator.title.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        let validator = if let Some(name) = self.validator.as_deref() {
            plutus.validators.iter().find(|v| v.title == name).ok_or_else(|| {
                miette!("validator '{}' not found. Available: {}", name, available)
            })?
        } else if plutus.validators.len() == 1 {
            &plutus.validators[0]
        } else {
            return Err(miette!(
                "plutus.json contains {} validators; use --validator. Available: {}",
                plutus.validators.len(),
                available
            ));
        };

        let compiled_code = validator.compiled_code.trim();
        let compiled_code = compiled_code
            .strip_prefix("0x")
            .or_else(|| compiled_code.strip_prefix("0X"))
            .unwrap_or(compiled_code);

        let cbor_bytes = hex::decode(compiled_code).into_diagnostic()?;
        let mut buffer = Vec::new();
        let program = Program::<DeBruijn>::from_cbor(&cbor_bytes, &mut buffer).into_diagnostic()?;

        compile_program_to_elf(
            &program,
            self.no_input,
            &self.output,
            RuntimeFlavor::Validator,
        )
        .await
    }
}
