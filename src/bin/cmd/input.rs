use miette::IntoDiagnostic;
use uplc::ast::{DeBruijn, Program};

#[derive(Copy, Clone, clap::ValueEnum)]
pub enum Encoding {
    Cbor,
    Flat,
    Text,
}

pub fn decode_program(
    bytes: Vec<u8>,
    encoding: Encoding,
    hex: bool,
) -> miette::Result<Program<DeBruijn>> {
    match encoding {
        Encoding::Text => {
            if hex {
                println!("warning: hex flag does nothing when encoding format is text")
            }

            let program = String::from_utf8(bytes).into_diagnostic()?;
            let program = uplc::parser::program(&program).into_diagnostic()?;
            program.try_into().into_diagnostic()
        }
        Encoding::Cbor => {
            let bytes = if hex { decode_hex_bytes(bytes)? } else { bytes };
            let mut buffer = Vec::new();
            Program::<DeBruijn>::from_cbor(&bytes, &mut buffer).into_diagnostic()
        }
        Encoding::Flat => {
            let bytes = if hex { decode_hex_bytes(bytes)? } else { bytes };
            Program::<DeBruijn>::from_flat(&bytes).into_diagnostic()
        }
    }
}

fn decode_hex_bytes(bytes: Vec<u8>) -> miette::Result<Vec<u8>> {
    let text = String::from_utf8(bytes).into_diagnostic()?;
    let cleaned = text.split_whitespace().collect::<String>();
    let cleaned = cleaned
        .strip_prefix("0x")
        .or_else(|| cleaned.strip_prefix("0X"))
        .unwrap_or(&cleaned);
    hex::decode(cleaned).into_diagnostic()
}
