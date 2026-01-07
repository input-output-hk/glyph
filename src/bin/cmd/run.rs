use super::input::{decode_program, Encoding};
use emulator::ExecutionResult;
use miette::{IntoDiagnostic, miette};
use tokio::{
    fs,
    io::{self, AsyncReadExt},
};

/// Compile Untyped Plutus Core into riscv32
#[derive(clap::Args)]
pub struct Args {
    // Optional file to read from
    #[clap(short, long)]
    input_file: Option<String>,

    /// Encoding of the input contents
    #[clap(long, default_value = "cbor")]
    encoding: Encoding,

    /// Input contents will be hex decoded
    #[clap(long)]
    hex: bool,

    /// The program to execute using the emulator
    #[clap(short, long)]
    program_file: String,
}

impl Args {
    pub async fn exec(self) -> miette::Result<()> {
        let program = if let Some(file_path) = self.input_file {
            fs::read(file_path).await.into_diagnostic()?
        } else {
            let mut buffer = Vec::new();

            io::stdin()
                .read_to_end(&mut buffer)
                .await
                .into_diagnostic()?;

            buffer
        };

        let riscv_input = if !program.is_empty() {
            let program = decode_program(program, self.encoding, self.hex)?;

            glyph::serialize(&program, 0xA0000000, false).into_diagnostic()?
        } else {
            Vec::new()
        };

        // let temp_dir = tempfile::tempdir().into_diagnostic()?;

        // temp_dir.close().into_diagnostic()?;

        let v = glyph::cek::run_file(&self.program_file, riscv_input).into_diagnostic()?;

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            other => {
                return Err(miette::miette!("execution failed: {other}"));
            }
        };

        if result_pointer == u32::MAX {
            return Err(miette!(
                "execution failed: validator returned failure (exit code 0xFFFFFFFF)"
            ));
        }

        Ok(())
    }
}
