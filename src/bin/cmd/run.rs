use emulator::ExecutionResult;
use miette::IntoDiagnostic;
use tokio::{
    fs,
    io::{self, AsyncReadExt},
};
use uplc::ast::{DeBruijn, Program};

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

#[derive(Copy, Clone, clap::ValueEnum)]
enum Encoding {
    Cbor,
    Flat,
    Text,
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
            let program: Program<DeBruijn> = match self.encoding {
                Encoding::Cbor => todo!(),
                Encoding::Flat => todo!(),
                Encoding::Text => {
                    if self.hex {
                        println!("warning: hex flag does nothing when encoding format is text")
                    }

                    let program = String::from_utf8(program).into_diagnostic()?;

                    let program = uplc::parser::program(&program).into_diagnostic()?;

                    program.try_into().into_diagnostic()?
                }
            };

            glyph::serialize(&program, 0xA0000000, false).into_diagnostic()?
        } else {
            Vec::new()
        };

        // let temp_dir = tempfile::tempdir().into_diagnostic()?;

        // temp_dir.close().into_diagnostic()?;

        let v = glyph::cek::run_file(&self.program_file, riscv_input).into_diagnostic()?;

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            _ => unreachable!("HOW?"),
        };

        assert_ne!(result_pointer, u32::MAX);

        Ok(())
    }
}
