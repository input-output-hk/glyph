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
    file: Option<String>,

    /// Encoding of the input contents
    #[clap(long, default_value = "cbor")]
    encoding: Encoding,

    /// Input contents will be hex decoded
    #[clap(long)]
    hex: bool,
}

#[derive(Copy, Clone, clap::ValueEnum)]
enum Encoding {
    Cbor,
    Flat,
    Text,
}

impl Args {
    pub async fn exec(self) -> miette::Result<()> {
        let program = if let Some(file_path) = self.file {
            fs::read(file_path).await.into_diagnostic()?
        } else {
            let mut buffer = Vec::new();

            io::stdin()
                .read_to_end(&mut buffer)
                .await
                .into_diagnostic()?;

            buffer
        };

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
            },
        };

        Ok(())
    }
}
