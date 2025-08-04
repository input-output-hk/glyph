use miette::IntoDiagnostic;
use tokio::{
    fs,
    io::{self, AsyncReadExt},
    process::Command,
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

    #[clap(short, long, default_value = "false")]
    no_input: bool,
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
            }
        };

        let thing = glyph::Cek::default();

        let riscv_program =
            glyph::serialize(&program, 0x90000000, !self.no_input).into_diagnostic()?;

        let riscv_program = thing.cek_assembly(riscv_program).generate();

        let temp_dir = tempfile::tempdir().into_diagnostic()?;

        let program_s_path = temp_dir.path().join("program.s");

        fs::write(program_s_path, riscv_program)
            .await
            .into_diagnostic()?;

        Command::new("riscv64-elf-as")
            .current_dir(temp_dir.path())
            .args([
                "-march=rv32im",
                "-mabi=ilp32",
                "-o",
                "program.o",
                "program.s",
            ])
            .status()
            .await
            .into_diagnostic()?;

        // create runtime.o file in temp_dir
        let runtime_o_path = temp_dir.path().join("runtime.o");
        fs::write(runtime_o_path, glyph::RUNTIME)
            .await
            .into_diagnostic()?;

        // create memset.o file in temp_dir
        let memset_o_path = temp_dir.path().join("memset.o");
        fs::write(memset_o_path, glyph::MEMSET)
            .await
            .into_diagnostic()?;

        // create link.ld file in temp_dir
        let link_ld_path = temp_dir.path().join("link.ld");
        fs::write(link_ld_path, glyph::LINKER_SCRIPT)
            .await
            .into_diagnostic()?;

        Command::new("riscv64-elf-ld")
            .current_dir(temp_dir.path())
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "program.elf",
                "-T",
                "link.ld",
                "program.o",
                "runtime.o",
                "memset.o",
            ])
            .status()
            .await
            .into_diagnostic()?;

        let program_elf_path = temp_dir.path().join("program.elf");

        fs::copy(program_elf_path, "program.elf")
            .await
            .into_diagnostic()?;

        temp_dir.close().into_diagnostic()?;

        Ok(())
    }
}
