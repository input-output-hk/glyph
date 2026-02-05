use super::input::{Encoding, decode_program};
use miette::{IntoDiagnostic, miette};
use std::path::Path;
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

#[derive(Copy, Clone)]
pub(super) enum RuntimeFlavor {
    Function,
    Validator,
}

const DEFAULT_INPUT_BSS_BYTES: i32 = 1024 * 1024;

impl RuntimeFlavor {
    fn entrypoint(self) -> &'static str {
        match self {
            RuntimeFlavor::Function => "init",
            RuntimeFlavor::Validator => "init2",
        }
    }

    fn object_bytes(self) -> &'static [u8] {
        match self {
            RuntimeFlavor::Function => glyph::RUNTIMEFUNCTION,
            RuntimeFlavor::Validator => glyph::RUNTIME,
        }
    }
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

        let program = decode_program(program, self.encoding, self.hex)?;

        compile_program_to_elf(
            &program,
            self.no_input,
            "program.elf",
            RuntimeFlavor::Function,
        )
        .await
    }
}

pub(super) async fn compile_program_to_elf(
    program: &Program<DeBruijn>,
    no_input: bool,
    output_path: &str,
    runtime: RuntimeFlavor,
) -> miette::Result<()> {
    let thing = glyph::Cek::default();

    let riscv_program = glyph::serialize(program, 0x90000000, !no_input).into_diagnostic()?;

    let riscv_program = thing
        .cek_assembly_with_entry(
            riscv_program,
            runtime.entrypoint(),
            (!no_input).then_some(DEFAULT_INPUT_BSS_BYTES),
        )
        .generate();

    let temp_dir = tempfile::tempdir().into_diagnostic()?;

    let program_s_path = temp_dir.path().join("program.s");

    fs::write(program_s_path, riscv_program)
        .await
        .into_diagnostic()?;

    let mut assembler = Command::new("riscv64-elf-as");
    assembler.current_dir(temp_dir.path()).args([
        "-march=rv32im",
        "-mabi=ilp32",
        "-o",
        "program.o",
        "program.s",
    ]);
    run_command(&mut assembler, "riscv64-elf-as").await?;

    // create runtime.o file in temp_dir
    let runtime_o_path = temp_dir.path().join("runtime.o");
    fs::write(runtime_o_path, runtime.object_bytes())
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

    let mut linker = Command::new("riscv64-elf-ld");
    linker.current_dir(temp_dir.path()).args([
        "-m",
        "elf32lriscv",
        "-o",
        "program.elf",
        "-T",
        "link.ld",
        "program.o",
        "runtime.o",
        "memset.o",
    ]);
    run_command(&mut linker, "riscv64-elf-ld").await?;

    let program_elf_path = temp_dir.path().join("program.elf");
    let output_path = Path::new(output_path);

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).await.into_diagnostic()?;
        }
    }

    fs::copy(program_elf_path, output_path)
        .await
        .into_diagnostic()?;

    temp_dir.close().into_diagnostic()?;

    Ok(())
}

async fn run_command(command: &mut Command, name: &str) -> miette::Result<()> {
    let output = command.output().await.into_diagnostic()?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut details = String::new();

    if !stderr.trim().is_empty() {
        details.push_str(stderr.trim());
    }

    if !stdout.trim().is_empty() {
        if !details.is_empty() {
            details.push('\n');
        }
        details.push_str(stdout.trim());
    }

    if details.is_empty() {
        return Err(miette!("{name} failed with status {}", output.status));
    }

    Err(miette!("{name} failed: {details}"))
}
