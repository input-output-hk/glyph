use std::io::{self, Write};
use std::path::PathBuf;

use clap::Subcommand;
use emulator::executor::utils::{FailConfiguration, FailExecute, FailOpcode, FailReads, FailWrite};
use miette::{IntoDiagnostic, miette};
use tracing::Level;

use glyph::bitvmx::{
    self, BundleOptions, ExecuteOptions, DEFAULT_INPUT_SECTION, MANIFEST_FILE_NAME,
    MAPPING_FILE_NAME, ROM_COMMITMENT_FILE_NAME,
};

/// BitVMX-CPU emulator surface for demos
#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Execute a RISC-V ELF using the BitVMX emulator
    Execute(ExecuteArgs),
    /// Generate the opcode -> Bitcoin Script mapping
    InstructionMapping(InstructionMappingArgs),
    /// Generate ROM commitment data for an ELF
    RomCommitment(RomCommitmentArgs),
    /// Generate mapping + commitment bundle with a manifest.json
    Bundle(BundleArgs),
}

#[derive(clap::Args, Debug)]
pub struct ExecuteArgs {
    /// ELF file to load
    #[arg(long, value_name = "FILE")]
    elf: Option<PathBuf>,

    /// Input as hex
    #[arg(long, value_name = "HEX")]
    input: Option<String>,

    /// Input as little endian
    #[arg(long)]
    input_as_little: bool,

    /// Print program stdout and exit code
    #[arg(long)]
    stdout: bool,

    /// Output the execution trace and hash list
    #[arg(long)]
    trace: bool,

    /// Enable debug logging
    #[arg(long)]
    debug: bool,

    /// Save checkpoints to the given directory (defaults to ./checkpoints)
    #[arg(
        long = "checkpoints",
        alias = "checkpoint-path",
        value_name = "DIR",
        num_args = 0..=1,
        default_missing_value = "checkpoints"
    )]
    checkpoints: Option<PathBuf>,

    /// Step number to continue execution
    #[arg(long, value_name = "Step")]
    step: Option<u64>,

    /// Maximum number of steps to execute
    #[arg(long, value_name = "LimitStep")]
    limit: Option<u64>,

    /// List of specific trace steps to print
    #[arg(long, value_name = "TraceList")]
    list: Option<String>,

    /// Memory dump at given step
    #[arg(long)]
    dump_mem: Option<u64>,

    /// Fail producing hash for a specific step
    #[arg(long)]
    fail_hash: Option<u64>,

    /// Fail producing hash but only for steps until a specific one
    #[arg(long)]
    fail_hash_until: Option<u64>,

    /// Fail producing the write value for a specific step
    #[arg(long, value_names = &["step", "fake_trace"], num_args = 2)]
    fail_execute: Option<Vec<String>>,

    /// Fail while reading the pc at the given step
    #[arg(long)]
    fail_pc: Option<u64>,

    /// Fail reading read_1 at a given step
    #[arg(
        long,
        value_names = &["step", "address_original", "value", "modified_address", "modified_last_step"],
        num_args = 5
    )]
    fail_read_1: Option<Vec<String>>,

    /// Fail reading read_2 at a given step
    #[arg(
        long,
        value_names = &["step", "address_original", "value", "modified_address", "modified_last_step"],
        num_args = 5
    )]
    fail_read_2: Option<Vec<String>>,

    /// Fail write at a given step
    #[arg(long, value_names = &["step", "address_original", "value", "modified_address"], num_args = 4)]
    fail_write: Option<Vec<String>>,

    /// Fail reading opcode at a given step
    #[arg(long, value_names = &["step", "opcode"], num_args = 2)]
    fail_opcode: Option<Vec<String>>,

    /// Section name where the input will be loaded
    #[arg(long, value_name = "SectionName")]
    input_section: Option<String>,

    /// Avoid hashing the trace
    #[arg(long)]
    no_hash: bool,

    /// Verify execution on chain
    #[arg(long)]
    verify: bool,

    /// Disable instruction mapping
    #[arg(long)]
    no_mapping: bool,

    /// Show sections when loading the ELF
    #[arg(long)]
    sections: bool,

    /// Should we save steps that are not checkpoints
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    save_non_checkpoint_steps: bool,
}

#[derive(clap::Args)]
pub struct InstructionMappingArgs {
    /// Output file (defaults to stdout)
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
}

#[derive(clap::Args)]
pub struct RomCommitmentArgs {
    /// ELF file to load
    #[arg(long, value_name = "FILE")]
    elf: PathBuf,

    /// Show sections while loading the ELF
    #[arg(long)]
    sections: bool,

    /// Output file (defaults to stdout)
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
}

#[derive(clap::Args)]
pub struct BundleArgs {
    /// ELF file to load
    #[arg(long, value_name = "FILE")]
    elf: PathBuf,

    /// Output directory for bundle artifacts
    #[arg(long, value_name = "DIR")]
    out_dir: PathBuf,

    /// Show sections while loading the ELF
    #[arg(long)]
    sections: bool,
}

impl Args {
    pub async fn exec(self) -> miette::Result<()> {
        match self.command {
            Command::Execute(args) => args.exec().await,
            Command::InstructionMapping(args) => args.exec().await,
            Command::RomCommitment(args) => args.exec().await,
            Command::Bundle(args) => args.exec().await,
        }
    }
}

impl ExecuteArgs {
    async fn exec(self) -> miette::Result<()> {
        if self.debug {
            init_tracing(Level::DEBUG);
        } else if self.sections {
            init_tracing(Level::INFO);
        }

        let input = match self.input.as_deref() {
            Some(input) => bitvmx::parse_hex_input(input).into_diagnostic()?,
            None => Vec::new(),
        };

        let trace_list = self
            .list
            .as_deref()
            .map(bitvmx::parse_trace_list)
            .transpose()
            .into_diagnostic()?;

        let fail_execute = self.fail_execute.as_ref().map(FailExecute::new);
        let fail_reads = if self.fail_read_1.is_some() || self.fail_read_2.is_some() {
            Some(FailReads::new(
                self.fail_read_1.as_ref(),
                self.fail_read_2.as_ref(),
            ))
        } else {
            None
        };
        let fail_write = self.fail_write.as_ref().map(FailWrite::new);
        let fail_opcode = self.fail_opcode.as_ref().map(FailOpcode::new);

        let fail_config = FailConfiguration {
            fail_hash: self.fail_hash,
            fail_resign_hash: None,
            fail_hash_until: self.fail_hash_until,
            fail_execute,
            fail_reads,
            fail_write,
            fail_pc: self.fail_pc,
            fail_opcode,
            fail_memory_protection: false,
            fail_execute_only_protection: false,
            fail_commitment_step: None,
            fail_commitment_hash: false,
            fail_selection_bits: None,
            fail_prover_challenge_step: false,
        };

        let opts = ExecuteOptions {
            elf_path: self.elf,
            step: self.step,
            limit: self.limit,
            input,
            input_section: self
                .input_section
                .unwrap_or_else(|| DEFAULT_INPUT_SECTION.to_string()),
            input_as_little: self.input_as_little,
            trace: self.trace,
            verify: self.verify,
            use_instruction_mapping: !self.no_mapping,
            stdout: self.stdout,
            debug: self.debug,
            no_hash: self.no_hash,
            trace_list,
            dump_mem: self.dump_mem,
            checkpoint_path: self.checkpoints,
            fail_config,
            save_non_checkpoint_steps: self.save_non_checkpoint_steps,
            show_sections: self.sections,
        };

        let result = bitvmx::execute_elf(opts).into_diagnostic()?;

        if self.trace || self.list.is_some() {
            let trace_output = bitvmx::format_trace(&result.trace);
            if !trace_output.is_empty() {
                if self.stdout {
                    eprint!("{trace_output}");
                } else {
                    print!("{trace_output}");
                }
            }
        }

        if self.stdout {
            if !result.stdout.is_empty() {
                print!("{}", result.stdout);
                if !result.stdout.ends_with('\n') {
                    print!("\n");
                }
            }
            if let Some(code) = result.exit_code {
                println!("Exit code: {}", code);
            } else {
                println!("Execution result: {}", result.execution);
            }
        }

        let is_halt = matches!(result.execution, emulator::ExecutionResult::Halt(_, _));
        let is_limit = matches!(
            result.execution,
            emulator::ExecutionResult::LimitStepReached(_)
        );
        if !is_halt && !is_limit {
            return Err(miette!("execution failed: {}", result.execution));
        }
        if is_limit && !self.stdout {
            eprintln!("Execution result: {}", result.execution);
        }

        io::stdout().flush().into_diagnostic()?;
        Ok(())
    }
}

impl InstructionMappingArgs {
    async fn exec(self) -> miette::Result<()> {
        let mapping = bitvmx::instruction_mapping().into_diagnostic()?;

        match self.out {
            Some(path) => {
                bitvmx::write_atomic(&path, mapping.as_bytes()).into_diagnostic()?;
                eprintln!("Wrote {}", path.to_string_lossy());
            }
            None => {
                print!("{mapping}");
            }
        }

        Ok(())
    }
}

impl RomCommitmentArgs {
    async fn exec(self) -> miette::Result<()> {
        if self.sections {
            init_tracing(Level::INFO);
        }

        let commitment = bitvmx::rom_commitment(&self.elf, self.sections).into_diagnostic()?;

        match self.out {
            Some(path) => {
                bitvmx::write_atomic(&path, commitment.as_bytes()).into_diagnostic()?;
                eprintln!("Wrote {}", path.to_string_lossy());
            }
            None => {
                print!("{commitment}");
            }
        }

        Ok(())
    }
}

impl BundleArgs {
    async fn exec(self) -> miette::Result<()> {
        if self.sections {
            init_tracing(Level::INFO);
        }

        let out_dir = self.out_dir.clone();
        let opts = BundleOptions {
            elf_path: self.elf,
            out_dir: out_dir.clone(),
            sections: self.sections,
        };
        let _manifest = bitvmx::bundle(opts).into_diagnostic()?;

        eprintln!(
            "Wrote {}, {} and {}",
            out_dir.join(MAPPING_FILE_NAME).display(),
            out_dir.join(ROM_COMMITMENT_FILE_NAME).display(),
            out_dir.join(MANIFEST_FILE_NAME).display(),
        );

        Ok(())
    }
}

fn init_tracing(level: Level) {
    let _ = tracing_subscriber::fmt()
        .without_time()
        .with_target(false)
        .with_max_level(level)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_execute_defaults() {
        let cli = crate::cmd::Cli::parse_from([
            "glyph",
            "bitvmx",
            "execute",
            "--elf",
            "program.elf",
        ]);

        let args = match cli.cmd {
            crate::cmd::Cmd::Bitvmx(args) => args,
            _ => panic!("expected bitvmx command"),
        };

        match args.command {
            Command::Execute(exec) => {
                assert_eq!(exec.elf.unwrap(), PathBuf::from("program.elf"));
                assert!(!exec.input_as_little);
                assert!(exec.checkpoints.is_none());
            }
            _ => panic!("expected execute command"),
        }
    }

    #[test]
    fn parse_execute_checkpoints_default_path() {
        let cli = crate::cmd::Cli::parse_from([
            "glyph",
            "bitvmx",
            "execute",
            "--elf",
            "program.elf",
            "--checkpoints",
        ]);

        let args = match cli.cmd {
            crate::cmd::Cmd::Bitvmx(args) => args,
            _ => panic!("expected bitvmx command"),
        };

        match args.command {
            Command::Execute(exec) => {
                assert_eq!(exec.checkpoints.unwrap(), PathBuf::from("checkpoints"));
            }
            _ => panic!("expected execute command"),
        }
    }
}
