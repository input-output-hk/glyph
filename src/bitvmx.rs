use std::{
    fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

extern crate emulator as bitvmx_emulator;

use bitcoin_script_riscv::riscv::instruction_mapping::{
    create_verification_script_mapping, get_key_from_instruction_and_micro,
    get_required_microinstruction,
};
use bitvmx_emulator::{
    EmulatorError, ExecutionResult,
    constants::REGISTERS_BASE_ADDRESS,
    executor::{
        fetcher::{FullTrace, execute_program},
        utils::FailConfiguration,
    },
    loader::program::{Code, Program, RomCommitment, load_elf},
};
use riscv_decode::decode as decode_riscv;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use thiserror::Error;

pub const DEFAULT_INPUT_SECTION: &str = ".input";
pub const MAPPING_FILE_NAME: &str = "instruction_mapping.txt";
pub const ROM_COMMITMENT_FILE_NAME: &str = "rom_commitment.txt";
pub const MANIFEST_FILE_NAME: &str = "manifest.json";

#[derive(Debug, Error)]
pub enum BitvmxError {
    #[error("invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("stdout capture failed: {0}")]
    OutputCapture(String),
    #[error(transparent)]
    Emulator(#[from] EmulatorError),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, BitvmxError>;

#[derive(Debug, Clone)]
pub struct ExecuteOptions {
    pub elf_path: Option<PathBuf>,
    pub step: Option<u64>,
    pub limit: Option<u64>,
    pub input: Vec<u8>,
    pub input_section: String,
    pub input_as_little: bool,
    pub trace: bool,
    pub verify: bool,
    pub use_instruction_mapping: bool,
    pub stdout: bool,
    pub debug: bool,
    pub no_hash: bool,
    pub trace_list: Option<Vec<u64>>,
    pub dump_mem: Option<u64>,
    pub checkpoint_path: Option<PathBuf>,
    pub fail_config: FailConfiguration,
    pub save_non_checkpoint_steps: bool,
    pub show_sections: bool,
}

impl Default for ExecuteOptions {
    fn default() -> Self {
        Self {
            elf_path: None,
            step: None,
            limit: None,
            input: Vec::new(),
            input_section: DEFAULT_INPUT_SECTION.to_string(),
            input_as_little: false,
            trace: false,
            verify: false,
            use_instruction_mapping: true,
            stdout: false,
            debug: false,
            no_hash: false,
            trace_list: None,
            dump_mem: None,
            checkpoint_path: None,
            fail_config: FailConfiguration::default(),
            save_non_checkpoint_steps: true,
            show_sections: false,
        }
    }
}

#[derive(Debug)]
pub struct ExecuteResult {
    pub execution: ExecutionResult,
    pub exit_code: Option<u32>,
    pub stdout: String,
    pub trace: FullTrace,
}

#[derive(Debug, Clone)]
pub struct BundleOptions {
    pub elf_path: PathBuf,
    pub out_dir: PathBuf,
    pub sections: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    pub elf_path: String,
    pub elf_sha256: String,
    pub mapping_file: String,
    pub mapping_sha256: String,
    pub commitment_file: String,
    pub commitment_sha256: String,
    pub tool_versions: ToolVersions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolVersions {
    pub glyph: String,
    pub bitvmx_cpu: Option<BitvmxCpuVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitvmxCpuVersion {
    pub version: Option<String>,
    pub git_sha: Option<String>,
}

pub fn execute_elf(opts: ExecuteOptions) -> Result<ExecuteResult> {
    if opts.elf_path.is_none() && opts.step.is_none() {
        return Err(BitvmxError::InvalidParameters(
            "either --elf or --step is required".to_string(),
        ));
    }
    if opts.elf_path.is_some() && opts.step.is_some() {
        return Err(BitvmxError::InvalidParameters(
            "choose either --elf or --step, not both".to_string(),
        ));
    }
    if opts.step.is_some() && opts.checkpoint_path.is_none() {
        return Err(BitvmxError::InvalidParameters(
            "--step requires --checkpoints to load a checkpoint".to_string(),
        ));
    }

    let ExecuteOptions {
        elf_path,
        step,
        limit,
        input,
        input_section,
        input_as_little,
        trace,
        verify,
        use_instruction_mapping,
        stdout,
        debug,
        no_hash,
        trace_list,
        dump_mem,
        checkpoint_path,
        fail_config,
        save_non_checkpoint_steps,
        show_sections,
    } = opts;

    let (mut program, input) = match elf_path {
        Some(path) => {
            let path_str = path.to_string_lossy().to_string();
            let program = load_elf(&path_str, show_sections)?;
            (program, input)
        }
        None => {
            let step = step.expect("step is required");
            let checkpoint_path = checkpoint_path
                .as_ref()
                .expect("checkpoint path is required");
            let program =
                Program::deserialize_from_file(checkpoint_path.to_string_lossy().as_ref(), step)?;
            (program, Vec::new())
        }
    };

    let checkpoint_path = checkpoint_path.map(|path| path.to_string_lossy().to_string());
    let print_trace = trace || trace_list.is_some();

    let run = || {
        execute_program(
            &mut program,
            input,
            &input_section,
            input_as_little,
            &checkpoint_path,
            limit,
            print_trace,
            verify,
            use_instruction_mapping,
            stdout,
            debug,
            no_hash,
            trace_list,
            dump_mem,
            fail_config,
            save_non_checkpoint_steps,
        )
    };

    let ((execution, trace), stdout_bytes) = if stdout {
        capture_stdout(run)?
    } else {
        (run(), Vec::new())
    };

    let stdout = String::from_utf8_lossy(&stdout_bytes).to_string();
    let exit_code = match execution {
        ExecutionResult::Halt(code, _) => Some(code),
        _ => None,
    };

    Ok(ExecuteResult {
        execution,
        exit_code,
        stdout,
        trace,
    })
}

pub fn instruction_mapping() -> Result<String> {
    let mapping = create_verification_script_mapping(REGISTERS_BASE_ADDRESS);
    let mut entries = mapping.into_iter().collect::<Vec<_>>();
    entries.sort_by(|(left, _), (right, _)| left.cmp(right));

    let mut output = String::new();
    for (key, (script, requires_witness)) in entries {
        let line = format!(
            "Key: {}, Script: {:?}, Size: {}, Witness: {}",
            key,
            script.to_hex_string(),
            script.len(),
            requires_witness
        );
        output.push_str(&line);
        output.push('\n');
    }
    Ok(output)
}

pub fn rom_commitment(elf_path: impl AsRef<Path>, sections: bool) -> Result<String> {
    let path = elf_path.as_ref().to_string_lossy().to_string();
    let program = load_elf(&path, sections)?;
    let commitment = build_rom_commitment(&program)?;
    Ok(format_rom_commitment(&commitment))
}

pub fn bundle(opts: BundleOptions) -> Result<BundleManifest> {
    let mapping = instruction_mapping()?;
    let commitment = rom_commitment(&opts.elf_path, opts.sections)?;

    fs::create_dir_all(&opts.out_dir)?;

    let mapping_path = opts.out_dir.join(MAPPING_FILE_NAME);
    let commitment_path = opts.out_dir.join(ROM_COMMITMENT_FILE_NAME);
    let manifest_path = opts.out_dir.join(MANIFEST_FILE_NAME);

    write_atomic(&mapping_path, mapping.as_bytes())?;
    write_atomic(&commitment_path, commitment.as_bytes())?;

    let elf_path_str = opts.elf_path.to_string_lossy().to_string();
    let elf_sha256 = sha256_file(&opts.elf_path)?;
    let mapping_sha256 = sha256_bytes(mapping.as_bytes());
    let commitment_sha256 = sha256_bytes(commitment.as_bytes());

    let manifest = BundleManifest {
        elf_path: elf_path_str,
        elf_sha256,
        mapping_file: MAPPING_FILE_NAME.to_string(),
        mapping_sha256,
        commitment_file: ROM_COMMITMENT_FILE_NAME.to_string(),
        commitment_sha256,
        tool_versions: ToolVersions {
            glyph: env!("CARGO_PKG_VERSION").to_string(),
            bitvmx_cpu: bitvmx_cpu_version(),
        },
    };

    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    write_atomic(&manifest_path, manifest_json.as_bytes())?;

    Ok(manifest)
}

pub fn format_trace(trace: &FullTrace) -> String {
    let mut output = String::new();
    for (step, hash) in trace {
        output.push_str(&step.to_csv());
        output.push(';');
        output.push_str(hash);
        output.push('\n');
    }
    output
}

pub fn parse_hex_input(input: &str) -> Result<Vec<u8>> {
    let trimmed = input.trim();
    let trimmed = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);

    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    if !trimmed.len().is_multiple_of(2) {
        return Err(BitvmxError::InvalidInput(
            "hex input must have an even number of digits".to_string(),
        ));
    }

    hex::decode(trimmed)
        .map_err(|err| BitvmxError::InvalidInput(format!("invalid hex input: {err}")))
}

pub fn parse_trace_list(list: &str) -> Result<Vec<u64>> {
    let mut values = Vec::new();
    for part in list.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed.parse::<u64>().map_err(|err| {
            BitvmxError::InvalidInput(format!("invalid trace list value '{trimmed}': {err}"))
        })?;
        values.push(value);
    }
    Ok(values)
}

pub fn write_atomic(path: impl AsRef<Path>, contents: &[u8]) -> Result<()> {
    let path = path.as_ref();
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;

    let mut temp = NamedTempFile::new_in(dir)?;
    temp.write_all(contents)?;
    temp.flush()?;
    match temp.persist(path) {
        Ok(_) => {}
        Err(err) => {
            if err.error.kind() == io::ErrorKind::AlreadyExists {
                fs::remove_file(path)?;
                err.file
                    .persist(path)
                    .map_err(|err| BitvmxError::Io(err.error))?;
            } else {
                return Err(BitvmxError::Io(err.error));
            }
        }
    }
    Ok(())
}

fn format_rom_commitment(commitment: &RomCommitment) -> String {
    let mut output = String::new();

    for entry in &commitment.code {
        output.push_str(&format!(
            "PC: 0x{:08x} Micro: {} Opcode: 0x{:08x} Key: {}",
            entry.address, entry.micro, entry.opcode, entry.key
        ));
        output.push('\n');
    }

    for (address, value) in &commitment.constants {
        output.push_str(&format!(
            "Address: 0x{:08x} value: 0x{:08x}",
            address, value
        ));
        output.push('\n');
    }

    for (start, size) in &commitment.zero_initialized {
        output.push_str(&format!(
            "Zero initialized range: start: 0x{:08x} size: 0x{:08x}",
            start, size
        ));
        output.push('\n');
    }

    output.push_str(&format!("Entrypoint: 0x{:08x}", commitment.entrypoint));
    output.push('\n');

    output
}

fn build_rom_commitment(program: &Program) -> Result<RomCommitment> {
    let mut rom_commitment = RomCommitment {
        entrypoint: program.pc.get_address(),
        code: Vec::new(),
        constants: Vec::new(),
        zero_initialized: Vec::new(),
    };

    for section in &program.sections {
        if section.is_code {
            let words = section.size / 4;
            for i in 0..words {
                let address = section.start + i * 4;
                let data = u32::from_be(section.data[i as usize]);
                let instruction = decode_riscv(data).map_err(|_| {
                    BitvmxError::InvalidInput(format!(
                        "code section with undecodeable instruction: 0x{:08x} at position: 0x{:08x}",
                        data, address
                    ))
                })?;
                let micros = get_required_microinstruction(&instruction);
                for micro in 0..micros {
                    let key = std::panic::catch_unwind(|| {
                        get_key_from_instruction_and_micro(&instruction, micro)
                    })
                    .map_err(|_| {
                        BitvmxError::InvalidInput(format!(
                            "instruction not supported: {:?}",
                            instruction
                        ))
                    })?;
                    rom_commitment.code.push(Code {
                        address,
                        micro,
                        opcode: data,
                        key,
                    });
                }
            }
        }
    }

    for section in &program.sections {
        if !section.is_code && section.initialized {
            let words = section.size / 4;
            for i in 0..words {
                let address = section.start + i * 4;
                let data = u32::from_be(section.data[i as usize]);
                rom_commitment.constants.push((address, data));
            }
        }
    }

    for section in &program.sections {
        if !section.is_code && !section.initialized {
            rom_commitment
                .zero_initialized
                .push((section.start, section.size));
        }
    }

    Ok(rom_commitment)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let hash = hasher.finalize();
    hex::encode(hash)
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(sha256_bytes(&bytes))
}

fn bitvmx_cpu_version() -> Option<BitvmxCpuVersion> {
    let version = option_env!("GLYPH_BITVMX_CPU_VERSION").map(|value| value.to_string());
    let git_sha = option_env!("GLYPH_BITVMX_CPU_GIT_SHA").map(|value| value.to_string());

    if version.is_none() && git_sha.is_none() {
        None
    } else {
        Some(BitvmxCpuVersion { version, git_sha })
    }
}

#[cfg(unix)]
fn capture_stdout<F, R>(func: F) -> Result<(R, Vec<u8>)>
where
    F: FnOnce() -> R,
{
    use std::os::unix::io::{AsRawFd, RawFd};

    struct StdoutRedirect {
        saved_fd: RawFd,
    }

    impl StdoutRedirect {
        fn new(target_fd: RawFd) -> io::Result<Self> {
            let stdout_fd = io::stdout().as_raw_fd();
            let saved_fd = unsafe { libc::dup(stdout_fd) };
            if saved_fd < 0 {
                return Err(io::Error::last_os_error());
            }

            if unsafe { libc::dup2(target_fd, stdout_fd) } < 0 {
                let err = io::Error::last_os_error();
                unsafe {
                    libc::close(saved_fd);
                }
                return Err(err);
            }

            Ok(Self { saved_fd })
        }
    }

    impl Drop for StdoutRedirect {
        fn drop(&mut self) {
            let stdout_fd = io::stdout().as_raw_fd();
            unsafe {
                libc::dup2(self.saved_fd, stdout_fd);
                libc::close(self.saved_fd);
            }
        }
    }

    let mut temp = NamedTempFile::new()?;
    let redirect = StdoutRedirect::new(temp.as_raw_fd())
        .map_err(|err| BitvmxError::OutputCapture(err.to_string()))?;

    let result = func();
    io::stdout().flush()?;
    drop(redirect);

    temp.seek(SeekFrom::Start(0))?;
    let mut buffer = Vec::new();
    temp.read_to_end(&mut buffer)?;
    Ok((result, buffer))
}

#[cfg(not(unix))]
fn capture_stdout<F, R>(_func: F) -> Result<(R, Vec<u8>)>
where
    F: FnOnce() -> R,
{
    Err(BitvmxError::OutputCapture(
        "stdout capture is only supported on unix platforms".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_input_accepts_prefix() {
        let bytes = parse_hex_input("0x0a0b").unwrap();
        assert_eq!(bytes, vec![0x0a, 0x0b]);
    }

    #[test]
    fn parse_trace_list_ignores_empty_entries() {
        let list = parse_trace_list("1,  ,2,,3").unwrap();
        assert_eq!(list, vec![1, 2, 3]);
    }

    #[test]
    fn write_atomic_writes_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("output.txt");
        write_atomic(&path, b"hello").unwrap();
        let contents = fs::read(path).unwrap();
        assert_eq!(contents, b"hello");
    }
}
