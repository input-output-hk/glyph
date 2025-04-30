//! UPLC to RISC-V compiler
//!
//! This crate provides functionality to compile Untyped Plutus Core (UPLC)
//! to RISC-V assembly code.

use thiserror::Error;
pub mod cek;

type Result<T> = std::result::Result<T, CompilationError>;

/// Errors that can occur during the compilation process
#[derive(Debug, Error)]
pub enum CompilationError {
    #[error("Parsing error: {0}")]
    Parse(String),

    #[error("Code generation error: {0}")]
    CodeGen(#[from] risc_v_gen::CodeGenError),

    #[error("Unsupported UPLC feature: {0}")]
    UnsupportedFeature(String),

    #[error("Invalid UPLC input: {0}")]
    InvalidInput(String),

    #[error("Evaluation error: {0}")]
    Evaluation(String),

    #[error("Term conversion error: {0}")]
    TermConversion(String),
}

/// Compilation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationMode {
    /// Compile with BitVMX compatibility
    BitVMX,
}

/// Main compiler structure
///
/// The `Compiler` struct is responsible for compiling Untyped Plutus Core (UPLC)
/// to RISC-V assembly code. It supports various compilation modes and
/// optimization levels.
///
/// # Examples
///
/// ```
/// use uplc_to_riscv::Compiler;
///
/// let compiler = Compiler::new();
/// let uplc_code = "(program 1.0.0 (con integer 42))";
/// let result = compiler.compile(uplc_code);
/// assert!(result.is_ok());
/// ```

#[test]
fn can_get_cek_assembly() {
    let cek = cek::Cek::default();

    let assembly = cek.cek_assembly();
    let assembly_string = assembly.generate();

    println!("Assembly {}", assembly_string);
}
