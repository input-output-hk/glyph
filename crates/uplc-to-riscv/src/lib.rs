//! UPLC to RISC-V compiler
//!
//! This crate provides functionality to compile Untyped Plutus Core (UPLC)
//! to RISC-V assembly code.

mod ir;

use risc_v_gen::{CodeGenerator, Instruction, Register};
use thiserror::Error;
use uplc::ast::{Name, Program};
use uplc::builtins::DefaultFunction;
use uplc::parser;
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
pub struct Compiler {}

impl Compiler {
    /// Create a new compiler with default settings
    ///
    /// This creates a compiler with no optimizations enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use uplc_to_riscv::Compiler;
    ///
    /// let compiler = Compiler::new();
    /// ```
    pub fn new() -> Self {
        Self {}
    }

    /// Compile UPLC code to RISC-V assembly
    ///
    /// This method compiles the given UPLC code to RISC-V assembly code.
    /// It currently only supports BitVMX-compatible compilation.
    ///
    /// # Arguments
    ///
    /// * `uplc_code` - The UPLC code to compile
    ///
    /// # Returns
    ///
    /// A `Result` containing the compiled RISC-V assembly code if successful,
    /// or a `CompilationError` if compilation fails.
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
    pub fn compile(&self, uplc_code: &str) -> Result<String> {
        // Parse the UPLC code
        let program: Program<Name> =
            parser::program(uplc_code).map_err(|e| CompilationError::Parse(e.to_string()))?;

        // Compile with BitVMX compatibility
        unreachable!("This is not yet implemented.")
    }

    /// Compile a UPLC builtin function to RISC-V code
    #[allow(dead_code)]
    fn compile_builtin(
        &self,
        generator: &mut CodeGenerator,
        builtin: DefaultFunction,
    ) -> Result<()> {
        match builtin {
            DefaultFunction::AddInteger => {
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp));
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp));
                generator.add_instruction(Instruction::Add(
                    Register::A0,
                    Register::A0,
                    Register::A1,
                ));
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4));
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
            },
            DefaultFunction::SubtractInteger => {
                generator.add_instruction(Instruction::Lw(Register::A0, 4, Register::Sp));
                generator.add_instruction(Instruction::Lw(Register::A1, 0, Register::Sp));
                generator.add_instruction(Instruction::Sub(
                    Register::A0,
                    Register::A0,
                    Register::A1,
                ));
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4));
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
            },
            DefaultFunction::MultiplyInteger => {
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp));
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp));
                generator.add_instruction(Instruction::Mul(
                    Register::A0,
                    Register::A0,
                    Register::A1,
                ));
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4));
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
            },
            DefaultFunction::DivideInteger => {
                generator.add_instruction(Instruction::Lw(Register::A0, 4, Register::Sp));
                generator.add_instruction(Instruction::Lw(Register::A1, 0, Register::Sp));
                generator.add_instruction(Instruction::Div(
                    Register::A0,
                    Register::A0,
                    Register::A1,
                ));
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4));
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
            },
            DefaultFunction::RemainderInteger => {
                generator.add_instruction(Instruction::Lw(Register::A0, 4, Register::Sp));
                generator.add_instruction(Instruction::Lw(Register::A1, 0, Register::Sp));
                generator.add_instruction(Instruction::Rem(
                    Register::A0,
                    Register::A0,
                    Register::A1,
                ));
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4));
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
            },
            DefaultFunction::LessThanInteger => {
                generator.add_instruction(Instruction::Lw(Register::A0, 4, Register::Sp));
                generator.add_instruction(Instruction::Lw(Register::A1, 0, Register::Sp));
                generator.add_instruction(Instruction::Slt(
                    Register::A0,
                    Register::A0,
                    Register::A1,
                ));
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4));
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
            },
            DefaultFunction::LessThanEqualsInteger => {
                generator.add_instruction(Instruction::Lw(Register::A0, 4, Register::Sp));
                generator.add_instruction(Instruction::Lw(Register::A1, 0, Register::Sp));
                generator.add_instruction(Instruction::Slt(
                    Register::A0,
                    Register::A1,
                    Register::A0,
                ));
                generator.add_instruction(Instruction::Xori(Register::A0, Register::A0, 1));
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4));
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
            },
            DefaultFunction::EqualsInteger => {
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp));
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp));
                generator.add_instruction(Instruction::Sub(
                    Register::A0,
                    Register::A0,
                    Register::A1,
                ));
                generator.add_instruction(Instruction::Seqz(Register::A0, Register::A0));
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4));
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
            },
            // ByteString operations
            DefaultFunction::AppendByteString => {
                // Load the two bytestring pointers
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp)); // Second bytestring
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp)); // First bytestring

                // Load the lengths of both bytestrings
                generator.add_instruction(Instruction::Lw(Register::T0, 0, Register::A0)); // Length of second bytestring
                generator.add_instruction(Instruction::Lw(Register::T1, 0, Register::A1)); // Length of first bytestring

                // Calculate the total length
                generator.add_instruction(Instruction::Add(
                    Register::T2,
                    Register::T0,
                    Register::T1,
                ));

                // Allocate memory for the new bytestring (length + data)
                generator.add_instruction(Instruction::Addi(Register::A0, Register::T2, 4)); // 4 bytes for length + data
                generator.add_instruction(Instruction::Jal(Register::Ra, "malloc".to_string()));

                // Store the total length at the beginning of the new bytestring
                generator.add_instruction(Instruction::Sw(Register::T2, 0, Register::A0));

                // Save the result pointer
                generator.add_instruction(Instruction::Mv(Register::T3, Register::A0));

                // Copy the first bytestring
                generator.add_instruction(Instruction::Addi(Register::A0, Register::T3, 4)); // Destination (result + 4)
                generator.add_instruction(Instruction::Addi(Register::A1, Register::A1, 4)); // Source (first bytestring + 4)
                generator.add_instruction(Instruction::Mv(Register::A2, Register::T1)); // Length of first bytestring
                generator.add_instruction(Instruction::Jal(Register::Ra, "memcpy".to_string()));

                // Copy the second bytestring
                generator.add_instruction(Instruction::Lw(Register::A1, 0, Register::Sp)); // Second bytestring
                generator.add_instruction(Instruction::Add(
                    Register::A0,
                    Register::T3,
                    Register::T1,
                )); // Destination (result + 4 + length1)
                generator.add_instruction(Instruction::Addi(Register::A0, Register::A0, 4)); // Adjust for header
                generator.add_instruction(Instruction::Addi(Register::A1, Register::A1, 4)); // Source (second bytestring + 4)
                generator.add_instruction(Instruction::Mv(Register::A2, Register::T0)); // Length of second bytestring
                generator.add_instruction(Instruction::Jal(Register::Ra, "memcpy".to_string()));

                // Clean up the stack and store the result
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4)); // Pop one value
                generator.add_instruction(Instruction::Sw(Register::T3, 0, Register::Sp));
                // Store result
            },
            DefaultFunction::ConsByteString => {
                // Load the byte and bytestring pointer
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp)); // ByteString
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp)); // Byte (as integer)

                // Load the length of the bytestring
                generator.add_instruction(Instruction::Lw(Register::T0, 0, Register::A0)); // Length of bytestring

                // Calculate the new length
                generator.add_instruction(Instruction::Addi(Register::T1, Register::T0, 1)); // New length = old length + 1

                // Allocate memory for the new bytestring (length + data)
                generator.add_instruction(Instruction::Addi(Register::A0, Register::T1, 4)); // 4 bytes for length + data
                generator.add_instruction(Instruction::Jal(Register::Ra, "malloc".to_string()));

                // Store the new length at the beginning of the new bytestring
                generator.add_instruction(Instruction::Sw(Register::T1, 0, Register::A0));

                // Save the result pointer
                generator.add_instruction(Instruction::Mv(Register::T3, Register::A0));

                // Store the byte at the beginning of the data
                generator.add_instruction(Instruction::Lw(Register::T4, 4, Register::Sp)); // Byte (as integer)
                generator.add_instruction(Instruction::Andi(Register::T4, Register::T4, 0xFF)); // Ensure it's just a byte
                generator.add_instruction(Instruction::Sb(Register::T4, 4, Register::T3)); // Store at result + 4

                // Copy the original bytestring
                generator.add_instruction(Instruction::Lw(Register::A1, 0, Register::Sp)); // Original bytestring
                generator.add_instruction(Instruction::Addi(Register::A0, Register::T3, 5)); // Destination (result + 4 + 1)
                generator.add_instruction(Instruction::Addi(Register::A1, Register::A1, 4)); // Source (original bytestring + 4)
                generator.add_instruction(Instruction::Mv(Register::A2, Register::T0)); // Length of original bytestring
                generator.add_instruction(Instruction::Jal(Register::Ra, "memcpy".to_string()));

                // Clean up the stack and store the result
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4)); // Pop one value
                generator.add_instruction(Instruction::Sw(Register::T3, 0, Register::Sp));
                // Store result
            },
            DefaultFunction::SliceByteString => {
                // Load the arguments
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp)); // ByteString
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp)); // Length to take
                generator.add_instruction(Instruction::Lw(Register::A2, 8, Register::Sp)); // Start index

                // Load the length of the bytestring
                generator.add_instruction(Instruction::Lw(Register::T0, 0, Register::A0)); // Length of bytestring

                // Check if start index is out of bounds
                generator.add_instruction(Instruction::Bge(
                    Register::A2,
                    Register::T0,
                    "slice_empty".to_string(),
                ));

                // Calculate how many bytes we can actually take
                generator.add_instruction(Instruction::Sub(
                    Register::T1,
                    Register::T0,
                    Register::A2,
                )); // Available bytes
                generator.add_instruction(Instruction::Bge(
                    Register::A1,
                    Register::T1,
                    "slice_remaining".to_string(),
                ));
                generator.add_instruction(Instruction::Mv(Register::T1, Register::A1)); // Use requested length
                generator.add_instruction(Instruction::Jal(
                    Register::Zero,
                    "slice_allocate".to_string(),
                ));

                // Label for taking remaining bytes
                generator.add_instruction(Instruction::Label("slice_remaining".to_string()));
                // T1 already contains available bytes

                // Label for allocating the result
                generator.add_instruction(Instruction::Label("slice_allocate".to_string()));
                // Allocate memory for the new bytestring (length + data)
                generator.add_instruction(Instruction::Addi(Register::A0, Register::T1, 4)); // 4 bytes for length + data
                generator.add_instruction(Instruction::Jal(Register::Ra, "malloc".to_string()));

                // Store the length at the beginning of the new bytestring
                generator.add_instruction(Instruction::Sw(Register::T1, 0, Register::A0));

                // Save the result pointer
                generator.add_instruction(Instruction::Mv(Register::T3, Register::A0));

                // Copy the slice
                generator.add_instruction(Instruction::Lw(Register::A1, 0, Register::Sp)); // Original bytestring
                generator.add_instruction(Instruction::Lw(Register::A2, 8, Register::Sp)); // Start index
                generator.add_instruction(Instruction::Addi(Register::A0, Register::T3, 4)); // Destination (result + 4)
                generator.add_instruction(Instruction::Add(
                    Register::A1,
                    Register::A1,
                    Register::A2,
                )); // Source (original + start)
                generator.add_instruction(Instruction::Addi(Register::A1, Register::A1, 4)); // Adjust for header
                generator.add_instruction(Instruction::Mv(Register::A2, Register::T1)); // Length to copy
                generator.add_instruction(Instruction::Jal(Register::Ra, "memcpy".to_string()));

                generator
                    .add_instruction(Instruction::Jal(Register::Zero, "slice_done".to_string()));

                // Label for empty slice
                generator.add_instruction(Instruction::Label("slice_empty".to_string()));
                // Allocate memory for an empty bytestring (just the length field)
                generator.add_instruction(Instruction::Li(Register::A0, 4)); // 4 bytes for length
                generator.add_instruction(Instruction::Jal(Register::Ra, "malloc".to_string()));

                // Store zero length
                generator.add_instruction(Instruction::Sw(Register::Zero, 0, Register::A0));

                // Save the result pointer
                generator.add_instruction(Instruction::Mv(Register::T3, Register::A0));

                // Label for cleanup
                generator.add_instruction(Instruction::Label("slice_done".to_string()));

                // Clean up the stack and store the result
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 8)); // Pop two values
                generator.add_instruction(Instruction::Sw(Register::T3, 0, Register::Sp));
                // Store result
            },
            DefaultFunction::LengthOfByteString => {
                // Load the bytestring pointer
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp));

                // Load the length field
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::A0));

                // Store the length on the stack
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
            },
            DefaultFunction::IndexByteString => {
                // Load the arguments
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp)); // Index
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp)); // ByteString

                // Load the length of the bytestring
                generator.add_instruction(Instruction::Lw(Register::T0, 0, Register::A1)); // Length of bytestring

                // Check if index is out of bounds
                generator.add_instruction(Instruction::Bge(
                    Register::A0,
                    Register::T0,
                    "index_error".to_string(),
                ));

                // Load the byte at the specified index
                generator.add_instruction(Instruction::Add(
                    Register::A0,
                    Register::A0,
                    Register::A1,
                )); // ByteString + index
                generator.add_instruction(Instruction::Addi(Register::A0, Register::A0, 4)); // Adjust for header
                generator.add_instruction(Instruction::Lbu(Register::A0, 0, Register::A0)); // Load byte

                generator
                    .add_instruction(Instruction::Jal(Register::Zero, "index_done".to_string()));

                // Label for index error
                generator.add_instruction(Instruction::Label("index_error".to_string()));
                // Return 0 for out of bounds (could be changed to error handling)
                generator.add_instruction(Instruction::Li(Register::A0, 0));

                // Label for cleanup
                generator.add_instruction(Instruction::Label("index_done".to_string()));

                // Clean up the stack and store the result
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4)); // Pop one value
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
                // Store result
            },
            DefaultFunction::EqualsByteString => {
                // Load the two bytestring pointers
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp)); // Second bytestring
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp)); // First bytestring

                // Load the lengths of both bytestrings
                generator.add_instruction(Instruction::Lw(Register::T0, 0, Register::A0)); // Length of second bytestring
                generator.add_instruction(Instruction::Lw(Register::T1, 0, Register::A1)); // Length of first bytestring

                // Check if lengths are equal
                generator.add_instruction(Instruction::Bne(
                    Register::T0,
                    Register::T1,
                    "not_equal".to_string(),
                ));

                // Compare the contents
                generator.add_instruction(Instruction::Addi(Register::A0, Register::A0, 4)); // Adjust for header
                generator.add_instruction(Instruction::Addi(Register::A1, Register::A1, 4)); // Adjust for header
                generator.add_instruction(Instruction::Mv(Register::A2, Register::T0)); // Length to compare
                generator.add_instruction(Instruction::Jal(Register::Ra, "memcmp".to_string()));

                // Check the result of memcmp (0 means equal)
                generator.add_instruction(Instruction::Seqz(Register::A0, Register::A0));

                generator
                    .add_instruction(Instruction::Jal(Register::Zero, "equals_done".to_string()));

                // Label for not equal
                generator.add_instruction(Instruction::Label("not_equal".to_string()));
                generator.add_instruction(Instruction::Li(Register::A0, 0)); // False

                // Label for cleanup
                generator.add_instruction(Instruction::Label("equals_done".to_string()));

                // Clean up the stack and store the result
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4)); // Pop one value
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
                // Store result
            },
            DefaultFunction::LessThanByteString => {
                // Load the two bytestring pointers
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp)); // Second bytestring
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp)); // First bytestring

                // Compare the contents
                generator.add_instruction(Instruction::Addi(Register::A0, Register::A0, 4)); // Adjust for header
                generator.add_instruction(Instruction::Addi(Register::A1, Register::A1, 4)); // Adjust for header
                generator.add_instruction(Instruction::Jal(Register::Ra, "memcmp".to_string()));

                // Check the result of memcmp (negative means first < second)
                generator.add_instruction(Instruction::Slti(Register::A0, Register::A0, 0));

                // Clean up the stack and store the result
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4)); // Pop one value
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
                // Store result
            },
            DefaultFunction::LessThanEqualsByteString => {
                // Load the two bytestring pointers
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::Sp)); // Second bytestring
                generator.add_instruction(Instruction::Lw(Register::A1, 4, Register::Sp)); // First bytestring

                // Compare the contents
                generator.add_instruction(Instruction::Addi(Register::A0, Register::A0, 4)); // Adjust for header
                generator.add_instruction(Instruction::Addi(Register::A1, Register::A1, 4)); // Adjust for header
                generator.add_instruction(Instruction::Jal(Register::Ra, "memcmp".to_string()));

                // Check the result of memcmp (non-positive means first <= second)
                generator.add_instruction(Instruction::Slti(Register::A0, Register::A0, 1));

                // Clean up the stack and store the result
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 4)); // Pop one value
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
                // Store result
            },
            DefaultFunction::IfThenElse => {
                // Check if condition (A0) is a boolean (type = 0)
                let _ = generator.add_instruction(Instruction::Lw(Register::T0, 0, Register::A0));

                // Check if type is boolean (0)
                let _ = generator.add_instruction(Instruction::Li(Register::T1, 0));
                let _ = generator.add_instruction(Instruction::Bne(
                    Register::T0,
                    Register::T1,
                    "if_error_call".to_string(),
                ));

                // Load the boolean value (true = 1, false = 0)
                let _ = generator.add_instruction(Instruction::Lw(Register::T0, 4, Register::A0));

                // Branch based on condition
                let _ = generator.add_instruction(Instruction::Beq(
                    Register::T0,
                    Register::Zero,
                    "if_false_call".to_string(),
                ));

                // If true, use then-branch (A1)
                let _ = generator.add_instruction(Instruction::Mv(Register::A0, Register::A1));
                let _ = generator
                    .add_instruction(Instruction::Jal(Register::Zero, "if_done_call".to_string()));

                // If false, use else-branch (already in A0)
                let _ = generator.add_instruction(Instruction::Label("if_false_call".to_string()));

                // End of if-then-else
                let _ = generator.add_instruction(Instruction::Label("if_done_call".to_string()));

                // Error handling for non-boolean condition
                let _ = generator.add_instruction(Instruction::Label("if_error_call".to_string()));

                // Default to false branch for now
                // A0 already contains the else-branch
            },
            DefaultFunction::MkPairData => {
                // Allocate memory for the pair (8 bytes)
                generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, -8));

                // A0 contains the first element, A1 contains the second element
                // Store the first element at offset 0
                generator.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));

                // Store the second element at offset 4
                generator.add_instruction(Instruction::Sw(Register::A1, 4, Register::Sp));

                // Return the pointer to the pair (stack pointer)
                generator.add_instruction(Instruction::Mv(Register::A0, Register::Sp));
            },
            DefaultFunction::FstPair => {
                // A0 contains the pair pointer
                // Check if the pair pointer is null
                generator.add_instruction(Instruction::Beq(
                    Register::A0,
                    Register::Zero,
                    "fst_pair_error".to_string(),
                ));

                // Load the first element from the pair (first 4 bytes)
                generator.add_instruction(Instruction::Lw(Register::A0, 0, Register::A0));

                // Jump to end of function
                generator
                    .add_instruction(Instruction::Jal(Register::Zero, "fst_pair_end".to_string()));

                // Error handling for null pair
                generator.add_instruction(Instruction::Label("fst_pair_error".to_string()));

                // Load error code for null pair
                generator.add_instruction(Instruction::Li(Register::A0, -1)); // Error code for null pair

                // End of function
                generator.add_instruction(Instruction::Label("fst_pair_end".to_string()));
            },
            DefaultFunction::SndPair => {
                // A0 contains the pair pointer
                // Check if the pair pointer is null
                generator.add_instruction(Instruction::Beq(
                    Register::A0,
                    Register::Zero,
                    "snd_pair_error".to_string(),
                ));

                // Load the second element from the pair (second 4 bytes)
                generator.add_instruction(Instruction::Lw(Register::A0, 4, Register::A0));

                // Jump to end of function
                generator
                    .add_instruction(Instruction::Jal(Register::Zero, "snd_pair_end".to_string()));

                // Error handling for null pair
                generator.add_instruction(Instruction::Label("snd_pair_error".to_string()));

                // Load error code for null pair
                generator.add_instruction(Instruction::Li(Register::A0, -1)); // Error code for null pair

                // End of function
                generator.add_instruction(Instruction::Label("snd_pair_end".to_string()));
            },
            _ => {
                return Err(CompilationError::UnsupportedFeature(format!(
                    "Unsupported builtin function: {:?}",
                    builtin
                )));
            },
        }

        Ok(())
    }

    /// Parse UPLC code into a Term
    #[allow(dead_code)]
    fn parse_uplc(&self, uplc_code: &str) -> Result<Program<Name>> {
        parser::program(uplc_code).map_err(|err| CompilationError::Parse(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let _compiler = Compiler::new();
        // Just testing that we can create a compiler instance for now
    }

    #[test]
    fn test_compile_simple_term() {
        let compiler = Compiler::new();
        let uplc_code = "(program 1.0.0 (con integer 42))";
        let result = compiler.compile(uplc_code);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_with_evaluation() {
        let compiler = Compiler::new();
        let uplc_code = "(program\n  1.0.0\n  [(lam x x) (con integer 42)]\n)";
        let result = compiler.compile(uplc_code);
        if let Err(ref err) = result {
            println!("Error: {:?}", err);
        }
        assert!(result.is_ok());
    }

    #[test]
    fn test_compile_single_line_format() {
        let compiler = Compiler::new();
        let uplc_code = "(program 1.0.0 (con integer 42))";
        let result = compiler.compile(uplc_code);
        assert!(result.is_ok());
    }
}
