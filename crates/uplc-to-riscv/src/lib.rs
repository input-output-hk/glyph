//! UPLC to RISC-V compiler
//!
//! This crate provides functionality to compile Untyped Plutus Core (UPLC)
//! to RISC-V assembly code.

pub mod bitvm_emulator;
pub mod bitvm_verification;
mod ir;
pub mod trace_generator;
extern crate bitvm_common;

// Use ExecutionTrace and ExecutionStep from bitvm-common
pub use bitvm_common::trace::{ExecutionStep, ExecutionTrace};

use ir::{lower_to_ir, IRInstr};
use risc_v_gen::{CodeGenerator, Instruction, Register};
use thiserror::Error;
use uplc::ast::{DeBruijn, Name, NamedDeBruijn, Program, Term};
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

    #[error("BitVMX code generation error: {0}")]
    BitVMXCodeGen(#[from] risc_v_gen::BitVMXCodeGenError),

    #[error("Unsupported UPLC feature: {0}")]
    UnsupportedFeature(String),

    #[error("Invalid UPLC input: {0}")]
    InvalidInput(String),

    #[error("Evaluation error: {0}")]
    Evaluation(String),

    #[error("Term conversion error: {0}")]
    TermConversion(String),

    #[error("BitVMX emulator error: {0}")]
    BitVMXEmulator(#[from] bitvm_emulator::BitVMXEmulatorError),
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
        self.compile_bitvm(&program)
    }

    /// Compile UPLC code to RISC-V assembly with execution trace
    pub fn compile_with_trace(&self, uplc_code: &str) -> Result<(String, ExecutionTrace)> {
        // Parse the UPLC code
        let program = self.parse_uplc(uplc_code)?;

        // Compile with BitVMX compatibility
        self.compile_bitvm_with_trace(&program)
    }

    /// Compile with BitVMX compatibility
    fn compile_bitvm(&self, program: &Program<Name>) -> Result<String> {
        // Convert Term<Name> to Term<DeBruijn>
        let debruijn_program = Program {
            version: program.version.clone(),
            term: TryInto::<Term<NamedDeBruijn>>::try_into(program.term.clone())
                .map_err(|e| CompilationError::TermConversion(e.to_string()))?,
        };

        // Lower to IR
        let debruijn_term: Term<DeBruijn> = debruijn_program.term.into();
        let ir_instructions = lower_to_ir(&debruijn_term);

        // Create a BitVMX code generator
        let mut bitvm_generator = risc_v_gen::BitVMXCodeGenerator::new();

        // Create an execution trace
        let mut execution_trace = ExecutionTrace::new();

        // Generate program setup (data segment, entry point, etc.)
        self.generate_bitvm_program_setup(&mut bitvm_generator, program)?;

        // Compile the optimized IR to RISC-V instructions
        self.compile_bitvm_ir(&mut bitvm_generator, &ir_instructions, &mut execution_trace)?;

        // Generate program teardown (return, etc.)
        self.generate_bitvm_program_teardown(&mut bitvm_generator)?;

        // Create a dispute resolver
        // let dispute_resolver = DisputeResolver::new(execution_trace.clone());

        // Generate both standard assembly and BitVMX trace
        let assembly = bitvm_generator.generate_assembly();
        let bitvm_trace = bitvm_generator.generate_bitvm_trace();

        // Combine both outputs
        let result = format!(
            "# Standard RISC-V Assembly\n\n{}\n\n# BitVMX Trace\n\n{}\n\n# Execution Trace\n\n# ---------------\n\n# {} steps\n",
            assembly,
            bitvm_trace,
            execution_trace.len()
        );

        Ok(result)
    }

    /// Compile a UPLC program to RISC-V assembly with BitVMX compatibility and return the execution trace
    fn compile_bitvm_with_trace(
        &self,
        program: &Program<Name>,
    ) -> Result<(String, ExecutionTrace)> {
        // Convert Term<Name> to Term<DeBruijn>
        let debruijn_program = Program {
            version: program.version.clone(),
            term: TryInto::<Term<NamedDeBruijn>>::try_into(program.term.clone())
                .map_err(|e| CompilationError::TermConversion(e.to_string()))?,
        };

        // Lower to IR
        let debruijn_term: Term<DeBruijn> = debruijn_program.term.into();
        let ir_instructions = lower_to_ir(&debruijn_term);

        // Create a BitVMX code generator
        let mut bitvm_generator = risc_v_gen::BitVMXCodeGenerator::new();

        // Create an execution trace
        let mut execution_trace = ExecutionTrace::new();

        // Generate program setup (data segment, entry point, etc.)
        self.generate_bitvm_program_setup(&mut bitvm_generator, program)?;

        // Compile the optimized IR to RISC-V instructions
        self.compile_bitvm_ir(&mut bitvm_generator, &ir_instructions, &mut execution_trace)?;

        // Generate program teardown (return, etc.)
        self.generate_bitvm_program_teardown(&mut bitvm_generator)?;

        // Generate both standard assembly and BitVMX trace
        let assembly = bitvm_generator.generate_assembly();
        let bitvm_trace = bitvm_generator.generate_bitvm_trace();

        // Combine both outputs
        let result = format!(
            "# Standard RISC-V Assembly\n\n{}\n\n# BitVMX Trace\n\n{}\n\n# Execution Trace\n\n# ---------------\n\n# {} steps\n",
            assembly,
            bitvm_trace,
            execution_trace.len()
        );

        Ok((result, execution_trace))
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

    /// Generate program setup for BitVMX
    fn generate_bitvm_program_setup(
        &self,
        generator: &mut risc_v_gen::BitVMXCodeGenerator,
        program: &Program<Name>,
    ) -> Result<()> {
        // Set up the program header
        let _ = generator
            .add_instruction(risc_v_gen::Instruction::Comment(
                format!("UPLC Program (version {:?})", program.version).to_string(),
            ))
            .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
        let _ = generator
            .add_instruction(risc_v_gen::Instruction::Global("_start".to_string()))
            .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
        let _ = generator
            .add_instruction(risc_v_gen::Instruction::Label("_start".to_string()))
            .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

        // Set up the stack pointer
        let _ = generator
            .add_instruction(risc_v_gen::Instruction::Li(
                risc_v_gen::Register::Sp,
                0x10000,
            ))
            .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

        // Set up the environment pointer (S0)
        let _ = generator
            .add_instruction(risc_v_gen::Instruction::Mv(
                risc_v_gen::Register::S0,
                risc_v_gen::Register::Sp,
            ))
            .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

        Ok(())
    }

    /// Generate program teardown for BitVMX
    fn generate_bitvm_program_teardown(
        &self,
        generator: &mut risc_v_gen::BitVMXCodeGenerator,
    ) -> Result<()> {
        // Set up the program footer
        let _ = generator
            .add_instruction(risc_v_gen::Instruction::Label("exit".to_string()))
            .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
        let _ = generator
            .add_instruction(risc_v_gen::Instruction::Li(risc_v_gen::Register::A0, 0))
            .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
        let _ = generator
            .add_instruction(risc_v_gen::Instruction::Li(risc_v_gen::Register::A7, 93)) // exit syscall
            .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
        let _ = generator
            .add_instruction(risc_v_gen::Instruction::Comment(
                "End of program".to_string(),
            ))
            .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

        Ok(())
    }

    /// Compile IR to RISC-V instructions with BitVMX compatibility
    fn compile_bitvm_ir(
        &self,
        generator: &mut risc_v_gen::BitVMXCodeGenerator,
        instructions: &[IRInstr],
        execution_trace: &mut ExecutionTrace,
    ) -> Result<()> {
        // Set initial segment type to ReadOnly for code
        generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);

        for instruction in instructions {
            match instruction {
                IRInstr::PushConst(value) => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(format!(
                            "Push constant {}",
                            value
                        )))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Load the constant into a register
                    generator
                        .add_instruction(risc_v_gen::Instruction::Li(
                            risc_v_gen::Register::A0,
                            *value as i32,
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-write segment for stack operations
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Push the value onto the stack
                    generator
                        .add_instruction(risc_v_gen::Instruction::Addi(
                            risc_v_gen::Register::Sp,
                            risc_v_gen::Register::Sp,
                            -4,
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
                    generator
                        .add_instruction(risc_v_gen::Instruction::Sw(
                            risc_v_gen::Register::A0,
                            0,
                            risc_v_gen::Register::Sp,
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch back to read-only segment for code
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);

                    // Create execution step
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read
                    .with_write(0x2000, 0x3000); // Placeholder memory write
                    execution_trace.add_step(step);
                },
                IRInstr::PushBool(value) => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(format!(
                            "Push boolean {}",
                            value
                        )))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Load boolean into register
                    generator
                        .add_instruction(risc_v_gen::Instruction::Li(
                            risc_v_gen::Register::A0,
                            if *value { 1 } else { 0 },
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-write segment for stack operations
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Push the value onto the stack
                    generator
                        .add_instruction(risc_v_gen::Instruction::Addi(
                            risc_v_gen::Register::Sp,
                            risc_v_gen::Register::Sp,
                            -4,
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
                    generator
                        .add_instruction(risc_v_gen::Instruction::Sw(
                            risc_v_gen::Register::A0,
                            0,
                            risc_v_gen::Register::Sp,
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch back to read-only segment for code
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);

                    // Create execution step
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read
                    .with_write(0x2000, if *value { 1 } else { 0 }); // Placeholder memory write
                    execution_trace.add_step(step);
                },
                IRInstr::PushVar(name) => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(format!(
                            "Push variable {}",
                            name
                        )))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-write segment for memory read
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Load variable from environment (for now, just load from a fixed offset)
                    let lw_instr = risc_v_gen::Instruction::Lw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::S0,
                    );
                    generator
                        .add_instruction(lw_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for the load instruction
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (S0)
                    .with_read2(0x2000, 0x3000, 0); // Placeholder memory read
                    execution_trace.add_step(step);

                    // Push the value onto the stack
                    generator
                        .add_instruction(risc_v_gen::Instruction::Addi(
                            risc_v_gen::Register::Sp,
                            risc_v_gen::Register::Sp,
                            -4,
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
                    generator
                        .add_instruction(risc_v_gen::Instruction::Sw(
                            risc_v_gen::Register::A0,
                            0,
                            risc_v_gen::Register::Sp,
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for the store instruction
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read (SP)
                    .with_write(0x2000, 0x3000); // Placeholder memory write
                    execution_trace.add_step(step);

                    // Switch back to read-only segment
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);
                },
                IRInstr::CallBuiltin(builtin) => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(format!(
                            "Call builtin {:?}",
                            builtin
                        )))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-write segment for memory reads
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Load arguments from stack
                    let lw_instr1 = risc_v_gen::Instruction::Lw(
                        risc_v_gen::Register::A1,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(lw_instr1.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for first load
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (SP)
                    .with_read2(0x2000, 0x3000, 0); // Placeholder memory read
                    execution_trace.add_step(step);

                    // Increment stack pointer
                    let addi_instr1 = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        4,
                    );
                    generator
                        .add_instruction(addi_instr1.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    let lw_instr2 = risc_v_gen::Instruction::Lw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(lw_instr2.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for second load
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (SP)
                    .with_read2(0x2000, 0x3000, 0); // Placeholder memory read
                    execution_trace.add_step(step);

                    // Increment stack pointer
                    let addi_instr2 = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        4,
                    );
                    generator
                        .add_instruction(addi_instr2.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch back to read-only segment for computation
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);

                    // Perform builtin operation
                    match builtin {
                        DefaultFunction::AddInteger => {
                            let add_instr = risc_v_gen::Instruction::Add(
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A1,
                            );
                            generator
                                .add_instruction(add_instr.clone())
                                .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                            // Create execution step for add operation
                            let pc = generator.current_pc();
                            let step = ExecutionStep::new(
                                pc,
                                pc,
                                0x12345678,
                                pc + 4,
                                execution_trace.current_step,
                            )
                            .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                            .with_read2(0x1004, 0x2004, 0); // Placeholder register read (A1)
                            execution_trace.add_step(step);
                        },
                        DefaultFunction::SubtractInteger => {
                            let sub_instr = risc_v_gen::Instruction::Sub(
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A1,
                                risc_v_gen::Register::A0,
                            );
                            generator
                                .add_instruction(sub_instr.clone())
                                .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                            // Create execution step for subtract operation
                            let pc = generator.current_pc();
                            let step = ExecutionStep::new(
                                pc,
                                pc,
                                0x12345678,
                                pc + 4,
                                execution_trace.current_step,
                            )
                            .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A1)
                            .with_read2(0x1004, 0x2004, 0); // Placeholder register read (A0)
                            execution_trace.add_step(step);
                        },
                        DefaultFunction::MultiplyInteger => {
                            let mul_instr = risc_v_gen::Instruction::Mul(
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A1,
                            );
                            generator
                                .add_instruction(mul_instr.clone())
                                .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                            // Create execution step for multiply operation
                            let pc = generator.current_pc();
                            let step = ExecutionStep::new(
                                pc,
                                pc,
                                0x12345678,
                                pc + 4,
                                execution_trace.current_step,
                            )
                            .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                            .with_read2(0x1004, 0x2004, 0); // Placeholder register read (A1)
                            execution_trace.add_step(step);
                        },
                        DefaultFunction::EqualsInteger => {
                            // Subtract the two values and check if the result is zero
                            let sub_instr = risc_v_gen::Instruction::Sub(
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A1,
                            );
                            generator
                                .add_instruction(sub_instr.clone())
                                .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                            // Create execution step for subtract operation
                            let pc = generator.current_pc();
                            let step = ExecutionStep::new(
                                pc,
                                pc,
                                0x12345678,
                                pc + 4,
                                execution_trace.current_step,
                            )
                            .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                            .with_read2(0x1004, 0x2004, 0); // Placeholder register read (A1)
                            execution_trace.add_step(step);

                            // Check if result is zero (set to 1 if zero, 0 otherwise)
                            let seqz_instr = risc_v_gen::Instruction::Seqz(
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A0,
                            );
                            generator
                                .add_instruction(seqz_instr.clone())
                                .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                            // Create execution step for seqz operation
                            let pc = generator.current_pc();
                            let step = ExecutionStep::new(
                                pc,
                                pc,
                                0x12345678,
                                pc + 4,
                                execution_trace.current_step,
                            )
                            .with_read1(0x1000, 0x2000, 0); // Placeholder register read (A0)
                            execution_trace.add_step(step);
                        },
                        DefaultFunction::LessThanInteger => {
                            // Set A0 to 1 if A0 < A1, 0 otherwise
                            let slt_instr = risc_v_gen::Instruction::Slt(
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A1,
                            );
                            generator
                                .add_instruction(slt_instr.clone())
                                .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                            // Create execution step for slt operation
                            let pc = generator.current_pc();
                            let step = ExecutionStep::new(
                                pc,
                                pc,
                                0x12345678,
                                pc + 4,
                                execution_trace.current_step,
                            )
                            .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                            .with_read2(0x1004, 0x2004, 0); // Placeholder register read (A1)
                            execution_trace.add_step(step);
                        },
                        DefaultFunction::LessThanEqualsInteger => {
                            // Set A0 to 1 if A1 < A0, 0 otherwise
                            let slt_instr = risc_v_gen::Instruction::Slt(
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A1,
                                risc_v_gen::Register::A0,
                            );
                            generator
                                .add_instruction(slt_instr.clone())
                                .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                            // Create execution step for slt operation
                            let pc = generator.current_pc();
                            let step = ExecutionStep::new(
                                pc,
                                pc,
                                0x12345678,
                                pc + 4,
                                execution_trace.current_step,
                            )
                            .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A1)
                            .with_read2(0x1004, 0x2004, 0); // Placeholder register read (A0)
                            execution_trace.add_step(step);

                            // Invert the result (1 if A0 <= A1, 0 otherwise)
                            let xori_instr = risc_v_gen::Instruction::Xori(
                                risc_v_gen::Register::A0,
                                risc_v_gen::Register::A0,
                                1,
                            );
                            generator
                                .add_instruction(xori_instr.clone())
                                .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                            // Create execution step for xori operation
                            let pc = generator.current_pc();
                            let step = ExecutionStep::new(
                                pc,
                                pc,
                                0x12345678,
                                pc + 4,
                                execution_trace.current_step,
                            )
                            .with_read1(0x1000, 0x2000, 0); // Placeholder register read (A0)
                            execution_trace.add_step(step);
                        },
                        DefaultFunction::IfThenElse => {
                            // Check if condition (A0) is a boolean (type = 0)
                            let _ = generator.add_instruction(Instruction::Lw(
                                Register::T0,
                                0,
                                Register::A0,
                            ));

                            // Check if type is boolean (0)
                            let _ = generator.add_instruction(Instruction::Li(Register::T1, 0));
                            let _ = generator.add_instruction(Instruction::Bne(
                                Register::T0,
                                Register::T1,
                                "if_error_call".to_string(),
                            ));

                            // Load the boolean value (true = 1, false = 0)
                            let _ = generator.add_instruction(Instruction::Lw(
                                Register::T0,
                                4,
                                Register::A0,
                            ));

                            // Branch based on condition
                            let _ = generator.add_instruction(Instruction::Beq(
                                Register::T0,
                                Register::Zero,
                                "if_false_call".to_string(),
                            ));

                            // If true, use then-branch (A1)
                            let _ = generator
                                .add_instruction(Instruction::Mv(Register::A0, Register::A1));
                            let _ = generator.add_instruction(Instruction::Jal(
                                Register::Zero,
                                "if_done_call".to_string(),
                            ));

                            // If false, use else-branch (already in A0)
                            let _ = generator
                                .add_instruction(Instruction::Label("if_false_call".to_string()));

                            // End of if-then-else
                            let _ = generator
                                .add_instruction(Instruction::Label("if_done_call".to_string()));

                            // Error handling for non-boolean condition
                            let _ = generator
                                .add_instruction(Instruction::Label("if_error_call".to_string()));

                            // Default to false branch for now
                            // A0 already contains the else-branch
                        },
                        DefaultFunction::MkCons => {
                            // Create a new cons cell
                            // A0 = head, A1 = tail
                            let _ = generator
                                .add_instruction(Instruction::Mv(Register::T0, Register::A0));

                            // Allocate memory for the cons cell (8 bytes)
                            let _ = generator.add_instruction(Instruction::Li(Register::A0, 8));
                            let _ = generator.add_instruction(Instruction::Jal(
                                Register::Ra,
                                "malloc".to_string(),
                            ));

                            // Store head in the first 4 bytes
                            let _ = generator.add_instruction(Instruction::Sw(
                                Register::T0,
                                0,
                                Register::A0,
                            ));

                            // Store tail in the second 4 bytes
                            let _ = generator.add_instruction(Instruction::Sw(
                                Register::A1,
                                4,
                                Register::A0,
                            ));
                        },
                        DefaultFunction::HeadList => {
                            // A0 contains the list pointer
                            // Check if the list pointer is null
                            let _ = generator.add_instruction(Instruction::Beq(
                                Register::A0,
                                Register::Zero,
                                "head_list_error".to_string(),
                            ));

                            // Load the head element from the list (first 4 bytes)
                            let _ = generator.add_instruction(Instruction::Lw(
                                Register::A0,
                                0,
                                Register::A0,
                            ));

                            // Jump to end of function
                            let _ = generator.add_instruction(Instruction::Jal(
                                Register::Zero,
                                "head_list_end".to_string(),
                            ));

                            // Error handling for null list
                            let _ = generator
                                .add_instruction(Instruction::Label("head_list_error".to_string()));

                            // Load error code for empty list
                            let _ = generator.add_instruction(Instruction::Li(Register::A0, -1)); // Error code for empty list

                            // End of function
                            let _ = generator
                                .add_instruction(Instruction::Label("head_list_end".to_string()));
                        },
                        DefaultFunction::TailList => {
                            // A0 contains the list pointer
                            // Check if the list pointer is null
                            let _ = generator.add_instruction(Instruction::Beq(
                                Register::A0,
                                Register::Zero,
                                "tail_list_error".to_string(),
                            ));

                            // Load the tail element from the list (second 4 bytes)
                            let _ = generator.add_instruction(Instruction::Lw(
                                Register::A0,
                                4,
                                Register::A0,
                            ));

                            // Jump to end of function
                            let _ = generator.add_instruction(Instruction::Jal(
                                Register::Zero,
                                "tail_list_end".to_string(),
                            ));

                            // Error handling for null list
                            let _ = generator
                                .add_instruction(Instruction::Label("tail_list_error".to_string()));

                            // Load error code for empty list
                            let _ = generator.add_instruction(Instruction::Li(Register::A0, -1)); // Error code for empty list

                            // End of function
                            let _ = generator
                                .add_instruction(Instruction::Label("tail_list_end".to_string()));
                        },
                        DefaultFunction::NullList => {
                            // A0 contains the list pointer
                            // Check if the list pointer is null (0 = true, 1 = false)
                            let _ = generator
                                .add_instruction(Instruction::Seqz(Register::A0, Register::A0));
                        },
                        DefaultFunction::MkPairData => {
                            // Create a new pair
                            // A0 = first element, A1 = second element
                            let _ = generator.add_instruction(Instruction::Addi(
                                Register::Sp,
                                Register::Sp,
                                -8,
                            ));

                            // Store the elements on the stack
                            let _ = generator.add_instruction(Instruction::Sw(
                                Register::A0,
                                0,
                                Register::Sp,
                            ));

                            // Store the second element
                            let _ = generator.add_instruction(Instruction::Sw(
                                Register::A1,
                                4,
                                Register::Sp,
                            ));

                            // Return the stack pointer as the pair pointer
                            let _ = generator
                                .add_instruction(Instruction::Mv(Register::A0, Register::Sp));
                        },
                        DefaultFunction::FstPair => {
                            // A0 contains the pair pointer
                            // Check if the pair pointer is null
                            let _ = generator.add_instruction(Instruction::Beq(
                                Register::A0,
                                Register::Zero,
                                "fst_pair_error".to_string(),
                            ));

                            // Load the first element from the pair (first 4 bytes)
                            let _ = generator.add_instruction(Instruction::Lw(
                                Register::A0,
                                0,
                                Register::A0,
                            ));

                            // Jump to end of function
                            let _ = generator.add_instruction(Instruction::Jal(
                                Register::Zero,
                                "fst_pair_end".to_string(),
                            ));

                            // Error handling for null pair
                            let _ = generator
                                .add_instruction(Instruction::Label("fst_pair_error".to_string()));

                            // Load error code for null pair
                            let _ = generator.add_instruction(Instruction::Li(Register::A0, -1)); // Error code for null pair

                            // End of function
                            let _ = generator
                                .add_instruction(Instruction::Label("fst_pair_end".to_string()));
                        },
                        DefaultFunction::SndPair => {
                            // A0 contains the pair pointer
                            // Check if the pair pointer is null
                            let _ = generator.add_instruction(Instruction::Beq(
                                Register::A0,
                                Register::Zero,
                                "snd_pair_error".to_string(),
                            ));

                            // Load the second element from the pair (second 4 bytes)
                            let _ = generator.add_instruction(Instruction::Lw(
                                Register::A0,
                                4,
                                Register::A0,
                            ));

                            // Jump to end of function
                            let _ = generator.add_instruction(Instruction::Jal(
                                Register::Zero,
                                "snd_pair_end".to_string(),
                            ));

                            // Error handling for null pair
                            let _ = generator
                                .add_instruction(Instruction::Label("snd_pair_error".to_string()));

                            // Load error code for null pair
                            let _ = generator.add_instruction(Instruction::Li(Register::A0, -1)); // Error code for null pair

                            // End of function
                            let _ = generator
                                .add_instruction(Instruction::Label("snd_pair_end".to_string()));
                        },
                        _ => {
                            return Err(CompilationError::UnsupportedFeature(format!(
                                "Unsupported builtin function: {:?}",
                                builtin
                            )));
                        },
                    }

                    // Switch to read-write segment for stack write
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Decrement stack pointer
                    let addi_instr3 = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        -4,
                    );
                    generator
                        .add_instruction(addi_instr3.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Store result on stack
                    let sw_instr = risc_v_gen::Instruction::Sw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(sw_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for store
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read (SP)
                    .with_write(0x2000, 0x3000); // Placeholder memory write
                    execution_trace.add_step(step);

                    // Switch back to read-only segment
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);
                },
                IRInstr::Apply => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(
                            "Apply function to argument".to_string(),
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-write segment for memory reads
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Load argument from stack
                    let lw_instr1 = risc_v_gen::Instruction::Lw(
                        risc_v_gen::Register::A1,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(lw_instr1.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for argument load
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (SP)
                    .with_read2(0x2000, 0x3000, 0); // Placeholder memory read
                    execution_trace.add_step(step);

                    // Increment stack pointer
                    let addi_instr1 = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        4,
                    );
                    generator
                        .add_instruction(addi_instr1.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Load function from stack
                    let lw_instr2 = risc_v_gen::Instruction::Lw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(lw_instr2.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for function load
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (SP)
                    .with_read2(0x2000, 0x3000, 0); // Placeholder memory read
                    execution_trace.add_step(step);

                    // Increment stack pointer
                    let addi_instr2 = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        4,
                    );
                    generator
                        .add_instruction(addi_instr2.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-only segment for function call
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);

                    // Call the function (indirect jump)
                    let jalr_instr = risc_v_gen::Instruction::Jalr(
                        risc_v_gen::Register::Ra,
                        risc_v_gen::Register::A0,
                        0,
                    );
                    generator
                        .add_instruction(jalr_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for function call
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        0x1000,
                        execution_trace.current_step,
                    ) // Placeholder next PC
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                    .with_read2(0x1004, 0x2004, 0); // Placeholder register read (Ra)
                    execution_trace.add_step(step);

                    // Switch to read-write segment for stack write
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Decrement stack pointer
                    let addi_instr3 = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        -4,
                    );
                    generator
                        .add_instruction(addi_instr3.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Store result on stack
                    let sw_instr = risc_v_gen::Instruction::Sw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(sw_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for store
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read (SP)
                    .with_write(0x2000, 0x3000); // Placeholder memory write
                    execution_trace.add_step(step);

                    // Switch back to read-only segment
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);
                },
                IRInstr::Lambda(name) => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(format!(
                            "Create lambda {}",
                            name
                        )))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-write segment for memory writes
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Store the current environment pointer
                    let sw_instr1 = risc_v_gen::Instruction::Sw(
                        risc_v_gen::Register::S0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(sw_instr1.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for environment store
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (S0)
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read (SP)
                    .with_write(0x2000, 0x3000); // Placeholder memory write
                    execution_trace.add_step(step);

                    // Decrement stack pointer for closure object
                    let addi_instr = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        -8,
                    );
                    generator
                        .add_instruction(addi_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for stack pointer update
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read
                    .with_write(0x1000, 0x2004); // Placeholder register write
                    execution_trace.add_step(step);

                    // Switch back to read-only segment
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);
                },
                IRInstr::Return => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(
                            "Return from function".to_string(),
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-write segment for memory read
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Load return value from stack
                    let lw_instr = risc_v_gen::Instruction::Lw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(lw_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for return value load
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (SP)
                    .with_read2(0x2000, 0x3000, 0); // Placeholder memory read
                    execution_trace.add_step(step);

                    // Increment stack pointer
                    let addi_instr = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        4,
                    );
                    generator
                        .add_instruction(addi_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch back to read-only segment for return
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);

                    // Return to caller
                    let jalr_instr = risc_v_gen::Instruction::Jalr(
                        risc_v_gen::Register::Zero,
                        risc_v_gen::Register::Ra,
                        0,
                    );
                    generator
                        .add_instruction(jalr_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for return jump
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        0x1000,
                        execution_trace.current_step,
                    ) // Placeholder next PC
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (Ra)
                    .with_read2(0x1004, 0x2004, 0); // Placeholder register read
                    execution_trace.add_step(step);
                },
                IRInstr::Delay => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(
                            "Delay computation".to_string(),
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
                    // Delay creates a thunk that will be evaluated later when forced
                    // We need to allocate memory for the thunk and store the necessary information

                    // Create a thunk (delayed computation)
                    // Allocate memory for the thunk
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Addi(
                        Register::A0,
                        Register::Zero,
                        8,
                    )); // Size of thunk (2 words)
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Jal(
                        Register::Ra,
                        "malloc".to_string(),
                    ));

                    // Set the thunk type (1 = delayed computation)
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Li(Register::T0, 1));
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Sw(
                        Register::T0,
                        0,
                        Register::A0,
                    ));

                    // Store the current stack pointer (environment) in the thunk
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Sw(
                        Register::Sp,
                        4,
                        Register::A0,
                    ));

                    // Return the thunk
                    let _ = generator
                        .add_instruction(risc_v_gen::Instruction::Mv(Register::A0, Register::A0));

                    // Push the thunk on the stack
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Addi(
                        Register::Sp,
                        Register::Sp,
                        -4,
                    ));
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Sw(
                        Register::A0,
                        0,
                        Register::Sp,
                    ));
                },
                IRInstr::Force => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(
                            "Force delayed computation".to_string(),
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
                    // Force evaluates a delayed computation (thunk)

                    // Load the thunk from the stack
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Lw(
                        Register::A0,
                        0,
                        Register::Sp,
                    ));
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Addi(
                        Register::Sp,
                        Register::Sp,
                        4,
                    ));

                    // Check if it's a thunk (type = 1)
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Lw(
                        Register::T0,
                        0,
                        Register::A0,
                    ));
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Li(Register::T1, 1));
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Bne(
                        Register::T0,
                        Register::T1,
                        "force_error".to_string(),
                    ));

                    // Load the environment pointer
                    let _ = generator.add_instruction(risc_v_gen::Instruction::Lw(
                        Register::T0,
                        4,
                        Register::A0,
                    ));

                    // Save the current stack pointer
                    let _ = generator
                        .add_instruction(risc_v_gen::Instruction::Mv(Register::T1, Register::Sp));

                    // Set the stack pointer to the environment
                    let _ = generator
                        .add_instruction(risc_v_gen::Instruction::Mv(Register::Sp, Register::T0));

                    // TODO: Execute the thunk code
                    // For now, we just restore the stack pointer

                    // Restore the stack pointer
                    let _ = generator
                        .add_instruction(risc_v_gen::Instruction::Mv(Register::Sp, Register::T1));

                    // Error handling
                    let _ = generator
                        .add_instruction(risc_v_gen::Instruction::Label("force_error".to_string()));
                    // TODO: Proper error handling
                },
                IRInstr::PushByteString(bytes) => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(format!(
                            "Push bytestring of length {}",
                            bytes.len()
                        )))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-write segment for memory writes
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Allocate space for bytestring on stack
                    let addi_instr1 = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        -(bytes.len() as i32 + 4),
                    );
                    generator
                        .add_instruction(addi_instr1.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for stack allocation
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read
                    .with_write(0x1000, 0x2004); // Placeholder register write
                    execution_trace.add_step(step);

                    // Store length
                    let li_instr =
                        risc_v_gen::Instruction::Li(risc_v_gen::Register::A0, bytes.len() as i32);
                    generator
                        .add_instruction(li_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    let sw_instr1 = risc_v_gen::Instruction::Sw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(sw_instr1.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for length store
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read (SP)
                    .with_write(0x2000, bytes.len() as u32); // Store actual length
                    execution_trace.add_step(step);

                    // Store bytes (in real implementation, would need to handle byte-by-byte)
                    let sw_instr2 = risc_v_gen::Instruction::Sw(
                        risc_v_gen::Register::A0,
                        4,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(sw_instr2.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for data store
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read (SP)
                    .with_write(0x2000, 0x3000); // Placeholder memory write
                    execution_trace.add_step(step);

                    // Switch back to read-only segment
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);
                },
                IRInstr::PushString(s) => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(format!(
                            "Push string \"{}\"",
                            s
                        )))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch to read-write segment for memory writes
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Allocate space for string on stack (length + chars)
                    let addi_instr1 = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        -(s.len() as i32 + 4),
                    );
                    generator
                        .add_instruction(addi_instr1.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for stack allocation
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read
                    .with_write(0x1000, 0x2004); // Placeholder register write
                    execution_trace.add_step(step);

                    // Store length
                    let li_instr =
                        risc_v_gen::Instruction::Li(risc_v_gen::Register::A0, s.len() as i32);
                    generator
                        .add_instruction(li_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    let sw_instr1 = risc_v_gen::Instruction::Sw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(sw_instr1.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for length store
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read (SP)
                    .with_write(0x2000, s.len() as u32); // Store actual length
                    execution_trace.add_step(step);

                    // Store string data (in real implementation, would need to handle char-by-char)
                    let sw_instr2 = risc_v_gen::Instruction::Sw(
                        risc_v_gen::Register::A0,
                        4,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(sw_instr2.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for data store
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read (SP)
                    .with_write(0x2000, 0x3000); // Placeholder memory write
                    execution_trace.add_step(step);

                    // Switch back to read-only segment
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);
                },
                IRInstr::PushUnit => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment("Push unit".to_string()))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Load zero into register
                    let li_instr = risc_v_gen::Instruction::Li(risc_v_gen::Register::A0, 0);
                    generator
                        .add_instruction(li_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for the load immediate instruction
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    ); // Placeholder opcode value
                    execution_trace.add_step(step);

                    // Switch to read-write segment for memory write
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    // Decrement stack pointer
                    let addi_instr = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        -4,
                    );
                    generator
                        .add_instruction(addi_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for the stack pointer update
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0x1000, 0x2000, 0); // Placeholder register read
                    execution_trace.add_step(step);

                    // Store value on stack
                    let sw_instr = risc_v_gen::Instruction::Sw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(sw_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for the store instruction
                    let pc = generator.current_pc();
                    let step = ExecutionStep::new(
                        pc,
                        pc,
                        0x12345678,
                        pc + 4,
                        execution_trace.current_step,
                    ) // Placeholder opcode value
                    .with_read1(0x1000, 0x2000, 0) // Placeholder register read (A0)
                    .with_read2(0x1004, 0x2004, 0) // Placeholder register read (SP)
                    .with_write(0x2000, 0x3000); // Placeholder memory write
                    execution_trace.add_step(step);

                    // Switch back to read-only segment
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);
                },
                IRInstr::CaseStart(num_branches) => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(format!(
                            "Case expression with {} branches",
                            num_branches
                        )))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Load the constructor value from the stack
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadWrite);

                    let lw_instr = risc_v_gen::Instruction::Lw(
                        risc_v_gen::Register::A0,
                        0,
                        risc_v_gen::Register::Sp,
                    );
                    generator
                        .add_instruction(lw_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for loading the constructor
                    let pc = generator.current_pc();
                    let opcode_addr = pc;
                    let opcode_value = 0; // Placeholder
                    let step = ExecutionStep::new(
                        pc,
                        opcode_addr,
                        opcode_value,
                        pc + 4,
                        execution_trace.current_step,
                    )
                    .with_read1(0, 0, 0); // Placeholder values
                    execution_trace.add_step(step);

                    // Pop the constructor value
                    let addi_instr = risc_v_gen::Instruction::Addi(
                        risc_v_gen::Register::Sp,
                        risc_v_gen::Register::Sp,
                        4,
                    );
                    generator
                        .add_instruction(addi_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Switch back to read-only segment for code
                    generator.set_segment_type(risc_v_gen::MemorySegmentType::ReadOnly);
                },
                IRInstr::CaseBranch(branch_index) => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(format!(
                            "Case branch {}",
                            branch_index
                        )))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Compare the constructor value with the branch index
                    let li_instr =
                        risc_v_gen::Instruction::Li(risc_v_gen::Register::A1, *branch_index as i32);
                    generator
                        .add_instruction(li_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for loading the branch index
                    let pc = generator.current_pc();
                    let opcode_addr = pc;
                    let opcode_value = 0; // Placeholder
                    let step = ExecutionStep::new(
                        pc,
                        opcode_addr,
                        opcode_value,
                        pc + 4,
                        execution_trace.current_step,
                    );
                    execution_trace.add_step(step);

                    // Compare A0 (constructor) with A1 (branch index)
                    let beq_instr = risc_v_gen::Instruction::Beq(
                        risc_v_gen::Register::A0,
                        risc_v_gen::Register::A1,
                        format!("case_branch_{}", branch_index),
                    );
                    generator
                        .add_instruction(beq_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for the branch
                    let pc = generator.current_pc();
                    let opcode_addr = pc;
                    let opcode_value = 0; // Placeholder
                    let step = ExecutionStep::new(
                        pc,
                        opcode_addr,
                        opcode_value,
                        pc + 4,
                        execution_trace.current_step,
                    );
                    execution_trace.add_step(step);

                    // Add a label for the branch
                    let label_instr =
                        risc_v_gen::Instruction::Label(format!("case_branch_{}", branch_index));
                    generator
                        .add_instruction(label_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;
                },
                IRInstr::CaseBranchEnd => {
                    generator
                        .add_instruction(risc_v_gen::Instruction::Comment(
                            "End of case branch".to_string(),
                        ))
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Jump to the end of the case expression
                    let j_instr = risc_v_gen::Instruction::Jal(
                        risc_v_gen::Register::Zero,
                        "case_end".to_string(),
                    );
                    generator
                        .add_instruction(j_instr.clone())
                        .map_err(|e| CompilationError::BitVMXCodeGen(e))?;

                    // Create execution step for the jump
                    let pc = generator.current_pc();
                    let opcode_addr = pc;
                    let opcode_value = 0; // Placeholder
                    let step = ExecutionStep::new(
                        pc,
                        opcode_addr,
                        opcode_value,
                        pc + 4,
                        execution_trace.current_step,
                    );
                    execution_trace.add_step(step);
                },
                IRInstr::CaseEnd => {
                    // End of case expression
                    // Nothing to do
                },
                _ => {
                    // Unsupported instruction
                    return Err(CompilationError::UnsupportedFeature(format!(
                        "Unsupported IR instruction: {:?}",
                        instruction
                    )));
                },
            }
        }

        Ok(())
    }

    /// Compile UPLC code to RISC-V assembly and execute it using the BitVMX-CPU emulator directly
    ///
    /// This method compiles the UPLC code to RISC-V assembly and then executes it
    /// using the BitVMX-CPU emulator directly, without spawning a subprocess.
    ///
    /// # Arguments
    ///
    /// * `uplc_code` - The UPLC code to compile and execute
    ///
    /// # Returns
    ///
    /// The execution trace of the program
    pub fn compile_and_execute_direct(&self, input: &str) -> Result<ExecutionTrace> {
        let assembly = self.compile(input)?;

        // Execute the assembly using the BitVMX emulator
        let trace = bitvm_emulator::execute_assembly(&assembly)?;

        Ok(trace)
    }

    /// Save the execution trace to a CSV file
    pub fn save_execution_trace(
        &self,
        trace: &ExecutionTrace,
        file_path: &std::path::Path,
    ) -> Result<()> {
        // Save the trace to a CSV file
        bitvm_emulator::save_trace_to_csv(trace, file_path)
            .map_err(|e| CompilationError::BitVMXEmulator(e))
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
