//! RISC-V Code Generation
//!
//! This crate provides utilities for generating RISC-V assembly code.

use thiserror::Error;

/// Errors that can occur during RISC-V code generation
#[derive(Debug, Error)]
pub enum CodeGenError {
    #[error("Invalid instruction: {0}")]
    InvalidInstruction(String),

    #[error("Register out of range: {0}")]
    InvalidRegister(u8),

    #[error("Immediate value out of range: {0}")]
    InvalidImmediate(i32),
}

/// Result type for code generation operations
pub type Result<T> = std::result::Result<T, CodeGenError>;

/// RISC-V register
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Register {
    #[default]
    Zero,
    Ra,
    Sp,
    Gp,
    Tp,
    T0,
    T1,
    T2,
    S0,
    S1,
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    T3,
    T4,
    T5,
    T6,
}

impl Register {
    /// Get the register number
    pub fn number(&self) -> u8 {
        match self {
            Self::Zero => 0,
            Self::Ra => 1,
            Self::Sp => 2,
            Self::Gp => 3,
            Self::Tp => 4,
            Self::T0 => 5,
            Self::T1 => 6,
            Self::T2 => 7,
            Self::S0 => 8,
            Self::S1 => 9,
            Self::A0 => 10,
            Self::A1 => 11,
            Self::A2 => 12,
            Self::A3 => 13,
            Self::A4 => 14,
            Self::A5 => 15,
            Self::A6 => 16,
            Self::A7 => 17,
            Self::S2 => 18,
            Self::S3 => 19,
            Self::S4 => 20,
            Self::S5 => 21,
            Self::S6 => 22,
            Self::S7 => 23,
            Self::S8 => 24,
            Self::S9 => 25,
            Self::S10 => 26,
            Self::S11 => 27,
            Self::T3 => 28,
            Self::T4 => 29,
            Self::T5 => 30,
            Self::T6 => 31,
        }
    }

    /// Get the ABI name of the register
    pub fn name(&self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Ra => "ra",
            Self::Sp => "sp",
            Self::Gp => "gp",
            Self::Tp => "tp",
            Self::T0 => "t0",
            Self::T1 => "t1",
            Self::T2 => "t2",
            Self::S0 => "s0",
            Self::S1 => "s1",
            Self::A0 => "a0",
            Self::A1 => "a1",
            Self::A2 => "a2",
            Self::A3 => "a3",
            Self::A4 => "a4",
            Self::A5 => "a5",
            Self::A6 => "a6",
            Self::A7 => "a7",
            Self::S2 => "s2",
            Self::S3 => "s3",
            Self::S4 => "s4",
            Self::S5 => "s5",
            Self::S6 => "s6",
            Self::S7 => "s7",
            Self::S8 => "s8",
            Self::S9 => "s9",
            Self::S10 => "s10",
            Self::S11 => "s11",
            Self::T3 => "t3",
            Self::T4 => "t4",
            Self::T5 => "t5",
            Self::T6 => "t6",
        }
    }
}

/// RISC-V instruction
#[derive(Debug, Clone)]
pub enum Instruction {
    // R-type instructions
    Add(Register, Register, Register),
    Sub(Register, Register, Register),
    And(Register, Register, Register),
    Or(Register, Register, Register),
    Xor(Register, Register, Register),
    Slt(Register, Register, Register),
    Sltu(Register, Register, Register),
    Sll(Register, Register, Register),
    Srl(Register, Register, Register),
    Sra(Register, Register, Register),

    // M-extension instructions
    Mul(Register, Register, Register),
    Mulh(Register, Register, Register),
    Mulhsu(Register, Register, Register),
    Mulhu(Register, Register, Register),
    Div(Register, Register, Register),
    Divu(Register, Register, Register),
    Rem(Register, Register, Register),
    Remu(Register, Register, Register),

    // I-type instructions
    Addi(Register, Register, i32),
    Andi(Register, Register, i32),
    Ori(Register, Register, i32),
    Xori(Register, Register, i32),
    Slti(Register, Register, i32),
    Sltiu(Register, Register, i32),
    Slli(Register, Register, i32),
    Srli(Register, Register, i32),
    Srai(Register, Register, i32),
    Lw(Register, i32, Register),
    Lh(Register, i32, Register),
    Lb(Register, i32, Register),
    Lhu(Register, i32, Register),
    Lbu(Register, i32, Register),
    Jalr(Register, Register, i32),

    // S-type instructions
    Sw(Register, i32, Register),
    Sh(Register, i32, Register),
    Sb(Register, i32, Register),

    // B-type instructions
    Beq(Register, Register, String),
    Bne(Register, Register, String),
    Blt(Register, Register, String),
    Bge(Register, Register, String),
    Bltu(Register, Register, String),
    Bgeu(Register, Register, String),

    // U-type instructions
    Lui(Register, i32),
    Auipc(Register, i32),

    // J-type instructions
    Jal(Register, String),

    // Pseudo-instructions
    J(String),
    Li(Register, i32),
    La(Register, String),
    Mv(Register, Register),
    Not(Register, Register),
    Neg(Register, Register),
    Seqz(Register, Register),
    Snez(Register, Register),
    Nop,

    // Label
    Label(String),

    // Directives
    Global(String),
    Section(String),
    Align(i32),
    Word(i32),
    Byte(i32),
    Ascii(String),
    Asciiz(String),
    Space(i32),

    // Comments
    Comment(String),

    // Exit
    Ecall,
}

/// Code generator for RISC-V assembly
#[derive(Debug, Default)]
pub struct CodeGenerator {
    instructions: Vec<Instruction>,
}

impl CodeGenerator {
    /// Create a new code generator
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    /// Add an instruction to the code generator
    pub fn add_instruction(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }

    /// Generate assembly code
    pub fn generate(&self) -> String {
        let mut result = String::new();

        for instruction in &self.instructions {
            match instruction {
                Instruction::Label(label) => {
                    result.push_str(&format!("{}:\n", label));
                },
                Instruction::Comment(comment) => {
                    result.push_str(&format!("    # {}\n", comment));
                },
                Instruction::Global(symbol) => {
                    result.push_str(&format!(".global {}\n", symbol));
                },
                Instruction::Section(section) => {
                    result.push_str(&format!(".{}\n", section));
                },
                Instruction::Align(align) => {
                    result.push_str(&format!("    .align {}\n", align));
                },
                Instruction::Word(value) => {
                    result.push_str(&format!("    .word {}\n", value));
                },
                Instruction::Byte(value) => {
                    result.push_str(&format!("    .byte {}\n", value));
                },
                Instruction::Ascii(string) => {
                    result.push_str(&format!("    .ascii \"{}\"\n", string));
                },
                Instruction::Asciiz(string) => {
                    result.push_str(&format!("    .asciiz \"{}\"\n", string));
                },
                Instruction::Space(size) => {
                    result.push_str(&format!("    .space {}\n", size));
                },
                _ => {
                    result.push_str("    ");
                    result.push_str(&self.format_instruction(instruction));
                    result.push('\n');
                },
            }
        }

        result
    }

    /// Format an instruction as a string
    fn format_instruction(&self, instruction: &Instruction) -> String {
        match instruction {
            Instruction::Add(rd, rs1, rs2) => {
                format!("add {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Sub(rd, rs1, rs2) => {
                format!("sub {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::And(rd, rs1, rs2) => {
                format!("and {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Or(rd, rs1, rs2) => {
                format!("or {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Xor(rd, rs1, rs2) => {
                format!("xor {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Slt(rd, rs1, rs2) => {
                format!("slt {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Sltu(rd, rs1, rs2) => {
                format!("sltu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Sll(rd, rs1, rs2) => {
                format!("sll {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Srl(rd, rs1, rs2) => {
                format!("srl {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Sra(rd, rs1, rs2) => {
                format!("sra {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Mul(rd, rs1, rs2) => {
                format!("mul {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Mulh(rd, rs1, rs2) => {
                format!("mulh {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Mulhsu(rd, rs1, rs2) => {
                format!("mulhsu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Mulhu(rd, rs1, rs2) => {
                format!("mulhu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Div(rd, rs1, rs2) => {
                format!("div {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Divu(rd, rs1, rs2) => {
                format!("divu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Rem(rd, rs1, rs2) => {
                format!("rem {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Remu(rd, rs1, rs2) => {
                format!("remu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            },
            Instruction::Addi(rd, rs1, imm) => {
                format!("addi {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Andi(rd, rs1, imm) => {
                format!("andi {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Ori(rd, rs1, imm) => {
                format!("ori {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Xori(rd, rs1, imm) => {
                format!("xori {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Slti(rd, rs1, imm) => {
                format!("slti {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Sltiu(rd, rs1, imm) => {
                format!("sltiu {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Slli(rd, rs1, imm) => {
                format!("slli {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Srli(rd, rs1, imm) => {
                format!("srli {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Srai(rd, rs1, imm) => {
                format!("srai {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Lw(rd, offset, rs1) => {
                format!("lw {}, {}({})", rd.name(), offset, rs1.name())
            },
            Instruction::Lh(rd, offset, rs1) => {
                format!("lh {}, {}({})", rd.name(), offset, rs1.name())
            },
            Instruction::Lb(rd, offset, rs1) => {
                format!("lb {}, {}({})", rd.name(), offset, rs1.name())
            },
            Instruction::Lhu(rd, offset, rs1) => {
                format!("lhu {}, {}({})", rd.name(), offset, rs1.name())
            },
            Instruction::Lbu(rd, offset, rs1) => {
                format!("lbu {}, {}({})", rd.name(), offset, rs1.name())
            },
            Instruction::Jalr(rd, rs1, imm) => {
                format!("jalr {}, {}, {}", rd.name(), rs1.name(), imm)
            },
            Instruction::Sw(rs2, offset, rs1) => {
                format!("sw {}, {}({})", rs2.name(), offset, rs1.name())
            },
            Instruction::Sh(rs2, offset, rs1) => {
                format!("sh {}, {}({})", rs2.name(), offset, rs1.name())
            },
            Instruction::Sb(rs2, offset, rs1) => {
                format!("sb {}, {}({})", rs2.name(), offset, rs1.name())
            },
            Instruction::Beq(rs1, rs2, label) => {
                format!("beq {}, {}, {}", rs1.name(), rs2.name(), label)
            },
            Instruction::Bne(rs1, rs2, label) => {
                format!("bne {}, {}, {}", rs1.name(), rs2.name(), label)
            },
            Instruction::Blt(rs1, rs2, label) => {
                format!("blt {}, {}, {}", rs1.name(), rs2.name(), label)
            },
            Instruction::Bge(rs1, rs2, label) => {
                format!("bge {}, {}, {}", rs1.name(), rs2.name(), label)
            },
            Instruction::Bltu(rs1, rs2, label) => {
                format!("bltu {}, {}, {}", rs1.name(), rs2.name(), label)
            },
            Instruction::Bgeu(rs1, rs2, label) => {
                format!("bgeu {}, {}, {}", rs1.name(), rs2.name(), label)
            },
            Instruction::Lui(rd, imm) => {
                format!("lui {}, {}", rd.name(), imm)
            },
            Instruction::Auipc(rd, imm) => {
                format!("auipc {}, {}", rd.name(), imm)
            },
            Instruction::Jal(rd, label) => {
                format!("jal {}, {}", rd.name(), label)
            },
            Instruction::Li(rd, imm) => {
                format!("li {}, {}", rd.name(), imm)
            },
            Instruction::La(rd, symbol) => {
                format!("la {}, {}", rd.name(), symbol)
            },
            Instruction::Mv(rd, rs) => {
                format!("mv {}, {}", rd.name(), rs.name())
            },
            Instruction::Not(rd, rs) => {
                format!("not {}, {}", rd.name(), rs.name())
            },
            Instruction::Neg(rd, rs) => {
                format!("neg {}, {}", rd.name(), rs.name())
            },
            Instruction::Seqz(rd, rs) => {
                format!("seqz {}, {}", rd.name(), rs.name())
            },
            Instruction::Snez(rd, rs) => {
                format!("snez {}, {}", rd.name(), rs.name())
            },
            Instruction::J(imm) => {
                format!("j {}", imm)
            },
            Instruction::Nop => "nop".to_string(),
            Instruction::Label(_) => unreachable!("Labels are handled separately"),
            Instruction::Global(_) => unreachable!("Global directives are handled separately"),
            Instruction::Section(_) => unreachable!("Section directives are handled separately"),
            Instruction::Align(_) => unreachable!("Align directives are handled separately"),
            Instruction::Word(_) => unreachable!("Word directives are handled separately"),
            Instruction::Byte(_) => unreachable!("Byte directives are handled separately"),
            Instruction::Ascii(_) => unreachable!("Ascii directives are handled separately"),
            Instruction::Asciiz(_) => unreachable!("Asciiz directives are handled separately"),
            Instruction::Space(_) => unreachable!("Space directives are handled separately"),
            Instruction::Comment(_) => unreachable!("Comments are handled separately"),
            Instruction::Ecall => "ecall".to_string(),
        }
    }

    pub fn save_to_file(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        file.write_all(self.generate().as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_generation() {
        let mut gen = CodeGenerator::new();

        gen.add_instruction(Instruction::Label("main".to_string()));
        gen.add_instruction(Instruction::Li(Register::A0, 42));
        gen.add_instruction(Instruction::Li(Register::A1, 58));
        gen.add_instruction(Instruction::Add(Register::A2, Register::A0, Register::A1));

        let asm = gen.generate();
        assert!(asm.contains("main:"));
        assert!(asm.contains("li a0, 42"));
        assert!(asm.contains("li a1, 58"));
        assert!(asm.contains("add a2, a0, a1"));
    }

    #[test]
    fn test_memory_instructions() {
        let mut gen = CodeGenerator::new();

        gen.add_instruction(Instruction::Li(Register::A0, 42));
        gen.add_instruction(Instruction::Sw(Register::A0, 0, Register::Sp));
        gen.add_instruction(Instruction::Lw(Register::A1, 0, Register::Sp));

        let asm = gen.generate();
        assert!(asm.contains("li a0, 42"));
        assert!(asm.contains("sw a0, 0(sp)"));
        assert!(asm.contains("lw a1, 0(sp)"));
    }

    #[test]
    fn test_branch_instructions() {
        let mut gen = CodeGenerator::new();

        gen.add_instruction(Instruction::Li(Register::A0, 42));
        gen.add_instruction(Instruction::Li(Register::A1, 58));
        gen.add_instruction(Instruction::Blt(
            Register::A0,
            Register::A1,
            "label".to_string(),
        ));
        gen.add_instruction(Instruction::Label("label".to_string()));

        let asm = gen.generate();
        assert!(asm.contains("li a0, 42"));
        assert!(asm.contains("li a1, 58"));
        assert!(asm.contains("blt a0, a1, label"));
        assert!(asm.contains("label:"));
    }

    #[test]
    fn test_directives() {
        let mut gen = CodeGenerator::new();

        gen.add_instruction(Instruction::Section(".text".to_string()));
        gen.add_instruction(Instruction::Global("main".to_string()));
        gen.add_instruction(Instruction::Label("main".to_string()));
        gen.add_instruction(Instruction::Comment("This is a comment".to_string()));

        let asm = gen.generate();
        assert!(asm.contains(".text"));
        assert!(asm.contains(".global main"));
        assert!(asm.contains("main:"));
        assert!(asm.contains("# This is a comment"));
    }
}
