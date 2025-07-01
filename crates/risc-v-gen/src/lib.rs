//! RISC-V Code gene.ration
//!
//! This crate provides utilities for gene.rating RISC-V assembly code.

use std::{fs::File, io::Write};

use strum_macros::EnumIter;
use thiserror::Error;

// pub mod elf;
pub mod emulator;

/// Errors that can occur during RISC-V code gene.ration
#[derive(Debug, Error)]
pub enum CodeGenError {
    #[error("Invalid instruction: {0}")]
    InvalidInstruction(String),

    #[error("Register out of range: {0}")]
    InvalidRegister(u8),

    #[error("Immediate value out of range: {0}")]
    InvalidImmediate(i32),
}

/// Result type for code gene.ration operations
pub type Result<T> = std::result::Result<T, CodeGenError>;

/// RISC-V register
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, EnumIter)]
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

#[derive(Debug, Clone)]
pub struct Assign {
    pub name: String,
    pub register: Option<Register>,
    pub assigned: bool,
    pub mutable: bool,
}

/// RISC-V instruction
#[derive(Debug, Clone)]
pub enum Instruction {
    // R-type instructions
    Add(Register, Register, Register),
    Sub(Register, Register, Register),
    Xor(Register, Register, Register),
    Or(Register, Register, Register),
    And(Register, Register, Register),
    Sll(Register, Register, Register),
    Srl(Register, Register, Register),
    Sra(Register, Register, Register),
    Slt(Register, Register, Register),
    Sltu(Register, Register, Register),

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
    Word(String),
    Byte(Vec<u8>),
    Ascii(String),
    Asciiz(String),
    Space(i32),

    // Comments
    Comment(String),

    // Exit
    Ecall,
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

impl Assign {
    pub fn make_var(name: impl ToString, register: Register) -> Assign {
        Assign {
            name: name.to_string(),
            register: Some(register),
            assigned: false,
            mutable: true,
        }
    }

    pub fn make_constant(name: impl ToString, register: Register) -> Assign {
        Assign {
            name: name.to_string(),
            register: Some(register),
            assigned: false,
            mutable: false,
        }
    }

    pub fn var_overwrite(&mut self, name: impl ToString) -> Assign {
        let Some(register) = self.register else {
            panic!(
                "Var already overwritten\n\nVar: {}, New Name: {}",
                self.name,
                name.to_string()
            )
        };

        self.register = None;
        Assign {
            name: name.to_string(),
            register: Some(register),
            assigned: false,
            mutable: true,
        }
    }

    pub fn constnt_overwrite(&mut self, name: impl ToString) -> Assign {
        let mut assign = self.var_overwrite(name);
        assign.mutable = false;
        assign
    }

    pub fn register_assign(&mut self) -> Register {
        let Some(register) = self.register else {
            panic!("Using Overwritten Var\n\nVar: {}", self.name)
        };

        if self.assigned && !self.mutable {
            panic!(
                "Mutating a Constant\n\nVar: {}, Register: {}",
                self.name,
                register.name()
            );
        }

        self.assigned = true;

        register
    }

    pub fn register_value(&self) -> Register {
        let Some(register) = self.register else {
            panic!("Using Overwritten Var\n\nVar: {}", self.name);
        };

        if !self.assigned {
            panic!(
                "Var Not Assigned\n\nVar: {}, Register: {}",
                self.name,
                register.name()
            );
        }

        register
    }
}

impl Instruction {
    // R-type instructions
    pub fn add(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Add(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn sub(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Sub(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn xor(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Xor(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn or(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Or(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn and(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::And(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn sll(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Sll(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn srl(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Srl(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn sra(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Sra(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn slt(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Slt(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn sltu(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Sltu(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    // M-extension instructions
    pub fn mul(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Mul(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn mulh(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Mulh(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn mulhsu(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Mulhsu(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn mulhu(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Mulhu(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn div(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Div(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn divu(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Divu(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn rem(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Rem(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    pub fn remu(rd: &mut Assign, rs1: &Assign, rs2: &Assign) -> Self {
        Self::Remu(
            rd.register_assign(),
            rs1.register_value(),
            rs2.register_value(),
        )
    }

    // I-type instructions
    pub fn addi(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Addi(rd.register_assign(), rs1.register_value(), imm)
    }

    pub fn andi(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Andi(rd.register_assign(), rs1.register_value(), imm)
    }

    pub fn ori(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Ori(rd.register_assign(), rs1.register_value(), imm)
    }

    pub fn xori(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Xori(rd.register_assign(), rs1.register_value(), imm)
    }

    pub fn slti(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Slti(rd.register_assign(), rs1.register_value(), imm)
    }

    pub fn sltiu(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Sltiu(rd.register_assign(), rs1.register_value(), imm)
    }

    pub fn slli(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Slli(rd.register_assign(), rs1.register_value(), imm)
    }

    pub fn srli(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Srli(rd.register_assign(), rs1.register_value(), imm)
    }

    pub fn srai(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Srai(rd.register_assign(), rs1.register_value(), imm)
    }

    pub fn lw(rd: &mut Assign, offset: i32, rs1: &Assign) -> Self {
        Self::Lw(rd.register_assign(), offset, rs1.register_value())
    }

    pub fn lh(rd: &mut Assign, offset: i32, rs1: &Assign) -> Self {
        Self::Lh(rd.register_assign(), offset, rs1.register_value())
    }

    pub fn lb(rd: &mut Assign, offset: i32, rs1: &Assign) -> Self {
        Self::Lb(rd.register_assign(), offset, rs1.register_value())
    }

    pub fn lhu(rd: &mut Assign, offset: i32, rs1: &Assign) -> Self {
        Self::Lhu(rd.register_assign(), offset, rs1.register_value())
    }

    pub fn lbu(rd: &mut Assign, offset: i32, rs1: &Assign) -> Self {
        Self::Lbu(rd.register_assign(), offset, rs1.register_value())
    }

    pub fn jalr(rd: &mut Assign, rs1: &Assign, imm: i32) -> Self {
        Self::Jalr(rd.register_assign(), rs1.register_value(), imm)
    }

    // S-type instructions
    pub fn sw(rs2: &Assign, offset: i32, rs1: &Assign) -> Self {
        Self::Sw(rs2.register_value(), offset, rs1.register_value())
    }

    pub fn sh(rs2: &Assign, offset: i32, rs1: &Assign) -> Self {
        Self::Sh(rs2.register_value(), offset, rs1.register_value())
    }

    pub fn sb(rs2: &Assign, offset: i32, rs1: &Assign) -> Self {
        Self::Sb(rs2.register_value(), offset, rs1.register_value())
    }

    // B-type instructions
    pub fn beq(rs1: &Assign, rs2: &Assign, label: String) -> Self {
        Self::Beq(rs1.register_value(), rs2.register_value(), label)
    }

    pub fn bne(rs1: &Assign, rs2: &Assign, label: String) -> Self {
        Self::Bne(rs1.register_value(), rs2.register_value(), label)
    }

    pub fn blt(rs1: &Assign, rs2: &Assign, label: String) -> Self {
        Self::Blt(rs1.register_value(), rs2.register_value(), label)
    }

    pub fn bge(rs1: &Assign, rs2: &Assign, label: String) -> Self {
        Self::Bge(rs1.register_value(), rs2.register_value(), label)
    }

    pub fn bltu(rs1: &Assign, rs2: &Assign, label: String) -> Self {
        Self::Bltu(rs1.register_value(), rs2.register_value(), label)
    }

    pub fn bgeu(rs1: &Assign, rs2: &Assign, label: String) -> Self {
        Self::Bgeu(rs1.register_value(), rs2.register_value(), label)
    }

    // U-type instructions
    pub fn lui(rd: &mut Assign, imm: i32) -> Self {
        Self::Lui(rd.register_assign(), imm)
    }

    pub fn auipc(rd: &mut Assign, imm: i32) -> Self {
        Self::Auipc(rd.register_assign(), imm)
    }

    // J-type instructions
    pub fn jal(rd: &mut Assign, label: String) -> Self {
        Self::Jal(rd.register_assign(), label)
    }

    // Pseudo-instructions
    pub fn li(rd: &mut Assign, imm: i32) -> Self {
        Self::Li(rd.register_assign(), imm)
    }

    pub fn la(rd: &mut Assign, label: String) -> Self {
        Self::La(rd.register_assign(), label)
    }

    pub fn mv(rd: &mut Assign, rs: &Assign) -> Self {
        Self::Mv(rd.register_assign(), rs.register_value())
    }

    pub fn not(rd: &mut Assign, rs: &Assign) -> Self {
        Self::Not(rd.register_assign(), rs.register_value())
    }

    pub fn neg(rd: &mut Assign, rs: &Assign) -> Self {
        Self::Neg(rd.register_assign(), rs.register_value())
    }

    pub fn seqz(rd: &mut Assign, rs: &Assign) -> Self {
        Self::Seqz(rd.register_assign(), rs.register_value())
    }

    pub fn snez(rd: &mut Assign, rs: &Assign) -> Self {
        Self::Snez(rd.register_assign(), rs.register_value())
    }

    // Pseudo-instructions (additional)
    pub fn j(label: String) -> Self {
        Self::J(label)
    }

    pub fn nop() -> Self {
        Self::Nop
    }

    // Label
    pub fn label(name: String) -> Self {
        Self::Label(name)
    }

    // Directives
    pub fn global(symbol: String) -> Self {
        Self::Global(symbol)
    }

    pub fn section(name: String) -> Self {
        Self::Section(name)
    }

    pub fn align(boundary: i32) -> Self {
        Self::Align(boundary)
    }

    pub fn word(value: String) -> Self {
        Self::Word(value)
    }

    pub fn byte(data: Vec<u8>) -> Self {
        Self::Byte(data)
    }

    pub fn ascii(text: String) -> Self {
        Self::Ascii(text)
    }

    pub fn asciiz(text: String) -> Self {
        Self::Asciiz(text)
    }

    pub fn space(size: i32) -> Self {
        Self::Space(size)
    }

    // Comments
    pub fn comment(text: String) -> Self {
        Self::Comment(text)
    }

    // Exit
    pub fn ecall() -> Self {
        Self::Ecall
    }
}

/// Code gene.rator for RISC-V assembly
#[derive(Debug, Default)]
pub struct CodeGenerator {
    pub instructions: Vec<Instruction>,
}

impl CodeGenerator {
    /// Create a new code gene.rator
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    /// Add an instruction to the code gene.rator
    pub fn add_instruction(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }

    /// gene.rate assembly code
    pub fn generate(&self) -> String {
        let mut result = String::new();

        for instruction in &self.instructions {
            match instruction {
                Instruction::Label(label) => {
                    result.push_str(&format!("{label}:\n"));
                },
                Instruction::Comment(comment) => {
                    result.push_str(&format!("    # {comment}\n"));
                },
                Instruction::Global(symbol) => {
                    result.push_str(&format!(".global {symbol}\n"));
                },
                Instruction::Section(section) => {
                    result.push_str(&format!(".section .{section}\n"));
                },
                Instruction::Align(align) => {
                    result.push_str(&format!("    .align {align}\n"));
                },
                Instruction::Word(value) => {
                    result.push_str(&format!("    .word {value}\n"));
                },
                Instruction::Byte(value) => {
                    result.push_str(&format!(
                        "    .byte {}\n",
                        value
                            .iter()
                            .map(|val| val.to_string())
                            .collect::<Vec<String>>()
                            .join(", ")
                    ));
                },
                Instruction::Ascii(string) => {
                    result.push_str(&format!("    .ascii \"{string}\"\n"));
                },
                Instruction::Asciiz(string) => {
                    result.push_str(&format!("    .asciiz \"{string}\"\n"));
                },
                Instruction::Space(size) => {
                    result.push_str(&format!("    .space {size}\n"));
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

    // Generate an ELF file from the code generator
    // pub fn generate_elf(&self, output_path: &Path, linker_script: &str) -> Result<()> {
    //     elf::build_elf(self, linker_script, output_path)
    // }
}

// Assemble assembly code from a string and generate an ELF file
//
// This function provides a streamlined way to:
// 1. Take assembly code as a string
// 2. Assemble it using lib_rv32_asm
// 3. Generate an ELF file with the provided or default linker script
//
// It mimics the CLI approach:
// `riscv64-unknown-elf-as test.s -march=rv32i -mabi=ilp32 -o test.o`
// `riscv64-unknown-elf-ld test.o -m elf32lriscv -o test.elf`
//
// pub fn assemble_and_link(
//     asm_code: &str,
//     output_path: &Path,
//     linker_script: Option<&str>,
// ) -> Result<()> {
//     elf::assemble_and_link(asm_code, output_path, linker_script)
// }

// // Re-export functions from the elf module
// pub use elf::{
//     assemble_instructions, build_elf, validate_imm_range, validate_section_overlap,
//     DEFAULT_LINKER_SCRIPT,
// };

// /// Parse a linker script from a string
// pub fn parse_linker_script(script: &str) -> Result<elf::LinkerScript> {
//     elf::LinkerScript::parse(script)
// }

#[cfg(test)]
mod tests {
    use crate::{Assign, CodeGenerator, Instruction, Register};

    #[test]
    fn test_code_generation() {
        let mut gene = CodeGenerator::new();

        let mut a0 = Assign::make_var("a0", Register::A0);
        let mut a1 = Assign::make_var("a1", Register::A1);
        let mut a2 = Assign::make_var("a2", Register::A2);

        gene.add_instruction(Instruction::label("main".to_string()));
        gene.add_instruction(Instruction::li(&mut a0, 42));
        gene.add_instruction(Instruction::li(&mut a1, 58));
        gene.add_instruction(Instruction::add(&mut a2, &a0, &a1));

        let asm = gene.generate();
        assert!(asm.contains("main:"));
        assert!(asm.contains("li a0, 42"));
        assert!(asm.contains("li a1, 58"));
        assert!(asm.contains("add a2, a0, a1"));
    }

    #[test]
    fn test_memory_instructions() {
        let mut gene = CodeGenerator::new();

        let mut a0 = Assign::make_var("a0", Register::A0);
        let mut a1 = Assign::make_var("a1", Register::A1);

        gene.add_instruction(Instruction::label("main".to_string()));
        gene.add_instruction(Instruction::li(&mut a0, 42));
        gene.add_instruction(Instruction::Sw(a0.register_value(), 0, Register::Sp));
        gene.add_instruction(Instruction::Lw(a1.register_assign(), 0, Register::Sp));

        let asm = gene.generate();
        assert!(asm.contains("li a0, 42"));
        assert!(asm.contains("sw a0, 0(sp)"));
        assert!(asm.contains("lw a1, 0(sp)"));
    }

    #[test]
    fn test_branch_instructions() {
        let mut gene = CodeGenerator::new();

        let mut a0 = Assign::make_var("a0", Register::A0);
        let mut a1 = Assign::make_var("a1", Register::A1);

        gene.add_instruction(Instruction::li(&mut a0, 42));
        gene.add_instruction(Instruction::li(&mut a1, 58));
        gene.add_instruction(Instruction::blt(&a0, &a1, "label".to_string()));
        gene.add_instruction(Instruction::label("label".to_string()));

        let asm = gene.generate();
        assert!(asm.contains("li a0, 42"));
        assert!(asm.contains("li a1, 58"));
        assert!(asm.contains("blt a0, a1, label"));
        assert!(asm.contains("label:"));
    }

    #[test]
    fn test_directives() {
        let mut gene = CodeGenerator::new();

        gene.add_instruction(Instruction::section("text".to_string()));
        gene.add_instruction(Instruction::global("main".to_string()));
        gene.add_instruction(Instruction::label("main".to_string()));
        gene.add_instruction(Instruction::comment("This is a comment".to_string()));

        let asm = gene.generate();
        assert!(asm.contains(".text"));
        assert!(asm.contains(".global main"));
        assert!(asm.contains("main:"));
        assert!(asm.contains("# This is a comment"));
    }

    #[test]
    fn test_directives2() {
        let mut gene = CodeGenerator::new();

        gene.add_instruction(Instruction::section("text".to_string()));
        gene.add_instruction(Instruction::global("_start".to_string()));
        gene.add_instruction(Instruction::label("_start".to_string()));
        gene.add_instruction(Instruction::comment("This is a comment".to_string()));

        gene.add_instruction(Instruction::Addi(Register::A0, Register::A1, 43));
        gene.add_instruction(Instruction::Li(Register::A7, 93));
        gene.add_instruction(Instruction::Ecall);

        // gene.save_to_file("test.s").unwrap();
    }
}
