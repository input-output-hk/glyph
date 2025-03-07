//! RISC-V Code Generation
//!
//! This crate provides utilities for generating RISC-V assembly code.

use thiserror::Error;

// Import from bitvm-common instead
use bitvm_common::memory;

// Re-export MemorySegmentType from bitvm-common
pub use bitvm_common::MemorySegmentType;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Register {
    Zero,
    Ra,
    Sp,
    Gp,
    Tp,
    T0, T1, T2,
    S0, S1,
    A0, A1, A2, A3, A4, A5, A6, A7,
    S2, S3, S4, S5, S6, S7, S8, S9, S10, S11,
    T3, T4, T5, T6,
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
}

/// Code generator for RISC-V assembly
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
                }
                Instruction::Comment(comment) => {
                    result.push_str(&format!("    # {}\n", comment));
                }
                Instruction::Global(symbol) => {
                    result.push_str(&format!("    .global {}\n", symbol));
                }
                Instruction::Section(section) => {
                    result.push_str(&format!("    .section {}\n", section));
                }
                Instruction::Align(align) => {
                    result.push_str(&format!("    .align {}\n", align));
                }
                Instruction::Word(value) => {
                    result.push_str(&format!("    .word {}\n", value));
                }
                Instruction::Byte(value) => {
                    result.push_str(&format!("    .byte {}\n", value));
                }
                Instruction::Ascii(string) => {
                    result.push_str(&format!("    .ascii \"{}\"\n", string));
                }
                Instruction::Asciiz(string) => {
                    result.push_str(&format!("    .asciiz \"{}\"\n", string));
                }
                Instruction::Space(size) => {
                    result.push_str(&format!("    .space {}\n", size));
                }
                _ => {
                    result.push_str("    ");
                    result.push_str(&self.format_instruction(instruction));
                    result.push('\n');
                }
            }
        }
        
        result
    }
    
    /// Format an instruction as a string
    fn format_instruction(&self, instruction: &Instruction) -> String {
        match instruction {
            Instruction::Add(rd, rs1, rs2) => {
                format!("add {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Sub(rd, rs1, rs2) => {
                format!("sub {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::And(rd, rs1, rs2) => {
                format!("and {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Or(rd, rs1, rs2) => {
                format!("or {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Xor(rd, rs1, rs2) => {
                format!("xor {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Slt(rd, rs1, rs2) => {
                format!("slt {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Sltu(rd, rs1, rs2) => {
                format!("sltu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Sll(rd, rs1, rs2) => {
                format!("sll {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Srl(rd, rs1, rs2) => {
                format!("srl {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Sra(rd, rs1, rs2) => {
                format!("sra {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Mul(rd, rs1, rs2) => {
                format!("mul {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Mulh(rd, rs1, rs2) => {
                format!("mulh {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Mulhsu(rd, rs1, rs2) => {
                format!("mulhsu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Mulhu(rd, rs1, rs2) => {
                format!("mulhu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Div(rd, rs1, rs2) => {
                format!("div {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Divu(rd, rs1, rs2) => {
                format!("divu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Rem(rd, rs1, rs2) => {
                format!("rem {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Remu(rd, rs1, rs2) => {
                format!("remu {}, {}, {}", rd.name(), rs1.name(), rs2.name())
            }
            Instruction::Addi(rd, rs1, imm) => {
                format!("addi {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Andi(rd, rs1, imm) => {
                format!("andi {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Ori(rd, rs1, imm) => {
                format!("ori {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Xori(rd, rs1, imm) => {
                format!("xori {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Slti(rd, rs1, imm) => {
                format!("slti {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Sltiu(rd, rs1, imm) => {
                format!("sltiu {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Slli(rd, rs1, imm) => {
                format!("slli {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Srli(rd, rs1, imm) => {
                format!("srli {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Srai(rd, rs1, imm) => {
                format!("srai {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Lw(rd, offset, rs1) => {
                format!("lw {}, {}({})", rd.name(), offset, rs1.name())
            }
            Instruction::Lh(rd, offset, rs1) => {
                format!("lh {}, {}({})", rd.name(), offset, rs1.name())
            }
            Instruction::Lb(rd, offset, rs1) => {
                format!("lb {}, {}({})", rd.name(), offset, rs1.name())
            }
            Instruction::Lhu(rd, offset, rs1) => {
                format!("lhu {}, {}({})", rd.name(), offset, rs1.name())
            }
            Instruction::Lbu(rd, offset, rs1) => {
                format!("lbu {}, {}({})", rd.name(), offset, rs1.name())
            }
            Instruction::Jalr(rd, rs1, imm) => {
                format!("jalr {}, {}, {}", rd.name(), rs1.name(), imm)
            }
            Instruction::Sw(rs2, offset, rs1) => {
                format!("sw {}, {}({})", rs2.name(), offset, rs1.name())
            }
            Instruction::Sh(rs2, offset, rs1) => {
                format!("sh {}, {}({})", rs2.name(), offset, rs1.name())
            }
            Instruction::Sb(rs2, offset, rs1) => {
                format!("sb {}, {}({})", rs2.name(), offset, rs1.name())
            }
            Instruction::Beq(rs1, rs2, label) => {
                format!("beq {}, {}, {}", rs1.name(), rs2.name(), label)
            }
            Instruction::Bne(rs1, rs2, label) => {
                format!("bne {}, {}, {}", rs1.name(), rs2.name(), label)
            }
            Instruction::Blt(rs1, rs2, label) => {
                format!("blt {}, {}, {}", rs1.name(), rs2.name(), label)
            }
            Instruction::Bge(rs1, rs2, label) => {
                format!("bge {}, {}, {}", rs1.name(), rs2.name(), label)
            }
            Instruction::Bltu(rs1, rs2, label) => {
                format!("bltu {}, {}, {}", rs1.name(), rs2.name(), label)
            }
            Instruction::Bgeu(rs1, rs2, label) => {
                format!("bgeu {}, {}, {}", rs1.name(), rs2.name(), label)
            }
            Instruction::Lui(rd, imm) => {
                format!("lui {}, {}", rd.name(), imm)
            }
            Instruction::Auipc(rd, imm) => {
                format!("auipc {}, {}", rd.name(), imm)
            }
            Instruction::Jal(rd, label) => {
                format!("jal {}, {}", rd.name(), label)
            }
            Instruction::Li(rd, imm) => {
                format!("li {}, {}", rd.name(), imm)
            }
            Instruction::La(rd, symbol) => {
                format!("la {}, {}", rd.name(), symbol)
            }
            Instruction::Mv(rd, rs) => {
                format!("mv {}, {}", rd.name(), rs.name())
            }
            Instruction::Not(rd, rs) => {
                format!("not {}, {}", rd.name(), rs.name())
            }
            Instruction::Neg(rd, rs) => {
                format!("neg {}, {}", rd.name(), rs.name())
            }
            Instruction::Seqz(rd, rs) => {
                format!("seqz {}, {}", rd.name(), rs.name())
            }
            Instruction::Snez(rd, rs) => {
                format!("snez {}, {}", rd.name(), rs.name())
            }
            Instruction::Nop => {
                "nop".to_string()
            }
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
        }
    }
}

/// BitVMX instruction with memory access tracking
/// 
/// This represents a RISC-V instruction with additional metadata for BitVMX:
/// - Memory addresses being read and written
/// - Explicit PC update
#[derive(Debug, Clone)]
pub struct BitVMXInstruction {
    /// The original RISC-V instruction
    pub original: Instruction,
    
    /// First memory address being read
    pub read_addr1: Option<u32>,
    
    /// First memory value being read
    pub read_value1: Option<u32>,
    
    /// Second memory address being read
    pub read_addr2: Option<u32>,
    
    /// Second memory value being read
    pub read_value2: Option<u32>,
    
    /// Opcode memory address
    pub opcode_addr: u32,
    
    /// Opcode value
    pub opcode_value: u32,
    
    /// Memory address being written
    pub write_addr: Option<u32>,
    
    /// Memory value being written
    pub write_value: Option<u32>,
    
    /// Next program counter value
    pub next_pc: u32,
    
    /// Whether this instruction is in a read-only segment
    pub is_readonly: bool,
    
    /// Memory segment type for this instruction
    pub segment_type: MemorySegmentType,
    
    /// Last memory location modified
    pub last_modified_addr: Option<u32>,
}

impl BitVMXInstruction {
    /// Create a new BitVMX instruction from a standard RISC-V instruction
    pub fn new(instruction: Instruction, pc: u32, segment_type: MemorySegmentType) -> Self {
        // Default values
        let mut read_addr1 = None;
        let mut read_value1 = None;
        let mut read_addr2 = None;
        let mut read_value2 = None;
        let mut write_addr = None;
        let mut write_value = None;
        let next_pc = pc + 4; // Default PC increment
        let opcode_addr = pc;
        let opcode_value = 0; // Placeholder, actual value determined at runtime
        let is_readonly = segment_type == MemorySegmentType::ReadOnly;
        let mut last_modified_addr = None;
        
        // Extract memory access information based on instruction type
        match &instruction {
            // Load instructions have one read address
            Instruction::Lw(rd, _offset, rs) | 
            Instruction::Lh(rd, _offset, rs) | 
            Instruction::Lb(rd, _offset, rs) | 
            Instruction::Lhu(rd, _offset, rs) | 
            Instruction::Lbu(rd, _offset, rs) => {
                // Memory address being read is base register + offset
                read_addr1 = Some(0xFFFF0000); // Placeholder, actual address determined at runtime
                read_value1 = Some(0); // Placeholder
                read_addr2 = Some(rs.number() as u32); // Register read
                read_value2 = Some(0); // Placeholder
                write_addr = Some(rd.number() as u32); // Register write
                write_value = Some(0); // Placeholder
                last_modified_addr = write_addr;
                
                // Validate memory access based on segment type
                if segment_type == MemorySegmentType::ReadOnly {
                    // In read-only segment, we can only read from memory, not write
                    // This is a read operation, so it's allowed
                }
            },
            
            // Store instructions have one read and one write address
            Instruction::Sw(rs, _offset, rd) | 
            Instruction::Sh(rs, _offset, rd) | 
            Instruction::Sb(rs, _offset, rd) => {
                read_addr1 = Some(rs.number() as u32); // Source register read
                read_value1 = Some(0); // Placeholder
                read_addr2 = Some(rd.number() as u32); // Base address register read
                read_value2 = Some(0); // Placeholder
                write_addr = Some(0xFFFF0000); // Memory write, placeholder
                write_value = Some(0); // Placeholder
                last_modified_addr = write_addr;
                
                // Validate memory access based on segment type
                if segment_type == MemorySegmentType::ReadOnly {
                    // In read-only segment, we can't write to memory
                    // This is a write operation, so it's not allowed
                    // In a real implementation, we would generate an error here
                }
            },
            
            // Branch instructions modify PC and read two registers
            Instruction::Beq(rs1, rs2, _label) | 
            Instruction::Bne(rs1, rs2, _label) | 
            Instruction::Blt(rs1, rs2, _label) | 
            Instruction::Bge(rs1, rs2, _label) | 
            Instruction::Bltu(rs1, rs2, _label) | 
            Instruction::Bgeu(rs1, rs2, _label) => {
                read_addr1 = Some(rs1.number() as u32); // First register read
                read_value1 = Some(0); // Placeholder
                read_addr2 = Some(rs2.number() as u32); // Second register read
                read_value2 = Some(0); // Placeholder
                // PC update handled by verification logic
            },
            
            // Jump instructions modify PC
            Instruction::Jal(rd, _label) => {
                write_addr = Some(rd.number() as u32); // Register write (link register)
                write_value = Some(pc + 4); // Return address
                last_modified_addr = write_addr;
                // PC update handled by verification logic
            },
            
            Instruction::Jalr(rd, rs, _offset) => {
                read_addr1 = Some(rs.number() as u32); // Base register read
                read_value1 = Some(0); // Placeholder
                write_addr = Some(rd.number() as u32); // Register write (link register)
                write_value = Some(pc + 4); // Return address
                last_modified_addr = write_addr;
                // PC update handled by verification logic
            },
            
            // R-type instructions read two registers and write one
            Instruction::Add(rd, rs1, rs2) | 
            Instruction::Sub(rd, rs1, rs2) | 
            Instruction::And(rd, rs1, rs2) | 
            Instruction::Or(rd, rs1, rs2) | 
            Instruction::Xor(rd, rs1, rs2) | 
            Instruction::Slt(rd, rs1, rs2) | 
            Instruction::Sltu(rd, rs1, rs2) | 
            Instruction::Sll(rd, rs1, rs2) | 
            Instruction::Srl(rd, rs1, rs2) | 
            Instruction::Sra(rd, rs1, rs2) | 
            Instruction::Mul(rd, rs1, rs2) | 
            Instruction::Div(rd, rs1, rs2) | 
            Instruction::Rem(rd, rs1, rs2) => {
                read_addr1 = Some(rs1.number() as u32); // First register read
                read_value1 = Some(0); // Placeholder
                read_addr2 = Some(rs2.number() as u32); // Second register read
                read_value2 = Some(0); // Placeholder
                write_addr = Some(rd.number() as u32); // Register write
                write_value = Some(0); // Placeholder
                last_modified_addr = write_addr;
            },
            
            // I-type instructions read one register and write one
            Instruction::Addi(rd, rs, _imm) | 
            Instruction::Andi(rd, rs, _imm) | 
            Instruction::Ori(rd, rs, _imm) | 
            Instruction::Xori(rd, rs, _imm) | 
            Instruction::Slti(rd, rs, _imm) | 
            Instruction::Sltiu(rd, rs, _imm) | 
            Instruction::Slli(rd, rs, _imm) | 
            Instruction::Srli(rd, rs, _imm) | 
            Instruction::Srai(rd, rs, _imm) => {
                read_addr1 = Some(rs.number() as u32); // Register read
                read_value1 = Some(0); // Placeholder
                write_addr = Some(rd.number() as u32); // Register write
                write_value = Some(0); // Placeholder
                last_modified_addr = write_addr;
            },
            
            // U-type instructions write one register
            Instruction::Lui(rd, _imm) | 
            Instruction::Auipc(rd, _imm) => {
                write_addr = Some(rd.number() as u32); // Register write
                write_value = Some(0); // Placeholder
                last_modified_addr = write_addr;
            },
            
            // Pseudo-instructions
            Instruction::Li(rd, _imm) => {
                write_addr = Some(rd.number() as u32); // Register write
                write_value = Some(0); // Placeholder
                last_modified_addr = write_addr;
            },
            
            Instruction::La(rd, _label) => {
                write_addr = Some(rd.number() as u32); // Register write
                write_value = Some(0); // Placeholder
                last_modified_addr = write_addr;
            },
            
            Instruction::Mv(rd, rs) => {
                read_addr1 = Some(rs.number() as u32); // Register read
                read_value1 = Some(0); // Placeholder
                write_addr = Some(rd.number() as u32); // Register write
                write_value = Some(0); // Placeholder
                last_modified_addr = write_addr;
            },
            
            Instruction::Not(rd, rs) | 
            Instruction::Neg(rd, rs) | 
            Instruction::Seqz(rd, rs) | 
            Instruction::Snez(rd, rs) => {
                read_addr1 = Some(rs.number() as u32); // Register read
                read_value1 = Some(0); // Placeholder
                write_addr = Some(rd.number() as u32); // Register write
                write_value = Some(0); // Placeholder
                last_modified_addr = write_addr;
            },
            
            // Other instructions don't access memory or registers
            _ => {}
        }
        
        Self {
            original: instruction,
            read_addr1,
            read_value1,
            read_addr2,
            read_value2,
            opcode_addr,
            opcode_value,
            write_addr,
            write_value,
            next_pc,
            is_readonly,
            segment_type,
            last_modified_addr,
        }
    }
    
    /// Format the instruction for BitVMX trace
    pub fn format_for_bitvm(&self) -> String {
        // Skip formatting for directives, labels, and comments
        match &self.original {
            Instruction::Label(_) | 
            Instruction::Global(_) | 
            Instruction::Section(_) | 
            Instruction::Align(_) | 
            Instruction::Word(_) | 
            Instruction::Byte(_) | 
            Instruction::Ascii(_) | 
            Instruction::Asciiz(_) | 
            Instruction::Space(_) | 
            Instruction::Comment(_) => return String::new(),
            _ => {}
        }
        
        // Format as: PC, Read1, Read2, Write, NextPC
        let read1 = match (self.read_addr1, self.read_value1) {
            (Some(addr), Some(val)) => format!("R(0x{:08x})=0x{:08x}", addr, val),
            _ => "R(none)".to_string(),
        };
        
        let read2 = match (self.read_addr2, self.read_value2) {
            (Some(addr), Some(val)) => format!("R(0x{:08x})=0x{:08x}", addr, val),
            _ => "R(none)".to_string(),
        };
        
        let write = match (self.write_addr, self.write_value) {
            (Some(addr), Some(val)) => format!("W(0x{:08x})=0x{:08x}", addr, val),
            _ => "W(none)".to_string(),
        };
        
        format!(
            "PC=0x{:08x} {} {} {} NextPC=0x{:08x} Segment={:?}",
            self.opcode_addr,
            read1,
            read2,
            write,
            self.next_pc,
            self.segment_type
        )
    }
    
    /// Validate memory access based on segment type
    pub fn validate_memory_access(&self) -> std::result::Result<(), CodeGenError> {
        // In read-only segment, we can't write to memory
        if self.segment_type == MemorySegmentType::ReadOnly {
            match &self.original {
                Instruction::Sw(_, _, _) | 
                Instruction::Sh(_, _, _) | 
                Instruction::Sb(_, _, _) => {
                    return Err(CodeGenError::InvalidInstruction(format!(
                        "Memory write not allowed in read-only segment: {:?}",
                        self.original
                    )));
                }
                _ => {}
            }
        }
        
        Ok(())
    }
}

/// RISC-V Code Generation Error with additional context
#[derive(Debug, Error)]
pub enum BitVMXCodeGenError {
    #[error("Memory access violation: {0}")]
    MemoryAccessViolation(String),
    
    #[error("Invalid instruction for BitVMX: {0}")]
    InvalidInstruction(String),
    
    #[error("Segment type error: {0}")]
    SegmentTypeError(String),
    
    #[error("Code generation error: {0}")]
    CodeGenError(#[from] CodeGenError),
}

/// BitVMX code generator for RISC-V assembly
pub struct BitVMXCodeGenerator {
    /// Standard code generator
    generator: CodeGenerator,
    
    /// Current program counter
    pc: u32,
    
    /// BitVMX instructions
    bitvm_instructions: Vec<BitVMXInstruction>,
    
    /// Memory segment type
    segment_type: MemorySegmentType,
    
    /// Memory access tracking
    memory_accesses: Vec<(u32, u32)>, // (address, value)
    
    /// Last memory location modified
    last_modified_addr: Option<u32>,
    
    /// Auto-segment mode (automatically switches segment type based on instruction)
    auto_segment: bool,
}

impl BitVMXCodeGenerator {
    /// Create a new BitVMX code generator
    pub fn new() -> Self {
        Self {
            generator: CodeGenerator::new(),
            pc: 0,
            bitvm_instructions: Vec::new(),
            segment_type: MemorySegmentType::ReadWrite,
            memory_accesses: Vec::new(),
            last_modified_addr: None,
            auto_segment: true,
        }
    }
    
    /// Set the memory segment type
    pub fn set_segment_type(&mut self, segment_type: MemorySegmentType) {
        self.segment_type = segment_type;
    }
    
    /// Get the current memory segment type
    pub fn get_segment_type(&self) -> MemorySegmentType {
        self.segment_type
    }
    
    /// Enable or disable auto-segment mode
    pub fn set_auto_segment(&mut self, auto_segment: bool) {
        self.auto_segment = auto_segment;
    }
    
    /// Get the current program counter
    pub fn current_pc(&self) -> u32 {
        self.pc
    }
    
    /// Add an instruction to the code generator with improved error handling
    pub fn add_instruction(&mut self, instruction: Instruction) -> std::result::Result<(), BitVMXCodeGenError> {
        // If auto-segment is enabled, set the appropriate segment type based on the instruction
        if self.auto_segment {
            match &instruction {
                // Memory write instructions require ReadWrite segment
                Instruction::Sw(_, _, _) | 
                Instruction::Sh(_, _, _) | 
                Instruction::Sb(_, _, _) => {
                    self.segment_type = MemorySegmentType::ReadWrite;
                },
                
                // Other instructions can use ReadOnly segment
                _ => {}
            }
        }
        
        // Create BitVMX instruction
        let bitvm_instruction = BitVMXInstruction::new(instruction.clone(), self.pc, self.segment_type);
        
        // Validate memory access with improved error handling
        match bitvm_instruction.validate_memory_access() {
            Ok(_) => {},
            Err(e) => {
                return Err(BitVMXCodeGenError::MemoryAccessViolation(format!(
                    "Memory access violation at PC {:#x}: {}",
                    self.pc, e
                )));
            }
        }
        
        // Track memory access
        if let Some(addr) = bitvm_instruction.write_addr {
            if let Some(value) = bitvm_instruction.write_value {
                self.memory_accesses.push((addr, value));
                self.last_modified_addr = Some(addr);
            }
        }
        
        // Add to standard generator - CodeGenerator::add_instruction doesn't return a Result
        self.generator.add_instruction(instruction.clone());
        
        // Add to BitVMX instructions
        self.bitvm_instructions.push(bitvm_instruction);
        
        // Update PC
        self.pc += 4;
        
        Ok(())
    }
    
    /// Generate standard assembly code
    pub fn generate_assembly(&self) -> String {
        self.generator.generate()
    }
    
    /// Generate BitVMX trace
    pub fn generate_bitvm_trace(&self) -> String {
        let mut result = String::new();
        
        for instruction in &self.bitvm_instructions {
            let formatted = instruction.format_for_bitvm();
            if !formatted.is_empty() {
                result.push_str(&formatted);
                result.push('\n');
            }
        }
        
        result
    }
    
    /// Get the last memory location modified
    pub fn get_last_modified_addr(&self) -> Option<u32> {
        self.last_modified_addr
    }
    
    /// Get all memory accesses
    pub fn get_memory_accesses(&self) -> &[(u32, u32)] {
        &self.memory_accesses
    }
    
    /// Validate all memory accesses with improved error reporting
    pub fn validate_all_memory_accesses(&self) -> std::result::Result<(), BitVMXCodeGenError> {
        for (i, instr) in self.bitvm_instructions.iter().enumerate() {
            match instr.validate_memory_access() {
                Ok(_) => {},
                Err(e) => {
                    return Err(BitVMXCodeGenError::MemoryAccessViolation(format!(
                        "Memory access violation at instruction {}: PC {:#x}: {}",
                        i, instr.next_pc - 4, e
                    )));
                }
            }
        }
        Ok(())
    }

    /// Check memory alignment for an instruction
    pub fn check_memory_alignment(&self, _instruction: &Instruction, address: u32) -> bool {
        let alignment = self.get_alignment(address);
        memory::is_aligned(address, alignment)
    }

    /// Get the alignment of an address
    pub fn get_alignment(&self, address: u32) -> u32 {
        address & 0x3
    }

    /// Validate memory access for an instruction
    pub fn validate_memory_access(&self, _instruction: &Instruction, address: u32) -> std::result::Result<(), BitVMXCodeGenError> {
        let alignment = self.get_alignment(address);
        
        // Use the memory module for validation
        match memory::is_valid_memory_operation(address, self.segment_type == MemorySegmentType::ReadWrite, alignment) {
            Ok(_) => Ok(()),
            Err(msg) => Err(BitVMXCodeGenError::MemoryAccessViolation(msg)),
        }
    }

    /// Add an instruction with memory alignment validation
    pub fn add_instruction_with_validation(&mut self, instruction: Instruction) -> std::result::Result<(), BitVMXCodeGenError> {
        // Extract memory address from the instruction
        let address = match &instruction {
            Instruction::Lw(_, offset, _base) | Instruction::Lh(_, offset, _base) | 
            Instruction::Lb(_, offset, _base) | Instruction::Lhu(_, offset, _base) | 
            Instruction::Lbu(_, offset, _base) | Instruction::Sw(_, offset, _base) | 
            Instruction::Sh(_, offset, _base) | Instruction::Sb(_, offset, _base) => {
                // In a real implementation, we would compute the actual address
                // For now, we'll just use a placeholder
                *offset as u32
            },
            _ => 0, // Non-memory instructions don't need validation
        };
        
        // Validate memory access if it's a memory instruction
        match &instruction {
            Instruction::Lw(_, _, _) | Instruction::Lh(_, _, _) | Instruction::Lb(_, _, _) | 
            Instruction::Lhu(_, _, _) | Instruction::Lbu(_, _, _) | Instruction::Sw(_, _, _) | 
            Instruction::Sh(_, _, _) | Instruction::Sb(_, _, _) => {
                self.validate_memory_access(&instruction, address)?;
            },
            _ => {}, // Non-memory instructions don't need validation
        }
        
        // Add the instruction
        self.add_instruction(instruction)
    }
}

/// Memory alignment masks for BitVMX compatibility
pub mod alignment_masks {
    use super::Instruction;

    /// Get the mask for the first round of a store operation
    pub fn get_mask_round_1(instruction: &Instruction, alignment: u32) -> (u32, u32, i8) {
        match instruction {
            Instruction::Sb(_, _, _) => match alignment {
                0 => (0xFFFF_FF00, 0x0000_00FF, 0),
                1 => (0xFFFF_00FF, 0x0000_00FF, 1),
                2 => (0xFF00_FFFF, 0x0000_00FF, 2),
                3 => (0x00FF_FFFF, 0x0000_00FF, 3),
                _ => panic!("Invalid alignment for Sb"),
            },
            Instruction::Sh(_, _, _) => match alignment {
                0 => (0xFFFF_0000, 0x0000_FFFF, 0),
                1 => (0xFF00_00FF, 0x0000_FFFF, 1),
                2 => (0x0000_FFFF, 0x0000_FFFF, 2),
                3 => (0x00FF_FFFF, 0x0000_00FF, 3),
                _ => panic!("Invalid alignment for Sh"),
            },
            Instruction::Sw(_, _, _) => match alignment {
                3 => (0x00FF_FFFF, 0x0000_00FF, 3),
                2 => (0x0000_FFFF, 0x0000_FFFF, 2),
                1 => (0x0000_00FF, 0x00FF_FFFF, 1),
                0 => (0x0000_0000, 0xFFFF_FFFF, 0), // Aligned case
                _ => panic!("Invalid alignment for Sw"),
            },
            _ => panic!("Instruction does not support get_mask_round_1"),
        }
    }

    /// Get the mask for the second round of a store operation
    pub fn get_mask_round_2(instruction: &Instruction, alignment: u32) -> (u32, u32, i8) {
        match instruction {
            Instruction::Sh(_, _, _) => match alignment {
                3 => (0xFFFF_FF00, 0x0000_FF00, -1),
                _ => panic!("Invalid alignment for Sh in round 2"),
            },
            Instruction::Sw(_, _, _) => match alignment {
                3 => (0xFF00_0000, 0xFFFF_FF00, -1),
                2 => (0xFFFF_0000, 0xFFFF_0000, -2),
                1 => (0xFFFF_FF00, 0xFF00_0000, -3),
                _ => panic!("Invalid alignment for Sw in round 2"),
            },
            _ => panic!("Instruction does not support get_mask_round_2"),
        }
    }

    /// Apply sign extension for load operations
    pub fn sign_extension(instruction: &Instruction, value: u32) -> u32 {
        match instruction {
            Instruction::Lb(_, _, _) => {
                if (value & 0x0000_0080) != 0 {
                    return 0xFFFFFF00 | value;
                }
                value
            }
            Instruction::Lh(_, _, _) => {
                if (value & 0x0000_8000) != 0 {
                    return 0xFFFF0000 | value;
                }
                value
            }
            _ => value,
        }
    }

    /// Get the mask for the first round of a load operation
    pub fn get_mask_round_1_for_load(instruction: &Instruction, alignment: u32) -> (u32, i8) {
        match instruction {
            Instruction::Lb(_, _, _) => match alignment {
                0 => (0x0000_00FF, 0),
                1 => (0x0000_FF00, -1),
                2 => (0x00FF_0000, -2),
                3 => (0xFF00_0000, -3),
                _ => panic!("Invalid alignment for Lb"),
            },
            Instruction::Lh(_, _, _) | Instruction::Lhu(_, _, _) => match alignment {
                0 => (0x0000_FFFF, 0),
                1 => (0x00FF_FF00, -1),
                2 => (0xFF00_0000, -2),
                3 => (0xFF00_0000, -3), // This will need a second round
                _ => panic!("Invalid alignment for Lh/Lhu"),
            },
            Instruction::Lw(_, _, _) => match alignment {
                0 => (0xFFFF_FFFF, 0),
                _ => (0x0000_0000, 0), // Non-aligned loads need special handling
            },
            _ => panic!("Instruction does not support get_mask_round_1_for_load"),
        }
    }

    /// Get the mask for the second round of a load operation
    pub fn get_mask_round_2_for_load(instruction: &Instruction, alignment: u32) -> (u32, i8) {
        match instruction {
            Instruction::Lh(_, _, _) | Instruction::Lhu(_, _, _) => match alignment {
                3 => (0x0000_00FF, 1),
                _ => panic!("Invalid alignment for Lh/Lhu in round 2"),
            },
            Instruction::Lw(_, _, _) => match alignment {
                1 => (0xFFFF_FF00, 1),
                2 => (0xFFFF_0000, 2),
                3 => (0xFF00_0000, 3),
                _ => panic!("Invalid alignment for Lw in round 2"),
            },
            _ => panic!("Instruction does not support get_mask_round_2_for_load"),
        }
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
        gen.add_instruction(Instruction::Blt(Register::A0, Register::A1, "label".to_string()));
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
        assert!(asm.contains(".section .text"));
        assert!(asm.contains(".global main"));
        assert!(asm.contains("main:"));
        assert!(asm.contains("# This is a comment"));
    }
} 