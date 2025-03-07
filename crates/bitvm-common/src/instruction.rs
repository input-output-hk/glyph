//! BitVMX Instruction Utilities
//!
//! This module provides utilities for working with RISC-V instructions in BitVMX.
//! It includes instruction parsing, encoding, and BitVMX-specific processing.

use riscv_decode::Instruction as RiscVInstruction;

/// BitVMX instruction format (a wrapper around RISC-V instructions with BitVMX metadata)
#[derive(Debug, Clone)]
pub struct BitVMXInstruction {
    /// The underlying RISC-V instruction
    pub instruction: RiscVInstruction,
    
    /// Program counter for this instruction
    pub pc: u32,
    
    /// First memory read address (if any)
    pub read_addr1: Option<u32>,
    
    /// Second memory read address (if any)
    pub read_addr2: Option<u32>,
    
    /// Memory write address (if any)
    pub write_addr: Option<u32>,
    
    /// Next program counter
    pub next_pc: u32,
    
    /// Micro-instruction step (for multi-step instructions)
    pub micro: u8,
}

impl BitVMXInstruction {
    /// Create a new BitVMX instruction
    pub fn new(instruction: RiscVInstruction, pc: u32) -> Self {
        Self {
            instruction,
            pc,
            read_addr1: None,
            read_addr2: None,
            write_addr: None,
            next_pc: pc + 4, // Default to next instruction
            micro: 0,
        }
    }
    
    /// Set the first memory read address
    pub fn with_read1(mut self, addr: u32) -> Self {
        self.read_addr1 = Some(addr);
        self
    }
    
    /// Set the second memory read address
    pub fn with_read2(mut self, addr: u32) -> Self {
        self.read_addr2 = Some(addr);
        self
    }
    
    /// Set the memory write address
    pub fn with_write(mut self, addr: u32) -> Self {
        self.write_addr = Some(addr);
        self
    }
    
    /// Set the next program counter
    pub fn with_next_pc(mut self, next_pc: u32) -> Self {
        self.next_pc = next_pc;
        self
    }
    
    /// Set the micro-instruction step
    pub fn with_micro(mut self, micro: u8) -> Self {
        self.micro = micro;
        self
    }
    
    /// Check if this is a jump instruction
    pub fn is_jump(&self) -> bool {
        matches!(
            self.instruction,
            RiscVInstruction::Jal(_) | RiscVInstruction::Jalr(_)
        )
    }
    
    /// Check if this is a branch instruction
    pub fn is_branch(&self) -> bool {
        matches!(
            self.instruction,
            RiscVInstruction::Beq(_) | RiscVInstruction::Bne(_) |
            RiscVInstruction::Blt(_) | RiscVInstruction::Bge(_) |
            RiscVInstruction::Bltu(_) | RiscVInstruction::Bgeu(_)
        )
    }
    
    /// Check if this is a memory load instruction
    pub fn is_load(&self) -> bool {
        matches!(
            self.instruction,
            RiscVInstruction::Lb(_) | RiscVInstruction::Lh(_) |
            RiscVInstruction::Lw(_) | RiscVInstruction::Lbu(_) |
            RiscVInstruction::Lhu(_)
        )
    }
    
    /// Check if this is a memory store instruction
    pub fn is_store(&self) -> bool {
        matches!(
            self.instruction,
            RiscVInstruction::Sb(_) | RiscVInstruction::Sh(_) |
            RiscVInstruction::Sw(_)
        )
    }
}

/// Parse a register name to its number
pub fn parse_register_name(reg_str: &str) -> Result<u32, String> {
    match reg_str {
        "zero" | "x0" => Ok(0),
        "ra" | "x1" => Ok(1),
        "sp" | "x2" => Ok(2),
        "gp" | "x3" => Ok(3),
        "tp" | "x4" => Ok(4),
        "t0" | "x5" => Ok(5),
        "t1" | "x6" => Ok(6),
        "t2" | "x7" => Ok(7),
        "s0" | "fp" | "x8" => Ok(8),
        "s1" | "x9" => Ok(9),
        "a0" | "x10" => Ok(10),
        "a1" | "x11" => Ok(11),
        "a2" | "x12" => Ok(12),
        "a3" | "x13" => Ok(13),
        "a4" | "x14" => Ok(14),
        "a5" | "x15" => Ok(15),
        "a6" | "x16" => Ok(16),
        "a7" | "x17" => Ok(17),
        "s2" | "x18" => Ok(18),
        "s3" | "x19" => Ok(19),
        "s4" | "x20" => Ok(20),
        "s5" | "x21" => Ok(21),
        "s6" | "x22" => Ok(22),
        "s7" | "x23" => Ok(23),
        "s8" | "x24" => Ok(24),
        "s9" | "x25" => Ok(25),
        "s10" | "x26" => Ok(26),
        "s11" | "x27" => Ok(27),
        "t3" | "x28" => Ok(28),
        "t4" | "x29" => Ok(29),
        "t5" | "x30" => Ok(30),
        "t6" | "x31" => Ok(31),
        _ => Err(format!("Invalid register name: {}", reg_str)),
    }
}

/// Parse an immediate value (decimal or hex)
pub fn parse_immediate(imm_str: &str) -> Result<u32, String> {
    if imm_str.starts_with("0x") {
        // Hexadecimal
        u32::from_str_radix(&imm_str[2..], 16).map_err(|e| format!("Invalid hexadecimal immediate: {}", e))
    } else {
        // Decimal
        imm_str.parse::<u32>().map_err(|e| format!("Invalid decimal immediate: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riscv_decode::Instruction::*;
    use riscv_decode::types::{JType, BType, IType, SType};
    
    #[test]
    fn test_parse_register_name() {
        assert_eq!(parse_register_name("x0").unwrap(), 0);
        assert_eq!(parse_register_name("zero").unwrap(), 0);
        assert_eq!(parse_register_name("ra").unwrap(), 1);
        assert_eq!(parse_register_name("x1").unwrap(), 1);
        assert_eq!(parse_register_name("sp").unwrap(), 2);
        assert_eq!(parse_register_name("a0").unwrap(), 10);
        assert_eq!(parse_register_name("t6").unwrap(), 31);
        assert!(parse_register_name("invalid").is_err());
    }
    
    #[test]
    fn test_parse_immediate() {
        assert_eq!(parse_immediate("42").unwrap(), 42);
        assert_eq!(parse_immediate("0x2A").unwrap(), 42);
        assert!(parse_immediate("invalid").is_err());
    }
    
    #[test]
    fn test_bitvm_instruction() {
        // Test jump instruction detection
        let jal_instr = BitVMXInstruction::new(Jal(JType(1)), 0x1000);
        assert!(jal_instr.is_jump());
        assert!(!jal_instr.is_branch());
        assert!(!jal_instr.is_load());
        assert!(!jal_instr.is_store());
        
        // Test branch instruction detection
        let beq_instr = BitVMXInstruction::new(Beq(BType(1)), 0x1000);
        assert!(!beq_instr.is_jump());
        assert!(beq_instr.is_branch());
        assert!(!beq_instr.is_load());
        assert!(!beq_instr.is_store());
        
        // Test load instruction detection
        let lw_instr = BitVMXInstruction::new(Lw(IType(1)), 0x1000);
        assert!(!lw_instr.is_jump());
        assert!(!lw_instr.is_branch());
        assert!(lw_instr.is_load());
        assert!(!lw_instr.is_store());
        
        // Test store instruction detection
        let sw_instr = BitVMXInstruction::new(Sw(SType(1)), 0x1000);
        assert!(!sw_instr.is_jump());
        assert!(!sw_instr.is_branch());
        assert!(!sw_instr.is_load());
        assert!(sw_instr.is_store());
    }
} 