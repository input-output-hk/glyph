//! BitVM Common Utilities
//!
//! This crate provides common utilities for integrating with BitVMX, the verification
//! protocol for Bitcoin transactions. It centralizes functionality that was previously
//! duplicated across the `uplc-to-riscv` and `risc-v-gen` crates.
//!
//! # Components
//!
//! - **Memory Segments**: Utilities for working with memory segments in BitVMX
//! - **Instruction Handling**: Common code for working with RISC-V instructions
//! - **Trace Generation**: Utilities for generating execution traces
//! - **Verification**: Support for BitVMX's verification protocol
//! - **Emulator**: Integration with the BitVMX-CPU emulator
//!
//! # Example
//!
//! ```rust,no_run
//! use bitvm_common::memory::{is_valid_memory_operation, MemorySegmentType};
//!
//! fn check_memory_access(addr: u32, is_write: bool) -> Result<(), String> {
//!     is_valid_memory_operation(addr, is_write, 4)
//! }
//! ```

pub mod memory;
pub mod instruction;
pub mod trace;
pub mod verification;
pub mod emulator;

// Re-export key types and functions for convenience
pub use memory::{MemorySegmentType, is_valid_memory_operation, is_aligned, is_in_segment, 
                  is_in_code_segment, is_in_data_segment, is_in_heap_segment, is_in_stack_segment};
pub use trace::{ExecutionTrace, ExecutionStep, TraceRWStepAdapter, convert_from_bitvm_cpu_trace, convert_from_string_trace};
pub use verification::{generate_initial_hash, generate_step_hash, hash_to_hex_string, 
                       create_verification_script, create_verification_script_mapping};
pub use instruction::BitVMXInstruction;
pub use emulator::execute_assembly;

/// Version of the BitVM Common library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// BitVMX compatibility constants
pub mod constants {
    /// BitVMX trace format version
    pub const TRACE_FORMAT_VERSION: &str = "0.1.0";
    
    /// BitVMX memory segment boundaries
    pub const CODE_SEGMENT_START: u32 = 0x00000000;
    pub const CODE_SEGMENT_END: u32 = 0x00100000;
    pub const DATA_SEGMENT_START: u32 = 0x00100000;
    pub const DATA_SEGMENT_END: u32 = 0x00200000;
    pub const HEAP_SEGMENT_START: u32 = 0x00200000;
    pub const HEAP_SEGMENT_END: u32 = 0x00300000;
    pub const STACK_SEGMENT_START: u32 = 0x00300000;
    pub const STACK_SEGMENT_END: u32 = 0x00400000;
    
    /// BitVMX alignment masks
    pub const ALIGNMENT_MASK_1: u32 = 0x0;
    pub const ALIGNMENT_MASK_2: u32 = 0x1;
    pub const ALIGNMENT_MASK_4: u32 = 0x3;
    pub const ALIGNMENT_MASK_8: u32 = 0x7;
} 