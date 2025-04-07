//! Memory utilities for BitVMX
//!
//! This module provides utilities for working with memory segments in BitVMX.
//! It handles memory segment types, alignment checks, and ensures valid memory operations.

use crate::constants::*;

/// Memory segment type for BitVMX
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySegmentType {
    /// Read-only segment (code and data)
    ReadOnly,

    /// Read-write segment (heap and stack)
    ReadWrite,
}

/// Check if an address is aligned according to BitVMX rules
///
/// # Arguments
///
/// * `addr` - The memory address to check
/// * `alignment` - The required alignment (1, 2, 4, or 8 bytes)
///
/// # Returns
///
/// `true` if the address is properly aligned, `false` otherwise
pub fn is_aligned(addr: u32, alignment: u32) -> bool {
    match alignment {
        1 => true,
        2 => (addr & ALIGNMENT_MASK_2) == 0,
        4 => (addr & ALIGNMENT_MASK_4) == 0,
        8 => (addr & ALIGNMENT_MASK_8) == 0,
        _ => false,
    }
}

/// Check if an address is in a specific memory segment
///
/// # Arguments
///
/// * `addr` - The memory address to check
/// * `start` - The start address of the segment
/// * `end` - The end address of the segment
///
/// # Returns
///
/// `true` if the address is in the segment, `false` otherwise
pub fn is_in_segment(addr: u32, start: u32, end: u32) -> bool {
    addr >= start && addr < end
}

/// Check if an address is in the code segment
pub fn is_in_code_segment(addr: u32) -> bool {
    is_in_segment(addr, CODE_SEGMENT_START, CODE_SEGMENT_END)
}

/// Check if an address is in the data segment
pub fn is_in_data_segment(addr: u32) -> bool {
    is_in_segment(addr, DATA_SEGMENT_START, DATA_SEGMENT_END)
}

/// Check if an address is in the heap segment
pub fn is_in_heap_segment(addr: u32) -> bool {
    is_in_segment(addr, HEAP_SEGMENT_START, HEAP_SEGMENT_END)
}

/// Check if an address is in the stack segment
pub fn is_in_stack_segment(addr: u32) -> bool {
    is_in_segment(addr, STACK_SEGMENT_START, STACK_SEGMENT_END)
}

/// Get the memory segment type for an address
///
/// # Arguments
///
/// * `addr` - The memory address
///
/// # Returns
///
/// The appropriate `MemorySegmentType` based on the address
pub fn get_segment_type(addr: u32) -> Option<MemorySegmentType> {
    if is_in_code_segment(addr) || is_in_data_segment(addr) {
        Some(MemorySegmentType::ReadOnly)
    } else if is_in_heap_segment(addr) || is_in_stack_segment(addr) {
        Some(MemorySegmentType::ReadWrite)
    } else {
        None
    }
}

/// Check if a memory operation is valid according to BitVMX rules
///
/// # Arguments
///
/// * `addr` - The memory address
/// * `is_write` - Whether this is a write operation
/// * `alignment` - The required alignment (1, 2, 4, or 8 bytes)
///
/// # Returns
///
/// `Ok(())` if the operation is valid, `Err` with an error message otherwise
pub fn is_valid_memory_operation(addr: u32, is_write: bool, alignment: u32) -> Result<(), String> {
    // Check alignment
    if !is_aligned(addr, alignment) {
        return Err(format!(
            "Unaligned memory access at address 0x{:08x} with alignment {}",
            addr, alignment
        ));
    }

    // Check segment permissions
    if is_in_code_segment(addr) || is_in_data_segment(addr) {
        if is_write {
            return Err(format!(
                "Write to read-only memory at address 0x{:08x}",
                addr
            ));
        }
    } else if !is_in_heap_segment(addr) && !is_in_stack_segment(addr) {
        return Err(format!(
            "Memory access outside valid segments at address 0x{:08x}",
            addr
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_aligned() {
        assert!(is_aligned(0x00000000, 4));
        assert!(is_aligned(0x00000004, 4));
        assert!(!is_aligned(0x00000001, 4));
        assert!(!is_aligned(0x00000002, 4));
        assert!(!is_aligned(0x00000003, 4));

        assert!(is_aligned(0x00000000, 2));
        assert!(is_aligned(0x00000002, 2));
        assert!(!is_aligned(0x00000001, 2));

        assert!(is_aligned(0x00000000, 1));
        assert!(is_aligned(0x00000001, 1));
    }

    #[test]
    fn test_is_in_segment() {
        assert!(is_in_code_segment(0x00000000));
        assert!(is_in_code_segment(0x000FFFFF));
        assert!(!is_in_code_segment(0x00100000));

        assert!(is_in_data_segment(0x00100000));
        assert!(is_in_data_segment(0x001FFFFF));
        assert!(!is_in_data_segment(0x00200000));

        assert!(is_in_heap_segment(0x00200000));
        assert!(is_in_heap_segment(0x002FFFFF));
        assert!(!is_in_heap_segment(0x00300000));

        assert!(is_in_stack_segment(0x00300000));
        assert!(is_in_stack_segment(0x003FFFFF));
        assert!(!is_in_stack_segment(0x00400000));
    }

    #[test]
    fn test_get_segment_type() {
        assert_eq!(
            get_segment_type(0x00000000),
            Some(MemorySegmentType::ReadOnly)
        );
        assert_eq!(
            get_segment_type(0x00100000),
            Some(MemorySegmentType::ReadOnly)
        );
        assert_eq!(
            get_segment_type(0x00200000),
            Some(MemorySegmentType::ReadWrite)
        );
        assert_eq!(
            get_segment_type(0x00300000),
            Some(MemorySegmentType::ReadWrite)
        );
        assert_eq!(get_segment_type(0x00400000), None);
    }

    #[test]
    fn test_is_valid_memory_operation() {
        // Valid read from code segment
        assert!(is_valid_memory_operation(0x00000000, false, 4).is_ok());

        // Invalid write to code segment
        assert!(is_valid_memory_operation(0x00000000, true, 4).is_err());

        // Valid read from data segment
        assert!(is_valid_memory_operation(0x00100000, false, 4).is_ok());

        // Invalid write to data segment
        assert!(is_valid_memory_operation(0x00100000, true, 4).is_err());

        // Valid read from heap segment
        assert!(is_valid_memory_operation(0x00200000, false, 4).is_ok());

        // Valid write to heap segment
        assert!(is_valid_memory_operation(0x00200000, true, 4).is_ok());

        // Valid read from stack segment
        assert!(is_valid_memory_operation(0x00300000, false, 4).is_ok());

        // Valid write to stack segment
        assert!(is_valid_memory_operation(0x00300000, true, 4).is_ok());

        // Invalid unaligned access
        assert!(is_valid_memory_operation(0x00000001, false, 4).is_err());

        // Invalid access outside segments
        assert!(is_valid_memory_operation(0x00400000, false, 4).is_err());
    }
}
