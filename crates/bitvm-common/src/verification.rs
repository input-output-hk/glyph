//! BitVMX Verification Utilities
//!
//! This module provides utilities for BitVMX's verification protocol.
//! It includes execution trace generation, hash chains, and verification game support.

use std::collections::HashMap;
use sha2::{Digest, Sha256};

/// Generate the initial hash for the verification hash chain
///
/// This matches the initial hash used by BitVMX-CPU.
///
/// # Returns
///
/// A Vec<u8> containing the initial hash
pub fn generate_initial_hash() -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(b"BitVMX Initial Hash");
    let result = hasher.finalize();
    result.to_vec()
}

/// Generate a hash for a step in the verification chain
///
/// # Arguments
///
/// * `previous_hash` - The hash of the previous step
/// * `data` - The data for this step
///
/// # Returns
///
/// A 32-byte array containing the new hash
pub fn generate_step_hash(previous_hash: &[u8], data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(previous_hash);
    hasher.update(data);
    let result = hasher.finalize();
    
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Convert a hash to a hex string
///
/// # Arguments
///
/// * `hash` - The hash bytes
///
/// # Returns
///
/// A string containing the hex representation of the hash
pub fn hash_to_hex_string(hash: &[u8]) -> String {
    hash.iter().map(|byte| format!("{:02x}", byte)).collect()
}

/// Create a BitVMX verification script for an instruction
///
/// This function creates a verification script for the given instruction and PC.
/// This is a placeholder implementation and should be replaced with the actual
/// BitVMX-CPU implementation.
///
/// # Arguments
///
/// * `pc` - The program counter value
/// * `opcode` - The opcode/instruction value
///
/// # Returns
///
/// * `Some(String)` - The verification script
/// * `None` - If no verification script could be created
pub fn create_verification_script(pc: u32, opcode: u32) -> Option<String> {
    // In a real implementation, this would use the BitVMX-CPU's instruction_mapping module
    // to create a verification script based on the instruction and PC
    
    // For now, we'll return a placeholder
    Some(format!("# BitVMX verification script for instruction 0x{:08x} at PC 0x{:08x}", opcode, pc))
}

/// Create a BitVMX verification script mapping for all supported instructions
///
/// This function creates a mapping of instruction keys to verification scripts.
/// This is a placeholder implementation and should be replaced with the actual
/// BitVMX-CPU implementation.
///
/// # Arguments
///
/// * `base_register_address` - The base register address
///
/// # Returns
///
/// A HashMap<String, String> mapping instruction keys to verification scripts
pub fn create_verification_script_mapping(base_register_address: u32) -> HashMap<String, String> {
    let mut mapping = HashMap::new();
    
    // In a real implementation, this would create a mapping of instruction keys to verification scripts
    // similar to BitVMX-CPU's create_verification_script_mapping function
    
    // Add some placeholder entries
    mapping.insert("add".to_string(), format!("# BitVMX verification script for add with base register address 0x{:08x}", base_register_address));
    mapping.insert("sub".to_string(), format!("# BitVMX verification script for sub with base register address 0x{:08x}", base_register_address));
    mapping.insert("beq".to_string(), format!("# BitVMX verification script for beq with base register address 0x{:08x}", base_register_address));
    mapping.insert("lw_0".to_string(), format!("# BitVMX verification script for lw_0 with base register address 0x{:08x}", base_register_address));
    mapping.insert("sw_0".to_string(), format!("# BitVMX verification script for sw_0 with base register address 0x{:08x}", base_register_address));
    
    mapping
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_initial_hash() {
        let hash = generate_initial_hash();
        assert_eq!(hash.len(), 32);
    }
    
    #[test]
    fn test_generate_step_hash() {
        let initial_hash = generate_initial_hash();
        let data = b"test data";
        let hash = generate_step_hash(&initial_hash, data);
        assert_eq!(hash.len(), 32);
    }
    
    #[test]
    fn test_hash_to_hex_string() {
        let hash = [
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
            0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
            0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
        ];
        let hex = hash_to_hex_string(&hash);
        assert_eq!(hex, format!("0123456789abcdeffedcba98765432100123456789abcdeffedcba9876543210"));
    }
    
    #[test]
    fn test_create_verification_script() {
        let script = create_verification_script(0x1000, 0x12345678);
        assert!(script.is_some());
        let script = script.unwrap();
        assert!(script.contains("0x12345678"));
        assert!(script.contains("0x00001000"));
    }
    
    #[test]
    fn test_create_verification_script_mapping() {
        let mapping = create_verification_script_mapping(0x1000);
        assert!(mapping.contains_key("add"));
        assert!(mapping.contains_key("sub"));
        assert!(mapping.contains_key("beq"));
        assert!(mapping.contains_key("lw_0"));
        assert!(mapping.contains_key("sw_0"));
    }
} 