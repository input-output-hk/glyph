use std::io;
use thiserror::Error;
use uplc::ast::{DeBruijn, Program};

mod serializer;
mod constants;
mod memory_layout;

pub use serializer::UPLCSerializer;
pub use memory_layout::{MemoryLayout, Address};

/// Error type for serialization failures
#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),
    
    #[error("Invalid term type: {0}")]
    InvalidTermType(String),
    
    #[error("Integer too large to serialize: {0}")]
    IntegerTooLarge(String),
    
    #[error("String too large to serialize: {0}")]
    StringTooLarge(String),
    
    #[error("ByteString too large to serialize: {0}")]
    ByteStringTooLarge(String),
    
    #[error("Data structure too complex: {0}")]
    DataTooComplex(String),
    
    #[error("Memory layout error: {0}")]
    MemoryLayoutError(String),
}

/// Result type for serialization operations
pub type Result<T> = std::result::Result<T, SerializationError>;

/// Serialize a UPLC program to a binary format suitable for the RISC-V CEK machine.
/// 
/// This serialization follows a specific binary format:
/// 
/// - Program header (12 bytes):
///   - Magic bytes (4 bytes): 'UPLC'
///   - Version (3 bytes): `(major, minor, patch)`
///   - Reserved (1 byte): 0x00
///   - Root term offset (4 bytes): Offset to the root term
/// 
/// - Term region: Contains serialized terms with the following format:
///   - Tag byte (1 byte): Identifies the term type
///   - Term-specific data (variable length)
/// 
/// - Constant pools: Separate regions for different types of constants:
///   - Integer pool: Stores integer constants
///   - ByteString pool: Stores bytestring constants
///   - String pool: Stores string constants
///   - Complex data pool: Stores other data structures
/// 
/// Each term is serialized according to its type (see `constants.rs` for tag values):
/// 
/// - Variable (0x00):
///   - DeBruijn index (4 bytes)
/// 
/// - Lambda (0x01):
///   - Body reference (4 bytes)
/// 
/// - Apply (0x02):
///   - Function reference (4 bytes)
///   - Argument reference (4 bytes)
/// 
/// - Force (0x03):
///   - Term reference (4 bytes)
/// 
/// - Delay (0x04):
///   - Term reference (4 bytes)
/// 
/// - Constant (0x05):
///   - Constant reference (4 bytes)
/// 
/// - Builtin (0x06):
///   - Builtin function ID (1 byte)
/// 
/// - Error (0x07):
///   - No additional data
/// 
/// - Constructor (0x08):
///   - Tag (2 bytes)
///   - Field count (2 bytes)
///   - Fields reference (4 bytes)
/// 
/// - Case (0x09):
///   - Match term reference (4 bytes)
///   - Branch count (2 bytes)
///   - Branches reference (4 bytes)
/// 
/// Constants are serialized in separate pools, each with a type tag and size-specific encoding.
/// 
/// # Arguments
/// 
/// * `program` - The UPLC program to serialize
/// 
/// # Returns
/// 
/// A `Result` containing the serialized program as a `Vec<u8>` or a `SerializationError`
pub fn serialize_program(program: &Program<DeBruijn>) -> Result<Vec<u8>> {
    let serializer = UPLCSerializer::new(program);
    serializer.serialize()
}

/// Parse a UPLC file and serialize it to binary format
/// 
/// This is a convenience function that parses a UPLC program from text format
/// and then serializes it to the binary format for the RISC-V CEK machine.
/// 
/// # Arguments
/// 
/// * `uplc_text` - The UPLC program text
/// 
/// # Returns
/// 
/// A `Result` containing the serialized program as a `Vec<u8>` or a `SerializationError`
pub fn parse_and_serialize(uplc_text: &str) -> Result<Vec<u8>> {
    let program = uplc::parser::program(uplc_text)
        .map_err(|e| SerializationError::InvalidTermType(format!("Parse error: {}", e)))?
        .to_debruijn()
        .map_err(|e| SerializationError::InvalidTermType(format!("DeBruijn conversion error: {}", e)))?;
    
    serialize_program(&program)
}

/// Deserialize binary format back to a UPLC program (placeholder for future implementation)
pub fn deserialize(_binary: &[u8]) -> Result<Program<DeBruijn>> {
    unimplemented!("Deserialization is not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_serialization() {
        // A simple program: (program 1.0.0 [(con integer 42)])
        let uplc_text = "(program 1.0.0 [(con integer 42)])";
        let binary = parse_and_serialize(uplc_text).unwrap();
        
        // Check basic structure
        assert_eq!(&binary[0..4], b"UPLC", "Magic bytes should be 'UPLC'");
        assert_eq!(binary[4], 1, "Major version should be 1");
        assert_eq!(binary[5], 0, "Minor version should be 0");
        assert_eq!(binary[6], 0, "Patch version should be 0");
        
        // The program should be successfully serialized
        assert!(binary.len() > 12, "Binary should be larger than just the header");
    }
    
    #[test]
    fn test_term_types() {
        // A program with various term types
        let uplc_text = r#"
        (program 1.0.0
          [
            (lam x
              [
                (force
                  (delay x)
                )
                (con integer 1)
              ]
            )
            (builtin addInteger)
          ]
        )
        "#;
        
        let binary = parse_and_serialize(uplc_text).unwrap();
        
        // We can't check exact binary contents as the memory layout might change,
        // but we can verify the program was serialized
        assert!(binary.len() > 20, "Binary should contain data beyond the header");
    }
    
    #[test]
    fn test_constants() {
        // Test integer constants
        let int_test = "(program 1.0.0 [(con integer 42)])";
        let int_binary = parse_and_serialize(int_test).unwrap();
        assert!(int_binary.len() > 12);
        
        // Test boolean constant
        let bool_test = "(program 1.0.0 [(con bool True)])";
        let bool_binary = parse_and_serialize(bool_test).unwrap();
        assert!(bool_binary.len() > 12);
        
        // Test unit constant
        let unit_test = "(program 1.0.0 [(con unit ())])";
        let unit_binary = parse_and_serialize(unit_test).unwrap();
        assert!(unit_binary.len() > 12);
        
        // Test string constant
        let string_test = r#"(program 1.0.0 [(con string "hello")])"#;
        let string_binary = parse_and_serialize(string_test).unwrap();
        assert!(string_binary.len() > 12);
        
        // Test bytestring constant
        let bytestring_test = "(program 1.0.0 [(con byteString #\"01020304\")])";
        let bytestring_binary = parse_and_serialize(bytestring_test).unwrap();
        assert!(bytestring_binary.len() > 12);
    }
    
    #[test]
    fn test_complex_program() {
        // A more complex program with nested terms
        let complex_text = r#"
        (program 1.0.0
          [
            (lam f 
              (lam x
                [
                  [
                    (force f)
                    x
                  ]
                  [
                    (builtin addInteger)
                    (con integer 40)
                    (con integer 2)
                  ]
                ]
              )
            )
            (delay
              (lam arg1
                (lam arg2
                  [
                    (lam cond
                      [
                        [
                          (builtin ifThenElse)
                          cond
                          (delay arg1)
                          (delay arg2)
                        ]
                      ]
                    )
                    [
                      (builtin lessThanInteger)
                      arg1
                      arg2
                    ]
                  ]
                )
              )
            )
          ]
        )
        "#;
        
        let binary = parse_and_serialize(complex_text).unwrap();
        
        // Again, we can't check exact contents, but we want to verify
        // that the complex program was serialized without errors
        assert!(binary.len() > 100, "Complex program should result in a sizeable binary");
    }
    
    #[test]
    fn test_error_handling() {
        // Test handling of invalid UPLC
        let invalid_uplc = "(program 1.0.0 [(invalid_term)])";
        let result = parse_and_serialize(invalid_uplc);
        assert!(result.is_err(), "Serializing invalid UPLC should return an error");
        
        // Test handling of too-large data (this is hard to test directly without
        // creating enormous terms, so we'll skip actual implementation)
    }
} 