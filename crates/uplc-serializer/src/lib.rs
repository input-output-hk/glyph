use std::io;
use thiserror::Error;
use uplc::ast::{DeBruijn, Program};

pub mod constants;
pub mod serializer;

pub use serializer::serialize;

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
/// Each term is serialized according to its type (see `constants.rs` for tag values):
///
/// - Variable (0x00):
///   - DeBruijn index (4 bytes)
///
/// - Lambda (0x01):
///   - Followed by body term (0 bytes)
///
/// - Apply (0x02):
///   - Argument reference (4 bytes)
///   - Followed by function term (0 bytes)
///
/// - Force (0x03):
///   - Followed by body term (0 bytes)
///
/// - Delay (0x04):
///  - Followed by body term (0 bytes)
///
/// - Constant (0x05):
///   - Followed by constant encoding (0 bytes)
///
/// - Builtin (0x06):
///   - Builtin function ID (1 byte)
///
/// - Error (0x07):
///   - No additional data
///
/// - Constructor (0x08):
///   - Tag (2 bytes)
///   - Field count (4 bytes)
///   - x Field references (4*x bytes)
///
/// - Case (0x09):
///   - Match term reference (4 bytes)
///   - Branch count (4 bytes)
///   - x Branch references (4*x bytes)
///
/// # Arguments
///
/// * `program` - The UPLC program to serialize
///
/// # Returns
///
/// A `Result` containing the serialized program as a `Vec<u8>` or a `SerializationError`
pub fn serialize_program(program: &Program<DeBruijn>) -> Result<Vec<u8>> {
    serialize(program, 0)
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
        .map_err(|e| {
            SerializationError::InvalidTermType(format!("DeBruijn conversion error: {}", e))
        })?;

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
        // A simple program: (program 1.0.0 (con integer 42))
        let uplc_text = "(program 1.0.0 (con integer 42))";
        let binary = parse_and_serialize(uplc_text).unwrap();

        // The program should be successfully serialized
        assert!(
            binary.len() > 1,
            "Binary should be larger than just the header"
        );
    }

    #[test]
    fn test_term_types() {
        // A program with various term types
        let uplc_text = r#"
        (program 1.0.0
          (lam x
            [
              (force
                (delay x)
              )
              (con integer 1)
            ]
          )
        )
        "#;

        let binary = parse_and_serialize(uplc_text).unwrap();

        // We can't check exact binary contents as the memory layout might change,
        // but we can verify the program was serialized
        assert!(
            binary.len() > 20,
            "Binary should contain data beyond the header"
        );
    }

    #[test]
    fn test_constants() {
        // Test integer constants
        let int_test = "(program 1.0.0 (con integer 42))";
        let int_binary = parse_and_serialize(int_test).unwrap();
        assert!(int_binary.len() > 1);

        // Test boolean constant
        let bool_test = "(program 1.0.0 (con bool True))";
        let bool_binary = parse_and_serialize(bool_test).unwrap();
        assert!(bool_binary.len() > 1);

        // Test unit constant
        let unit_test = "(program 1.0.0 (con unit ()))";
        let unit_binary = parse_and_serialize(unit_test).unwrap();
        assert!(unit_binary.len() > 1);

        // Test bytestring constant
        let bytestring_test = "(program 1.0.0 (con bytestring #01020304))";
        let bytestring_binary = parse_and_serialize(bytestring_test).unwrap();
        assert!(bytestring_binary.len() > 1);
    }

    #[test]
    fn test_complex_program() {
        // A more complex program with nested terms
        let complex_text = r#"
        (program 1.0.0
          (lam f
            (lam x
              (force
                [
                  (builtin addInteger)
                  x
                  (con integer 2)
                ]
              )
            )
          )
        )
        "#;

        let binary = parse_and_serialize(complex_text).unwrap();

        // Again, we can't check exact contents, but we want to verify
        // that the complex program was serialized without errors
        assert!(
            binary.len() > 0,
            "Binary should contain substantial data for a complex program"
        );
    }

    #[test]
    fn test_error_handling() {
        // Test handling of invalid UPLC
        let invalid_uplc = "(program 1.0.0 (invalid node))";
        let result = parse_and_serialize(invalid_uplc);
        assert!(
            result.is_err(),
            "Serializing invalid UPLC should return an error"
        );
    }
}
