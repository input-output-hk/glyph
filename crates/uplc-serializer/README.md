# UPLC Serializer

A Rust library for serializing Untyped Plutus Core (UPLC) programs into a compact binary format suitable for execution on a RISC-V CEK machine.

## Overview

This crate provides functionality for converting UPLC programs from their AST representation (using the `uplc` crate) into a binary format that can be efficiently executed by our RISC-V CEK machine implementation. The serialization process follows a defined memory layout and binary format as described in the serialization plan.

## Features

- Serializes all UPLC term types (variables, lambdas, applications, etc.)
- Handles various constant types (integers, bytestrings, strings, booleans, etc.)
- Supports term deduplication for reduced binary size
- Memory region separation for different types of data
- Comprehensive error handling

## Binary Format Specification

Each UPLC program is serialized according to the following format:

### Program Structure

1. **Program Header (12 bytes):**
   - Magic bytes (4 bytes): `'UPLC'`
   - Version (3 bytes): `(major, minor, patch)`
   - Reserved (1 byte): `0x00`
   - Root term address (4 bytes): Pointer to the root term

2. **Memory Regions:**
   - Term Region (0x00010000 - 0x0001FFFF): Contains serialized terms
   - Integer Pool (0x00020000 - 0x0002FFFF): Integer constants
   - ByteString Pool (0x00030000 - 0x0003FFFF): ByteString constants
   - String Pool (0x00040000 - 0x0004FFFF): String constants
   - Complex Data Pool (0x00050000 - 0x0005FFFF): Other constants

### Term Format

Each term starts with a 1-byte tag identifying its type:

- `0x00`: Variable
  - DeBruijn index (4 bytes)
- `0x01`: Lambda
  - Body reference (4 bytes)
- `0x02`: Apply
  - Function reference (4 bytes)
  - Argument reference (4 bytes)
- `0x03`: Force
  - Term reference (4 bytes)
- `0x04`: Delay
  - Term reference (4 bytes)
- `0x05`: Constant
  - Constant reference (4 bytes)
- `0x06`: Builtin
  - Builtin function ID (1 byte)
- `0x07`: Error
  - No additional data
- `0x08`: Constructor
  - Tag (2 bytes)
  - Field count (2 bytes)
  - Fields reference (4 bytes)
- `0x09`: Case
  - Match term reference (4 bytes)
  - Branch count (2 bytes)
  - Branches reference (4 bytes)

### Constants Format

Constants are stored in constant pools with the following format:

- Integer Constants (`0x00`):
  - Size indicator (1 byte): 0x01 (1 byte), 0x02 (2 bytes), 0x04 (4 bytes), 0x08 (8 bytes), or 0xFF (BigInt)
  - For fixed-size integers: value in little-endian format
  - For BigInt: sign byte (0x00 for positive, 0x01 for negative), length (4 bytes), magnitude bytes

- ByteString Constants (`0x01`):
  - Length (4 bytes)
  - Raw byte data

- String Constants (`0x02`):
  - Length (4 bytes)
  - UTF-8 encoded string data

- Unit Constant (`0x03`):
  - No additional data

- Boolean Constants (`0x04`):
  - Boolean value (0x00 for false, 0x01 for true)

- Data Constants (`0x05`):
  - Data variant tag (1 byte)
  - Variant-specific encoding

## Usage

### Basic Usage

```rust
use uplc_serializer::parse_and_serialize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // UPLC program as text
    let uplc_text = r#"
    (program 1.0.0
      [
        (lam x
          [
            (builtin addInteger)
            x
            (con integer 1)
          ]
        )
        (con integer 41)
      ]
    )
    "#;

    // Parse and serialize
    let binary = parse_and_serialize(uplc_text)?;
    
    // Use the binary data (e.g., write to file, send to interpreter)
    std::fs::write("program.bin", binary)?;
    
    Ok(())
}
```

### Serializing from UPLC AST

```rust
use uplc_serializer::serialize_program;
use uplc::ast::Program;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse UPLC program
    let program_text = "(program 1.0.0 [(con integer 42)])";
    let program = uplc::parser::program(program_text)?.to_debruijn()?;
    
    // Serialize directly from Program AST
    let binary = serialize_program(&program)?;
    
    // Use the binary data
    
    Ok(())
}
```

## Examples

Check out the examples directory for more detailed usage examples:

- `simple_serialization.rs`: Basic serialization of a simple UPLC program
- `complex_serialization.rs`: Advanced serialization with various term types

## Development

### Running the Examples

```
cargo run --example simple_serialization
cargo run --example complex_serialization
```

### Running Tests

```
cargo test
```

## Future Improvements

- Implementation of deserialization functionality
- Better handling of complex Plutus data structures
- Optimizations for term sharing and binary size reduction
- Integration with the RISC-V CEK machine

## License

This project is licensed under the MIT OR Apache-2.0 License. 