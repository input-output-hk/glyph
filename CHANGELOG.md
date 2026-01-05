# Changelog

## [next] - YYYY-MM-DD

### Added

### Changed

### Removed

## [0.1.7] - Initial release

### Added

#### Core Architecture
- Integration with the `uplc` crate for parsing UPLC code
- CEK Machine written in Zig and compiled to RISC-V
- Serializer written in Rust; Serializes canonical UPLC into a format compatible with the RISC-V CEK

#### UPLC Term Support
- Integer constants and operations
- Lambda expressions and applications
- Builtin functions (core set of arithmetic and logic)
- Delay/Force term types with proper handling
- ByteString operations
- Error term type
- Constructor term type
- Case expressions
- Pair operations (mkPairData, fstPair, sndPair)
- List operations
- If-then-else conditional logic

#### Tools and Testing Infrastructure
- UPLC conformance test infrastructure
