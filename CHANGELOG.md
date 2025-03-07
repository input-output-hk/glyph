# Changelog

## [Unreleased] - Initial release

### Added

#### Core Architecture
- Integration with the `uplc` crate for parsing UPLC code
- Modular architecture with separate crates (`uplc-to-riscv`, `risc-v-gen`, `bitvm-common`)

#### BitVMX Compatibility
- BitVMX-specific code generation with PC tracking
- Memory segmentation (read-only vs. read-write)
- Execution trace generation for BitVMX verification
- Hash chain implementation for verification proofs
- Dispute resolution utilities for BitVMX verification
- BitVMX-compatible memory alignment and validation
- Direct integration with BitVMX-CPU emulator for RISC-V code validation

#### Compilation Pipeline
- Intermediate Representation (IR) system for UPLC terms
- Conversion from UPLC terms to IR instructions
- IR to RISC-V compilation

#### UPLC Term Support
- Integer constants and operations (add, subtract, multiply, compare)
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
- Command-line interface with BitVMX-specific options
- Support for configurable optimization levels
- Trace generation with CSV output
- Hash chain file generation for verification
- Comprehensive unit tests for core components
- Integration tests for end-to-end compilation
- BitVMX compatibility tests
- UPLC conformance test infrastructure

#### Documentation
- API documentation for all public interfaces
- Example programs demonstrating BitVMX verification
- Architecture documentation to aid new contributors
