# Changelog

## [Unreleased] - Initial release

### Added
- Integration with the `uplc` crate for parsing UPLC code
- Implementation of RISC-V code generator
- Implementation of UPLC to RISC-V compiler
- BitVMX-specific code generation with PC tracking
- Execution trace generation for BitVMX verification
- Memory segmentation (read-only vs. read-write)
- Memory access tracking with last_step fields
- Intermediate Representation (IR) system for UPLC terms
- Support for integer constants, lambda expressions, and applications
- Support for basic integer operations as builtins
- Support for case expressions in the IR and code generation
- Hash chain implementation for BitVMX verification
- Dispute resolution utilities for BitVMX verification
- Command-line interface for the compiler
- Support for configurable optimization levels
- Basic documentation for the API
- Integration tests for the compiler
- Unit tests for core components
- Dual-approach strategy with complementary UPLC-to-RISC-V and UPLC-to-LLVM implementations
- Error handling throughout the codebase
- BitVMXCodeGenerator with detailed error reporting
- Memory access validation with context-aware error messages