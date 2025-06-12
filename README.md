# UPLC to RISC-V Compiler

> [!WARNING]
>
> This project is currently under active development and is NOT
> ready for production use. It is in an experimental state with
> many features still being implemented and tested.

test

A compiler that translates Untyped Plutus Core (UPLC) to RISC-V assembly code.

## Project Structure

The project is organized into multiple crates:

- `uplc-parser`: A parser for UPLC code
- `risc-v-gen`: A RISC-V assembly code generator
- `uplc-to-risc`: The main compiler library
- `cli`: A command-line interface for the compiler

## Building

To build the project, you need to have Rust and Cargo installed. Then, run:

```bash
cargo build --release
```

## Usage

### Command-Line Interface

The CLI provides several commands for working with UPLC code:

#### Compile UPLC to RISC-V

```bash
uplc-to-risc-cli compile --input input.uplc --output output.s
```

Options:
- `--mode`: Compilation mode (direct, evaluate, optimize)
- `--optimize`: Optimization level (none, default, aggressive)

### Library Usage

```rust
use uplc_to_risc::{Compiler, CompilationMode, OptimizationLevel};

// Create a compiler with default settings
let compiler = Compiler::new();

// Or with custom settings
let compiler = Compiler::new()
    .with_mode(CompilationMode::Optimize)
    .with_optimization_level(OptimizationLevel::Aggressive);

// Compile UPLC code to RISC-V assembly
let uplc_code = "(program (1 0 0) (con integer 42))";
let risc_v_code = compiler.compile(uplc_code)?;

// Evaluate UPLC code
let result = compiler.evaluate(uplc_code)?;
```

## Features

- Parsing of UPLC code
- Compilation to RISC-V assembly
- Evaluation of UPLC terms using a CEK machine
- Optimization of intermediate code
- Support for various RISC-V instructions and directives

## Production Readiness

This project is currently in active development and requires several key improvements before it can be considered production-ready:

### Current Progress

- ✅ Successfully integrated the `uplc` crate as a dependency
- ✅ Created a modular architecture with separate crates
- ✅ Implemented BitVMX-specific code generation with PC tracking
- ✅ Added support for integer constants, lambda expressions, applications, and basic builtins
- ✅ Implemented Delay/Force term types and ByteString operations
- ✅ Added support for case expressions in the IR and code generation
- ✅ Improved error handling and documentation

### Remaining Work for Production Readiness

1. **Complete UPLC Term Support**
   - Implement remaining UPLC term types (Error handling, Constructors)
   - Add support for all required builtin functions (Boolean, List, Pair, Cryptographic operations)

2. **BitVMX Compatibility**
   - Enhance memory segmentation (read-only vs. read-write)
   - Complete execution trace generation for verification
   - Implement hash chain support and dispute resolution utilities

3. **Testing and Documentation**
   - Create a comprehensive test suite covering all components
   - Develop detailed documentation for API usage and BitVMX integration
   - Add examples demonstrating smart contracts and validation logic

4. **Performance and Optimization**
   - Implement BitVMX-specific optimizations
   - Optimize memory usage and reduce code size
   - Benchmark against other UPLC compilers

5. **Ecosystem Integration**
   - Create seamless integration with Aiken compiler
   - Ensure compatibility with Cardano's smart contract platform
   - Complete integration with BitVMX runtime
