# UPLC to RISC-V Compiler

This crate provides the main functionality for compiling Untyped Plutus Core (UPLC) to RISC-V assembly code.

## Features

- Compile UPLC code to RISC-V assembly
- Evaluate UPLC terms using a CEK machine
- Optimize intermediate code
- Multiple compilation modes
- Configurable optimization levels

## Usage

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
let risc_v_code = compiler.compile(uplc_code).unwrap();
println!("{}", risc_v_code);

// Evaluate UPLC code
let result = compiler.evaluate(uplc_code).unwrap();
println!("Result: {}", result);
```

## Compilation Modes

The compiler supports three compilation modes:

- `Direct`: Compile UPLC code directly to RISC-V assembly without evaluation or optimization.
- `Evaluate`: Evaluate UPLC code using the CEK machine, then compile the result to RISC-V assembly.
- `Optimize`: Compile UPLC code to intermediate representation (IR), optimize the IR, then compile to RISC-V assembly.

## Optimization Levels

The compiler supports three optimization levels:

- `None`: No optimization.
- `Default`: Basic optimizations like constant folding and dead code elimination.
- `Aggressive`: More aggressive optimizations that may take longer to compile but produce more efficient code.

## Intermediate Representation (IR)

The compiler uses an intermediate representation (IR) to facilitate optimization and code generation. The IR consists of simple instructions that are easier to optimize and translate to RISC-V assembly.

## CEK Machine

The compiler includes a CEK (Control, Environment, Kontinuation) machine for evaluating UPLC terms. The CEK machine is a simple abstract machine that can evaluate lambda calculus terms efficiently.

## Error Handling

The compiler provides detailed error messages for various error conditions:

```rust
use uplc_to_risc::{Compiler, CompilationError};

let compiler = Compiler::new();
let invalid_uplc = "(program (1 0 0) (invalid))";
let result = compiler.compile(invalid_uplc);

match result {
    Ok(_) => println!("Compilation succeeded"),
    Err(CompilationError::Parse(err)) => println!("Parse error: {}", err),
    Err(CompilationError::CodeGen(err)) => println!("Code generation error: {}", err),
    Err(CompilationError::UnsupportedFeature(feature)) => println!("Unsupported feature: {}", feature),
    Err(CompilationError::InvalidInput(msg)) => println!("Invalid input: {}", msg),
    Err(CompilationError::Evaluation(msg)) => println!("Evaluation error: {}", msg),
}
``` 