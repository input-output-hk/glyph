# UPLC to RISC-V Compiler

This project provides a compiler from Untyped Plutus Core (UPLC) to RISC-V assembly code.

## Crates

The project is organized into several crates:

- `uplc-to-risc`: The main compiler library that orchestrates the compilation process.
- `uplc-parser`: A parser for Untyped Plutus Core.
- `risc-v-gen`: A code generator for RISC-V assembly.
- `uplc-to-risc-cli`: A command-line interface for the compiler.

## Building

To build the project, run:

```bash
cd crates
cargo build
```

## Usage

To compile a UPLC file to RISC-V assembly:

```bash
cd crates
cargo run --bin uplc-to-risc -- compile -i input.uplc -o output.s
```

Or, using the installed binary:

```bash
uplc-to-risc compile -i input.uplc -o output.s
```

You can also pipe UPLC code to the compiler:

```bash
cat input.uplc | uplc-to-risc compile > output.s
```