# UPLC to RISC-V CLI

This crate provides a command-line interface for the UPLC to RISC-V compiler.

## Installation

```bash
cargo install uplc-to-risc-cli
```

Or build from source:

```bash
cargo build --release
```

## Usage

### Compile UPLC to RISC-V

```bash
uplc-to-risc-cli compile --input input.uplc --output output.s
```

Options:
- `--mode`: Compilation mode (direct, evaluate, optimize)
- `--optimize`: Optimization level (none, default, aggressive)

### Evaluate UPLC

```bash
uplc-to-risc-cli evaluate --input input.uplc
```

## Examples

### Simple Addition

```bash
echo '(program (1 0 0) (app (app (builtin addInteger) (con integer 40)) (con integer 2)))' > add.uplc
uplc-to-risc-cli compile --input add.uplc --output add.s
```

### Factorial

```bash
uplc-to-risc-cli compile --input factorial.uplc --output factorial.s --mode optimize --optimize aggressive
```

### Evaluation

```bash
uplc-to-risc-cli evaluate --input factorial.uplc
```

## Integration with RISC-V Toolchain

The generated RISC-V assembly code can be assembled and linked using the RISC-V GNU toolchain:

```bash
riscv64-elf-gcc -march=rv32i -mabi=ilp32 -nostdlib -o program program.s
```

Or for simulation:

```bash
riscv64-elf-gcc -march=rv32i -mabi=ilp32 -nostdlib -T link.ld -o program program.s
spike pk program
``` 