# Glyph: A UPLC to RISC-V Compilation Pipeline

## Project Structure

The project is organized into a CLI tool with a built-in serializer for UPLC, as well as a runtime written in Zig.

- `bin`: A CLI tool to interact with Glyph.
- `serializer`: A parser and serializer for UPLC code.
- `runtime`: A CEK Machine with multiple entry-points written in Zig and compiled to RISC-V.

## Installing Glyph
Installing Glyph from a prebuilt binary can be done with the following command(s). Replace the version with the most up-to-date release.

Using Shell Script:
```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/input-output-hk/glyph/releases/download/v0.1.7/glyph-installer.sh | sh
```

Using Powershell Script:
```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/input-output-hk/glyph/releases/download/v0.1.7/glyph-installer.ps1 | iex"
```

## Building

### Requirements
To build the project, you need to have Rust and Cargo installed, as well as a risc-v toolchain.

On Debian based systems:
```bash
sudo apt-get install -y gcc-riscv64-unknown-elf
```

On MacOS:
```bash
brew install riscv64-elf-gcc
```

### Build Command(s)
Then, run:

```bash
cargo build
cargo test
```

## Usage

### Command-Line Interface

The CLI provides several commands for working with UPLC code:

#### Compile UPLC to RISC-V

```bash
uplc-to-risc-cli compile --input input.uplc --output output.s
```

## Features

- Parsing and serialization of UPLC code
- Evaluation of serialized UPLC terms using a CEK machine

## Production Readiness

This project is currently in active development and requires some key improvements before it can be considered production-ready:
