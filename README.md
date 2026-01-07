# Glyph: UPLC to RISC-V Compilation Pipeline

Glyph compiles Untyped Plutus Core (UPLC) to RISC-V and ships a CEK runtime written in Zig.

## Project Structure

The project is organized into a CLI tool with a built-in serializer for UPLC, as well as a runtime written in Zig.

- `src/bin`: A CLI tool to interact with Glyph.
- `src/serializer.rs`: A parser and serializer for UPLC code.
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
To build the project, you need Rust and Cargo, Zig, and a RISC-V toolchain.

On Debian based systems:
```bash
sudo apt-get install -y gcc-riscv64-unknown-elf
```

On MacOS:
```bash
brew install riscv64-elf-gcc
```

Zig should be available on your PATH (see https://ziglang.org/download/).

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
glyph compile --encoding text --file program.uplc
```

This produces `program.elf` in the current working directory.

#### Run a RISC-V program with optional UPLC input

```bash
glyph run --program-file program.elf --encoding text --file input.uplc
```

If your input program does not require a script argument, use `--no-input` with `compile`
and omit `--file` when running.

#### Input encodings

- `--encoding text` parses human-readable UPLC text.
- `--encoding cbor` expects CBOR-encoded flat bytes (default).
- `--encoding flat` expects flat-encoded bytes.
- `--hex` decodes hex input for `cbor` or `flat` encodings.

#### Build from `plutus.json`

The `build` subcommand compiles validator bundles from `plutus.json`.

```bash
glyph build --file plutus.json --validator example.main.else --output program.elf
```

When the bundle only has a single validator, `--validator` can be omitted.

## Features

- Parsing and serialization of UPLC code
- Evaluation of serialized UPLC terms using a CEK machine

## Status

This project is in active development and should be considered experimental until the first stable release.
