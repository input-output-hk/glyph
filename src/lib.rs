pub mod cek;
mod emulator;
mod serializer;

pub use cek::Cek;
pub use serializer::*;

// CEK Semantics written in zig
pub const RUNTIMEFUNCTION: &[u8] = include_bytes!("../runtime/zig-out/lib/runtimeFunction.o");
pub const RUNTIME: &[u8] = include_bytes!("../runtime/zig-out/lib/runtime.o");

// Some weird thing cause idk
pub const MEMSET: &[u8] = include_bytes!("../runtime/zig-out/lib/memset.o");

// Templated linker script with params
pub const LINKER_SCRIPT: &str = include_str!("../linker/link.ld");
