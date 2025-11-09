const std = @import("std");
/// Zig version. When writing code that supports multiple versions of Zig, prefer
/// feature detection (i.e. with `@hasDecl` or `@hasField`) over version checks.
pub const zig_version = std.SemanticVersion.parse(zig_version_string) catch unreachable;
pub const zig_version_string = "0.15.0-dev.833+5f7780c53";
pub const zig_backend = std.builtin.CompilerBackend.stage2_llvm;

pub const output_mode: std.builtin.OutputMode = .Obj;
pub const link_mode: std.builtin.LinkMode = .static;
pub const unwind_tables: std.builtin.UnwindTables = .none;
pub const is_test = false;
pub const single_threaded = false;
pub const abi: std.Target.Abi = .none;
pub const cpu: std.Target.Cpu = .{
    .arch = .riscv32,
    .model = &std.Target.riscv.cpu.baseline_rv32,
    .features = std.Target.riscv.featureSet(&.{
        .@"32bit",
        .i,
        .m,
        .zaamo,
        .zalrsc,
        .zmmul,
    }),
};
pub const os: std.Target.Os = .{
    .tag = .freestanding,
    .version_range = .{ .none = {} },
};
pub const target: std.Target = .{
    .cpu = cpu,
    .os = os,
    .abi = abi,
    .ofmt = object_format,
    .dynamic_linker = .none,
};
pub const object_format: std.Target.ObjectFormat = .elf;
pub const mode: std.builtin.OptimizeMode = .ReleaseFast;
pub const link_libc = false;
pub const link_libcpp = false;
pub const have_error_return_tracing = false;
pub const valgrind_support = false;
pub const sanitize_thread = false;
pub const fuzz = false;
pub const position_independent_code = false;
pub const position_independent_executable = false;
pub const strip_debug_info = false;
pub const code_model: std.builtin.CodeModel = .default;
pub const omit_frame_pointer = false;
