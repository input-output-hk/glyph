const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Force RISC-V 32-bit Linux target for tests
    const riscv_build_target = b.resolveTargetQuery(.{
        .cpu_arch = .riscv32,
        .os_tag = .freestanding,
    });

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const obj = b.addObject(.{
        .name = "core",
        .root_source_file = b.path("src/root.zig"),
        .target = riscv_build_target,
        .optimize = optimize,
    });

    const loc = obj.getEmittedBin();

    const file = b.addInstallFile(loc, "lib/runtime.o");

    b.getInstallStep().dependOn(&file.step);

    // Force RISC-V 32-bit Linux target for tests
    const riscv_test_target = b.resolveTargetQuery(.{
        .cpu_arch = .riscv32,
        .os_tag = .linux,
    });

    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = riscv_test_target,
        .optimize = optimize,
    });

    const qemu_run = b.addSystemCommand(&.{"qemu-riscv32"});
    qemu_run.addArtifactArg(lib_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&qemu_run.step);
    qemu_run.step.dependOn(&lib_unit_tests.step);
}
