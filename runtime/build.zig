const std = @import("std");

pub fn build(b: *std.Build) !void {
    const riscv_build_target_query = try std.Build.parseTargetQuery(.{
        .arch_os_abi = "riscv32-freestanding",
        .cpu_features = "baseline+i+m",
    });

    // Force RISC-V 32-bit freestanding target for builds
    const riscv_build_target = b.resolveTargetQuery(riscv_build_target_query);

    const optimize = b.standardOptimizeOption(.{});

    const obj = b.addObject(.{
        .name = "runtime",
        .root_source_file = b.path("src/root.zig"),
        .target = riscv_build_target,
        .optimize = .ReleaseFast,
    });

    const loc = obj.getEmittedBin();

    const file = b.addInstallFile(loc, "lib/runtime.o");

    b.getInstallStep().dependOn(&file.step);

    // Force RISC-V 32-bit Linux target for tests
    const riscv_test_target_query = try std.Build.parseTargetQuery(.{
        .arch_os_abi = "riscv32-linux",
        .cpu_features = "baseline+i+m",
    });

    const riscv_test_target = b.resolveTargetQuery(riscv_test_target_query);

    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = riscv_test_target,
        .optimize = optimize,
    });

    const qemu_run = b.addSystemCommand(&.{"qemu-riscv32"});
    qemu_run.addArtifactArg(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&qemu_run.step);
    qemu_run.step.dependOn(&lib_unit_tests.step);
}
