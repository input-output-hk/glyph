const std = @import("std");

pub fn build(b: *std.Build) !void {
    const riscvBuildTargetQuery = try std.Build.parseTargetQuery(.{
        .arch_os_abi = "riscv32-freestanding",
        .cpu_features = "baseline+i+m-f-d-c-a-zicsr",
    });

    // Force RISC-V 32-bit freestanding target for builds
    const riscvBuildTarget = b.resolveTargetQuery(riscvBuildTargetQuery);

    const optimize = b.standardOptimizeOption(.{});

    const obj = b.addObject(.{
        .name = "runtime",
        .root_source_file = b.path("src/root.zig"),
        .target = riscvBuildTarget,
        .optimize = .ReleaseFast,
    });

    const loc = obj.getEmittedBin();

    const file = b.addInstallFile(loc, "lib/runtime.o");

    b.getInstallStep().dependOn(&file.step);

    const memsetObj = b.addObject(.{
        .name = "memset",
        .root_source_file = b.path("src/memset.zig"),
        .target = riscvBuildTarget,
        .optimize = .ReleaseFast,
    });

    const memLoc = memsetObj.getEmittedBin();

    const otherFile = b.addInstallFile(memLoc, "lib/memset.o");

    b.getInstallStep().dependOn(&otherFile.step);

    // Force RISC-V 32-bit Linux target for tests
    const riscv_test_target_query = try std.Build.parseTargetQuery(.{
        .arch_os_abi = "riscv32-linux",
        .cpu_features = "baseline+i+m-f-d-c",
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
