const std = @import("std");

pub fn build(b: *std.Build) !void {
    const riscvBuildTargetQuery = try std.Build.parseTargetQuery(.{
        .arch_os_abi = "riscv32-freestanding",
        .cpu_features = "baseline+i+m-f-d-c-a-zicsr",
    });

    // Force RISC-V 32-bit freestanding target for builds
    const riscvBuildTarget = b.resolveTargetQuery(riscvBuildTargetQuery);

    const optimize = b.standardOptimizeOption(.{});

    // NEW: Get the blst dependency (fetched from build.zig.zon)
    const blst_dep = b.dependency("blst", .{
        .target = riscvBuildTarget,
        .optimize = .ReleaseFast,
    });

    // NEW: Build blst as a static library in portable mode
    const blst_lib = b.addStaticLibrary(.{
        .name = "blst",
        .target = riscvBuildTarget,
        .optimize = .ReleaseFast,
    });

    // NEW: Add include paths from the dependency
    blst_lib.addIncludePath(blst_dep.path("bindings"));
    blst_lib.addIncludePath(blst_dep.path("src"));

    // NEW: Compile blst's main source file with portable flag (no assembly)
    blst_lib.addCSourceFile(.{
        .file = blst_dep.path("src/server.c"),
        .flags = &[_][]const u8{
            "-D__BLST_PORTABLE__",
            "-O3", // Extra optimization (optional)
            "-D__FREESTANDING__",
        },
    });

    const obj = b.addObject(.{
        .name = "runtime",
        .root_source_file = b.path("src/runtimeValidator.zig"),
        .target = riscvBuildTarget,
        .optimize = .ReleaseFast,
    });

    // NEW: Link blst to the runtime object
    obj.linkLibrary(blst_lib);

    const loc = obj.getEmittedBin();

    const file = b.addInstallFile(loc, "lib/runtime.o");

    b.getInstallStep().dependOn(&file.step);

    const objRuntimeFunction = b.addObject(.{
        .name = "runtimeFunction",
        .root_source_file = b.path("src/runtimeFunction.zig"),
        .target = riscvBuildTarget,
        .optimize = .ReleaseFast,
    });

    const locrf = objRuntimeFunction.getEmittedBin();

    const filerf = b.addInstallFile(locrf, "lib/runtimeFunction.o");

    b.getInstallStep().dependOn(&filerf.step);

    const memsetObj = b.addObject(.{
        .name = "memset",
        .root_source_file = b.path("src/memset.zig"),
        .target = riscvBuildTarget,
        .optimize = .ReleaseFast,
    });

    // NEW: Link blst to memsetObj (if it uses BLS functions; otherwise optional)
    memsetObj.linkLibrary(blst_lib);

    const memLoc = memsetObj.getEmittedBin();

    const otherFile = b.addInstallFile(memLoc, "lib/memset.o");

    b.getInstallStep().dependOn(&otherFile.step);

    // Force RISC-V 32-bit Linux target for tests
    const riscv_test_target_query = try std.Build.parseTargetQuery(.{
        .arch_os_abi = "riscv32-linux",
        .cpu_features = "baseline+i+m-f-d-c",
    });

    const riscv_test_target = b.resolveTargetQuery(riscv_test_target_query);

    // NEW: Get the blst dependency (fetched from build.zig.zon)
    const blst_test_dep = b.dependency("blst", .{
        .target = riscv_test_target,
        .optimize = .ReleaseFast,
    });

    // Build blst for test target
    const blst_test_lib = b.addStaticLibrary(.{
        .name = "blst",
        .target = riscv_test_target,
        .link_libc = true,
        .optimize = .ReleaseFast,
    });

    blst_test_lib.addIncludePath(blst_test_dep.path("bindings"));
    blst_test_lib.addIncludePath(blst_test_dep.path("src"));

    blst_test_lib.addCSourceFile(.{
        .file = blst_test_dep.path("src/server.c"),
        .flags = &[_][]const u8{
            "-D__BLST_PORTABLE__",
            "-O3",
        },
    });

    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/runtimeFunction.zig"),
        .target = riscv_test_target,
        .optimize = optimize,
    });

    lib_unit_tests.addIncludePath(blst_test_dep.path("bindings"));
    lib_unit_tests.addIncludePath(blst_test_dep.path("src"));

    // lib_unit_tests.linkLibC();
    // NEW: Link blst to unit tests
    lib_unit_tests.linkLibrary(blst_test_lib);

    const qemu_run = b.addSystemCommand(&.{"qemu-riscv32"});
    qemu_run.addArtifactArg(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&qemu_run.step);
    qemu_run.step.dependOn(&lib_unit_tests.step);
}
