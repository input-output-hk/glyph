const std = @import("std");
const testing = std.testing;
const expr = @import("expr.zig");
const Term = expr.Term;
const cek = @import("cek.zig");
const Heap = @import("Heap.zig");
const Env = cek.Env;

pub export fn init() u32 {
    const initial_term_addr: u32 = 0x90000000;
    const ptr: *const Term = @ptrFromInt(initial_term_addr);
    const x = @intFromPtr(ptr) + 4;

    return x;
}

test "basic functionality" {
    const memory: []const u32 = &.{ 1, 0, 1 };
    const ptr: *const Term = @ptrCast(memory);

    const nextTerm = switch (ptr.*) {
        .delay => ptr.termBody(),
        else => @panic("NOOOO"),
    };

    const debruijnIndex = switch (nextTerm.*) {
        .tvar => nextTerm.debruijnIndex(),
        else => @panic("NOOOO"),
    };

    try testing.expect(debruijnIndex == 1);
}

test "apply functionality" {
    const argument: []const u32 = &.{ 1, 0, 2 };
    const argPointer: *const u32 = @ptrCast(argument);
    const thing: u32 = @truncate(@intFromPtr(argPointer));
    const function: []const u32 = &.{ 3, thing, 2, 0, 1 };
    const ptr: *const Term = @ptrCast(function);

    const applyStruct = switch (ptr.*) {
        .apply => ptr.appliedTerms(),
        else => @panic("NOOOOOOOOOO"),
    };

    const funcVar = switch (applyStruct.function.*) {
        .lambda => applyStruct.function.termBody().debruijnIndex(),
        else => @panic("NOOOOOOOOOO"),
    };

    const argVar = switch (applyStruct.argument.*) {
        .delay => applyStruct.argument.termBody().debruijnIndex(),
        else => @panic("NOOOOOOOOOO"),
    };

    try testing.expect(funcVar == 1);
    try testing.expect(argVar == 2);
}

test "constr functionality" {
    const field1: []const u32 = &.{ 1, 0, 5 };
    const field1Pointer: *const u32 = @ptrCast(field1);
    const field2: []const u32 = &.{ 2, 0, 2 };
    const field2Pointer: *const u32 = @ptrCast(field2);
    const constr: []const u32 = &.{
        8,
        55,
        2,
        @truncate(@intFromPtr(field1Pointer)),
        @truncate(@intFromPtr(field2Pointer)),
    };
    const ptr: *const Term = @ptrCast(constr);

    const constrStruct = switch (ptr.*) {
        .constr => ptr.constrValues(),
        else => @panic("NO"),
    };

    const field1Index = switch (constrStruct.fields.list[0].*) {
        .delay => constrStruct.fields.list[0].termBody().debruijnIndex(),
        else => @panic("OOOO"),
    };

    const field2Index = switch (constrStruct.fields.list[1].*) {
        .lambda => constrStruct.fields.list[1].termBody().debruijnIndex(),
        else => @panic("NOOOOOOOOOO"),
    };

    try testing.expect(constrStruct.tag == 55);
    try testing.expect(constrStruct.fields.length == 2);
    try testing.expect(field1Index == 5);
    try testing.expect(field2Index == 2);
}

test "case functionality" {
    const branch1: []const u32 = &.{ 1, 0, 2 };
    const branch2: []const u32 = &.{ 2, 0, 1 };
    const constr: []const u32 = &.{6};
    const branch1Pointer: *const u32 = @ptrCast(branch1);
    const branch2Pointer: *const u32 = @ptrCast(branch2);
    const constrPointer: *const u32 = @ptrCast(constr);

    const case: []const u32 = &.{
        9,
        @truncate(@intFromPtr(constrPointer)),
        2,
        @truncate(@intFromPtr(branch1Pointer)),
        @truncate(@intFromPtr(branch2Pointer)),
    };
    const ptr: *const Term = @ptrCast(case);

    std.debug.print("HERE\n", .{});
    const caseValues = switch (ptr.*) {
        .case => ptr.caseValues(),
        else => @panic("NOOOOOOOOOO"),
    };

    switch (caseValues.constr.*) {
        .terror => {},
        else => @panic("NOOOOOOOOOO"),
    }

    const branch1Var = switch (caseValues.branches.list[0].*) {
        .delay => caseValues.branches.list[0].termBody().debruijnIndex(),
        else => @panic("NOOOOOOOOOO"),
    };

    const branch2Var = switch (caseValues.branches.list[1].*) {
        .lambda => caseValues.branches.list[1].termBody().debruijnIndex(),
        else => @panic("NOOOOOOOOOO"),
    };

    try testing.expect(caseValues.branches.length == 2);
    try testing.expect(branch1Var == 2);
    try testing.expect(branch2Var == 1);
}

test "heap functionality" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);

    var env: Env = Env.init(&heap);

    const argument: []const u32 = &.{ 1, 0, 2 };
    const argPointer: *const u32 = @ptrCast(argument);
    const thing: u32 = @truncate(@intFromPtr(argPointer));
    const function: []const u32 = &.{ 3, thing, 2, 0, 1 };
    const ptr: *const Term = @ptrCast(function);

    const applyStruct = switch (ptr.*) {
        .apply => ptr.appliedTerms(),
        else => @panic("NOO"),
    };

    const funcVar = switch (applyStruct.function.*) {
        .lambda => applyStruct.function.termBody().debruijnIndex(),
        else => @panic("N"),
    };

    const argVar = switch (applyStruct.argument.*) {
        .delay => env.createDelay(applyStruct.argument.termBody()),
        else => @panic("NOOOO"),
    };

    const argIndex = switch (argVar.*) {
        .delay => |other| other.body.debruijnIndex(),
        else => @panic("NO"),
    };

    try testing.expect(funcVar == 1);
    try testing.expect(argIndex == 2);
}
