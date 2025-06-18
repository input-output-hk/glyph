const std = @import("std");
const testing = std.testing;
const expr = @import("expr.zig");
const Term = expr.Term;
const Constant = expr.Constant;
const cek = @import("cek.zig");
const Heap = @import("Heap.zig");
const Env = cek.Env;
const allocType = cek.allocType;

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

test "cek machine basic functionality" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var env = Env.init(&heap);

    // Create a constant value
    const constant = allocType(env.heap, Constant);
    constant.* = Constant{ .integer = 42 };
    const value = env.mkValue(.{ .constant = constant });
    const extended_env = env.extend(value);

    // Create a term that references the value
    const term: []const u32 = &.{ 0, 1 }; // var with debruijn index 1
    const ptr: *const Term = @ptrCast(term);

    const result = extended_env.run(ptr);
    try testing.expect(result.* == .constant);
    try testing.expect(result.constant.*.integer == 42);
}

test "apply lambda var constant" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var env = Env.init(&heap);

    // Allocate memory for our terms
    const memory = try std.heap.page_allocator.alignedAlloc(u32, @enumFromInt(2), 8);
    defer std.heap.page_allocator.free(memory);

    // Write constant term [4, 42] at offset 0
    memory[0] = 4; // constant tag
    memory[1] = 42; // value

    // Write lambda term [2, 1] at offset 2
    memory[2] = 2; // lambda tag
    memory[3] = 1; // debruijn index

    // Write apply term [3, ptr_to_lambda, ptr_to_constant] at offset 4
    memory[4] = 3; // apply tag
    memory[5] = 2; // pointer to lambda term (offset 2)
    memory[6] = 0; // pointer to constant term (offset 0)
    memory[7] = 0; // padding

    // Run the CEK machine
    const ptr: *const Term = @ptrCast(memory.ptr);
    const result = env.run(ptr);

    try testing.expect(result.* == .constant);
    try testing.expect(result.constant.*.integer == 42);
}

test "case with constructor" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var env = Env.init(&heap);

    // Allocate memory for our terms
    const memory = try std.heap.page_allocator.alignedAlloc(u32, @enumFromInt(2), 12);
    defer std.heap.page_allocator.free(memory);

    // Write constant terms for constructor fields
    memory[0] = 4; // constant tag
    memory[1] = 99; // first field value
    memory[2] = 4; // constant tag
    memory[3] = 13; // second field value

    // Write constructor term [8, 1, 2, ptr_to_field1, ptr_to_field2] at offset 4
    memory[4] = 8; // constr tag
    memory[5] = 1; // tag value
    memory[6] = 2; // fields length
    memory[7] = 0; // pointer to first field (offset 0)
    memory[8] = 2; // pointer to second field (offset 2)

    // Write case term [9, ptr_to_constr, 2, ptr_to_branch1, ptr_to_branch2] at offset 9
    memory[9] = 9; // case tag
    memory[10] = 4; // pointer to constructor (offset 4)
    memory[11] = 2; // branches length

    const ptr: *const Term = @ptrCast(memory.ptr);
    const result = env.run(ptr);

    try testing.expect(result.* == .constant);
    try testing.expect(result.constant.*.integer == 13);
}

test "force delay error" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var env = Env.init(&heap);

    // Allocate memory for our terms
    const memory = try std.heap.page_allocator.alignedAlloc(u32, @enumFromInt(2), 8);
    defer std.heap.page_allocator.free(memory);

    // Write error term [6] at offset 0
    memory[0] = 6; // error tag

    // Write delay term [5, ptr_to_error] at offset 1
    memory[1] = 5; // delay tag
    memory[2] = 0; // pointer to error term (offset 0)

    // Write force term [4, ptr_to_delay] at offset 3
    memory[3] = 4; // force tag
    memory[4] = 1; // pointer to delay term (offset 1)

    memory[5] = 0; // padding
    memory[6] = 0; // padding
    memory[7] = 0; // padding

    const ptr: *const Term = @ptrCast(memory.ptr);

    // This should panic with "evaluation failure"
    _ = env.run(ptr);
    @panic("Should have failed with evaluation failure");
}
