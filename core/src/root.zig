//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const testing = std.testing;

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

const Term = enum(u32) {
    tvar,
    delay,
    lambda,
    apply,
    constant,
    force,
    builtin,
    constr,
    case,

    fn debruijnIndex(ptr: *const Term) u32 {
        const dbIndex: *u32 = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        return dbIndex.*;
    }

    fn termBody(ptr: *const Term) *const Term {
        const nextTerm: *const Term = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        return nextTerm;
    }

    fn appliedTerms(ptr: *const Term) Apply {
        const argTerm: **const Term = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        const funcTerm: *const Term = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32) * 2);

        return .{
            .function = funcTerm,
            .argument = argTerm.*,
        };
    }

    fn defaultFunction(ptr: *const Term) DefaultFunction {
        const func: *const DefaultFunction = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        return func.*;
    }

    fn constrValues(ptr: *const Term) .{ .tag = u32, .fields = *const TermList } {
        const tag: *const u32 = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        const fields: *const TermList = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32) * 2);

        return .{ .tag = tag.*, .fields = fields };
    }

    fn caseValues(ptr: *const Term) .{ .constr = *const Term, .branches = *const TermList } {
        const constr: *const Term = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        const branches: *const TermList = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32) * 2);

        return .{ .constr = constr, .branches = branches };
    }
};

const Apply = struct { function: *const Term, argument: *const Term };

const TermList = extern struct { length: u32, list: []*Term };

const DefaultFunction = enum(u32) {
    addInteger,
    subtractInteger,
};
