//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const testing = std.testing;

pub export fn init() u32 {
    const initial_term_addr: u32 = 0x90000000;
    const ptr: *const Tag = @ptrFromInt(initial_term_addr);

    const x = @intFromPtr(ptr) + 4;

    return x;
}

test "basic add functionality" {
    const memory: []const u32 = &.{ 1, 0, 1 };
    const ptr: usize = @intFromPtr(&memory);
    const ptr2: **const Tag = @ptrFromInt(ptr);

    const nextTag = switch (ptr2.*.*) {
        .delay => ptr2.*.tagBody(),
        else => @panic("NOOOO"),
    };

    const debruijnIndex = switch (nextTag.*) {
        .@"var" => nextTag.debruijnIndex(),
        else => @panic("NOOOO"),
    };

    try testing.expect(debruijnIndex == 1);
}

const Tag = enum(u32) {
    @"var",
    delay,
    lambda,
    apply,
    constant,
    force,
    builtin,
    constr,
    case,

    fn debruijnIndex(ptr: *const Tag) u32 {
        const dbIndex: *u32 = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        return dbIndex.*;
    }

    fn tagBody(ptr: *const Tag) *const Tag {
        const nextTag: *const Tag = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        return nextTag;
    }

    fn appliedTags(ptr: *const Tag) .{ .function = *const Tag, .argument = *const Tag } {
        const argTag: **const Tag = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        const funcTag: *const Tag = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32) * 2);

        return .{
            .function = funcTag,
            .argument = *argTag,
        };
    }
};
