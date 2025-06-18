const std = @import("std");

heap_ptr: [*]u8,

const Self = @This();

pub fn createTestHeap(arena: *std.heap.ArenaAllocator) !Self {
    const heapMemory = try arena.allocator().alloc(u32, 10000);
    const heapPointer: [*]u8 = @ptrCast(heapMemory);

    return Self{ .heap_ptr = heapPointer };
}
