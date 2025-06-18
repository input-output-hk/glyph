const std = @import("std");

heap_ptr: [*]u8,

const Self = @This();

pub fn createTestHeap(arena: *std.heap.ArenaAllocator) !Self {
    const heapMemory = try arena.allocator().alloc(u32, 10000);
    const heapPointer: [*]u8 = @ptrCast(heapMemory);

    return Self{ .heap_ptr = heapPointer };
}

pub fn allocType(heap: *Self, comptime T: type) *T {
    if (@alignOf(T) < 4) {
        // How to print type?
        @compileError("Type {} is not aligned to 4 bytes");
    }
    // Align the heap pointer for the type
    const ptr_bytes: [*]align(@alignOf(T)) u8 = @alignCast(heap.heap_ptr);
    const ptr: *T = @ptrCast(ptr_bytes);
    heap.heap_ptr = @ptrFromInt(@intFromPtr(heap.heap_ptr) + @sizeOf(T));
    return ptr;
}

pub fn allocArray(heap: *Self, comptime T: type, len: u32) [*]T {
    if (@alignOf(T) < 4) {
        // How to print type?
        @compileError("Type {} is not aligned to 4 bytes");
    }
    const ptr_bytes: [*]align(@alignOf(T)) u8 = @alignCast(heap.heap_ptr);
    const ptr: [*]T = @ptrCast(ptr_bytes);
    heap.heap_ptr = @ptrFromInt(@intFromPtr(heap.heap_ptr) + @sizeOf(T) * len);
    return ptr;
}
