const std = @import("std");
const testing = std.testing;
const expr = @import("expr.zig");
const Term = expr.Term;
const Constant = expr.Constant;
const cek = @import("cek.zig");
const Heap = @import("Heap.zig");
const Env = cek.Env;
const Frames = cek.Frames;
const allocType = cek.allocType;
const Machine = cek.Machine;
const utils = @import("utils.zig");

pub export fn init2() void {
    const initial_term_addr: u32 = 0x90000000;
    const initial_term: *const Term = @ptrFromInt(initial_term_addr);
    const heap_addr: u32 = 0xC0000000;
    const frame_addr: u32 = 0xD0000000;

    var frames = Frames.createFrames(frame_addr);
    var heap = Heap.createHeap(heap_addr);

    var runtime = Machine.init(&heap, &frames);

    runtime.runValidator(initial_term);

    utils.exit(0);
}
