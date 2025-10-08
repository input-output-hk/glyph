export fn memset(dest: *anyopaque, c: c_int, n: u32) void {
    const ptr = @as([*]u32, @ptrCast(@alignCast(dest)));
    const byte_value = @as(u32, @bitCast(c));
    const n_word = (n + 3) / 4;

    var i: u32 = 0;
    while (i < n_word) : (i += 1) {
        ptr[i] = byte_value;
    }
}

export fn memcpy(dest: *anyopaque, src: *const anyopaque, n: u32) void {
    const dest_ptr = @as([*]u32, @ptrCast(@alignCast(dest)));
    const src_ptr = @as([*]const u32, @ptrCast(@alignCast(src)));
    const n_word = (n + 3) / 4;

    var i: u32 = 0;
    while (i < n_word) : (i += 1) {
        dest_ptr[i] = src_ptr[i];
    }
}

// Compiler runtime function for 64-bit unsigned division
// Required by RISC-V 32-bit targets for u64 division
export fn __udivdi3(a: u64, b: u64) u64 {
    return a / b;
}
