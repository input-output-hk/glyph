export fn memset(dest: *anyopaque, c: c_int, n: u32) void {
    const ptr = @as([*]u32, @ptrCast(@alignCast(dest)));
    const byte_value = @as(u32, @bitCast(c));

    var i: u32 = 0;
    while (i < n) : (i += 1) {
        ptr[i] = byte_value;
    }
}
