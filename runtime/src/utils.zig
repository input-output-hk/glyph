pub fn exit(exitVal: u32) noreturn {
    asm volatile (
        \\li a7, 93
        \\ecall
        :
        : [exitVal] "{a0}" (exitVal),
    );
    unreachable;
}

fn padTo4(comptime str: []const u8) [((str.len + 3) / 4) * 4]u8 {
    const padded_len = ((str.len + 3) / 4) * 4;
    var result: [padded_len]u8 = [_]u8{0} ** padded_len;

    for (str, 0..) |char, i| {
        result[i] = char;
    }

    return result;
}

pub fn printString(comptime str: []const u8) void {
    const val = padTo4(str);

    for (val) |character| {
        printChar(character);
    }
}

fn printChar(val: u8) void {
    var charAddress: [*]u32 = @ptrFromInt(0xA000_1000);

    charAddress[0] = @as(u32, val) << 24;

    asm volatile (
        \\li a7, 116
        \\ecall
    );
}
