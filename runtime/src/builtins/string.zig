const std = @import("std");
const expr = @import("../expr.zig");
const utils = @import("../utils.zig");
const runtime_value = @import("../value.zig");
const ConstantType = expr.ConstantType;
const Constant = expr.Constant;
const String = expr.String;
const Bytes = expr.Bytes;

const BuiltinContext = runtime_value.BuiltinContext;
const LinkedValues = runtime_value.LinkedValues;
const Value = runtime_value.Value;
const createConst = runtime_value.createConst;

inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
}

pub const StringBytes = struct {
    byte_len: u32,
    is_packed: bool,
};

pub fn analyzeString(str: String) StringBytes {
    var is_packed = false;
    var probe: u32 = 0;
    while (probe < str.length) : (probe += 1) {
        if (str.bytes[probe] > 0xFF) {
            is_packed = true;
            break;
        }
    }

    return .{
        .byte_len = if (is_packed) packedLength(str) else unpackedLength(str),
        .is_packed = is_packed,
    };
}

fn unpackedLength(str: String) u32 {
    var len = str.length;
    while (len > 0 and (str.bytes[len - 1] & 0xFF) == 0) {
        len -= 1;
    }
    return len;
}

fn packedLength(str: String) u32 {
    var total: u32 = str.length * 4;
    if (total == 0) return 0;

    var idx = str.length;
    while (idx > 0) : (idx -= 1) {
        const word = str.bytes[idx - 1];
        var shift: u5 = 24;
        while (true) {
            const byte_val = (word >> shift) & 0xFF;
            if (byte_val == 0) {
                total -= 1;
                if (total == 0) return 0;
            } else {
                return total;
            }

            if (shift == 0) break;
            shift -= 8;
        }
    }

    return 0;
}

pub fn extractStringByte(str: String, view: StringBytes, byte_index: u32) u8 {
    if (byte_index >= view.byte_len) return 0;

    if (!view.is_packed) {
        // Unpacked: one byte per word
        return @truncate(str.bytes[byte_index]);
    } else {
        // Packed: 4 bytes per word in little-endian
        const word_idx = byte_index / 4;
        const byte_in_word = byte_index % 4;
        const word = str.bytes[word_idx];
        const shift: u5 = @intCast(byte_in_word * 8);
        return @as(u8, @truncate((word >> shift) & 0xFF));
    }
}

// String functions
pub fn appendString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapString();
    const x = args.next.?.value.unwrapString();

    const x_view = analyzeString(x);
    const y_view = analyzeString(y);
    const total_bytes = x_view.byte_len + y_view.byte_len;

    // Calculate word count for packed format (4 bytes per word)
    const word_count: u32 = if (total_bytes == 0) 0 else (total_bytes + 3) / 4;

    // type_length 4 bytes, type pointer 4 bytes, value pointer 4 bytes, length 4 bytes, then packed words
    var result = m.heap.createArray(u32, word_count + 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.stringType());
    result[2] = @intFromPtr(result + 3);
    result[3] = word_count;

    var data = result + 4;
    // Zero out all words (including padding bytes)
    var zero_index: u32 = 0;
    while (zero_index < word_count) : (zero_index += 1) {
        data[zero_index] = 0;
    }

    // Pack the bytes from x into words
    var i: u32 = 0;
    while (i < x_view.byte_len) : (i += 1) {
        const byte = extractStringByte(x, x_view, i);
        const word_idx = i / 4;
        const byte_pos = i % 4;
        data[word_idx] |= @as(u32, byte) << @intCast(byte_pos * 8);
    }

    // Pack the bytes from y into words, continuing from where x left off
    var j: u32 = 0;
    while (j < y_view.byte_len) : (j += 1) {
        const byte = extractStringByte(y, y_view, j);
        const total_byte_idx = x_view.byte_len + j;
        const word_idx = total_byte_idx / 4;
        const byte_pos = total_byte_idx % 4;
        data[word_idx] |= @as(u32, byte) << @intCast(byte_pos * 8);
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn equalsString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapString();
    const x = args.next.?.value.unwrapString();

    const equality = x.equals(&y);

    var result = m.heap.createArray(u32, 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(equality);

    return createConst(m.heap, @ptrCast(result));
}

pub fn encodeUtf8(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const str = args.value.unwrapString();
    const view = analyzeString(str);

    const byte_len = view.byte_len;
    var result = m.heap.createArray(u32, byte_len + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = byte_len;

    const data = result + 4;
    // Strings may be stored packed (four bytes per word) or unpacked; extractStringByte
    // normalizes each byte so we can lay out a standard bytestring payload.
    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        data[i] = extractStringByte(str, view, i);
    }

    return createConst(m.heap, @ptrCast(result));
}

fn isUtf8Continuation(byte: u8) bool {
    return (byte & 0xC0) == 0x80;
}

fn loadUtf8Byte(bytes: Bytes, idx: usize) u8 {
    return @as(u8, @truncate(bytes.bytes[idx]));
}

fn validateUtf8(bytes: Bytes) bool {
    const len: usize = @intCast(bytes.length);
    var i: usize = 0;

    while (i < len) {
        const b0 = loadUtf8Byte(bytes, i);

        if (b0 < 0x80) {
            i += 1;
            continue;
        }

        if (b0 < 0xC2) {
            return false; // Reject overlong forms (0xC0,0xC1) and stray continuations.
        }

        if (b0 < 0xE0) {
            if (i + 1 >= len) return false;
            const b1 = loadUtf8Byte(bytes, i + 1);
            if (!isUtf8Continuation(b1)) return false;
            i += 2;
            continue;
        }

        if (b0 < 0xF0) {
            if (i + 2 >= len) return false;
            const b1 = loadUtf8Byte(bytes, i + 1);
            const b2 = loadUtf8Byte(bytes, i + 2);
            if (!isUtf8Continuation(b1) or !isUtf8Continuation(b2)) return false;
            if (b0 == 0xE0 and b1 < 0xA0) return false; // Overlong encodings.
            if (b0 == 0xED and b1 >= 0xA0) return false; // UTF-16 surrogate range.
            i += 3;
            continue;
        }

        if (b0 < 0xF5) {
            if (i + 3 >= len) return false;
            const b1 = loadUtf8Byte(bytes, i + 1);
            const b2 = loadUtf8Byte(bytes, i + 2);
            const b3 = loadUtf8Byte(bytes, i + 3);
            if (!isUtf8Continuation(b1) or !isUtf8Continuation(b2) or !isUtf8Continuation(b3)) return false;
            if (b0 == 0xF0 and b1 < 0x90) return false; // Overlong encodings.
            if (b0 == 0xF4 and b1 >= 0x90) return false; // Beyond U+10FFFF.
            i += 4;
            continue;
        }

        return false; // Reject 5+ byte sequences (> 0xF4 lead byte).
    }

    return true;
}

pub fn decodeUtf8(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const bytes = args.value.unwrapBytestring();

    if (!validateUtf8(bytes)) {
        builtinEvaluationFailure();
    }

    const byte_len: u32 = bytes.length;
    const word_count: u32 = if (byte_len == 0) 0 else (byte_len + 3) / 4;

    // Strings store bytes packed little-endian into u32 words (4 bytes per word).
    // The first four slots mirror Constant header layout (type length, type ptr, etc.).
    var result = m.heap.createArray(u32, word_count + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.stringType());
    result[2] = @intFromPtr(result + 3);
    result[3] = word_count;

    const data = result + 4;
    var zero_idx: u32 = 0;
    while (zero_idx < word_count) : (zero_idx += 1) {
        data[zero_idx] = 0;
    }

    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        const byte = @as(u8, @truncate(bytes.bytes[i]));
        const word_idx = i / 4;
        const byte_pos = i % 4;
        data[word_idx] |= @as(u32, byte) << @intCast(byte_pos * 8);
    }

    return createConst(m.heap, @ptrCast(result));
}
