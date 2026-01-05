const std = @import("std");
const expr = @import("../expr.zig");
const Heap = @import("../Heap.zig");
const utils = @import("../utils.zig");
const runtime_value = @import("../value.zig");
const ConstantType = expr.ConstantType;
const Constant = expr.Constant;
const BigInt = expr.BigInt;
const Bytes = expr.Bytes;

const BuiltinContext = runtime_value.BuiltinContext;
const LinkedValues = runtime_value.LinkedValues;
const Value = runtime_value.Value;
const createConst = runtime_value.createConst;

inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
}

pub const ByteStringAllocation = struct {
    constant_words: [*]u32,
    data_words: [*]u32,
};

pub fn initByteStringAllocation(heap: *Heap, byte_len: u32) ByteStringAllocation {
    const header_words: u32 = 4; // Constant header plus byte length word.
    if (byte_len > std.math.maxInt(u32) - header_words) {
        builtinEvaluationFailure();
    }

    const total_words = header_words + byte_len;
    const buf = heap.createArray(u32, total_words);
    buf[0] = 1;
    buf[1] = @intFromPtr(ConstantType.bytesType());
    buf[2] = @intFromPtr(buf + 3);
    buf[3] = byte_len;

    // ByteStrings keep one byte per word, so the data slice is `byte_len` words long.
    return .{
        .constant_words = buf,
        .data_words = buf + 4,
    };
}

// ByteString functions
pub fn appendByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapBytestring();
    const x = args.next.?.value.unwrapBytestring();

    const length = x.length + y.length;

    // type_length 4 bytes, type pointer 4 bytes, value pointer 4 bytes, length 4 bytes, list of words 4 * (x length + y length)
    var result = m.heap.createArray(u32, length + 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = length;

    var resultPtr = result + 4;

    var i: u32 = 0;
    while (i < x.length) : (i += 1) {
        resultPtr[0] = x.bytes[i];
        resultPtr += 1;
    }

    i = 0;
    while (i < y.length) : (i += 1) {
        resultPtr[i] = y.bytes[i];
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn consByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapBytestring();
    const x = args.next.?.value.unwrapInteger();

    if (x.length > 1 or x.words[0] > 255 or x.sign == 1) {
        builtinEvaluationFailure();
    }

    const length = y.length + 1;

    // type_length 4 bytes, integer 4 bytes, value pointer 4 bytes, length 4 bytes, list of words 4 * (y length + 1)
    var result = m.heap.createArray(u32, length + 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = length;
    result[4] = x.words[0];

    var resultPtr = result + 5;

    var i: u32 = 0;
    while (i < y.length) : (i += 1) {
        resultPtr[i] = y.bytes[i];
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn sliceByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const bytes = args.value.unwrapBytestring();

    // second arg take
    const take = args.next.?.value.unwrapInteger();

    const takeAmount: u32 = if (take.sign == 1) blk: {
        break :blk 0;
    } else if (take.length > 1) blk: {
        break :blk std.math.maxInt(u32);
    } else blk: {
        break :blk take.words[0];
    };

    // first arg drop
    const drop = args.next.?.next.?.value.unwrapInteger();

    const dropAmount: u32 = if (drop.sign == 1) blk: {
        break :blk 0;
    } else if (drop.length > 1) blk: {
        break :blk std.math.maxInt(u32);
    } else blk: {
        break :blk drop.words[0];
    };

    const bytestringLen = bytes.length;

    const leftover = if (dropAmount > bytestringLen) blk: {
        break :blk 0;
    } else blk: {
        break :blk bytestringLen - dropAmount;
    };

    const finalTake = if (takeAmount > leftover) blk: {
        break :blk leftover;
    } else blk: {
        break :blk takeAmount;
    };

    // type 4 bytes, length 4 bytes, value pointer 4 bytes, list of words 4 * (finalTake)
    var result = m.heap.createArray(u32, finalTake + 4);

    const offset = bytes.bytes + bytestringLen - leftover;

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = finalTake;

    var resultPtr = result + 4;

    var i: u32 = 0;
    while (i < finalTake) : (i += 1) {
        resultPtr[i] = offset[i];
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn lengthOfByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const x = args.value.unwrapBytestring();

    var result = m.heap.createArray(u32, 6);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.integerType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 0;
    result[4] = 1;
    result[5] = x.length;

    return createConst(m.heap, @ptrCast(result));
}

pub fn indexByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const index = args.value.unwrapInteger();
    const bytes = args.next.?.value.unwrapBytestring();

    if (index.sign == 1 or index.length > 1) {
        builtinEvaluationFailure();
    }

    // Zero is encoded with zero words, so guard before touching `words[0]`.
    const idx: u32 = if (index.length == 0) 0 else index.words[0];

    if (idx >= bytes.length) {
        builtinEvaluationFailure();
    }

    var result = m.heap.createArray(u32, 6);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.integerType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 0;
    result[4] = 1;
    result[5] = bytes.bytes[idx];

    return createConst(m.heap, @ptrCast(result));
}

pub fn equalsByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapBytestring();
    const x = args.next.?.value.unwrapBytestring();

    const equality = x.compareBytes(&y);

    var result = m.heap.createArray(u32, 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(equality[0]);

    return createConst(m.heap, @ptrCast(result));
}

pub fn lessThanByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapBytestring();
    const x = args.next.?.value.unwrapBytestring();

    const xPtr = &x;

    const equality = xPtr.compareBytes(&y);

    var result = m.heap.createArray(u32, 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(
        !equality[0] and (@intFromPtr(xPtr) != @intFromPtr(equality[1])),
    );

    return createConst(m.heap, @ptrCast(result));
}

pub fn lessThanEqualsByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapBytestring();
    const x = args.next.?.value.unwrapBytestring();

    const xPtr = &x;

    const equality = xPtr.compareBytes(&y);

    var result = m.heap.createArray(u32, 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(
        equality[0] or (@intFromPtr(xPtr) != @intFromPtr(equality[1])),
    );

    return createConst(m.heap, @ptrCast(result));
}

// Conversions
pub fn integerToByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const input = args.value.unwrapInteger();
    const width = args.next.?.value.unwrapInteger();
    const big_endian = args.next.?.next.?.value.unwrapBool();

    // Negative input is not allowed
    if (input.sign == 1) {
        builtinEvaluationFailure();
    }

    // Negative width is not allowed
    if (width.sign == 1) {
        builtinEvaluationFailure();
    }

    // Width must fit in a u32
    if (width.length > 1) {
        builtinEvaluationFailure();
    }

    const width_val: u32 = if (width.length == 0) 0 else width.words[0];

    // Check if width is too large (max 8192 bytes)
    if (width_val > 8192) {
        builtinEvaluationFailure();
    }

    // Reinterpret limbs as a little-endian byte slice so we can trim and pad
    // at byte granularity, matching how the host sees results.
    const word_len: usize = @intCast(input.length);
    const raw_bytes_all: []const u8 = std.mem.sliceAsBytes(input.words[0..word_len]);

    var mag_len: usize = raw_bytes_all.len;
    while (mag_len > 0 and raw_bytes_all[mag_len - 1] == 0) {
        mag_len -= 1;
    }
    const magnitude = raw_bytes_all[0..mag_len];

    if (mag_len > 8192) {
        builtinEvaluationFailure();
    }

    const mag_len_u32: u32 = @intCast(mag_len);
    const width_len: usize = @intCast(width_val);
    const output_len: u32 = if (width_val == 0) mag_len_u32 else blk: {
        if (mag_len > width_len) {
            builtinEvaluationFailure();
        }
        break :blk width_val;
    };

    // Allocate and initialize result
    var result = m.heap.createArray(u32, output_len + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = output_len;

    var out_bytes = result + 4;
    const out_len: usize = @intCast(output_len);

    if (big_endian) {
        // Big-endian: place least-significant bytes at the tail.
        var dst: usize = out_len;
        var src: usize = 0;
        while (src < magnitude.len) : (src += 1) {
            dst -= 1;
            out_bytes[dst] = magnitude[src];
        }

        while (dst > 0) {
            dst -= 1;
            out_bytes[dst] = 0;
        }
    } else {
        // Little-endian: copy magnitude directly and pad the tail.
        var src: usize = 0;
        while (src < magnitude.len) : (src += 1) {
            out_bytes[src] = magnitude[src];
        }

        var pad_idx = magnitude.len;
        while (pad_idx < out_len) : (pad_idx += 1) {
            out_bytes[pad_idx] = 0;
        }
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn byteStringToInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const bytes = args.value.unwrapBytestring();
    const big_endian = args.next.?.value.unwrapBool(); // True => big-endian, False => little-endian

    if (bytes.length == 0) {
        var zero = m.heap.createArray(u32, 6);
        zero[0] = 1;
        zero[1] = @intFromPtr(ConstantType.integerType());
        zero[2] = @intFromPtr(zero + 3);
        zero[3] = 0;
        zero[4] = 1;
        zero[5] = 0;
        return createConst(m.heap, @ptrCast(zero));
    }

    const max_words = (bytes.length + 3) / 4;
    var result = m.heap.createArray(u32, max_words + 5);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.integerType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 0;

    var words = result + 5;
    var word_index: u32 = 0;
    var byte_in_word: u32 = 0;
    var current_word: u32 = 0;

    if (big_endian) {
        var remaining = bytes.length;
        while (remaining > 0) {
            remaining -= 1;
            const byte_val: u32 = bytes.bytes[remaining] & 0xFF;
            const shift: u5 = @intCast(byte_in_word * 8);
            current_word |= byte_val << shift;
            byte_in_word += 1;

            if (byte_in_word == 4) {
                words[word_index] = current_word;
                word_index += 1;
                byte_in_word = 0;
                current_word = 0;
            }
        }
    } else {
        var idx: u32 = 0;
        while (idx < bytes.length) : (idx += 1) {
            const byte_val: u32 = bytes.bytes[idx] & 0xFF;
            const shift: u5 = @intCast(byte_in_word * 8);
            current_word |= byte_val << shift;
            byte_in_word += 1;

            if (byte_in_word == 4) {
                words[word_index] = current_word;
                word_index += 1;
                byte_in_word = 0;
                current_word = 0;
            }
        }
    }

    if (byte_in_word != 0) {
        words[word_index] = current_word;
        word_index += 1;
    }

    var actual_words = word_index;
    while (actual_words > 1 and words[actual_words - 1] == 0) {
        actual_words -= 1;
    }

    if (actual_words == 0) {
        words[0] = 0;
        actual_words = 1;
    }

    result[4] = actual_words;

    if (actual_words < max_words) {
        m.heap.reclaimHeap(u32, max_words - actual_words);
    }

    return createConst(m.heap, @ptrCast(result));
}

// Logical
pub fn andByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const rhs = args.value.unwrapBytestring();
    const lhs = args.next.?.value.unwrapBytestring();
    const pad_to_max = args.next.?.next.?.value.unwrapBool();

    const min_len = @min(lhs.length, rhs.length);
    const max_len = @max(lhs.length, rhs.length);
    const result_len = if (pad_to_max) max_len else min_len;

    // When padding to the longer length we conceptually extend the shorter input with 0xFF bytes.
    var result = m.heap.createArray(u32, result_len + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = result_len;

    var out_bytes = result + 4;
    var i: u32 = 0;
    while (i < result_len) : (i += 1) {
        const lhs_byte: u32 = if (i < lhs.length) lhs.bytes[i] & 0xFF else 0xFF;
        const rhs_byte: u32 = if (i < rhs.length) rhs.bytes[i] & 0xFF else 0xFF;
        out_bytes[i] = lhs_byte & rhs_byte;
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn orByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const rhs = args.value.unwrapBytestring();
    const lhs = args.next.?.value.unwrapBytestring();
    const pad_to_max = args.next.?.next.?.value.unwrapBool();

    const min_len = @min(lhs.length, rhs.length);
    const max_len = @max(lhs.length, rhs.length);
    const result_len = if (pad_to_max) max_len else min_len;

    var result = m.heap.createArray(u32, result_len + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = result_len;

    var out_bytes = result + 4;
    var i: u32 = 0;
    while (i < result_len) : (i += 1) {
        // Padding with zeroes leaves the longer operand unchanged.
        const lhs_byte: u32 = if (i < lhs.length) lhs.bytes[i] & 0xFF else 0;
        const rhs_byte: u32 = if (i < rhs.length) rhs.bytes[i] & 0xFF else 0;
        out_bytes[i] = lhs_byte | rhs_byte;
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn xorByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const rhs = args.value.unwrapBytestring();
    const lhs = args.next.?.value.unwrapBytestring();
    const pad_to_max = args.next.?.next.?.value.unwrapBool();

    const min_len = @min(lhs.length, rhs.length);
    const max_len = @max(lhs.length, rhs.length);
    const result_len = if (pad_to_max) max_len else min_len;

    var result = m.heap.createArray(u32, result_len + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = result_len;

    var out_bytes = result + 4;
    var i: u32 = 0;
    while (i < result_len) : (i += 1) {
        // Zero-extension preserves the longer operand when padding.
        const lhs_byte: u32 = if (i < lhs.length) lhs.bytes[i] & 0xFF else 0;
        const rhs_byte: u32 = if (i < rhs.length) rhs.bytes[i] & 0xFF else 0;
        out_bytes[i] = lhs_byte ^ rhs_byte;
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn complementByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const input = args.value.unwrapBytestring();

    var result = m.heap.createArray(u32, input.length + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = input.length;

    var i: u32 = 0;
    while (i < input.length) : (i += 1) {
        const byte: u8 = @as(u8, @truncate(input.bytes[i]));
        result[4 + i] = ~byte; // bytes are stored as u32 slots, so truncate before inverting
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn readBit(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const index = args.value.unwrapInteger();
    const bytes = args.next.?.value.unwrapBytestring();

    if (index.sign == 1 or index.length > 1) {
        builtinEvaluationFailure();
    }

    const idx: u64 = if (index.length == 0) 0 else index.words[0];
    const total_bits: u64 = @as(u64, bytes.length) * 8;

    if (idx >= total_bits) {
        builtinEvaluationFailure();
    }

    const byte_from_lsb: u32 = @intCast(idx / 8);
    const bit_offset: u3 = @intCast(idx % 8);
    const byte_index: u32 = bytes.length - 1 - byte_from_lsb;
    const byte_value: u8 = @as(u8, @truncate(bytes.bytes[byte_index]));
    // Bits are numbered from the least-significant end of the byte string.
    const bit_is_set = ((byte_value >> bit_offset) & 1) == 1;

    var result = m.heap.createArray(u32, 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(bit_is_set);

    return createConst(m.heap, @ptrCast(result));
}

pub fn writeBits(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const set_bit = args.value.unwrapBool();
    const index_list = args.next.?.value.unwrapList();
    const bytes = args.next.?.next.?.value.unwrapBytestring();

    // writeBits expects a list of integer indices to toggle; reject mismatched element types.
    if (index_list.type_length == 0) {
        builtinEvaluationFailure();
    }
    const inner_types = index_list.inner_type;
    if (@intFromPtr(inner_types) == 0 or inner_types[0] != ConstantType.integer) {
        builtinEvaluationFailure();
    }

    var result = m.heap.createArray(u32, bytes.length + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = bytes.length;

    const out_bytes = result + 4;
    var i: u32 = 0;
    while (i < bytes.length) : (i += 1) {
        out_bytes[i] = bytes.bytes[i] & 0xFF;
    }

    const total_bits: u64 = @as(u64, bytes.length) * 8;

    var cursor = index_list.items;
    while (cursor) |node| {
        const index_const = Constant{
            .length = index_list.type_length,
            .type_list = inner_types,
            .value = node.value,
        };
        const bit_index = index_const.bigInt();
        if (bit_index.sign == 1 or bit_index.length > 1) {
            builtinEvaluationFailure();
        }
        const idx: u64 = if (bit_index.length == 0) 0 else bit_index.words[0];
        if (idx >= total_bits) {
            builtinEvaluationFailure();
        }

        const byte_from_lsb: u32 = @intCast(idx / 8);
        const bit_offset: u3 = @intCast(idx % 8);
        // Bits are numbered from the least-significant end of the ByteString, so
        // work backwards from the tail when locating the target byte.
        const byte_index: u32 = bytes.length - 1 - byte_from_lsb;

        var byte_value: u8 = @truncate(out_bytes[byte_index]);
        const mask: u8 = @as(u8, 1) << bit_offset;
        if (set_bit) {
            byte_value |= mask;
        } else {
            byte_value &= ~mask;
        }
        out_bytes[byte_index] = byte_value;

        cursor = node.next;
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn replicateByte(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const byte_val = args.value.unwrapInteger();
    const count_val = args.next.?.value.unwrapInteger();

    if (count_val.sign == 1 or count_val.length > 1) {
        builtinEvaluationFailure();
    }
    const count: u32 = if (count_val.length == 0) 0 else count_val.words[0];
    if (count > 8192) {
        builtinEvaluationFailure();
    }

    if (byte_val.sign == 1 or byte_val.length > 1) {
        builtinEvaluationFailure();
    }
    const byte_word: u32 = if (byte_val.length == 0) 0 else byte_val.words[0];
    if (byte_word > 255) {
        builtinEvaluationFailure();
    }

    var result = m.heap.createArray(u32, count + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = count;

    // Replication is capped at 8192 bytes by the Plutus spec.
    var out = result + 4;
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        out[i] = byte_word;
    }

    return createConst(m.heap, @ptrCast(result));
}

// Bitwise
pub fn shiftByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const shift = args.value.unwrapInteger();
    const input = args.next.?.value.unwrapBytestring();

    var result = m.heap.createArray(u32, input.length + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = input.length;
    const out_bytes = result + 4;

    if (input.length == 0) {
        return createConst(m.heap, @ptrCast(result));
    }

    if (bigIntIsZero(shift)) {
        copyBytesInto(out_bytes, input);
        return createConst(m.heap, @ptrCast(result));
    }

    const bit_len: u64 = @as(u64, input.length) * 8;
    const maybe_shift = shiftAmountWithinBitLen(shift, bit_len);

    // Any shift that covers the full bit width produces an all-zero ByteString.
    if (maybe_shift == null) {
        var i: u32 = 0;
        while (i < input.length) : (i += 1) {
            out_bytes[i] = 0;
        }
        return createConst(m.heap, @ptrCast(result));
    }

    const abs_shift = maybe_shift.?;
    if (abs_shift == 0) {
        copyBytesInto(out_bytes, input);
        return createConst(m.heap, @ptrCast(result));
    }

    const byte_shift: u32 = @intCast(abs_shift / 8);
    const bit_shift_raw: u3 = @intCast(abs_shift % 8);

    // Positive counts shift left, negative ones shift right.
    if (shift.sign == 1) {
        shiftRightBytes(out_bytes, input, byte_shift, bit_shift_raw);
    } else {
        shiftLeftBytes(out_bytes, input, byte_shift, bit_shift_raw);
    }

    return createConst(m.heap, @ptrCast(result));
}

// Shift counts can be much larger than the byte string, but the width itself fits in u64.
fn shiftAmountWithinBitLen(shift: BigInt, bit_len: u64) ?u64 {
    var trimmed_len: usize = shift.length;
    while (trimmed_len > 0 and shift.words[trimmed_len - 1] == 0) {
        trimmed_len -= 1;
    }

    if (trimmed_len == 0) {
        return 0;
    }

    if (trimmed_len > 2) {
        return null;
    }

    var value: u64 = @as(u64, shift.words[0]);
    if (trimmed_len == 2) {
        value |= @as(u64, shift.words[1]) << 32;
    }

    if (value >= bit_len) {
        return null;
    }

    return value;
}

fn shiftLeftBytes(out: [*]u32, input: Bytes, byte_shift: u32, bit_shift_raw: u3) void {
    const len = input.length;
    const len_u64: u64 = len;

    if (bit_shift_raw == 0) {
        var i: u32 = 0;
        while (i < len) : (i += 1) {
            const src_idx = @as(u64, i) + @as(u64, byte_shift);
            out[i] = if (src_idx < len_u64)
                input.bytes[@intCast(src_idx)] & 0xFF
            else
                0;
        }
        return;
    }

    const left_shift: u4 = @intCast(bit_shift_raw);
    const right_shift: u4 = @intCast(8 - left_shift);

    var i: u32 = 0;
    while (i < len) : (i += 1) {
        const src_idx = @as(u64, i) + @as(u64, byte_shift);
        const curr: u8 = if (src_idx < len_u64)
            @as(u8, @truncate(input.bytes[@intCast(src_idx)]))
        else
            0;
        const next_idx = src_idx + 1;
        const next: u8 = if (next_idx < len_u64)
            @as(u8, @truncate(input.bytes[@intCast(next_idx)]))
        else
            0;

        const high = (@as(u16, curr) << left_shift) & 0xFF;
        const low = @as(u16, next) >> right_shift;
        out[i] = @intCast(high | low);
    }
}

fn shiftRightBytes(out: [*]u32, input: Bytes, byte_shift: u32, bit_shift_raw: u3) void {
    const len = input.length;

    if (bit_shift_raw == 0) {
        var i: u32 = 0;
        while (i < len) : (i += 1) {
            if (byte_shift > i) {
                out[i] = 0;
            } else {
                const src_idx: u32 = i - byte_shift;
                out[i] = input.bytes[src_idx] & 0xFF;
            }
        }
        return;
    }

    const right_shift: u4 = @intCast(bit_shift_raw);
    const left_shift: u4 = @intCast(8 - right_shift);

    var i: u32 = 0;
    while (i < len) : (i += 1) {
        if (byte_shift > i) {
            out[i] = 0;
            continue;
        }

        const curr_idx: u32 = i - byte_shift;
        const curr: u8 = @as(u8, @truncate(input.bytes[curr_idx]));
        const prev: u8 = if (curr_idx == 0)
            0
        else
            @as(u8, @truncate(input.bytes[curr_idx - 1]));

        const high = (@as(u16, prev) << left_shift) & 0xFF;
        const low = @as(u16, curr) >> right_shift;
        out[i] = @intCast(high | low);
    }
}

fn divRemWideStep(remainder: u32, limb: u32, divisor: u32) u32 {
    // Bit-by-bit long division so we never emit 64-bit div/mod helpers on RV32.
    const divisor64 = @as(u64, divisor);
    var rem = @as(u64, remainder);
    var mask: u32 = 0x8000_0000;
    while (mask != 0) : (mask >>= 1) {
        rem = (rem << 1) | @as(u64, @intFromBool((limb & mask) != 0));
        if (rem >= divisor64) {
            rem -= divisor64;
        }
    }
    return @intCast(rem);
}

fn divRemWideStep64(remainder: u64, limb: u32, divisor: u64) u64 {
    // Same idea as divRemWideStep but promoted to 64-bit divisors.  We still
    // walk every bit to avoid emitting helper calls for 128-bit div/mod on RV64.
    var rem: u128 = remainder;
    const divisor128: u128 = divisor;
    var mask: u32 = 0x8000_0000;
    while (mask != 0) : (mask >>= 1) {
        rem = (rem << 1) | @as(u128, @intFromBool((limb & mask) != 0));
        if (rem >= divisor128) {
            rem -= divisor128;
        }
    }
    return @intCast(rem);
}

fn bigIntModuloSmall(value: BigInt, modulus: u64) u64 {
    if (modulus == 0 or value.length == 0) {
        return 0;
    }

    if (modulus <= std.math.maxInt(u32)) {
        var rem: u32 = 0;
        var idx = value.length;
        const divisor: u32 = @intCast(modulus);
        while (idx > 0) {
            idx -= 1;
            rem = divRemWideStep(rem, value.words[idx], divisor);
        }
        return rem;
    }

    var rem64: u64 = 0;
    var idx = value.length;
    while (idx > 0) {
        idx -= 1;
        rem64 = divRemWideStep64(rem64, value.words[idx], modulus);
    }
    return rem64;
}

fn normalizeRotationAmount(shift: BigInt, bit_len: u64) u64 {
    if (bit_len == 0) {
        return 0;
    }

    if (shift.length == 0) {
        return 0;
    }

    // Shift counts can be arbitrarily large (and negative), so reduce them
    // modulo the bit width and flip the direction for negative inputs.
    const remainder = bigIntModuloSmall(shift, bit_len);
    if (remainder == 0) {
        return 0;
    }

    if (shift.sign == 1) {
        return bit_len - remainder;
    }

    return remainder;
}

fn copyBytesInto(out: [*]u32, input: Bytes) void {
    var i: u32 = 0;
    while (i < input.length) : (i += 1) {
        out[i] = input.bytes[i] & 0xFF;
    }
}

pub fn rotateByteString(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const shift = args.value.unwrapInteger();
    const input = args.next.?.value.unwrapBytestring();

    var result = m.heap.createArray(u32, input.length + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = input.length;
    const out_bytes = result + 4;

    const bit_len: u64 = @as(u64, input.length) * 8;
    const normalized = normalizeRotationAmount(shift, bit_len);

    if (normalized == 0) {
        copyBytesInto(out_bytes, input);
        return createConst(m.heap, @ptrCast(result));
    }

    const byte_shift: u32 = @intCast(normalized / 8);
    const bit_shift_raw: u3 = @intCast(normalized % 8);
    const len_u64: u64 = input.length;

    if (bit_shift_raw == 0) {
        var i: u32 = 0;
        while (i < input.length) : (i += 1) {
            var idx_u64 = @as(u64, i) + @as(u64, byte_shift);
            if (idx_u64 >= len_u64) {
                idx_u64 -= len_u64;
            }
            const idx: u32 = @intCast(idx_u64);
            const byte_val = @as(u8, @truncate(input.bytes[idx]));
            out_bytes[i] = @intCast(byte_val);
        }
    } else {
        const left_shift: u4 = @intCast(bit_shift_raw);
        const right_shift: u4 = @intCast(8 - left_shift);
        var i: u32 = 0;
        while (i < input.length) : (i += 1) {
            var curr_idx_u64 = @as(u64, i) + @as(u64, byte_shift);
            if (curr_idx_u64 >= len_u64) {
                curr_idx_u64 -= len_u64;
            }
            var next_idx_u64 = curr_idx_u64 + 1;
            if (next_idx_u64 == len_u64) {
                next_idx_u64 = 0;
            }

            const curr_idx: u32 = @intCast(curr_idx_u64);
            const next_idx: u32 = @intCast(next_idx_u64);
            const curr = @as(u8, @truncate(input.bytes[curr_idx]));
            const next = @as(u8, @truncate(input.bytes[next_idx]));

            const high = (@as(u16, curr) << left_shift) & 0xFF;
            const low = @as(u16, next) >> right_shift;
            const rotated: u16 = @intCast(high | low);
            out_bytes[i] = @intCast(rotated);
        }
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn countSetBits(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const bytes = args.value.unwrapBytestring();

    var total: u64 = 0;
    var i: u32 = 0;
    while (i < bytes.length) : (i += 1) {
        var byte = @as(u8, @truncate(bytes.bytes[i]));
        while (byte != 0) {
            total += @as(u64, byte & 1);
            byte >>= 1;
        }
    }

    if (total == 0) {
        const zero = [_]u32{0};
        return createIntegerValueFromLimbs(m, zero[0..1], true);
    }

    if (total <= std.math.maxInt(u32)) {
        const limbs = [_]u32{@intCast(total)};
        return createIntegerValueFromLimbs(m, limbs[0..1], true);
    }

    // Result fits in 64 bits (at most 8 * bytes.length), so two limbs are sufficient.
    const limbs = [_]u32{
        @intCast(total & 0xFFFF_FFFF),
        @intCast(total >> 32),
    };
    return createIntegerValueFromLimbs(m, limbs[0..2], true);
}

pub fn findFirstSetBit(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const bytes = args.value.unwrapBytestring();
    const negative_one = [_]u32{1};

    if (bytes.length == 0) {
        return createIntegerValueFromLimbs(m, negative_one[0..1], false);
    }

    var idx = bytes.length;
    while (idx > 0) {
        idx -= 1;
        const byte = @as(u8, @truncate(bytes.bytes[idx]));
        if (byte == 0) continue;

        // ByteStrings are big-endian, so walk from the least-significant byte (end)
        // towards the start until we see the first set bit.
        var tz: u32 = 0;
        var shifted = byte;
        while ((shifted & 1) == 0) {
            shifted >>= 1;
            tz += 1;
        }

        const trailing_bytes: u64 = @intCast(bytes.length - 1 - idx);
        const bit_index: u64 = trailing_bytes * 8 + @as(u64, tz);

        if (bit_index == 0) {
            const zero = [_]u32{0};
            return createIntegerValueFromLimbs(m, zero[0..1], true);
        }

        if (bit_index <= std.math.maxInt(u32)) {
            const limbs = [_]u32{@intCast(bit_index)};
            return createIntegerValueFromLimbs(m, limbs[0..1], true);
        }

        // Result fits in 64 bits (at most 8 * bytes.length), so two limbs are sufficient.
        const limbs = [_]u32{
            @intCast(bit_index & 0xFFFF_FFFF),
            @intCast(bit_index >> 32),
        };
        return createIntegerValueFromLimbs(m, limbs[0..2], true);
    }

    return createIntegerValueFromLimbs(m, negative_one[0..1], false);
}

inline fn bigIntIsZero(n: BigInt) bool {
    // Some UPLC constants pad zero with extra limbs, so check every limb.
    var i: u32 = 0;
    while (i < n.length) : (i += 1) {
        if (n.words[i] != 0) {
            return false;
        }
    }
    return true;
}

fn createIntegerValueFromLimbs(
    m: *BuiltinContext,
    limbs: []const u32,
    positive: bool,
) *const Value {
    const len_u32: u32 = @intCast(limbs.len);
    const buf = m.heap.createArray(u32, @intCast(limbs.len + 2));
    buf[0] = @intFromBool(!positive);
    buf[1] = len_u32;
    var i: usize = 0;
    while (i < limbs.len) : (i += 1) {
        buf[2 + i] = limbs[i];
    }

    const con = Constant{
        .length = 1,
        .type_list = @ptrCast(ConstantType.integerType()),
        .value = @intFromPtr(buf),
    };
    return createConst(m.heap, m.heap.create(Constant, &con));
}
