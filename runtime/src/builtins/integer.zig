const std = @import("std");
const expr = @import("../expr.zig");
const utils = @import("../utils.zig");
const runtime_value = @import("../value.zig");
const ConstantType = expr.ConstantType;
const Constant = expr.Constant;
const BigInt = expr.BigInt;

const BuiltinContext = runtime_value.BuiltinContext;
const LinkedValues = runtime_value.LinkedValues;
const Value = runtime_value.Value;
const createConst = runtime_value.createConst;

inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
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

const DivisionMode = enum {
    trunc,
    floor,
};

const DivStep = struct { quotient: u32, remainder: u32 };

fn divRemWideStep(remainder: u32, limb: u32, divisor: u32) DivStep {
    // Bit-by-bit long division so we never emit 64-bit div/mod helpers on RV32.
    const divisor64 = @as(u64, divisor);
    var rem = @as(u64, remainder);
    var q: u32 = 0;
    var mask: u32 = 0x8000_0000;
    while (mask != 0) : (mask >>= 1) {
        rem = (rem << 1) | @as(u64, @intFromBool((limb & mask) != 0));
        q = (q << 1);
        if (rem >= divisor64) {
            rem -= divisor64;
            q |= 1;
        }
    }
    return .{ .quotient = q, .remainder = @intCast(rem) };
}

inline fn effectiveBigIntLen(n: BigInt) usize {
    var len: usize = n.length;
    while (len > 0 and n.words[len - 1] == 0) {
        len -= 1;
    }
    return len;
}

fn normalizeLimbsInPlace(buf: []u32) usize {
    var len = buf.len;
    while (len > 1 and buf[len - 1] == 0) {
        len -= 1;
    }
    return len;
}

fn incrementMagnitude(buf: []u32) void {
    var idx: usize = 0;
    while (idx < buf.len) : (idx += 1) {
        const sum = @as(u64, buf[idx]) + 1;
        buf[idx] = @intCast(sum & 0xFFFF_FFFF);
        if (sum <= std.math.maxInt(u32)) {
            return;
        }
    }
}

fn normalizeInto(dst: []u32, src: []const u32, shift: u6) u32 {
    var carry: u64 = 0;
    var i: usize = 0;
    while (i < src.len) : (i += 1) {
        const combined = (@as(u64, src[i]) << shift) | carry;
        dst[i] = @intCast(combined & 0xFFFF_FFFF);
        carry = combined >> 32;
    }
    return @intCast(carry);
}

fn denormalizeInPlace(buf: []u32, shift: u6) void {
    // Undo the normalization shift applied before division.
    if (shift == 0) return;
    const shift_amt: u5 = @intCast(shift);
    const inverse_shift: u5 = @intCast(@as(u6, 32) - shift);
    const mask: u32 = (@as(u32, 1) << shift_amt) - 1;
    var carry: u32 = 0;
    var idx: usize = buf.len;
    while (idx > 0) {
        idx -= 1;
        const word = buf[idx];
        buf[idx] = (word >> shift_amt) | (carry << inverse_shift);
        carry = word & mask;
    }
}

fn hasNonZero(slice: []const u32) bool {
    for (slice) |word| {
        if (word != 0) return true;
    }
    return false;
}

fn subtractMulAt(accum: []u32, sub: []const u32, factor: u64, offset: usize) bool {
    const base: u64 = 0x1_0000_0000;
    const factor32: u64 = factor & 0xFFFF_FFFF;
    var borrow: u64 = 0;
    var carry: u64 = 0;
    var i: usize = 0;
    while (i < sub.len) : (i += 1) {
        const product = @as(u64, sub[i]) * factor32 + carry;
        const prod_low = product & 0xFFFF_FFFF;
        carry = product >> 32;

        const idx = offset + i;
        const lhs = @as(u64, accum[idx]);
        const subtrahend = prod_low + borrow;
        if (lhs >= subtrahend) {
            accum[idx] = @intCast(lhs - subtrahend);
            borrow = 0;
        } else {
            accum[idx] = @intCast((lhs + base) - subtrahend);
            borrow = 1;
        }
    }
    const idx = offset + sub.len;
    const lhs_tail = @as(u64, accum[idx]);
    const tail_sub = carry + borrow;
    if (lhs_tail >= tail_sub) {
        accum[idx] = @intCast(lhs_tail - tail_sub);
        return false;
    } else {
        accum[idx] = @intCast((lhs_tail + base) - tail_sub);
        return true;
    }
}

fn addAtOffset(accum: []u32, addend: []const u32, offset: usize) void {
    var carry: u64 = 0;
    var i: usize = 0;
    while (i < addend.len) : (i += 1) {
        const idx = offset + i;
        const sum = @as(u64, accum[idx]) + @as(u64, addend[i]) + carry;
        accum[idx] = @intCast(sum & 0xFFFF_FFFF);
        carry = sum >> 32;
    }
    const idx = offset + addend.len;
    const sum_tail = @as(u64, accum[idx]) + carry;
    accum[idx] = @intCast(sum_tail & 0xFFFF_FFFF);
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

fn finalizeRemainderValue(
    m: *BuiltinContext,
    remainder_limbs: []const u32,
    remainder_nonzero: bool,
    numer_positive: bool,
    denom_positive: bool,
    mode: DivisionMode,
    denom: BigInt,
    denom_len: usize,
) *const Value {
    if (!remainder_nonzero) {
        const zero_limbs = [_]u32{0};
        return createIntegerValueFromLimbs(m, zero_limbs[0..1], true);
    }

    if (mode == .floor and (numer_positive != denom_positive)) {
        // Adjust by |denom| - remainder so the result shares the denominator's sign.
        return subtractMagnitudeForMod(m, denom, denom_len, remainder_limbs);
    }

    return createIntegerValueFromLimbs(m, remainder_limbs, numer_positive);
}

fn subtractMagnitudeForMod(
    m: *BuiltinContext,
    denom: BigInt,
    denom_len: usize,
    remainder_limbs: []const u32,
) *const Value {
    var denom_view = denom;
    denom_view.length = @intCast(denom_len);

    const remainder_big = BigInt{
        .sign = 0,
        .length = @intCast(remainder_limbs.len),
        .words = remainder_limbs.ptr,
    };

    return subSignedIntegers(m, denom_view, remainder_big);
}

fn handleSmallerMagnitude(
    m: *BuiltinContext,
    numer: BigInt,
    numer_positive: bool,
    denom_positive: bool,
    mode: DivisionMode,
) *const Value {
    const remainder_nonzero = !bigIntIsZero(numer);
    var magnitude = m.heap.createArray(u32, 1);
    magnitude[0] = 0;
    var result_positive = true;
    if (mode == .floor and remainder_nonzero and (numer_positive != denom_positive)) {
        magnitude[0] = 1;
        result_positive = false;
    }
    return createIntegerValueFromLimbs(m, magnitude[0..1], result_positive);
}

fn divideMultiLimb(
    m: *BuiltinContext,
    numer: BigInt,
    denom: BigInt,
    mode: DivisionMode,
) *const Value {
    const numer_len = effectiveBigIntLen(numer);
    const denom_len = effectiveBigIntLen(denom);
    const numer_positive = numer.sign == 0;
    const denom_positive = denom.sign == 0;
    if (numer_len < denom_len) {
        return handleSmallerMagnitude(m, numer, numer_positive, denom_positive, mode);
    }

    var result_positive = numer_positive == denom_positive;
    const v_high = denom.words[denom_len - 1];
    const shift: u6 = @intCast(@clz(v_high));

    const u_storage = m.heap.createArray(u32, @intCast(numer_len + 1));
    const v_storage = m.heap.createArray(u32, @intCast(denom_len));
    const q_capacity = numer_len - denom_len + 1;
    const quotient_storage = m.heap.createArray(u32, @intCast(q_capacity + 1));
    @memset(quotient_storage[0 .. q_capacity + 1], 0);

    const numer_slice = numer.words[0..numer_len];
    const denom_slice = denom.words[0..denom_len];
    const carry = normalizeInto(u_storage[0..numer_len], numer_slice, shift);
    u_storage[numer_len] = carry;
    _ = normalizeInto(v_storage[0..denom_len], denom_slice, shift);

    var idx: usize = q_capacity;
    while (idx > 0) : (idx -= 1) {
        const pos = idx - 1;
        const u_high = u_storage[pos + denom_len];
        const u_next = u_storage[pos + denom_len - 1];
        const numerator64 = (@as(u64, u_high) << 32) | @as(u64, u_next);
        var q_hat: u64 = if (u_high == v_storage[denom_len - 1])
            std.math.maxInt(u32)
        else
            numerator64 / @as(u64, v_storage[denom_len - 1]);
        var r_hat = numerator64 - q_hat * @as(u64, v_storage[denom_len - 1]);
        if (denom_len > 1) {
            const base: u64 = 0x1_0000_0000;
            while (q_hat == base or q_hat * @as(u64, v_storage[denom_len - 2]) >
                (r_hat << 32) + @as(u64, u_storage[pos + denom_len - 2]))
            {
                q_hat -= 1;
                r_hat += @as(u64, v_storage[denom_len - 1]);
                if (r_hat >= base) break;
            }
        }
        if (subtractMulAt(u_storage[0 .. numer_len + 1], v_storage[0..denom_len], q_hat, pos)) {
            addAtOffset(u_storage[0 .. numer_len + 1], v_storage[0..denom_len], pos);
            q_hat -= 1;
        }
        quotient_storage[pos] = @intCast(q_hat);
    }

    const remainder_nonzero = hasNonZero(u_storage[0..denom_len]);
    if (mode == .floor and remainder_nonzero and (numer_positive != denom_positive)) {
        incrementMagnitude(quotient_storage[0 .. q_capacity + 1]);
        result_positive = false;
    }

    const final_len = normalizeLimbsInPlace(quotient_storage[0 .. q_capacity + 1]);
    if (final_len == 1 and quotient_storage[0] == 0) {
        result_positive = true;
    }

    return createIntegerValueFromLimbs(m, quotient_storage[0..final_len], result_positive);
}

fn remainderMultiLimb(
    m: *BuiltinContext,
    numer: BigInt,
    denom: BigInt,
    mode: DivisionMode,
) *const Value {
    const numer_len = effectiveBigIntLen(numer);
    const denom_len = effectiveBigIntLen(denom);
    const numer_positive = numer.sign == 0;
    const denom_positive = denom.sign == 0;

    if (numer_len == 0) {
        const zero_limbs = [_]u32{0};
        return createIntegerValueFromLimbs(m, zero_limbs[0..1], true);
    }

    if (denom_len == 0) {
        builtinEvaluationFailure();
    }

    if (numer_len < denom_len) {
        const remainder_slice = numer.words[0..numer_len];
        const remainder_nonzero = hasNonZero(remainder_slice);
        return finalizeRemainderValue(
            m,
            remainder_slice,
            remainder_nonzero,
            numer_positive,
            denom_positive,
            mode,
            denom,
            denom_len,
        );
    }

    const v_high = denom.words[denom_len - 1];
    const shift: u6 = @intCast(@clz(v_high));

    const u_storage = m.heap.createArray(u32, @intCast(numer_len + 1));
    const v_storage = m.heap.createArray(u32, @intCast(denom_len));

    const numer_slice = numer.words[0..numer_len];
    const denom_slice = denom.words[0..denom_len];
    const carry = normalizeInto(u_storage[0..numer_len], numer_slice, shift);
    u_storage[numer_len] = carry;
    _ = normalizeInto(v_storage[0..denom_len], denom_slice, shift);

    const q_capacity = numer_len - denom_len + 1;
    var idx: usize = q_capacity;
    while (idx > 0) : (idx -= 1) {
        const pos = idx - 1;
        const u_high = u_storage[pos + denom_len];
        const u_next = u_storage[pos + denom_len - 1];
        const numerator64 = (@as(u64, u_high) << 32) | @as(u64, u_next);
        var q_hat: u64 = if (u_high == v_storage[denom_len - 1])
            std.math.maxInt(u32)
        else
            numerator64 / @as(u64, v_storage[denom_len - 1]);
        var r_hat = numerator64 - q_hat * @as(u64, v_storage[denom_len - 1]);
        if (denom_len > 1) {
            const base: u64 = 0x1_0000_0000;
            while (q_hat == base or q_hat * @as(u64, v_storage[denom_len - 2]) >
                (r_hat << 32) + @as(u64, u_storage[pos + denom_len - 2]))
            {
                q_hat -= 1;
                r_hat += @as(u64, v_storage[denom_len - 1]);
                if (r_hat >= base) break;
            }
        }
        if (subtractMulAt(u_storage[0 .. numer_len + 1], v_storage[0..denom_len], q_hat, pos)) {
            addAtOffset(u_storage[0 .. numer_len + 1], v_storage[0..denom_len], pos);
        }
    }

    var remainder_slice = u_storage[0..denom_len];
    denormalizeInPlace(remainder_slice, shift);
    const rem_len = normalizeLimbsInPlace(remainder_slice);
    remainder_slice = remainder_slice[0..rem_len];
    const remainder_nonzero = hasNonZero(remainder_slice);

    return finalizeRemainderValue(
        m,
        remainder_slice,
        remainder_nonzero,
        numer_positive,
        denom_positive,
        mode,
        denom,
        denom_len,
    );
}

fn quotientBySingleLimb(
    m: *BuiltinContext,
    numer: BigInt,
    denom: BigInt,
    mode: DivisionMode,
) ?*const Value {
    const denom_len = effectiveBigIntLen(denom);
    if (denom_len != 1) return null;

    const denom_abs: u32 = denom.words[0];
    if (denom_abs == 0) return null;

    var numer_len = effectiveBigIntLen(numer);
    if (numer_len == 0) numer_len = 1;

    const quotient_storage = m.heap.createArray(u32, @intCast(numer_len + 1));
    quotient_storage[numer_len] = 0;

    var remainder: u32 = 0;
    var idx = numer_len;
    while (idx > 0) : (idx -= 1) {
        const limb = numer.words[idx - 1];
        const step = divRemWideStep(remainder, limb, denom_abs);
        remainder = step.remainder;
        quotient_storage[idx - 1] = step.quotient;
    }

    const numer_positive = numer.sign == 0;
    const denom_positive = denom.sign == 0;
    var result_positive = numer_positive == denom_positive;
    const remainder_nonzero = remainder != 0;

    if (mode == .floor and remainder_nonzero and (numer_positive != denom_positive)) {
        incrementMagnitude(quotient_storage[0 .. numer_len + 1]);
        result_positive = false;
    }

    const q_len = normalizeLimbsInPlace(quotient_storage[0 .. numer_len + 1]);
    if (q_len == 1 and quotient_storage[0] == 0) {
        result_positive = true;
    }

    return createIntegerValueFromLimbs(m, quotient_storage[0..q_len], result_positive);
}

fn remainderBySingleLimb(
    m: *BuiltinContext,
    numer: BigInt,
    denom: BigInt,
    mode: DivisionMode,
) ?*const Value {
    const denom_len = effectiveBigIntLen(denom);
    if (denom_len != 1) return null;

    const denom_abs: u32 = denom.words[0];
    if (denom_abs == 0) return null;

    var numer_len = effectiveBigIntLen(numer);
    if (numer_len == 0) numer_len = 1;

    var remainder: u32 = 0;
    var idx = numer_len;
    while (idx > 0) : (idx -= 1) {
        const limb = numer.words[idx - 1];
        const step = divRemWideStep(remainder, limb, denom_abs);
        remainder = step.remainder;
    }

    var remainder_word = [_]u32{remainder};
    const numer_positive = numer.sign == 0;
    const denom_positive = denom.sign == 0;
    const remainder_nonzero = remainder != 0;

    return finalizeRemainderValue(
        m,
        remainder_word[0..1],
        remainder_nonzero,
        numer_positive,
        denom_positive,
        mode,
        denom,
        denom_len,
    );
}

fn computeQuotient(
    m: *BuiltinContext,
    numer: BigInt,
    denom: BigInt,
    mode: DivisionMode,
) *const Value {
    if (quotientBySingleLimb(m, numer, denom, mode)) |value| {
        return value;
    }
    return divideMultiLimb(m, numer, denom, mode);
}

fn computeRemainder(
    m: *BuiltinContext,
    numer: BigInt,
    denom: BigInt,
    mode: DivisionMode,
) *const Value {
    if (remainderBySingleLimb(m, numer, denom, mode)) |value| {
        return value;
    }
    return remainderMultiLimb(m, numer, denom, mode);
}

// Public builtin functions

pub fn addInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapInteger();
    const x = args.next.?.value.unwrapInteger();

    if (x.sign == y.sign) {
        return addSignedIntegers(m, x, y);
    } else {
        return subSignedIntegers(m, x, y);
    }
}

pub fn subInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    var y = args.value.unwrapInteger();
    y.sign ^= 1;

    const x = args.next.?.value.unwrapInteger();

    if (x.sign == y.sign) {
        return addSignedIntegers(m, x, y);
    } else {
        return subSignedIntegers(m, x, y);
    }
}

pub fn multiplyInteger(m: *BuiltinContext, args: *const LinkedValues) *const Value {
    const b = args.value.unwrapInteger();
    const a = args.next.?.value.unwrapInteger();

    const result_sign: u32 = a.sign ^ b.sign;

    const a_words = a.words[0..a.length];
    const b_words = b.words[0..b.length];

    // Allocate space for the worst-case length: |a| + |b|
    const max_len = a.length + b.length;
    const resultPtr = m.heap.createArray(u32, max_len + 5); // bump-allocate
    var result = resultPtr + 5;

    // The buffer comes back with arbitrary bytes - clear it so that the
    // length-trimming logic sees genuine zeroes in the untouched limbs.
    @memset(result[0..max_len], 0);

    // Core multiplication loop (32-bit words split into 16-bit halves)
    for (a_words, 0..) |a_word, i| {
        const a_low = a_word & 0xFFFF;
        const a_high = a_word >> 16;

        for (b_words, 0..) |b_word, j| {
            const b_low = b_word & 0xFFFF;
            const b_high = b_word >> 16;

            // 16-bit partial products, promoted to u64 to avoid overflow
            const p1: u64 = @as(u64, a_low) * b_low; // bits  0-31
            const p2: u64 = @as(u64, a_low) * b_high; // bits 16-47
            const p3: u64 = @as(u64, a_high) * b_low; // bits 16-47
            const p4: u64 = @as(u64, a_high) * b_high; // bits 32-63

            // Re-assemble a 64-bit product from the four partials
            const low: u64 = p1 & 0xFFFF;
            const mid: u64 = (p1 >> 16) + (p2 & 0xFFFF) + (p3 & 0xFFFF);
            const high: u64 = (p2 >> 16) + (p3 >> 16) + p4;

            const product: u64 = low + ((mid & 0xFFFF) << 16); // 32-bit value
            var carry: u64 = (mid >> 16) + high; // carry >= 0, < 2^32

            // Add the 32-bit product into result[i + j] with carry propagation
            var idx: usize = i + j;
            var tmp: u64 = @as(u64, result[idx]) + product;
            result[idx] = @truncate(tmp);
            carry += tmp >> 32;

            idx += 1;
            while (carry != 0) : (idx += 1) {
                tmp = @as(u64, result[idx]) + carry;
                result[idx] = @truncate(tmp);
                carry = tmp >> 32;
            }
        }
    }

    // Trim leading zero words
    var final_len: u32 = max_len;
    while (final_len > 1 and result[final_len - 1] == 0) {
        final_len -= 1;
    }
    // length of types is 1 for integer
    resultPtr[0] = 1;
    resultPtr[1] = @intFromPtr(ConstantType.integerType());
    resultPtr[2] = @intFromPtr(resultPtr + 3);
    resultPtr[3] = result_sign;
    resultPtr[4] = @intCast(final_len);

    return createConst(m.heap, @ptrCast(resultPtr));
}

pub fn divideInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const d = args.value.unwrapInteger();
    const n = args.next.?.value.unwrapInteger();

    if (bigIntIsZero(d)) {
        builtinEvaluationFailure();
    }

    if (bigIntIsZero(n)) {
        const buf = m.heap.createArray(u32, 3);
        buf[0] = 0;
        buf[1] = 1;
        buf[2] = 0;
        const con = Constant{
            .length = 1,
            .type_list = @ptrCast(ConstantType.integerType()),
            .value = @intFromPtr(buf),
        };
        return createConst(m.heap, m.heap.create(Constant, &con));
    }

    return computeQuotient(m, n, d, .floor);
}

pub fn quotientInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const d = args.value.unwrapInteger();
    const n = args.next.?.value.unwrapInteger();

    runtime_value.numer_len_debug = n.length;
    runtime_value.denom_len_debug = d.length;

    if (bigIntIsZero(d)) {
        builtinEvaluationFailure();
    }

    if (bigIntIsZero(n)) {
        const buf = m.heap.createArray(u32, 3);
        buf[0] = 0;
        buf[1] = 1;
        buf[2] = 0;
        const con = Constant{
            .length = 1,
            .type_list = @ptrCast(ConstantType.integerType()),
            .value = @intFromPtr(buf),
        };
        return createConst(m.heap, m.heap.create(Constant, &con));
    }

    return computeQuotient(m, n, d, .trunc);
}

pub fn remainderInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const d = args.value.unwrapInteger();
    const n = args.next.?.value.unwrapInteger();

    runtime_value.numer_len_debug = n.length;
    runtime_value.denom_len_debug = d.length;

    if (bigIntIsZero(d)) {
        builtinEvaluationFailure();
    }

    if (bigIntIsZero(n)) {
        const zero_limbs = [_]u32{0};
        return createIntegerValueFromLimbs(m, zero_limbs[0..1], true);
    }

    return computeRemainder(m, n, d, .trunc);
}

pub fn modInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const d = args.value.unwrapInteger();
    const n = args.next.?.value.unwrapInteger();

    runtime_value.numer_len_debug = n.length;
    runtime_value.denom_len_debug = d.length;

    if (bigIntIsZero(d)) {
        builtinEvaluationFailure();
    }

    if (bigIntIsZero(n)) {
        const zero_limbs = [_]u32{0};
        return createIntegerValueFromLimbs(m, zero_limbs[0..1], true);
    }

    return computeRemainder(m, n, d, .floor);
}

pub fn equalsInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapInteger();
    const x = args.next.?.value.unwrapInteger();

    const equality = x.compareMagnitude(&y);

    var result = m.heap.createArray(u32, 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(equality[0]);

    return createConst(m.heap, @ptrCast(result));
}

pub fn lessThanInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapInteger();
    const x = args.next.?.value.unwrapInteger();

    const xPtr = &x;

    const equality = xPtr.compareMagnitude(&y);

    var result = m.heap.createArray(u32, 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(
        !equality[0] and (@intFromPtr(xPtr) != @intFromPtr(equality[1])),
    );

    return createConst(m.heap, @ptrCast(result));
}

pub fn lessThanEqualsInteger(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value.unwrapInteger();
    const x = args.next.?.value.unwrapInteger();

    const xPtr = &x;

    const equality = xPtr.compareMagnitude(&y);

    var result = m.heap.createArray(u32, 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(
        equality[0] or (@intFromPtr(xPtr) != @intFromPtr(equality[1])),
    );

    return createConst(m.heap, @ptrCast(result));
}

pub fn addSignedIntegers(
    m: *BuiltinContext,
    x: BigInt,
    y: BigInt,
) *const Value {
    // We overallocate and then claim later if necessary
    // type_length 4 bytes, integer 4 bytes, pointer to value 4 bytes,
    // sign 4 bytes, length 4 bytes, list of words 4 * (max length + 1)
    const maxLength = @max(x.length, y.length);
    const resultLength = maxLength + 5 + 1;

    var result = m.heap.createArray(u32, resultLength);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.integerType());
    // pointer to big int value
    result[2] = @intFromPtr(result + 3);
    result[3] = x.sign;

    var i: u32 = 0;
    var carry: u32 = 0;
    var resultWords = result + 5;

    while (i < maxLength) : (i += 1) {
        var xWord: u32 = 0;
        if (x.length > i) {
            xWord = x.words[i];
        }

        var yWord: u32 = 0;
        if (y.length > i) {
            yWord = y.words[i];
        }

        const wordResult: u32 = xWord +% yWord;
        const carryResult: u32 = wordResult +% carry;

        carry = @intFromBool((wordResult < xWord) or (carryResult < wordResult));

        resultWords[i] = carryResult;
    }

    // length = to max_length + 1 if carry else max_length
    result[4] = if (carry == 1) blk: {
        resultWords[maxLength] = 1;
        break :blk maxLength + 1;
    } else blk: {
        // We reclaim one unused heap word here
        m.heap.reclaimHeap(u32, 1);
        break :blk maxLength;
    };

    return createConst(m.heap, @ptrCast(result));
}

pub fn subSignedIntegers(
    m: *BuiltinContext,
    x: BigInt,
    y: BigInt,
) *const Value {
    const compare = x.compareMagnitude(&y);

    // equal values so we return 0
    if (compare[0]) {
        var result = m.heap.createArray(u32, 6);
        //type length
        result[0] = 1;
        result[1] = @intFromPtr(ConstantType.integerType());
        // pointer to big int value
        result[2] = @intFromPtr(result + 3);
        // sign
        result[3] = 0;
        // length
        result[4] = 1;
        // zero-value
        result[5] = 0;

        return createConst(m.heap, @ptrCast(result));
    }

    const greater: *const BigInt = compare[1];
    const lesser: *const BigInt = compare[2];

    const maxLength = greater.length;
    const resultLength = maxLength + 5;

    var result = m.heap.createArray(u32, resultLength);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.integerType());
    // pointer to big int value
    result[2] = @intFromPtr(result + 3);
    result[3] = greater.sign;

    var i: u32 = 0;
    var carry: u32 = 0;
    var resultWords = result + 5;
    var reclaim: u32 = 0;
    var finalLength: u32 = 0;

    while (i < maxLength) : (i += 1) {
        var lesserWord: u32 = 0;
        if (lesser.length > i) {
            lesserWord = lesser.words[i];
        }

        const greaterWord = greater.words[i];

        const wordResult: u32 = greaterWord -% lesserWord;
        const carryResult: u32 = wordResult -% carry;

        carry = @intFromBool((wordResult > greaterWord) or (carryResult > wordResult));

        resultWords[i] = carryResult;

        if (carryResult == 0) {
            reclaim += 1;
        } else {
            finalLength += reclaim;
            finalLength += 1;
            reclaim = 0;
        }
    }

    // carry should always be 0 after this since we subtracted greater from the lesser value
    result[4] = finalLength;
    m.heap.reclaimHeap(u32, reclaim);
    return createConst(m.heap, @ptrCast(result));
}
