const Heap = @import("Heap.zig");
const expr = @import("./expr.zig");
const std = @import("std");
const testing = std.testing;
const Term = expr.Term;
const TermList = expr.TermList;
const ConstantType = expr.ConstantType;
const Constant = expr.Constant;
const DefaultFunction = expr.DefaultFunction;
const BigInt = expr.BigInt;
const Bytes = expr.Bytes;
const String = expr.String;
const List = expr.List;
const ListNode = expr.ListNode;
const G1Element = expr.G1Element;
const G2Element = expr.G2Element;
const MlResult = expr.MlResult;
const Data = expr.Data;
const DataListNode = expr.DataListNode;
const DataPairNode = expr.DataPairNode;
const ConstrData = expr.ConstrData;
const utils = @import("utils.zig");

const DataTag = std.meta.Tag(Data);

const UplcDataType = [1]u32{
    @intFromEnum(ConstantType.data),
};

inline fn dataTypePtr() [*]const ConstantType {
    return @ptrCast(&UplcDataType);
}

// Runtime-built Data constants reuse this shared type descriptor.
inline fn runtimeDataTypeAddr() usize {
    return @intFromPtr(dataTypePtr());
}

const blst = @cImport({
    @cInclude("blst.h");
    @cInclude("blst_aux.h");
});

const Frame = union(enum(u32)) {
    no_frame,
    frame_await_arg: struct { function: *const Value },
    frame_await_fun_term: struct { env: ?*Env, argument: *const Term },
    frame_await_fun_value: struct { argument: *const Value },
    frame_force,
    frame_constr: struct {
        env: ?*Env,
        tag: u32,
        fields: TermList,
        resolved_fields: ?*LinkedValues,
    },
    frame_case: struct {
        env: ?*Env,
        branches: TermList,
    },
};

pub const Frames = struct {
    frame_ptr: [*]u8,

    const Self = @This();

    pub fn createTestFrames(arena: *std.heap.ArenaAllocator) !Frames {
        const frameMemory = try arena.allocator().alloc(Frame, 1000);
        const framePointer: [*]u8 = @ptrCast(frameMemory);

        return Self{ .frame_ptr = framePointer };
    }

    pub fn createFrames(ptr: u32) Frames {
        const framePointer: [*]u8 = @ptrFromInt(ptr);

        return Self{ .frame_ptr = framePointer };
    }

    pub fn addFrame(self: *Self, frame: *const Frame) void {
        const ptr_bytes: [*]align(4) u8 = @alignCast(self.frame_ptr);
        @memcpy(ptr_bytes, std.mem.asBytes(frame));

        self.frame_ptr = @ptrFromInt(@intFromPtr(self.frame_ptr) + @sizeOf(Frame));
        frame_debug = @intFromPtr(self.frame_ptr);
    }

    pub fn popFrame(self: *Self) Frame {
        self.frame_ptr = @ptrFromInt(@intFromPtr(self.frame_ptr) - @sizeOf(Frame));
        frame_debug = @intFromPtr(self.frame_ptr);

        const frame = std.mem.bytesToValue(Frame, self.frame_ptr);

        return frame;
    }
};

pub export var frame_debug: u32 = 0;
pub export var numer_len_debug: u32 = 0;
pub export var denom_len_debug: u32 = 0;

const ValueList = struct { length: u32, list: [*]*const Value };

const LinkedValues = struct {
    value: *const Value,
    next: ?*const LinkedValues,

    fn create(heap: *Heap, comptime T: type, arg: T, types: *const ConstantType) *LinkedValues {
        const val = createConst(heap, arg.createConstant(types, heap));

        return heap.create(LinkedValues, &.{ .value = val, .next = null });
    }

    fn extend(
        self: *const LinkedValues,
        heap: *Heap,
        comptime T: type,
        arg: T,
        types: *const ConstantType,
    ) *LinkedValues {
        const val = createConst(heap, arg.createConstant(types, heap));

        return heap.create(LinkedValues, &.{ .value = val, .next = self });
    }
};

const Builtin = struct {
    fun: DefaultFunction,
    force_count: u8,
    arity: u8,
    args: ?*LinkedValues,
};

const Value = union(enum(u32)) {
    constant: *const Constant,
    delay: struct {
        env: ?*Env,
        body: *const Term,
    },
    lambda: struct {
        env: ?*Env,
        body: *const Term,
    },
    builtin: Builtin,
    constr: struct { tag: u32, values: ?*const LinkedValues },

    pub fn isUnit(ptr: *const Value) bool {
        switch (ptr.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .unit => return true,
                    else => return false,
                }
            },
            else => return false,
        }
    }

    pub fn unwrapUnit(v: *const Value) void {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .unit => return,
                    else => {
                        utils.printString("Not a unit constant\n");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
            else => {
                utils.printString("Not a constant\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }
    pub fn unwrapConstant(v: *const Value) *const Constant {
        switch (v.*) {
            .constant => |c| {
                return c;
            },
            else => {
                utils.printString("Not a constant\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    pub fn unwrapInteger(v: *const Value) BigInt {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .integer => {
                        return c.bigInt();
                    },
                    else => {
                        utils.printString("Not an integer constant\n");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
            else => {
                utils.printString("Not a constant\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    pub fn unwrapBytestring(v: *const Value) Bytes {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .bytes => {
                        return c.innerBytes();
                    },
                    else => {
                        utils.printString("Not a bytestring constant\n");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
            else => {
                utils.printString("Not a constant\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    pub fn unwrapString(v: *const Value) String {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .string => {
                        return c.string();
                    },
                    else => {
                        utils.printString("Not a string constant\n");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
            else => {
                utils.printString("Not a constant\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    pub fn unwrapBool(v: *const Value) bool {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .boolean => {
                        return c.bln();
                    },
                    else => {
                        utils.printString("Not a boolean constant\n");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
            else => {
                utils.printString("Not a constant\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    pub fn unwrapList(v: *const Value) List {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .list => {
                        return c.list();
                    },
                    else => {
                        utils.printString("Not a list constant\n");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
            else => {
                utils.printString("Not a constant\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    pub fn unwrapG1(v: *const Value) G1Element {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .bls12_381_g1_element => {
                        return c.g1Element();
                    },
                    else => {
                        utils.printString("Not a G1 element\\n");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
            else => {
                utils.printString("Not a constant\\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    pub fn unwrapG2(v: *const Value) G2Element {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .bls12_381_g2_element => {
                        return c.g2Element();
                    },
                    else => {
                        utils.printString("Not a G2 element\\n");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
            else => {
                utils.printString("Not a constant\\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    pub fn unwrapMlResult(v: *const Value) MlResult {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .bls12_381_mlresult => {
                        return c.mlResult();
                    },
                    else => {
                        utils.printString("Not a Miller Loop Result\\n");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
            else => {
                utils.printString("Not a constant\\n");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }
};

pub const Env = struct {
    value: *const Value,
    next: ?*Env,

    const Self = @This();

    pub fn init(v: *const Value, heap: *Heap) *Self {
        return heap.create(Env, &.{ .value = v, .next = null });
    }

    pub fn preprend(self: *Self, v: *const Value, heap: *Heap) *Self {
        return heap.create(Env, &.{ .value = v, .next = self });
    }

    pub fn lookupVar(self: *Self, idx: u32) *const Value {
        var cur: ?*Self = self;
        var i = idx - 1;

        while (cur) |n| : (i -= 1) {
            if (i == 0) {
                return n.value;
            }
            cur = n.next;
        }
        unreachable;
    }

    test "init" {
        var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer allocator.deinit();

        var heap = try Heap.createTestHeap(&allocator);

        const t = Term.terror;
        const v = createDelay(&heap, null, &t);

        const env = Env.init(v, &heap);

        try testing.expectEqualDeep(env.value, v);
        try testing.expect(env.next == null);
    }

    test "lookup" {
        var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer allocator.deinit();

        var heap = try Heap.createTestHeap(&allocator);

        const t = Term.terror;
        const v = createDelay(&heap, null, &t);

        const v2 = createLambda(&heap, null, &t);

        var env = Env.init(v, &heap);
        env = env.preprend(v2, &heap);

        const value = env.lookupVar(2);

        try testing.expectEqualDeep(value, v);
    }
};

pub fn createConst(heap: *Heap, c: *Constant) *Value {
    return heap.create(
        Value,
        &.{ .constant = c },
    );
}

pub fn createDelay(heap: *Heap, env: ?*Env, b: *const Term) *Value {
    return heap.create(
        Value,
        &.{
            .delay = .{
                .env = env,
                .body = b,
            },
        },
    );
}

pub fn createLambda(heap: *Heap, env: ?*Env, b: *const Term) *Value {
    return heap.create(
        Value,
        &.{
            .lambda = .{
                .env = env,
                .body = b,
            },
        },
    );
}

pub fn createBuiltin(heap: *Heap, f: DefaultFunction) *Value {
    return heap.create(Value, &.{
        .builtin = .{
            .fun = f,
            .force_count = f.forceCount(),
            .arity = f.arity(),
            .args = null,
        },
    });
}

pub fn forceBuiltin(heap: *Heap, b: *const Builtin) *Value {
    return heap.create(Value, &.{
        .builtin = .{
            .fun = b.fun,
            .force_count = b.force_count - 1,
            .arity = b.arity,
            .args = null,
        },
    });
}

pub fn createConstr(heap: *Heap, tag: u32, vls: ?*LinkedValues) *Value {
    return heap.create(
        Value,
        &Value{
            .constr = .{ .tag = tag, .values = vls },
        },
    );
}

pub const State = union(enum(u32)) {
    compute: struct {
        env: ?*Env,
        term: *const Term,
    },
    ret: struct {
        value: *const Value,
    },
    done: *const Value,
};

// The host consumes the same "unpacked" layout that the runtime works with
// (one byte per u32 word for byte-oriented constants), so no conversion is
// currently required before handing results back.
fn prepareValueForHost(value: *const Value) void {
    switch (value.*) {
        .constant => |c| packConstantForHost(c),
        else => {},
    }
}

fn packConstantForHost(constant: *const Constant) void {
    _ = constant;
    // Left intentionally blank; byte/string constants already match the host layout.
}

const empty_bytes = [_]u8{0};

const PackedBytes = struct {
    ptr: [*]const u8,
    len: usize,
};

fn materializeBytes(heap: *Heap, source: Bytes) PackedBytes {
    if (source.length == 0) {
        return .{
            .ptr = empty_bytes[0..].ptr,
            .len = 0,
        };
    }

    const byte_len: usize = @intCast(source.length);
    const word_len: u32 = @intCast((byte_len + 3) / 4);
    const buffer_words = heap.createArray(u32, word_len);
    const buffer_bytes: [*]u8 = @ptrCast(buffer_words);

    var i: usize = 0;
    while (i < byte_len) : (i += 1) {
        buffer_bytes[i] = @truncate(source.bytes[i]);
    }

    // Zero any padding bytes (at most three) so future reads see deterministic data.
    var pad_index = byte_len;
    const padded_len = @as(usize, word_len) * @sizeOf(u32);
    while (pad_index < padded_len) : (pad_index += 1) {
        buffer_bytes[pad_index] = 0;
    }

    return .{
        .ptr = buffer_bytes,
        .len = byte_len,
    };
}

pub const builtinFunctions = [_]*const fn (*Machine, *LinkedValues) *const Value{
    &addInteger,
    &subInteger,
    &multiplyInteger,
    &divideInteger,
    &quotientInteger,
    &remainderInteger,
    &modInteger,
    &equalsInteger,
    &lessThanInteger,
    &lessThanEqualsInteger,
    &appendByteString,
    &consByteString,
    &sliceByteString,
    &lengthOfByteString,
    &indexByteString,
    &equalsByteString,
    &lessThanByteString,
    &lessThanEqualsByteString,
    &sha2_256,
    &sha3_256,
    &blake2b_256,
    &verifyEd25519Signature,
    &appendString,
    &equalsString,
    &encodeUtf8,
    &decodeUtf8,
    &ifThenElse,
    &chooseUnit,
    &trace,
    &fstPair,
    &sndPair,
    &chooseList,
    &mkCons,
    &headList,
    &tailList,
    &nullList,
    &chooseData,
    &constrData,
    &mapData,
    &listData,
    &iData,
    &bData,
    &unConstrData,
    &unMapData,
    &unListData,
    &unIData,
    &unBData,
    &equalsData,
    &mkPairData,
    &mkNilData,
    &mkNilPairData,
    &serialiseData,
    &verifyEcdsaSecp256k1Signature,
    &verifySchnorrSecp256k1Signature,
    &bls12_381_G1_Add,
    &bls12_381_G1_Neg,
    &bls12_381_G1_ScalarMul,
    &bls12_381_G1_Equal,
    &bls12_381_G1_Compress,
    &bls12_381_G1_Uncompress,
    &bls12_381_G1_HashToGroup,
    &bls12_381_G2_Add,
    &bls12_381_G2_Neg,
    &bls12_381_G2_ScalarMul,
    &bls12_381_G2_Equal,
    &bls12_381_G2_Compress,
    &bls12_381_G2_Uncompress,
    &bls12_381_G2_HashToGroup,
    &bls12_381_MillerLoop,
    &bls12_381_MulMlResult,
    &bls12_381_FinalVerify,
    &keccak_256,
    &blake2b_224,
    &integerToByteString,
    &byteStringToInteger,
    &andByteString,
    &orByteString,
    &xorByteString,
    &complementByteString,
    &readBit,
    &writeBits,
    &replicateByte,
    &shiftByteString,
    &rotateByteString,
    &countSetBits,
    &findFirstSetBit,
    &ripemd_160,
};

pub fn addInteger(m: *Machine, args: *LinkedValues) *const Value {
    const y = args.value.unwrapInteger();

    const x = args.next.?.value.unwrapInteger();

    if (x.sign == y.sign) {
        return addSignedIntegers(m, x, y);
    } else {
        return subSignedIntegers(m, x, y);
    }
}

pub fn subInteger(m: *Machine, args: *LinkedValues) *const Value {
    var y = args.value.unwrapInteger();
    y.sign ^= 1;

    const x = args.next.?.value.unwrapInteger();

    if (x.sign == y.sign) {
        return addSignedIntegers(m, x, y);
    } else {
        return subSignedIntegers(m, x, y);
    }
}

pub fn multiplyInteger(m: *Machine, args: *const LinkedValues) *const Value {
    const b = args.value.unwrapInteger();

    const a = args.next.?.value.unwrapInteger();

    const result_sign: u32 = a.sign ^ b.sign;

    const a_words = a.words[0..a.length];
    const b_words = b.words[0..b.length];

    // Allocate space for the worst-case length: |a| + |b|
    const max_len = a.length + b.length;
    const resultPtr = m.heap.createArray(u32, max_len + 5); // bump-allocate
    var result = resultPtr + 5;

    // The buffer comes back with arbitrary bytes ‑ clear it so that the
    // length‑trimming logic sees genuine zeroes in the untouched limbs.
    @memset(result[0..max_len], 0);

    // // Core multiplication loop (32-bit words split into 16-bit halves)
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
            var carry: u64 = (mid >> 16) + high; // carry ≥ 0, < 2³²

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

fn computeQuotient(
    m: *Machine,
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
    m: *Machine,
    numer: BigInt,
    denom: BigInt,
    mode: DivisionMode,
) *const Value {
    if (remainderBySingleLimb(m, numer, denom, mode)) |value| {
        return value;
    }
    return remainderMultiLimb(m, numer, denom, mode);
}

fn finalizeRemainderValue(
    m: *Machine,
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
    m: *Machine,
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

fn handleSmallerMagnitude(
    m: *Machine,
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
    m: *Machine,
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
    m: *Machine,
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

fn createIntegerValueFromLimbs(
    m: *Machine,
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

fn quotientBySingleLimb(
    m: *Machine,
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
    m: *Machine,
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

pub fn divideInteger(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn quotientInteger(m: *Machine, args: *LinkedValues) *const Value {
    const d = args.value.unwrapInteger();
    const n = args.next.?.value.unwrapInteger();

    numer_len_debug = n.length;
    denom_len_debug = d.length;

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

pub fn remainderInteger(m: *Machine, args: *LinkedValues) *const Value {
    const d = args.value.unwrapInteger();
    const n = args.next.?.value.unwrapInteger();

    numer_len_debug = n.length;
    denom_len_debug = d.length;

    if (bigIntIsZero(d)) {
        builtinEvaluationFailure();
    }

    if (bigIntIsZero(n)) {
        const zero_limbs = [_]u32{0};
        return createIntegerValueFromLimbs(m, zero_limbs[0..1], true);
    }

    return computeRemainder(m, n, d, .trunc);
}

pub fn modInteger(m: *Machine, args: *LinkedValues) *const Value {
    const d = args.value.unwrapInteger();
    const n = args.next.?.value.unwrapInteger();

    numer_len_debug = n.length;
    denom_len_debug = d.length;

    if (bigIntIsZero(d)) {
        builtinEvaluationFailure();
    }

    if (bigIntIsZero(n)) {
        const zero_limbs = [_]u32{0};
        return createIntegerValueFromLimbs(m, zero_limbs[0..1], true);
    }

    return computeRemainder(m, n, d, .floor);
}

pub fn equalsInteger(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn lessThanInteger(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn lessThanEqualsInteger(m: *Machine, args: *LinkedValues) *const Value {
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

// ByteString functions
pub fn appendByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn consByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn sliceByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn lengthOfByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn indexByteString(m: *Machine, args: *LinkedValues) *const Value {
    const y = args.value.unwrapInteger();

    const x = args.next.?.value.unwrapBytestring();

    var result = m.heap.createArray(u32, 6);

    // Must be at least 1 byte, y < max(u32), y >= 0, y < x.length
    if (x.length == 0 or y.length > 1 or y.sign == 1 or y.words[0] > x.length - 1) {
        utils.printString("Integer larger than bytestring length or negative");
        utils.exit(std.math.maxInt(u32));
    }

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.integerType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 0;
    result[4] = 1;
    // This will work due to above check
    result[5] = x.bytes[y.words[0]];

    return createConst(m.heap, @ptrCast(result));
}

pub fn equalsByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn lessThanByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn lessThanEqualsByteString(m: *Machine, args: *LinkedValues) *const Value {
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

// Cryptography and hash functions
pub fn sha2_256(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn sha3_256(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn blake2b_256(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn verifyEd25519Signature(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

const StringBytes = struct {
    byte_len: u32,
    is_packed: bool,
};

fn analyzeString(str: String) StringBytes {
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

const ByteWriter = struct {
    dst: [*]u32, // dest words, one byte per word (unpacked form)
    index: u32 = 0,

    fn init(dst: [*]u32) ByteWriter {
        return .{ .dst = dst };
    }

    fn write(self: *ByteWriter, value: u8) void {
        self.dst[self.index] = value;
        self.index += 1;
    }
};

fn appendStringBytes(writer: *ByteWriter, str: String, view: StringBytes) void {
    if (view.byte_len == 0) return;

    if (!view.is_packed) {
        var i: u32 = 0;
        while (i < view.byte_len) : (i += 1) {
            writer.write(@truncate(str.bytes[i]));
        }
        return;
    }

    var remaining = view.byte_len;
    var word_idx: u32 = 0;
    var byte_idx: u2 = 0;

    while (remaining > 0 and word_idx < str.length) {
        const word = str.bytes[word_idx];
        const shift: u5 = @intCast(@as(u32, byte_idx) * 8);
        const byte_val = @as(u8, @truncate((word >> shift) & 0xFF));
        writer.write(byte_val);

        remaining -= 1;
        byte_idx += 1;
        if (byte_idx == 4) {
            byte_idx = 0;
            word_idx += 1;
        }
    }
}

fn extractStringByte(str: String, view: StringBytes, byte_index: u32) u8 {
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
pub fn appendString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn equalsString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn encodeUtf8(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn decodeUtf8(m: *Machine, args: *LinkedValues) *const Value {
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

// Bool function
pub fn ifThenElse(_: *Machine, args: *LinkedValues) *const Value {
    const otherwise = args.value;
    const then = args.next.?.value;
    const cond = args.next.?.next.?.value.unwrapBool();

    if (cond) {
        return then;
    } else {
        return otherwise;
    }
}

// Unit function
// Our LinkedValues list stores the most-recent argument at the head, so for
// `chooseUnit` the branch argument sits in `args.value` and the unit value is
// at `args.next?.value`.
pub fn chooseUnit(_: *Machine, args: *LinkedValues) *const Value {
    const then = args.value;
    const unit_node = args.next orelse {
        utils.printlnString("chooseUnit expects a unit argument");
        utils.exit(std.math.maxInt(u32));
    };

    // The scrutinee must still be an actual unit value, even though the branch
    // result is returned unchanged.  `unwrapUnit` enforces that without forcing
    // any structure on the branch argument itself (which may be non-constant).
    _ = unit_node.value.unwrapUnit();

    return then;
}

// Tracing function
pub fn trace(_: *Machine, args: *LinkedValues) *const Value {
    const then = args.value;
    const msg = args.next.?.value.unwrapString();

    var i: u32 = 0;
    while (i < msg.length) : (i += 1) {
        utils.printChar(@truncate(msg.bytes[i]));
    }

    return then;
}

// Pairs functions
pub fn fstPair(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn sndPair(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// List functions
// In Plutus the builtin receives arguments in the order (list, nil_branch, cons_branch),
// but our LinkedValues list is reversed (last applied argument first), so `args.value`
// corresponds to the cons branch and `args.next?.value` to the nil branch.
pub fn chooseList(_: *Machine, args: *LinkedValues) *const Value {
    const otherwise = args.value;

    const then = args.next.?.value;

    const list = args.next.?.next.?.value.unwrapList();

    if (list.length > 0) {
        return otherwise;
    } else {
        return then;
    }
}

pub fn mkCons(m: *Machine, args: *LinkedValues) *const Value {
    const y = args.value;

    const list = y.unwrapList();

    const yType = y.constant;

    const newItem = args.next.?.value.unwrapConstant();

    if (newItem.matchingTypes(list.inner_type, list.type_length)) {
        const prev = if (list.length > 0) blk: {
            break :blk list.items.?;
        } else blk: {
            break :blk null;
        };

        const node = m.heap.create(ListNode, &ListNode{ .value = newItem.rawValue(), .next = prev });

        // (Type length + 1) * 4 bytes + 4 byte to hold type length + 4 byte for list length + 4 byte for pointer to first list item (or null)
        var result = m.heap.createArray(u32, 5);

        result[0] = yType.length;
        result[1] = @intFromPtr(yType.type_list);
        result[2] = @intFromPtr(result + 3);
        result[3] = list.length + 1;
        result[4] = @intFromPtr(node);

        return createConst(m.heap, @ptrCast(result));
    } else {
        utils.printlnString("item does not match list type");
        utils.exit(std.math.maxInt(u32));
    }
}

pub fn headList(m: *Machine, args: *LinkedValues) *const Value {
    const y = args.value;

    const list = y.unwrapList();

    if (list.length == 0) {
        // Per the spec, headList is partial and must signal an evaluation failure on empty lists.
        builtinEvaluationFailure();
    }

    const c = Constant{
        .length = list.type_length,
        .type_list = list.inner_type,
        .value = list.items.?.value,
    };

    const con = m.heap.create(Constant, &c);

    return createConst(m.heap, con);
}

pub fn tailList(m: *Machine, args: *LinkedValues) *const Value {
    const y = args.value;

    const list = y.unwrapList();

    if (list.length > 0) {
        const result = m.heap.createArray(u32, 2);
        result[0] = list.length - 1;
        result[1] = @intFromPtr(list.items.?.next);

        const c = Constant{
            .length = list.type_length,
            .type_list = list.inner_type,
            .value = @intFromPtr(result),
        };

        const con = m.heap.create(Constant, &c);

        return createConst(m.heap, con);
    } else {
        utils.printlnString("called tailList on an empty list");
        utils.exit(std.math.maxInt(u32));
    }
}

pub fn nullList(m: *Machine, args: *LinkedValues) *const Value {
    const list = args.value.unwrapList();

    var result = m.heap.createArray(u32, 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(list.length == 0);

    return createConst(m.heap, @ptrCast(result));
}

// Data functions
pub fn chooseData(_: *Machine, args: *LinkedValues) *const Value {
    const bytes_branch = args.value;
    const int_branch = args.next.?.value;
    const list_branch = args.next.?.next.?.value;
    const map_branch = args.next.?.next.?.next.?.value;
    const constr_branch = args.next.?.next.?.next.?.next.?.value;
    const data_value = args.next.?.next.?.next.?.next.?.next.?.value.unwrapConstant();

    const variant = decodeDataVariant(data_value);

    return switch (variant) {
        .constr => constr_branch,
        .map => map_branch,
        .list => list_branch,
        .integer => int_branch,
        .bytes => bytes_branch,
    };
}

const serialized_data_const_tag: u32 = 0x05; // Mirrors serializer/constants.rs::const_tag::DATA

fn decodeDataVariant(con: *const Constant) DataTag {
    const raw_ptr = con.rawValue();
    if (raw_ptr == 0) {
        utils.printlnString("null Data pointer");
        utils.exit(std.math.maxInt(u32));
    }

    const uses_runtime_layout = @intFromPtr(con.type_list) == runtimeDataTypeAddr();
    // Serialized constants keep an embedded payload instead of referencing the shared type.
    if (!uses_runtime_layout) {
        if (con.length != serialized_data_const_tag) {
            utils.printlnString("chooseData expects a Data constant");
            utils.exit(std.math.maxInt(u32));
        }

        // Serialized constants store the Data payload inline after the header.
        return readSerializedDataTag(raw_ptr);
    }

    const ty = con.constType().*;
    if (ty != .data) {
        utils.printlnString("chooseData expects a Data constant");
        utils.exit(std.math.maxInt(u32));
    }

    return readHeapDataTag(raw_ptr);
}

fn readSerializedDataTag(raw_ptr: u32) DataTag {
    if (raw_ptr == 0) {
        utils.printlnString("null Data pointer");
        utils.exit(std.math.maxInt(u32));
    }

    // Serialized constants place the variant tag at the start of the payload the
    // Constant points to, so we can read it directly.
    const words: [*]const u32 = @ptrFromInt(raw_ptr);
    return tagFromWord(words[0]);
}

fn readHeapDataTag(raw_ptr: u32) DataTag {
    if (raw_ptr == 0) {
        utils.printlnString("null Data pointer");
        utils.exit(std.math.maxInt(u32));
    }

    const heap_data: *const Data = @ptrFromInt(raw_ptr);
    return std.meta.activeTag(heap_data.*);
}

fn tagFromWord(raw: u32) DataTag {
    return std.meta.intToEnum(DataTag, raw) catch {
        utils.printlnString("invalid Data tag");
        utils.exit(std.math.maxInt(u32));
    };
}

pub fn constrData(m: *Machine, args: *LinkedValues) *const Value {
    // First arg: list of Data (fields)
    const fields_list = args.value.unwrapList();

    // Second arg: integer (constructor tag)
    const tag_int = args.next.?.value.unwrapInteger();

    // Convert integer to u32 tag
    if (tag_int.sign == 1 or tag_int.length > 1) {
        utils.printlnString("constrData: tag must be a non-negative integer that fits in u32");
        utils.exit(std.math.maxInt(u32));
    }
    const tag: u32 = tag_int.words[0];

    // Convert List<Data> to DataListNode chain
    var data_list_head: ?*DataListNode = null;
    var current_list_node = fields_list.items;

    // Build the DataListNode chain in reverse by prepending
    while (current_list_node) |node| {
        // Each ListNode.value is a u32 pointer to a Constant
        // The Constant.value is a pointer to the Data
        const const_ptr: *const Constant = @ptrFromInt(node.value);
        const data_ptr: *const Data = @ptrFromInt(const_ptr.value);

        // Create new DataListNode
        const data_node = m.heap.create(DataListNode, &DataListNode{
            .value = data_ptr,
            .next = data_list_head,
        });
        data_list_head = data_node;

        current_list_node = node.next;
    }

    // Reverse the list to maintain original order
    var reversed_head: ?*DataListNode = null;
    var current = data_list_head;
    while (current) |node| {
        const next = node.next;
        const new_node = m.heap.create(DataListNode, &DataListNode{
            .value = node.value,
            .next = reversed_head,
        });
        reversed_head = new_node;
        current = next;
    }

    // Create ConstrData
    const constr_payload = ConstrData{
        .tag = tag,
        .fields = reversed_head,
    };

    // Create Data union with constr variant
    const data = Data{ .constr = constr_payload };
    const data_ptr = m.heap.create(Data, &data);

    // Create Constant wrapping the Data
    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn mapData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn listData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn iData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bData(m: *Machine, args: *LinkedValues) *const Value {
    const bytes = args.value.unwrapBytestring();

    // ByteStrings are immutable, so it is safe to share their backing buffer with Data.
    const payload = Data{ .bytes = bytes };
    const data_ptr = m.heap.create(Data, &payload);

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn unConstrData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn unMapData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn unListData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn unIData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn unBData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn equalsData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Misc constructors
pub fn mkPairData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn mkNilData(m: *Machine, args: *LinkedValues) *const Value {
    const unit_arg = args.value;

    // mkNilData takes a unit argument to fix its polymorphic type.
    unit_arg.unwrapUnit();

    var result = m.heap.createArray(u32, 5);
    result[0] = 2;
    result[1] = @intFromPtr(ConstantType.listDataType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 0;
    result[4] = 0;

    return createConst(m.heap, @ptrCast(result));
}

pub fn mkNilPairData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn serialiseData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn verifyEcdsaSecp256k1Signature(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn verifySchnorrSecp256k1Signature(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// BLS Builtins

fn integerToScalarBytes(bi: BigInt) [32]u8 {
    const scalar = integerToBlstScalar(bi);
    return scalar.b;
}

fn integerToBlstScalar(bi: BigInt) blst.blst_scalar {
    // Reduce the (possibly huge or negative) integer modulo the scalar field
    // using BLST's wide-input helpers so we match Plutus semantics.
    var scalar = std.mem.zeroes(blst.blst_scalar);

    const word_len: usize = @intCast(bi.length);
    if (word_len != 0) {
        const words = bi.words[0..word_len];
        const magnitude = std.mem.sliceAsBytes(words);
        _ = blst.blst_scalar_from_le_bytes(&scalar, magnitude.ptr, magnitude.len);
    }

    if (bi.sign != 0) {
        var tmp_fr: blst.blst_fr = undefined;
        blst.blst_fr_from_scalar(&tmp_fr, &scalar);
        blst.blst_fr_cneg(&tmp_fr, &tmp_fr, true);
        blst.blst_scalar_from_fr(&scalar, &tmp_fr);
    }

    return scalar;
}

const bls12_381_fp_modulus_be = [_]u8{
    0x1a, 0x01, 0x11, 0xea, 0x39, 0x7f, 0xe6, 0x9a,
    0x4b, 0x1b, 0xa7, 0xb6, 0x43, 0x4b, 0xac, 0xd7,
    0x64, 0x77, 0x4b, 0x84, 0xf3, 0x85, 0x12, 0xbf,
    0x67, 0x30, 0xd2, 0xa0, 0xf6, 0xb0, 0xf6, 0x24,
    0x1e, 0xab, 0xff, 0xfe, 0xb1, 0x53, 0xff, 0xff,
    0xb9, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xaa, 0xab,
};

fn decodeFpFromBendian(dst: *blst.blst_fp, chunk: [*]const u8) bool {
    blst.blst_fp_from_bendian(dst, chunk);
    // Enforce canonical encoding because blst_fp_from_bendian never fails.
    return std.mem.lessThan(u8, chunk[0..48], bls12_381_fp_modulus_be[0..]);
}

fn blst_fp12_from_bendian(ret: *blst.blst_fp12, bytes: [*]const u8) c_int {
    var offset: usize = 0;
    // Inverse of blst_bendian_from_fp12: iterate fp2 limbs first and interleave fp6 components
    // so we consume the 12 field elements in the same order they were emitted.
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        var j: usize = 0;
        while (j < 2) : (j += 1) {
            if (!decodeFpFromBendian(&ret.fp6[j].fp2[i].fp[0], bytes + offset))
                return blst.BLST_BAD_ENCODING;
            offset += 48;
            if (!decodeFpFromBendian(&ret.fp6[j].fp2[i].fp[1], bytes + offset))
                return blst.BLST_BAD_ENCODING;
            offset += 48;
        }
    }

    // Miller loop outputs are not guaranteed to lie in the final exponent subgroup,
    // so we only validate per-coordinate decoding here.
    return blst.BLST_SUCCESS;
}

pub fn bls12_381_G1_Add(m: *Machine, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG1();
    const p = args.next.?.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);
    var q_bytes: [48]u8 = undefined;
    @memcpy(&q_bytes, q.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);
    var aff_q: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_q: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_q, &aff_q);

    var point_r: blst.blst_p1 = undefined;
    blst.blst_p1_add(&point_r, &point_p, &point_q);

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p1_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G1_Neg(m: *Machine, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    blst.blst_p1_cneg(&point_p, true);

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p1_compress(out_bytes, &point_p);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G1_ScalarMul(m: *Machine, args: *LinkedValues) *const Value {
    // Spec orders arguments as (G1, scalar); the newest (scalar) is in the tail.
    const p = args.value.unwrapG1();
    const scalar = args.next.?.value.unwrapInteger();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    const scalar_words = integerToBlstScalar(scalar);

    var point_r: blst.blst_p1 = undefined;
    blst.blst_p1_mult(
        &point_r,
        &point_p,
        &scalar_words.b,
        @sizeOf(blst.blst_scalar) * 8,
    );

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p1_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G1_Equal(m: *Machine, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG1();
    const p = args.next.?.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);
    var q_bytes: [48]u8 = undefined;
    @memcpy(&q_bytes, q.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    var aff_q: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_q: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_q, &aff_q);

    const equal = blst.blst_p1_is_equal(&point_p, &point_q);

    var result = m.heap.createArray(u32, 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(equal);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G1_Compress(m: *Machine, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..48]);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    var out_bytes: [48]u8 = undefined;
    blst.blst_p1_compress(&out_bytes, &point_p);

    var result = m.heap.createArray(u32, 52);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 48;
    for (0..48) |i| {
        result[4 + i] = out_bytes[i];
    }
    return createConst(m.heap, @ptrCast(result));
}

inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
}

// ByteStrings are stored with one byte value per u32 word (unpacked format).
// This function extracts them into a continuous byte array for BLST library calls.
fn unpackWordPackedBytes(comptime expected_len: usize, bs: Bytes, out: *[expected_len]u8) bool {
    const byte_count: usize = @intCast(bs.length);
    if (byte_count != expected_len) {
        return false;
    }

    var i: usize = 0;
    while (i < byte_count) : (i += 1) {
        out[i] = @truncate(bs.bytes[i]);
    }
    return true;
}

pub fn bls12_381_G1_Uncompress(m: *Machine, args: *LinkedValues) *const Value {
    const bs = args.value.unwrapBytestring();
    var in_bytes: [48]u8 = undefined;
    if (!unpackWordPackedBytes(48, bs, &in_bytes)) {
        builtinEvaluationFailure();
    }

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &in_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);
    if (!blst.blst_p1_in_g1(&point_p)) {
        builtinEvaluationFailure();
    }

    var canonical: [48]u8 = undefined;
    blst.blst_p1_compress(&canonical, &point_p);
    if (!std.mem.eql(u8, canonical[0..], in_bytes[0..])) {
        // Input encoding must round-trip through compression.
        builtinEvaluationFailure();
    }

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    @memcpy(out_bytes[0..48], canonical[0..]);
    return createConst(m.heap, @ptrCast(result));
}
// Alias kept so callers using the spec's "decompress" name still resolve.
pub const bls12_381_G1_Decompress = bls12_381_G1_Uncompress;

pub fn bls12_381_G1_HashToGroup(m: *Machine, args: *LinkedValues) *const Value {
    const dst = args.value.unwrapBytestring();
    const msg = args.next.?.value.unwrapBytestring();

    const dst_packed = materializeBytes(m.heap, dst);
    const msg_packed = materializeBytes(m.heap, msg);
    if (dst_packed.len > 255) {
        // PLT spec enforces DST <= 255 bytes.
        builtinEvaluationFailure();
    }

    var point_r: blst.blst_p1 = undefined;
    blst.blst_hash_to_g1(&point_r, msg_packed.ptr, msg_packed.len, dst_packed.ptr, dst_packed.len, null, 0);

    var result = m.heap.createArray(u32, 15);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g1ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p1_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Add(m: *Machine, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG2();
    const p = args.next.?.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);
    var q_bytes: [96]u8 = undefined;
    @memcpy(&q_bytes, q.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var aff_q: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_q: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_q, &aff_q);

    var point_r: blst.blst_p2 = undefined;
    blst.blst_p2_add(&point_r, &point_p, &point_q);

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p2_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Neg(m: *Machine, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    blst.blst_p2_cneg(&point_p, true);

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p2_compress(out_bytes, &point_p);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_ScalarMul(m: *Machine, args: *LinkedValues) *const Value {
    // Spec orders arguments as (G2, scalar); the newest (scalar) is in the tail.
    const p = args.value.unwrapG2();
    const scalar = args.next.?.value.unwrapInteger();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    const scalar_words = integerToBlstScalar(scalar);

    var point_r: blst.blst_p2 = undefined;
    blst.blst_p2_mult(
        &point_r,
        &point_p,
        &scalar_words.b,
        @sizeOf(blst.blst_scalar) * 8,
    );

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p2_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Equal(m: *Machine, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG2();
    const p = args.next.?.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);
    var q_bytes: [96]u8 = undefined;
    @memcpy(&q_bytes, q.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var aff_q: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_q: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_q, &aff_q);

    const equal = blst.blst_p2_is_equal(&point_p, &point_q);

    var result = m.heap.createArray(u32, 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(equal);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Compress(m: *Machine, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes[0..96]);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var out_bytes: [96]u8 = undefined;
    blst.blst_p2_compress(&out_bytes, &point_p);

    var result = m.heap.createArray(u32, 100);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 96;
    for (0..96) |i| {
        result[4 + i] = out_bytes[i];
    }
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Uncompress(m: *Machine, args: *LinkedValues) *const Value {
    const bs = args.value.unwrapBytestring();
    var in_bytes: [96]u8 = undefined;
    if (!unpackWordPackedBytes(96, bs, &in_bytes)) {
        builtinEvaluationFailure();
    }

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &in_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);
    if (!blst.blst_p2_in_g2(&point_p)) {
        builtinEvaluationFailure();
    }

    var canonical: [96]u8 = undefined;
    blst.blst_p2_compress(&canonical, &point_p);
    if (!std.mem.eql(u8, canonical[0..], in_bytes[0..])) {
        builtinEvaluationFailure();
    }

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    @memcpy(out_bytes[0..96], canonical[0..]);
    return createConst(m.heap, @ptrCast(result));
}
// Alias kept so callers using the spec's "decompress" name still resolve.
pub const bls12_381_G2_Decompress = bls12_381_G2_Uncompress;

pub fn bls12_381_G2_HashToGroup(m: *Machine, args: *LinkedValues) *const Value {
    const dst = args.value.unwrapBytestring();
    const msg = args.next.?.value.unwrapBytestring();

    const dst_packed = materializeBytes(m.heap, dst);
    const msg_packed = materializeBytes(m.heap, msg);
    if (dst_packed.len > 255) {
        builtinEvaluationFailure();
    }

    var point_r: blst.blst_p2 = undefined;
    blst.blst_hash_to_g2(&point_r, msg_packed.ptr, msg_packed.len, dst_packed.ptr, dst_packed.len, null, 0);

    var result = m.heap.createArray(u32, 27);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.g2ElementType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_p2_compress(out_bytes, &point_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_MillerLoop(m: *Machine, args: *LinkedValues) *const Value {
    // Spec orders arguments as (G1, G2); the newest (G2) sits at the list head.
    const g2 = args.value.unwrapG2();
    const g1 = args.next.?.value.unwrapG1();

    var g1_bytes: [48]u8 = undefined;
    @memcpy(&g1_bytes, g1.bytes[0..48]);
    var g2_bytes: [96]u8 = undefined;
    @memcpy(&g2_bytes, g2.bytes[0..96]);

    var aff_g1: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_g1, &g1_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var chk_g1: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&chk_g1, &aff_g1);
    if (!blst.blst_p1_in_g1(&chk_g1)) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }

    var aff_g2: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_g2, &g2_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var chk_g2: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&chk_g2, &aff_g2);
    if (!blst.blst_p2_in_g2(&chk_g2)) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }

    var ml: blst.blst_fp12 = undefined;
    blst.blst_miller_loop(&ml, &aff_g2, &aff_g1);

    var result = m.heap.createArray(u32, 147);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.mlResultType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_bendian_from_fp12(out_bytes, &ml);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_MulMlResult(m: *Machine, args: *LinkedValues) *const Value {
    const b = args.value.unwrapMlResult();
    const a = args.next.?.value.unwrapMlResult();

    var a_bytes: [576]u8 = undefined;
    @memcpy(&a_bytes, a.bytes[0..576]);
    var b_bytes: [576]u8 = undefined;
    @memcpy(&b_bytes, b.bytes[0..576]);

    var fp_a: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_a, &a_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid MlResult\n");
        utils.exit(std.math.maxInt(u32));
    }

    var fp_b: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_b, &b_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid MlResult\n");
        utils.exit(std.math.maxInt(u32));
    }

    var fp_r: blst.blst_fp12 = undefined;
    blst.blst_fp12_mul(&fp_r, &fp_a, &fp_b);

    var result = m.heap.createArray(u32, 147);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.mlResultType());
    result[2] = @intFromPtr(result + 3);
    const out_bytes: [*]u8 = @ptrCast(result + 3);
    blst.blst_bendian_from_fp12(out_bytes, &fp_r);
    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_FinalVerify(m: *Machine, args: *LinkedValues) *const Value {
    const ml2 = args.value.unwrapMlResult();
    const ml1 = args.next.?.value.unwrapMlResult();

    var ml1_bytes: [576]u8 = undefined;
    @memcpy(&ml1_bytes, ml1.bytes[0..576]);
    var ml2_bytes: [576]u8 = undefined;
    @memcpy(&ml2_bytes, ml2.bytes[0..576]);

    var fp_ml1: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_ml1, &ml1_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid MlResult\n");
        utils.exit(std.math.maxInt(u32));
    }

    var fp_ml2: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_ml2, &ml2_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid MlResult\n");
        utils.exit(std.math.maxInt(u32));
    }

    const res = blst.blst_fp12_finalverify(&fp_ml1, &fp_ml2);

    var result = m.heap.createArray(u32, 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(res);
    return createConst(m.heap, @ptrCast(result));
}

pub fn keccak_256(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn blake2b_224(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Conversions
pub fn integerToByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn byteStringToInteger(m: *Machine, args: *LinkedValues) *const Value {
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
pub fn andByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn orByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn xorByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn complementByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn readBit(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn writeBits(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn replicateByte(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Bitwise
pub fn shiftByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn rotateByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn countSetBits(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn findFirstSetBit(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Ripemd_160
pub fn ripemd_160(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn addSignedIntegers(
    m: *Machine,
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
    m: *Machine,
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

pub const Machine = struct {
    heap: *Heap,
    frames: *Frames,

    const Self = @This();

    pub fn init(heap: *Heap, frames: *Frames) Self {
        return Self{
            .heap = heap,
            .frames = frames,
        };
    }

    pub fn runValidator(self: *Self, t: *const Term) void {
        self.frames.addFrame(&.no_frame);

        var state = State{
            .compute = .{
                .env = null,
                .term = t,
            },
        };

        while (true) {
            switch (state) {
                .compute => |c| {
                    state = self.compute(c.env, c.term);
                },
                .ret => |r| {
                    state = self.ret(r.value);
                },
                .done => |d| {
                    if (d.isUnit()) {
                        return;
                    } else {
                        utils.printString("Returned term other than unit\n");
                        utils.exit(std.math.maxInt(u32));
                    }
                },
            }
        }
    }

    pub fn runFunction(self: *Self, t: *const Term) *const Value {
        self.frames.addFrame(&.no_frame);

        var state = State{
            .compute = .{
                .env = null,
                .term = t,
            },
        };

        while (true) {
            switch (state) {
                .compute => |c| {
                    state = self.compute(c.env, c.term);
                },
                .ret => |r| {
                    state = self.ret(r.value);
                },
                .done => |d| {
                    prepareValueForHost(d);
                    return d;
                },
            }
        }
    }

    fn compute(self: *Self, env: ?*Env, t: *const Term) State {
        switch (t.*) {
            .tvar => return State{
                .ret = .{
                    .value = env.?.lookupVar(t.debruijnIndex()),
                },
            },
            .delay => return State{
                .ret = .{
                    .value = createDelay(self.heap, env, t.termBody()),
                },
            },
            .lambda => return State{
                .ret = .{
                    .value = createLambda(self.heap, env, t.termBody()),
                },
            },

            .apply => {
                const p = t.appliedTerms();

                self.frames.addFrame(
                    &Frame{
                        .frame_await_fun_term = .{
                            .env = env,
                            .argument = p.argument,
                        },
                    },
                );
                return State{
                    .compute = .{
                        .env = env,
                        .term = p.function,
                    },
                };
            },
            .constant => return State{
                .ret = .{
                    .value = createConst(
                        self.heap,
                        t.constantValue(self.heap),
                    ),
                },
            },
            .force => {
                self.frames.addFrame(&.frame_force);
                return State{
                    .compute = .{
                        .env = env,
                        .term = t.termBody(),
                    },
                };
            },

            .terror => {
                utils.printString("Eval Failure\n");
                utils.exit(std.math.maxInt(u32));
            },

            .builtin => return State{
                .ret = .{
                    .value = createBuiltin(self.heap, t.defaultFunction()),
                },
            },

            .constr => {
                const c = t.constrValues();

                if (c.fields.length == 0) {
                    return State{
                        .ret = .{
                            .value = createConstr(
                                self.heap,
                                c.tag,
                                null,
                            ),
                        },
                    };
                }

                const rest = TermList{
                    .length = c.fields.length - 1,
                    .list = c.fields.list + 1,
                };

                self.frames.addFrame(
                    &Frame{
                        .frame_constr = .{
                            .env = env,
                            .tag = c.tag,
                            .fields = rest,
                            .resolved_fields = null,
                        },
                    },
                );

                return State{
                    .compute = .{
                        .env = env,
                        .term = c.fields.list[0],
                    },
                };
            },

            .case => {
                const cs = t.caseValues();
                self.frames.addFrame(&Frame{
                    .frame_case = .{
                        .env = env,
                        .branches = cs.branches,
                    },
                });

                return State{
                    .compute = .{
                        .env = env,
                        .term = cs.constr,
                    },
                };
            },
        }
    }

    fn ret(self: *Self, v: *const Value) State {
        const frame = self.frames.popFrame();

        switch (frame) {
            .no_frame => return State{ .done = v },

            .frame_force => return self.forceEval(v),

            .frame_await_arg => |f| return self.applyEval(f.function, v),

            .frame_await_fun_value => |f| return self.applyEval(v, f.argument),

            .frame_await_fun_term => |arg| {
                self.frames.addFrame(
                    &Frame{
                        .frame_await_arg = .{
                            .function = v,
                        },
                    },
                );

                return State{
                    .compute = .{
                        .env = arg.env,
                        .term = arg.argument,
                    },
                };
            },

            .frame_constr => |f| {
                const nextResolved = self.heap.create(
                    LinkedValues,
                    &LinkedValues{ .value = v, .next = f.resolved_fields },
                );

                if (f.fields.length == 0) {
                    return State{
                        .ret = .{
                            .value = createConstr(
                                self.heap,
                                f.tag,
                                nextResolved,
                            ),
                        },
                    };
                } else {
                    const rest = TermList{
                        .length = f.fields.length - 1,
                        .list = f.fields.list + 1,
                    };

                    self.frames.addFrame(
                        &Frame{
                            .frame_constr = .{
                                .tag = f.tag,
                                .env = f.env,
                                .fields = rest,
                                .resolved_fields = nextResolved,
                            },
                        },
                    );

                    return State{
                        .compute = .{
                            .env = f.env,
                            .term = f.fields.list[0],
                        },
                    };
                }
            },

            .frame_case => |f| {
                switch (v.*) {
                    .constr => |c| {
                        if (c.tag >= f.branches.length) {
                            utils.printString("constructor tag out of range");
                            utils.exit(std.math.maxInt(u32));
                        }

                        const branch = f.branches.list[c.tag];

                        var fields = c.values;
                        while (fields != null) : (fields = fields.?.next) {
                            self.frames.addFrame(
                                &Frame{
                                    .frame_await_fun_value = .{
                                        .argument = fields.?.value,
                                    },
                                },
                            );
                        }

                        return State{
                            .compute = .{
                                .env = f.env,
                                .term = branch,
                            },
                        };
                    },
                    else => {
                        utils.printString("case on non-constructor");
                        utils.exit(std.math.maxInt(u32));
                    },
                }
            },
        }
    }

    fn forceEval(self: *Self, v: *const Value) State {
        switch (v.*) {
            .delay => |d| return State{
                .compute = .{
                    .env = d.env,
                    .term = d.body,
                },
            },

            .builtin => |b| {
                if (b.force_count == 0) {
                    utils.printString("builtin term argument expected");
                    utils.exit(std.math.maxInt(u32));
                }

                return State{
                    .ret = .{
                        .value = forceBuiltin(self.heap, &b),
                    },
                };
            },
            else => {
                utils.printString("non-polymorphic instantiation");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    fn applyEval(
        self: *Self,
        funVal: *const Value,
        argVal: *const Value,
    ) State {
        switch (funVal.*) {
            .lambda => |lam| {
                const newEnv = if (lam.env) |env| blk: {
                    break :blk env.preprend(argVal, self.heap);
                } else blk: {
                    break :blk Env.init(argVal, self.heap);
                };

                return State{
                    .compute = .{
                        .env = newEnv,
                        .term = lam.body,
                    },
                };
            },

            .builtin => |b| {
                if (b.force_count != 0) {
                    utils.printString("unexpected built-in term argument");
                    utils.exit(std.math.maxInt(u32));
                }

                if (b.arity == 0) {
                    utils.printString("unexpected built-in term argument");
                    utils.exit(std.math.maxInt(u32));
                }

                const nextArity = b.arity - 1;

                const nextArg = self.heap.create(
                    LinkedValues,
                    &LinkedValues{
                        .value = argVal,
                        .next = b.args,
                    },
                );

                const builtinValue = blk: {
                    if (nextArity == 0) {
                        break :blk self.callBuiltin(b.fun, nextArg);
                    } else {
                        break :blk self.heap.create(
                            Value,
                            &Value{
                                .builtin = .{
                                    .fun = b.fun,
                                    .force_count = b.force_count,
                                    .arity = nextArity,
                                    .args = nextArg,
                                },
                            },
                        );
                    }
                };

                return State{
                    .ret = .{
                        .value = builtinValue,
                    },
                };
            },

            else => {
                utils.printString("apply on non-callable");
                utils.exit(std.math.maxInt(u32));
            },
        }
    }

    fn callBuiltin(self: *Self, df: DefaultFunction, args: *LinkedValues) *const Value {
        const index = @intFromEnum(df);
        return builtinFunctions[index](self, args);
    }
};

test "lambda compute" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const v = Term.tvar;

    const expected = createLambda(&heap, null, &v);

    const memory: []const u32 = &.{ 2, 0, 1 };
    const ptr: *const Term = @ptrCast(memory);

    machine.frames.addFrame(&.no_frame);

    const state = machine.compute(null, ptr);

    switch (state) {
        .ret => |r| {
            const frame = machine.frames.popFrame();
            try testing.expectEqualDeep(frame, .no_frame);
            try testing.expectEqualDeep(r.value, expected);
        },
        else => {
            @panic("HOW");
        },
    }
}

test "apply compute and ret" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const v = Term.tvar;

    const expected = createLambda(&heap, null, &v);

    const argument: []const u32 = &.{ 1, 0, 2 };
    const argPointer: *const u32 = @ptrCast(argument);
    const thing: u32 = @truncate(@intFromPtr(argPointer));
    const function: []const u32 = &.{ 3, thing, 2, 0, 1 };
    const ptr: *const Term = @ptrCast(function);

    machine.frames.addFrame(&.no_frame);

    const state = machine.compute(null, ptr);

    const next = switch (state) {
        .compute => |c| machine.compute(c.env, c.term),
        else => @panic("HOW1"),
    };

    const finally = switch (next) {
        .ret => |r| machine.ret(r.value),
        else => @panic("HOW2"),
    };

    switch (finally) {
        .compute => |c| {
            const frame = machine.frames.popFrame();
            try testing.expectEqualDeep(
                frame,
                Frame{
                    .frame_await_arg = .{
                        .function = expected,
                    },
                },
            );
            const noFrame = machine.frames.popFrame();

            try testing.expectEqualDeep(noFrame, .no_frame);
            try testing.expectEqualDeep(c.term.*, Term.delay);
        },
        else => @panic("HOW3"),
    }
}

test "constr compute" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const field1: []const u32 = &.{ 1, 0, 5 };
    const field1Pointer: *const u32 = @ptrCast(field1);
    const field2: []const u32 = &.{ 2, 0, 2 };
    const field2Pointer: *const u32 = @ptrCast(field2);
    const constr: []const u32 = &.{
        8,
        55,
        2,
        @truncate(@intFromPtr(field1Pointer)),
        @truncate(@intFromPtr(field2Pointer)),
    };
    const ptr: *const Term = @ptrCast(constr);

    machine.frames.addFrame(&.no_frame);

    const state = machine.compute(null, ptr);

    switch (state) {
        .compute => |c| {
            try testing.expect(c.env == null);
            try testing.expectEqualDeep(c.term, &Term.delay);
            try testing.expectEqualDeep(c.term.termBody(), &Term.tvar);
            try testing.expectEqualDeep(c.term.termBody().debruijnIndex(), 5);
        },
        else => @panic("HOW1"),
    }

    const frame = machine.frames.popFrame();

    switch (frame) {
        .frame_constr => |c| {
            try testing.expect(c.env == null);
            try testing.expect(c.fields.length == 1);
            try testing.expectEqualDeep(c.fields.list[0], &Term.lambda);
            try testing.expectEqualDeep(c.fields.list[0].termBody(), &Term.tvar);
            try testing.expectEqualDeep(c.fields.list[0].termBody().debruijnIndex(), 2);
        },
        else => @panic("HOW2"),
    }
}

test "constr compute ret" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const field1: []const u32 = &.{ 1, 0, 5 };
    const field1Pointer: *const u32 = @ptrCast(field1);
    const field2: []const u32 = &.{ 2, 0, 2 };
    const field2Pointer: *const u32 = @ptrCast(field2);
    const constr: []const u32 = &.{
        8,
        55,
        2,
        @truncate(@intFromPtr(field1Pointer)),
        @truncate(@intFromPtr(field2Pointer)),
    };
    const ptr: *const Term = @ptrCast(constr);

    machine.frames.addFrame(&.no_frame);

    const state = machine.compute(null, ptr);

    const nextState = switch (state) {
        .compute => |c| blk: {
            break :blk machine.compute(c.env, c.term);
        },
        else => @panic("HERE???"),
    };

    const final = switch (nextState) {
        .ret => |r| blk: {
            break :blk machine.ret(r.value);
        },
        else => @panic("EHRER"),
    };

    switch (final) {
        .compute => |c| {
            try testing.expect(c.env == null);
            try testing.expectEqualDeep(c.term, &Term.lambda);
            try testing.expectEqualDeep(c.term.termBody(), &Term.tvar);
            try testing.expectEqualDeep(c.term.termBody().debruijnIndex(), 2);
        },
        else => @panic("DNFSJ"),
    }

    switch (machine.frames.popFrame()) {
        .frame_constr => |c| {
            try testing.expect(c.env == null);
            try testing.expect(c.fields.length == 0);
            try testing.expectEqualDeep(c.resolved_fields.?.value, &Value{
                .delay = .{
                    .env = null,
                    .body = &Term.tvar,
                },
            });
            try testing.expect(c.tag == 55);
        },
        else => unreachable,
    }
}

test "case compute ret" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const field1: []const u32 = &.{ 1, 0, 5 };
    const field1Pointer: *const u32 = @ptrCast(field1);
    const field2: []const u32 = &.{ 2, 0, 2 };
    const field2Pointer: *const u32 = @ptrCast(field2);
    const constr: []const u32 = &.{
        8,
        0,
        2,
        @truncate(@intFromPtr(field1Pointer)),
        @truncate(@intFromPtr(field2Pointer)),
    };
    const constrPointer: *const u32 = @ptrCast(constr);

    const branch: []const u32 = &.{ 2, 2, 5, 0, 4 };
    const branchPointer: *const u32 = @ptrCast(branch);

    const case: []const u32 = &.{
        9,
        @truncate(@intFromPtr(constrPointer)),
        1,
        @truncate(@intFromPtr(branchPointer)),
    };
    const ptr: *const Term = @ptrCast(case);

    machine.frames.addFrame(&.no_frame);

    const state = machine.compute(null, ptr);

    var nextState = switch (state) {
        .compute => |c| blk: {
            break :blk machine.compute(c.env, c.term);
        },
        else => @panic("HERE???1"),
    };

    nextState = switch (nextState) {
        .compute => |c| blk: {
            break :blk machine.compute(c.env, c.term);
        },
        else => {
            @panic("HERE???2");
        },
    };

    nextState = switch (nextState) {
        .ret => |r| blk: {
            break :blk machine.ret(r.value);
        },
        else => @panic("HERE???3"),
    };

    nextState = switch (nextState) {
        .compute => |c| blk: {
            break :blk machine.compute(c.env, c.term);
        },
        else => @panic("HERE???4"),
    };

    const final = switch (nextState) {
        .ret => |r| blk: {
            break :blk machine.ret(r.value);
        },
        else => {
            @panic("HERE???5");
        },
    };

    const firstField = LinkedValues{
        .value = &Value{
            .delay = .{
                .body = &Term.tvar,
                .env = null,
            },
        },
        .next = null,
    };

    const fields = LinkedValues{
        .value = &Value{
            .lambda = .{
                .body = &Term.tvar,
                .env = null,
            },
        },
        .next = &firstField,
    };

    switch (final) {
        .ret => |r| {
            try testing.expectEqualDeep(
                r.value,
                &Value{
                    .constr = .{
                        .tag = 0,
                        .values = &fields,
                    },
                },
            );
        },
        else => @panic("DNFSJ"),
    }

    switch (machine.frames.popFrame()) {
        .frame_case => |c| {
            try testing.expect(c.env == null);
            try testing.expect(c.branches.length == 1);
            try testing.expectEqualDeep(c.branches.list[0], &Term.lambda);
            try testing.expectEqualDeep(c.branches.list[0].termBody(), &Term.lambda);
        },
        else => unreachable,
    }
}

test "constant compute" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const unitType = ConstantType.unitType();
    const term: []const u32 = &.{
        4,
        1,
        @intFromPtr(unitType),
    };

    const ptr: *const Term = @ptrCast(term);

    machine.frames.addFrame(&.no_frame);

    machine.runValidator(ptr);
}

test "constant compute big int" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const intType = ConstantType.integerType();

    const term: []const u32 = &.{ 4, 1, @intFromPtr(intType), 1, 1, 11 };

    const ptr: *const Term = @ptrCast(term);

    machine.frames.addFrame(&.no_frame);

    const state = machine.compute(null, ptr);

    switch (state) {
        .ret => |r| {
            switch (r.value.constant.constType().*) {
                .integer => {
                    try testing.expect(r.value.constant.length == 1);
                    try testing.expect(r.value.constant.constType().* == .integer);
                    const bigInt = r.value.constant.bigInt();
                    try testing.expectEqual(bigInt.length, 1);
                    try testing.expect(bigInt.sign == 1);
                    try testing.expect(bigInt.words[0] == 11);
                },
                else => @panic("How?"),
            }
        },
        else => @panic("How?"),
    }
}

test "add same signed integers" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const xWords: [*]const u32 = &.{ 5, 6, 7 };

    const yWords: [*]const u32 = &.{ 5, 99 };

    const resultWords: [*]const u32 = &.{ 10, 105, 7 };

    const x = BigInt{
        .length = 3,
        .sign = 0,
        .words = xWords,
    };

    const y = BigInt{
        .length = 2,
        .sign = 0,
        .words = yWords,
    };

    const newVal = addSignedIntegers(&machine, x, y);

    switch (newVal.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .integer => {
                    const result = c.bigInt();
                    try testing.expect(result.length == 3);
                    try testing.expect(result.sign == 0);
                    try testing.expectEqual(result.words[0], resultWords[0]);
                    try testing.expect(result.words[1] == resultWords[1]);
                    try testing.expect(result.words[2] == resultWords[2]);
                },
                else => @panic("TODO"),
            }
        },
        else => @panic("TODO"),
    }
}

test "add same signed integers overflow" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const xWords: [*]const u32 = &.{ 5, 6 };

    const yWords: [*]const u32 = &.{
        5,
        std.math.maxInt(u32),
    };

    const resultWords: [*]const u32 = &.{ 10, 5, 1 };

    const x = BigInt{
        .length = 2,
        .sign = 0,
        .words = xWords,
    };

    const y = BigInt{
        .length = 2,
        .sign = 0,
        .words = yWords,
    };

    const newVal = addSignedIntegers(&machine, x, y);

    switch (newVal.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .integer => {
                    const result = c.bigInt();
                    try testing.expect(result.length == 3);
                    try testing.expect(result.sign == 0);
                    try testing.expect(result.words[0] == resultWords[0]);
                    try testing.expect(result.words[1] == resultWords[1]);
                    try testing.expect(result.words[2] == resultWords[2]);
                },
                else => @panic("TODO"),
            }
        },
        else => @panic("TODO"),
    }
}

test "sub signed integers overflow" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const xWords: [*]const u32 = &.{ 5, 6, 7 };

    const yWords: [*]const u32 = &.{ 5, 99 };

    const resultWords: [*]const u32 = &.{ 0, 4294967203, 6 };

    const x = BigInt{
        .length = 3,
        .sign = 1,
        .words = xWords,
    };

    const y = BigInt{
        .length = 2,
        .sign = 0,
        .words = yWords,
    };

    const newVal = subSignedIntegers(&machine, x, y);

    switch (newVal.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .integer => {
                    const result = c.bigInt();
                    try testing.expect(result.length == 3);
                    try testing.expect(result.sign == 1);
                    try testing.expectEqual(result.words[0], resultWords[0]);
                    try testing.expectEqual(result.words[1], resultWords[1]);
                    try testing.expect(result.words[2] == resultWords[2]);
                },
                else => @panic("TODO"),
            }
        },
        else => @panic("TODO"),
    }
}

test "sub signed integers reclaim" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    var machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const xWords: [*]const u32 = &.{ 8, 6, 1 };

    const yWords: [*]const u32 = &.{ 10, 5, 1 };

    const resultWords: [*]const u32 = &.{
        4294967294,
    };

    const x = BigInt{
        .length = 2,
        .sign = 0,
        .words = xWords,
    };

    const y = BigInt{
        .length = 2,
        .sign = 1,
        .words = yWords,
    };

    const newVal = subSignedIntegers(&machine, x, y);

    switch (newVal.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .integer => {
                    const result = c.bigInt();
                    try testing.expectEqual(result.length, 1);
                    try testing.expect(result.sign == 0);
                    try testing.expectEqual(result.words[0], resultWords[0]);
                },
                else => @panic("TODO"),
            }
        },
        else => @panic("TODO"),
    }
}

test "multiply same‑signed single‑word integers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const a_words: [*]const u32 = &.{3};
    const b_words: [*]const u32 = &.{4};

    const a = expr.BigInt{ .sign = 0, .length = 1, .words = a_words };
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = b_words };

    const args = LinkedValues.create(&heap, *const expr.BigInt, &a, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &b, ConstantType.integerType());

    const newVal = multiplyInteger(&machine, args);

    const result_words: [*]const u32 = &.{12};

    switch (newVal.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const result = c.bigInt();
                try testing.expect(result.sign == 0);
                try testing.expectEqual(result.length, 1);
                try testing.expect(result.words[0] == result_words[0]);
            },
            else => @panic("TODO"),
        },
        else => @panic("TODO"),
    }
}

test "multiply same‑signed integers with 32‑bit carry" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const a_words: [*]const u32 = &.{std.math.maxInt(u32)}; // 0xFFFFFFFF
    const b_words: [*]const u32 = &.{2};

    const a = expr.BigInt{ .sign = 0, .length = 1, .words = a_words };
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = b_words };

    const args = LinkedValues.create(&heap, *const expr.BigInt, &a, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &b, ConstantType.integerType());

    const newVal = multiplyInteger(&machine, args);

    const result_words: [*]const u32 = &.{ 0xFFFFFFFE, 1 }; // low word, high carry

    switch (newVal.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const result = c.bigInt();
                try testing.expect(result.length == 2);
                try testing.expect(result.sign == 0);
                try testing.expect(result.words[0] == result_words[0]);
                try testing.expect(result.words[1] == result_words[1]);
            },
            else => @panic("TODO"),
        },
        else => @panic("TODO"),
    }
}

test "multiply differing‑signed integers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const a_words: [*]const u32 = &.{3};
    const b_words: [*]const u32 = &.{4};

    const a = expr.BigInt{ .sign = 1, .length = 1, .words = a_words }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = b_words }; // positive

    const args = LinkedValues.create(&heap, *const expr.BigInt, &a, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &b, ConstantType.integerType());

    const newVal = multiplyInteger(&machine, args);

    const result_words: [*]const u32 = &.{12};

    switch (newVal.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const result = c.bigInt();
                try testing.expect(result.length == 1);
                try testing.expect(result.sign == 1);
                try testing.expect(result.words[0] == result_words[0]);
            },
            else => @panic("TODO"),
        },
        else => @panic("TODO"),
    }
}

test "divide: numerator == 0 → 0" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var mach = Machine{ .heap = &heap, .frames = &frames };

    const n = expr.BigInt{ .sign = 0, .length = 1, .words = &.{0} }; // 0
    const d = expr.BigInt{ .sign = 0, .length = 1, .words = &.{1234} }; // any ≠ 0

    const args = LinkedValues.create(&heap, *const expr.BigInt, &n, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &d, ConstantType.integerType());

    const res_val = divideInteger(&mach, args);

    switch (res_val.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const r = c.bigInt();
                try testing.expect(r.sign == 0);
                try testing.expect(r.length == 1);
                try testing.expect(r.words[0] == 0);
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

test "divide: 1 / 2 floors to 0" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var mach = Machine{ .heap = &heap, .frames = &frames };

    const n = expr.BigInt{ .sign = 0, .length = 1, .words = &.{1} }; // 1
    const d = expr.BigInt{ .sign = 0, .length = 1, .words = &.{2} }; // 2

    const args = LinkedValues.create(&heap, expr.BigInt, n, ConstantType.integerType())
        .extend(&heap, expr.BigInt, d, ConstantType.integerType());

    const res_val = divideInteger(&mach, args);

    switch (res_val.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const r = c.bigInt();
                try testing.expect(r.sign == 0);
                try testing.expect(r.length == 1);
                try testing.expect(r.words[0] == 0);
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

test "divide: (-503) / (-1777777777) = 0" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var mach = Machine{ .heap = &heap, .frames = &frames };

    //const a_words = &.{@as(u32, 503)}; // magnitude(503)
    //const b_words = &.{@as(u32, 1_777_777_777)}; // magnitude(1 777 777 777)

    const n = expr.BigInt{
        .sign = 1,
        .length = 1,
        .words = &[_]u32{503}, // OK: *const [1]u32 → coerces to [*]const u32
    };
    const d = expr.BigInt{
        .sign = 1,
        .length = 1,
        .words = &[_]u32{1_777_777_777},
    };

    const args = LinkedValues.create(&heap, expr.BigInt, n, ConstantType.integerType())
        .extend(&heap, expr.BigInt, d, ConstantType.integerType());

    //const n = expr.BigInt{ .sign = 1, .length = 1, .words = &a_words }; // −503
    //const d = expr.BigInt{ .sign = 1, .length = 1, .words = &b_words }; // −1777777777

    const res_val = divideInteger(&mach, args);

    switch (res_val.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const r = c.bigInt();
                try testing.expect(r.sign == 0);
                try testing.expect(r.length == 1);
                try testing.expect(r.words[0] == 0);
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

test "divide: (-503) / (+1777777777) floors to −1" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var mach = Machine{ .heap = &heap, .frames = &frames };

    // const a_words = &.{@as(u32, 503)};
    // const b_words = &.{@as(u32, 1_777_777_777)};
    // const n = expr.BigInt{ .sign = 1, .length = 1, .words = &a_words }; // −503
    // const d = expr.BigInt{ .sign = 0, .length = 1, .words = &b_words }; // +1777777777
    const n = expr.BigInt{
        .sign = 1,
        .length = 1,
        .words = &[_]u32{503}, // OK: *const [1]u32 → coerces to [*]const u32
    };
    const d = expr.BigInt{
        .sign = 0,
        .length = 1,
        .words = &[_]u32{1_777_777_777},
    };

    const args = LinkedValues.create(&heap, expr.BigInt, n, ConstantType.integerType())
        .extend(&heap, expr.BigInt, d, ConstantType.integerType());

    const res_val = divideInteger(&mach, args);

    switch (res_val.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const r = c.bigInt();
                try testing.expect(r.sign == 1); // negative
                try testing.expect(r.length == 1);
                try testing.expect(r.words[0] == 1); // magnitude 1
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

test "divide: (+503) / (−1777777777) floors to −1" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var mach = Machine{ .heap = &heap, .frames = &frames };

    // const a_words = &.{@as(u32, 503)};
    // const b_words = &.{@as(u32, 1_777_777_777)};
    // const n = expr.BigInt{ .sign = 0, .length = 1, .words = &a_words }; // +503
    // const d = expr.BigInt{ .sign = 1, .length = 1, .words = &b_words }; // −1777777777
    const n = expr.BigInt{
        .sign = 0,
        .length = 1,
        .words = &[_]u32{503}, // OK: *const [1]u32 → coerces to [*]const u32
    };
    const d = expr.BigInt{
        .sign = 1,
        .length = 1,
        .words = &[_]u32{1_777_777_777},
    };

    const args = LinkedValues.create(&heap, expr.BigInt, n, ConstantType.integerType())
        .extend(&heap, expr.BigInt, d, ConstantType.integerType());

    const res_val = divideInteger(&mach, args);

    switch (res_val.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const r = c.bigInt();
                try testing.expect(r.sign == 1); // negative
                try testing.expect(r.length == 1);
                try testing.expect(r.words[0] == 1); // magnitude 1
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

test "divide: (+503) / (+1777777777) = 0" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var mach = Machine{ .heap = &heap, .frames = &frames };

    // const a_words = &.{@as(u32, 503)};
    // const b_words = &.{@as(u32, 1_777_777_777)};
    // const n = expr.BigInt{ .sign = 0, .length = 1, .words = &a_words }; // +503
    // const d = expr.BigInt{ .sign = 0, .length = 1, .words = &b_words }; // +1777777777
    const n = expr.BigInt{
        .sign = 0,
        .length = 1,
        .words = &[_]u32{503}, // OK: *const [1]u32 → coerces to [*]const u32
    };
    const d = expr.BigInt{
        .sign = 0,
        .length = 1,
        .words = &[_]u32{1_777_777_777},
    };

    const args = LinkedValues.create(&heap, expr.BigInt, n, ConstantType.integerType())
        .extend(&heap, expr.BigInt, d, ConstantType.integerType());

    const res_val = divideInteger(&mach, args);

    switch (res_val.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const r = c.bigInt();
                try testing.expect(r.sign == 0);
                try testing.expect(r.length == 1);
                try testing.expect(r.words[0] == 0);
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

test "divide: multi-limb exact (positive / positive)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var mach = Machine{ .heap = &heap, .frames = &frames };

    const a_words: [*]const u32 = &.{ 4, 4, 1 }; // (2^32 + 2)^2 = 2^64 + 4*2^32 + 4
    const b_words: [*]const u32 = &.{ 2, 1 }; // 2^32 + 2

    const n = expr.BigInt{ .sign = 0, .length = 3, .words = a_words };
    const d = expr.BigInt{ .sign = 0, .length = 2, .words = b_words };

    const args = LinkedValues.create(&heap, expr.BigInt, n, ConstantType.integerType())
        .extend(&heap, expr.BigInt, d, ConstantType.integerType());

    const res_val = divideInteger(&mach, args);

    switch (res_val.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const r = c.bigInt();
                try testing.expect(r.sign == 0);
                try testing.expect(r.length == 2);
                try testing.expect(r.words[0] == 2);
                try testing.expect(r.words[1] == 1); // 2^32 + 2
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

test "divide: multi-limb with remainder and signs differ (positive / negative)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var mach = Machine{ .heap = &heap, .frames = &frames };

    const a_words: [*]const u32 = &.{ 5, 4, 1 }; // (2^32 + 2)^2 + 1 = 2^64 + 4*2^32 + 5
    const b_words: [*]const u32 = &.{ 2, 1 }; // 2^32 + 2

    const n = expr.BigInt{ .sign = 0, .length = 3, .words = a_words };
    const d = expr.BigInt{ .sign = 1, .length = 2, .words = b_words };

    const args = LinkedValues.create(&heap, expr.BigInt, n, ConstantType.integerType())
        .extend(&heap, expr.BigInt, d, ConstantType.integerType());

    const res_val = divideInteger(&mach, args);

    switch (res_val.*) {
        .constant => |c| switch (c.constType().*) {
            .integer => {
                const r = c.bigInt();
                try testing.expect(r.sign == 1);
                try testing.expect(r.length == 2);
                try testing.expect(r.words[0] == 3);
                try testing.expect(r.words[1] == 1); // -(2^32 + 3)
            },
            else => unreachable,
        },
        else => unreachable,
    }
}

// test "divide: division‑by‑zero panics" {
//     const expected_msg = "divideInteger: division by zero";

//     try testing.expectPanic(expected_msg, struct {
//         fn doPanic() void {
//             var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//             defer arena.deinit();

//             var heap = Heap.createTestHeap(&arena) catch unreachable;
//             var frames = Frames.createTestFrames(&arena) catch unreachable;
//             var mach = Machine{ .heap = &heap, .frames = &frames };

//             const n = expr.BigInt{ .sign = 0, .length = 1, .words = &.{1} }; // 1
//             const d = expr.BigInt{ .sign = 0, .length = 1, .words = &.{0} }; // 0  ← boom

//             _ = divideInteger(&mach, createIntArgs(&mach, n, d));
//         }
//     }.doPanic);
// }

test "equals integer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aWords: [*]const u32 = &.{ 705032704, 1 };
    const bWords: [*]const u32 = &.{5};
    const cWords: [*]const u32 = &.{ 705032699, 1 };

    const a = expr.BigInt{ .sign = 1, .length = 2, .words = aWords }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = bWords }; // positive
    const c = expr.BigInt{ .sign = 1, .length = 2, .words = cWords };

    const resultWords = subSignedIntegers(&machine, a, b);

    const args = LinkedValues.create(&heap, *const expr.BigInt, &c, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &resultWords.constant.bigInt(), ConstantType.integerType());

    const newVal = equalsInteger(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .boolean => {
                    const val = con.bln();
                    try testing.expect(val);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "less than integer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aWords: [*]const u32 = &.{ 705032704, 1 };
    const bWords: [*]const u32 = &.{5};
    const cWords: [*]const u32 = &.{ 705032698, 1 };

    const a = expr.BigInt{ .sign = 1, .length = 2, .words = aWords }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = bWords }; // positive
    const c = expr.BigInt{ .sign = 1, .length = 2, .words = cWords };

    const resultWords = subSignedIntegers(&machine, a, b);

    const args = LinkedValues.create(&heap, *const expr.BigInt, &c, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &resultWords.constant.bigInt(), ConstantType.integerType());

    const newVal = lessThanInteger(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .boolean => {
                    const val = con.bln();
                    try testing.expect(val);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "less than equals integer less" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aWords: [*]const u32 = &.{ 705032704, 1 };
    const bWords: [*]const u32 = &.{5};
    const cWords: [*]const u32 = &.{ 705032698, 1 };

    const a = expr.BigInt{ .sign = 1, .length = 2, .words = aWords }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = bWords }; // positive
    const c = expr.BigInt{ .sign = 1, .length = 2, .words = cWords };

    const resultWords = subSignedIntegers(&machine, a, b);

    const args = LinkedValues
        .create(&heap, *const expr.BigInt, &c, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &resultWords.constant.bigInt(), ConstantType.integerType());

    const newVal = lessThanEqualsInteger(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .boolean => {
                    const val = con.bln();
                    try testing.expect(val);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "less than equals integer equal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aWords: [*]const u32 = &.{ 705032704, 1 };
    const bWords: [*]const u32 = &.{5};
    const cWords: [*]const u32 = &.{ 705032699, 1 };

    const a = expr.BigInt{ .sign = 1, .length = 2, .words = aWords }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = bWords }; // positive
    const c = expr.BigInt{ .sign = 1, .length = 2, .words = cWords };

    const resultWords = subSignedIntegers(&machine, a, b);

    const args = LinkedValues
        .create(&heap, *const expr.BigInt, &c, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &resultWords.constant.bigInt(), ConstantType.integerType());

    const newVal = lessThanEqualsInteger(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .boolean => {
                    const val = con.bln();
                    try testing.expect(val);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "append bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 255, 254 };
    const bBytes: [*]const u32 = &.{ 0, 255, 1 };
    const resultBytes: [*]const u32 = &.{ 255, 254, 0, 255, 1 };

    const a = expr.Bytes{ .length = 2, .bytes = aBytes };
    const b = expr.Bytes{ .length = 3, .bytes = bBytes };

    const result = expr.Bytes{ .length = 5, .bytes = resultBytes };

    const args = LinkedValues
        .create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = appendByteString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .bytes => {
                    const val = con.innerBytes();
                    try testing.expectEqual(val.length, result.length);
                    try testing.expectEqual(val.bytes[0], 255);
                    try testing.expectEqual(val.bytes[1], 254);
                    try testing.expectEqual(val.bytes[2], 0);
                    try testing.expectEqual(val.bytes[3], 255);
                    try testing.expectEqual(val.bytes[4], 1);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "cons bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aByte: [*]const u32 = &.{37};
    const bBytes: [*]const u32 = &.{ 0, 255, 1 };
    const resultBytes: [*]const u32 = &.{ 37, 0, 255, 1 };

    const a = expr.BigInt{ .length = 1, .sign = 0, .words = aByte };
    const b = expr.Bytes{ .length = 3, .bytes = bBytes };

    const result = expr.Bytes{ .length = 4, .bytes = resultBytes };

    const args = LinkedValues
        .create(&heap, *const expr.BigInt, &a, ConstantType.integerType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = consByteString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .bytes => {
                    const val = con.innerBytes();
                    try testing.expectEqual(val.length, result.length);
                    try testing.expectEqual(val.bytes[0], 37);
                    try testing.expectEqual(val.bytes[1], 0);
                    try testing.expectEqual(val.bytes[2], 255);
                    try testing.expectEqual(val.bytes[3], 1);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "slice bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 37, 65, 77, 255, 88 };
    const dropWord: [*]const u32 = &.{2};
    const takeWord: [*]const u32 = &.{4};
    const resultBytes: [*]const u32 = &.{ 77, 255, 88 };

    const a = expr.Bytes{ .length = 5, .bytes = aBytes };
    const drop = expr.BigInt{ .length = 1, .sign = 0, .words = dropWord };
    const take = expr.BigInt{ .length = 1, .sign = 0, .words = takeWord };

    const result = expr.Bytes{ .length = 3, .bytes = resultBytes };

    const args = LinkedValues
        .create(&heap, *const expr.BigInt, &drop, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &take, ConstantType.integerType())
        .extend(&heap, *const expr.Bytes, &a, ConstantType.bytesType());

    const newVal = sliceByteString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .bytes => {
                    const val = con.innerBytes();
                    try testing.expectEqual(val.length, result.length);
                    try testing.expectEqual(val.bytes[0], result.bytes[0]);
                    try testing.expectEqual(val.bytes[1], result.bytes[1]);
                    try testing.expectEqual(val.bytes[2], result.bytes[2]);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "length bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 0, 255, 1, 4 };
    const resultBytes: [*]const u32 = &.{4};

    const a = expr.Bytes{ .length = 4, .bytes = aBytes };

    const result = expr.BigInt{ .length = 1, .sign = 0, .words = resultBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType());

    const newVal = lengthOfByteString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .integer => {
                    const val = con.bigInt();
                    try testing.expectEqual(val.length, result.length);
                    try testing.expectEqual(val.sign, result.sign);
                    try testing.expectEqual(val.words[0], result.words[0]);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "index bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 0, 255, 1, 72, 6 };
    const index: [*]const u32 = &.{3};
    const resultBytes: [*]const u32 = &.{72};

    const a = expr.Bytes{ .length = 5, .bytes = aBytes };
    const b = expr.BigInt{ .length = 1, .sign = 0, .words = index };

    const result = expr.BigInt{ .length = 1, .sign = 0, .words = resultBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.BigInt, &b, ConstantType.integerType());

    const newVal = indexByteString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .integer => {
                    const val = con.bigInt();
                    try testing.expectEqual(val.length, result.length);
                    try testing.expectEqual(val.sign, result.sign);
                    try testing.expectEqual(val.words[0], result.words[0]);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "equals bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 0, 255, 1, 72, 6 };

    const a = expr.Bytes{ .length = 5, .bytes = aBytes };
    const b = expr.Bytes{ .length = 5, .bytes = aBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = equalsByteString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .boolean => {
                    const val = con.bln();
                    try testing.expect(val);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "less than bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 0, 254, 1, 72, 6, 99 };
    const bBytes: [*]const u32 = &.{ 0, 255, 1, 72, 6 };

    const a = expr.Bytes{ .length = 6, .bytes = aBytes };
    const b = expr.Bytes{ .length = 5, .bytes = bBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = lessThanByteString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .boolean => {
                    const val = con.bln();
                    try testing.expect(val);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "less than equal bytes less" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 0, 254, 1, 72, 6, 99 };
    const bBytes: [*]const u32 = &.{ 0, 255, 1, 72, 6 };

    const a = expr.Bytes{ .length = 6, .bytes = aBytes };
    const b = expr.Bytes{ .length = 5, .bytes = bBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = lessThanEqualsByteString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .boolean => {
                    const val = con.bln();
                    try testing.expect(val);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "less than equal bytes equal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 0, 254, 1, 72, 6, 99 };

    const a = expr.Bytes{ .length = 6, .bytes = aBytes };
    const b = expr.Bytes{ .length = 6, .bytes = aBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = lessThanEqualsByteString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .boolean => {
                    const val = con.bln();
                    try testing.expect(val);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "append string" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 255, 254 };
    const bBytes: [*]const u32 = &.{ 0, 255, 1 };
    const resultBytes: [*]const u32 = &.{ 255, 254, 0, 255, 1 };

    const a = expr.String{ .length = 2, .bytes = aBytes };
    const b = expr.String{ .length = 3, .bytes = bBytes };

    const result = expr.String{ .length = 5, .bytes = resultBytes };

    const args = LinkedValues
        .create(&heap, *const expr.String, &a, ConstantType.stringType())
        .extend(&heap, *const expr.String, &b, ConstantType.stringType());

    const newVal = appendString(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .string => {
                    const val = con.string();
                    try testing.expectEqual(val.length, result.length);

                    try testing.expectEqual(val.bytes[0], 255);
                    try testing.expectEqual(val.bytes[1], 254);
                    try testing.expectEqual(val.bytes[2], 0);
                    try testing.expectEqual(val.bytes[3], 255);
                    try testing.expectEqual(val.bytes[4], 1);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "if then else" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 0, 254 };
    const bBytes: [*]const u32 = &.{ 1, 253, 3 };

    const a = expr.Bytes{ .length = 2, .bytes = aBytes };
    const b = expr.Bytes{ .length = 3, .bytes = bBytes };
    const x = expr.Bool{ .val = @intFromBool(true) };

    const args = LinkedValues
        .create(&heap, *const expr.Bool, &x, ConstantType.booleanType())
        .extend(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = ifThenElse(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .bytes => {
                    const val = con.innerBytes();
                    try testing.expectEqual(val.length, 2);
                    try testing.expectEqual(val.bytes[0], 0);
                    try testing.expectEqual(val.bytes[1], 254);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "chooseList Empty" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 0, 254 };
    const bBytes: [*]const u32 = &.{ 1, 253, 3 };

    const a = expr.Bytes{ .length = 2, .bytes = aBytes };
    const b = expr.Bytes{ .length = 3, .bytes = bBytes };

    const listIntType = [2]ConstantType{
        ConstantType.list,
        ConstantType.integer,
    };

    const types: *const ConstantType = @ptrCast(&listIntType);

    const list = expr.List{
        .type_length = 1,
        .inner_type = @ptrCast(ConstantType.integerType()),
        .length = 0,
        .items = null,
    };

    const args = LinkedValues
        .create(&heap, *const expr.List, &list, types)
        .extend(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = chooseList(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .bytes => {
                    const val = con.innerBytes();
                    try testing.expectEqual(val.length, 2);
                    try testing.expectEqual(val.bytes[0], 0);
                    try testing.expectEqual(val.bytes[1], 254);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "chooseList Something" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 0, 254 };
    const bBytes: [*]const u32 = &.{ 1, 253, 3 };

    const a = expr.Bytes{ .length = 2, .bytes = aBytes };
    const b = expr.Bytes{ .length = 3, .bytes = bBytes };

    const listItem = expr.BigInt{
        .sign = 0,
        .length = 3,
        .words = bBytes,
    };

    const listIntType = [2]ConstantType{
        ConstantType.list,
        ConstantType.integer,
    };

    const types: *const ConstantType = @ptrCast(&listIntType);

    const constantInt = listItem.createConstant(ConstantType.integerType(), machine.heap);

    var listNode = ListNode{
        .value = constantInt.rawValue(),
        .next = null,
    };

    const list = expr.List{
        .type_length = 1,
        .inner_type = @ptrCast(ConstantType.integerType()),
        .length = 1,
        .items = &listNode,
    };

    const args = LinkedValues
        .create(&heap, *const expr.List, &list, types)
        .extend(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = chooseList(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .bytes => {
                    const val = con.innerBytes();
                    try testing.expectEqual(val.length, 3);
                    try testing.expectEqual(val.bytes[0], 1);
                    try testing.expectEqual(val.bytes[1], 253);
                    try testing.expectEqual(val.bytes[2], 3);
                },
                else => {
                    @panic("TODO");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}

test "mkCons" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const aBytes: [*]const u32 = &.{ 5, 254 };
    const bBytes: [*]const u32 = &.{ 1, 253, 3 };

    const a = expr.BigInt{ .sign = 0, .length = 2, .words = aBytes };
    const b = expr.BigInt{ .sign = 1, .length = 3, .words = bBytes };

    const listIntType = [2]ConstantType{
        ConstantType.list,
        ConstantType.integer,
    };

    const types: *const ConstantType = @ptrCast(&listIntType);

    const constantInt = a.createConstant(ConstantType.integerType(), machine.heap);

    var listNode = ListNode{
        .value = constantInt.rawValue(),
        .next = null,
    };

    const list = expr.List{
        .type_length = 1,
        .inner_type = @ptrCast(ConstantType.integerType()),
        .length = 1,
        .items = &listNode,
    };

    const args = LinkedValues
        .create(&heap, *const expr.BigInt, &b, ConstantType.integerType())
        .extend(&heap, *const expr.List, &list, types);

    const newVal = mkCons(&machine, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .list => {
                    const val = con.list();

                    try testing.expectEqual(val.length, 2);
                    try testing.expectEqual(val.type_length, 1);
                    try testing.expectEqual(val.inner_type[0], ConstantType.integerType().*);

                    const bResult: Constant = Constant{
                        .length = 1,
                        .type_list = @ptrCast(ConstantType.integerType()),
                        .value = val.items.?.value,
                    };

                    const bInt = bResult.bigInt();

                    try testing.expectEqual(bInt.sign, 1);
                    try testing.expectEqual(bInt.length, 3);
                    try testing.expectEqual(bInt.words[0], b.words[0]);
                    try testing.expectEqual(bInt.words[1], b.words[1]);
                    try testing.expectEqual(bInt.words[2], b.words[2]);
                },
                else => {
                    @panic("TODO HERE");
                },
            }
        },
        else => {
            @panic("TODO");
        },
    }
}
