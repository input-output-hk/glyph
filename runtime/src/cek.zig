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

const UnConstrReturnTypeDescriptor = [4]u32{
    @intFromEnum(ConstantType.pair),
    @intFromEnum(ConstantType.integer),
    @intFromEnum(ConstantType.list),
    @intFromEnum(ConstantType.data),
};

// Reusable descriptors for [(Data, Data)] results (unMapData et al.).
const DataPairTypeDescriptor = [3]u32{
    @intFromEnum(ConstantType.pair),
    @intFromEnum(ConstantType.data),
    @intFromEnum(ConstantType.data),
};

const DataPairListTypeDescriptor = [4]u32{
    @intFromEnum(ConstantType.list),
    @intFromEnum(ConstantType.pair),
    @intFromEnum(ConstantType.data),
    @intFromEnum(ConstantType.data),
};

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

// Pair constants carry two payload pointers with their component type descriptors
// stored sequentially after the leading `.pair` tag. This view reconstructs both.
const PairConstantView = struct {
    first_value: u32,
    second_value: u32,
    first_type: [*]const ConstantType,
    second_type: [*]const ConstantType,
    first_type_len: u32,
    second_type_len: u32,
};

const PairPayload = extern struct {
    first: u32,
    second: u32,
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
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }
    pub fn unwrapConstant(v: *const Value) *const Constant {
        switch (v.*) {
            .constant => |c| {
                return c;
            },
            else => builtinEvaluationFailure(),
        }
    }

    pub fn unwrapInteger(v: *const Value) BigInt {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .integer => {
                        return c.bigInt();
                    },
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }

    pub fn unwrapBytestring(v: *const Value) Bytes {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .bytes => {
                        return c.innerBytes();
                    },
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }

    pub fn unwrapString(v: *const Value) String {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .string => {
                        return c.string();
                    },
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }

    pub fn unwrapBool(v: *const Value) bool {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .boolean => {
                        return c.bln();
                    },
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }

    pub fn unwrapList(v: *const Value) List {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .list => {
                        return c.list();
                    },
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }

    pub fn unwrapPair(v: *const Value) PairConstantView {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .pair => return pairConstantView(c),
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }

    pub fn unwrapG1(v: *const Value) G1Element {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .bls12_381_g1_element => {
                        return c.g1Element();
                    },
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }

    pub fn unwrapG2(v: *const Value) G2Element {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .bls12_381_g2_element => {
                        return c.g2Element();
                    },
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }

    pub fn unwrapMlResult(v: *const Value) MlResult {
        switch (v.*) {
            .constant => |c| {
                switch (c.constType().*) {
                    .bls12_381_mlresult => {
                        return c.mlResult();
                    },
                    else => builtinEvaluationFailure(),
                }
            },
            else => builtinEvaluationFailure(),
        }
    }
};

fn pairConstantView(constant: *const Constant) PairConstantView {
    const types: [*]const ConstantType = @ptrCast(constant.constType());
    const payload: *const PairPayload = @ptrFromInt(constant.value);

    if (payload.first == 0 or payload.second == 0) {
        utils.printlnString("Pair constant missing component payload");
        utils.exit(std.math.maxInt(u32));
    }

    const first_type = types + 1;
    const first_span = typeDescriptorSpan(first_type);
    const second_type = first_type + first_span;
    const second_span = typeDescriptorSpan(second_type);

    return PairConstantView{
        .first_value = payload.first,
        .second_value = payload.second,
        .first_type = first_type,
        .second_type = second_type,
        .first_type_len = @intCast(first_span),
        .second_type_len = @intCast(second_span),
    };
}

fn typeDescriptorSpan(cursor: [*]const ConstantType) usize {
    return switch (cursor[0]) {
        .list => 1 + typeDescriptorSpan(cursor + 1),
        .pair => blk: {
            const first_len = typeDescriptorSpan(cursor + 1);
            const second_len = typeDescriptorSpan(cursor + 1 + first_len);
            break :blk 1 + first_len + second_len;
        },
        else => 1,
    };
}

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
        if (idx == 0) {
            builtinEvaluationFailure();
        }

        var cur: ?*Self = self;
        var remaining = idx;

        while (cur) |node| {
            if (remaining == 1) {
                return node.value;
            }
            cur = node.next;
            remaining -= 1;
        }

        // Walking past the end means the term referenced more binders
        // than are in scope, which Plutus reports as evaluation failure.
        builtinEvaluationFailure();
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
    if (constant.length == serialized_data_const_tag) {
        if (serializedPayloadWordCount(constant.value) != null) {
            normalizeSerializedDataConstantForHost(constant);
        }
    }
    // Other constant kinds already match the host layout consumed by the host tests.
}

fn normalizeSerializedDataConstantForHost(constant: *const Constant) void {
    // Serialized Data constants (length == 0x05) reuse the same header fields as every
    // other constant even though their payload format differs. The host side only needs
    // to read the type descriptor to decide how to compare the result, so repoint the
    // descriptor to the shared Data type and trim the reported type length to 1.
    const mutable: *Constant = @constCast(constant);
    mutable.length = 1;
    mutable.type_list = dataTypePtr();
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
// Zig's std.crypto version of SHA2 currently produces incorrect digests when
// cross-compiled for our freestanding riscv32 target, so we ship a compact
// local implementation that matches the spec and accepts streamed byte input.
const Sha256Ctx = struct {
    h: [8]u32 = sha256_initial_state,
    buf: [64]u8 = undefined,
    buf_len: usize = 0,
    total_len: u64 = 0,

    fn init() Sha256Ctx {
        return Sha256Ctx{};
    }

    fn updateSlice(self: *Sha256Ctx, data: []const u8) void {
        var offset: usize = 0;
        self.total_len += data.len;

        if (self.buf_len != 0) {
            const space = 64 - self.buf_len;
            const to_copy = @min(space, data.len);
            @memcpy(self.buf[self.buf_len .. self.buf_len + to_copy], data[0..to_copy]);
            self.buf_len += to_copy;
            offset += to_copy;
            if (self.buf_len == 64) {
                self.processBlock(&self.buf);
                self.buf_len = 0;
            }
        }

        while (offset + 64 <= data.len) : (offset += 64) {
            const block_slice = data[offset..][0..64];
            const block_ptr: *const [64]u8 = @ptrCast(block_slice.ptr);
            self.processBlock(block_ptr);
        }

        const remaining = data.len - offset;
        if (remaining != 0) {
            @memcpy(self.buf[0..remaining], data[offset..]);
            self.buf_len = remaining;
        }
    }

    fn finalize(self: *Sha256Ctx, out: *[32]u8) void {
        // Append the single '1' bit (0x80 byte) then pad with zeros until we have room
        // for the 64-bit big-endian message length.
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;

        if (self.buf_len > 56) {
            while (self.buf_len < 64) : (self.buf_len += 1) {
                self.buf[self.buf_len] = 0;
            }
            self.processBlock(&self.buf);
            self.buf_len = 0;
        }

        while (self.buf_len < 56) : (self.buf_len += 1) {
            self.buf[self.buf_len] = 0;
        }

        const bit_len: u64 = self.total_len * 8;
        var shift: usize = 0;
        while (shift < 8) : (shift += 1) {
            const idx = 7 - shift;
            const shift_amt: u6 = @intCast(idx * 8);
            self.buf[56 + shift] = @as(u8, @intCast((bit_len >> shift_amt) & 0xFF));
        }
        self.processBlock(&self.buf);
        self.buf_len = 0;

        for (self.h, 0..) |word, idx| {
            const base = idx * 4;
            out[base + 0] = @as(u8, @intCast(word >> 24));
            out[base + 1] = @as(u8, @intCast((word >> 16) & 0xFF));
            out[base + 2] = @as(u8, @intCast((word >> 8) & 0xFF));
            out[base + 3] = @as(u8, @intCast(word & 0xFF));
        }
    }

    inline fn rotr(value: u32, shift: u5) u32 {
        if (shift == 0) return value;
        const amt: u32 = shift;
        return (value >> amt) | (value << @as(u5, @intCast(32 - amt)));
    }

    fn processBlock(self: *Sha256Ctx, block: *const [64]u8) void {
        var w: [64]u32 = undefined;
        const bytes: [*]const u8 = @ptrCast(block);

        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const idx = i * 4;
            w[i] = (@as(u32, bytes[idx]) << 24) |
                (@as(u32, bytes[idx + 1]) << 16) |
                (@as(u32, bytes[idx + 2]) << 8) |
                @as(u32, bytes[idx + 3]);
        }

        while (i < 64) : (i += 1) {
            const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        var a = self.h[0];
        var b = self.h[1];
        var c = self.h[2];
        var d = self.h[3];
        var e = self.h[4];
        var f = self.h[5];
        var g = self.h[6];
        var h = self.h[7];

        i = 0;
        while (i < 64) : (i += 1) {
            const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const ch = (e & f) ^ (~e & g);
            const temp1 = h + s1 + ch + sha256_round_constants[i] + w[i];
            const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const maj = (a & b) ^ (a & c) ^ (b & c);
            const temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        self.h[0] += a;
        self.h[1] += b;
        self.h[2] += c;
        self.h[3] += d;
        self.h[4] += e;
        self.h[5] += f;
        self.h[6] += g;
        self.h[7] += h;
    }
};

const sha256_initial_state = [8]u32{
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
};

const sha256_round_constants = [64]u32{
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
};

const sha3_rate_bytes: usize = 136;

const keccak_rho_offsets = [25]u8{
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

const keccak_round_constants = [24]u64{
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
};

inline fn rotl64(val: u64, shift: u6) u64 {
    if (shift == 0) return val; // avoid undefined >> 64 when rho offset is zero
    const inv: u6 = @intCast(@as(u7, 64) - @as(u7, shift));
    return (val << shift) | (val >> inv);
}

fn keccakF1600(state: *[25]u64) void {
    var round: usize = 0;
    while (round < keccak_round_constants.len) : (round += 1) {
        var c: [5]u64 = undefined;
        var d: [5]u64 = undefined;

        var x: usize = 0;
        while (x < 5) : (x += 1) {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }

        x = 0;
        while (x < 5) : (x += 1) {
            d[x] = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
        }

        x = 0;
        while (x < 5) : (x += 1) {
            var y: usize = 0;
            while (y < 5) : (y += 1) {
                state[x + 5 * y] ^= d[x];
            }
        }

        var b: [25]u64 = undefined;
        var y: usize = 0;
        while (y < 5) : (y += 1) {
            x = 0;
            while (x < 5) : (x += 1) {
                const idx = x + 5 * y;
                const new_x = y;
                const new_y = (2 * x + 3 * y) % 5;
                b[new_x + 5 * new_y] = rotl64(state[idx], @intCast(keccak_rho_offsets[idx]));
            }
        }

        y = 0;
        while (y < 5) : (y += 1) {
            x = 0;
            while (x < 5) : (x += 1) {
                const idx = x + 5 * y;
                const b1 = b[((x + 1) % 5) + 5 * y];
                const b2 = b[((x + 2) % 5) + 5 * y];
                state[idx] = b[idx] ^ ((~b1) & b2);
            }
        }

        state[0] ^= keccak_round_constants[round];
    }
}

const Sha3_256Ctx = struct {
    state: [25]u64 = [_]u64{0} ** 25,
    pos: usize = 0,

    pub fn init() Sha3_256Ctx {
        return .{};
    }

    pub fn updateSlice(self: *Sha3_256Ctx, bytes: []const u8) void {
        for (bytes) |byte| {
            self.absorbByte(byte);
        }
    }

    fn absorbByte(self: *Sha3_256Ctx, byte: u8) void {
        const lane_index = self.pos / 8;
        const lane_shift: u6 = @intCast((self.pos % 8) * 8);
        self.state[lane_index] ^= @as(u64, byte) << lane_shift;
        self.pos += 1;

        if (self.pos == sha3_rate_bytes) {
            keccakF1600(&self.state);
            self.pos = 0;
        }
    }

    pub fn finalize(self: *Sha3_256Ctx, out: *[32]u8) void {
        const lane_index = self.pos / 8;
        const lane_shift: u6 = @intCast((self.pos % 8) * 8);
        // SHA3-256 uses the 0x06 domain suffix followed by the mandatory 1-bit trailer.
        self.state[lane_index] ^= @as(u64, 0x06) << lane_shift;

        const last_index = (sha3_rate_bytes - 1) / 8;
        const last_shift: u6 = @intCast(((sha3_rate_bytes - 1) % 8) * 8);
        self.state[last_index] ^= @as(u64, 0x80) << last_shift;

        keccakF1600(&self.state);
        self.pos = 0;
        self.squeeze(out[0..]);
    }

    fn squeeze(self: *Sha3_256Ctx, out: []u8) void {
        var produced: usize = 0;
        while (produced < out.len) : (produced += 1) {
            if (self.pos == sha3_rate_bytes) {
                keccakF1600(&self.state);
                self.pos = 0;
            }

            const lane_index = self.pos / 8;
            const lane_shift: u6 = @intCast((self.pos % 8) * 8);
            out[produced] = @truncate(self.state[lane_index] >> lane_shift);
            self.pos += 1;
        }
    }
};

pub fn sha2_256(m: *Machine, args: *LinkedValues) *const Value {
    const input = args.value.unwrapBytestring();
    var hasher = Sha256Ctx.init();

    var chunk: [64]u8 = undefined;
    var chunk_len: usize = 0;
    const byte_len: usize = @intCast(input.length);

    var i: usize = 0;
    while (i < byte_len) : (i += 1) {
        chunk[chunk_len] = @as(u8, @truncate(input.bytes[i]));
        chunk_len += 1;

        if (chunk_len == chunk.len) {
            hasher.updateSlice(chunk[0..]);
            chunk_len = 0;
        }
    }

    if (chunk_len != 0) {
        hasher.updateSlice(chunk[0..chunk_len]);
    }

    var digest: [32]u8 = undefined;
    hasher.finalize(&digest);

    var result = m.heap.createArray(u32, 32 + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 32;
    for (digest, 0..) |byte, idx| {
        result[4 + idx] = byte;
    }

    return createConst(m.heap, @ptrCast(result));
}

pub fn sha3_256(m: *Machine, args: *LinkedValues) *const Value {
    const input = args.value.unwrapBytestring();
    var hasher = Sha3_256Ctx.init();

    const byte_len: usize = @intCast(input.length);
    var chunk: [64]u8 = undefined;
    var chunk_len: usize = 0;

    var i: usize = 0;
    while (i < byte_len) : (i += 1) {
        chunk[chunk_len] = @as(u8, @truncate(input.bytes[i]));
        chunk_len += 1;

        if (chunk_len == chunk.len) {
            hasher.updateSlice(chunk[0..]);
            chunk_len = 0;
        }
    }

    if (chunk_len != 0) {
        hasher.updateSlice(chunk[0..chunk_len]);
    }

    var digest: [32]u8 = undefined;
    hasher.finalize(&digest);

    var result = m.heap.createArray(u32, 32 + 4);
    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.bytesType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 32;
    for (digest, 0..) |byte, idx| {
        result[4 + idx] = byte;
    }

    return createConst(m.heap, @ptrCast(result));
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
    const cond_value = args.next.?.next.?.value;

    // Avoid unwrapBool so invalid inputs trigger a silent evaluation failure
    // instead of touching the (unimplemented) debug console.
    const cond = switch (cond_value.*) {
        .constant => |c| switch (c.constType().*) {
            .boolean => c.bln(),
            else => builtinEvaluationFailure(),
        },
        else => builtinEvaluationFailure(),
    };

    return if (cond) then else otherwise;
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

    // The Plutus trace builtin is effectful only through the surrounding
    // execution environment, so on the CEK we simply force the string argument
    // (to preserve error behaviour) and return the value.  Writing to the
    // original memory-mapped console at 0xA000_1000 causes `SectionNotFound`
    // errors under the conformance runner, which doesn't emulate that device.
    _ = msg;

    return then;
}

// Pairs functions
pub fn fstPair(m: *Machine, args: *LinkedValues) *const Value {
    const pair = args.value.unwrapPair();
    const c = Constant{
        .length = pair.first_type_len,
        .type_list = pair.first_type,
        .value = pair.first_value,
    };
    const result = m.heap.create(Constant, &c);
    return createConst(m.heap, result);
}

pub fn sndPair(m: *Machine, args: *LinkedValues) *const Value {
    const pair = args.value.unwrapPair();
    const c = Constant{
        .length = pair.second_type_len,
        .type_list = pair.second_type,
        .value = pair.second_value,
    };
    const result = m.heap.create(Constant, &c);
    return createConst(m.heap, result);
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
    }

    // mkCons is partial: mismatched element types must signal an evaluation failure.
    builtinEvaluationFailure();
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
    const list_const = y.unwrapConstant();
    const list = list_const.list();

    if (list.length == 0) {
        // tailList is partial per the spec; empty lists must fail.
        builtinEvaluationFailure();
    }

    const result = m.heap.createArray(u32, 2);
    result[0] = list.length - 1;
    result[1] = @intFromPtr(list.items.?.next);

    const c = Constant{
        .length = list_const.length,
        .type_list = list_const.type_list,
        .value = @intFromPtr(result),
    };

    const con = m.heap.create(Constant, &c);

    return createConst(m.heap, con);
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
const large_constr_tag_flag: u32 = 0x80000000; // Mirrors serializer LARGE_CONSTR_TAG_FLAG

// Data constants are either runtime pointers (type list matches `dataTypePtr`) or
// serialized payloads tagged with `serialized_data_const_tag`. Reject everything else
// so callers can safely materialize the payload.
fn ensureDataConstant(con: *const Constant, comptime type_error_msg: []const u8) void {
    _ = type_error_msg; // Preserve signature for clearer call sites; failure always bubbles up.
    const uses_runtime_layout = @intFromPtr(con.type_list) == runtimeDataTypeAddr();
    if (uses_runtime_layout) {
        if (con.constType().* == .data) {
            return;
        }

        builtinEvaluationFailure();
    }

    if (con.length == serialized_data_const_tag) {
        return;
    }

    builtinEvaluationFailure();
}

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

fn buildDataListFromConstantList(
    m: *Machine,
    list: List,
    comptime type_error_msg: []const u8,
    comptime null_payload_msg: []const u8,
) ?*DataListNode {
    ensureListHoldsData(list, type_error_msg);

    var head: ?*DataListNode = null;
    var tail: ?*DataListNode = null;
    var cursor = list.items;

    while (cursor) |node| {
        const data_ptr = materializeDataElement(m, node.value, null_payload_msg);
        const new_node = m.heap.create(DataListNode, &DataListNode{
            .value = data_ptr,
            .next = null,
        });

        if (tail) |t| {
            t.next = new_node;
        } else {
            head = new_node;
        }
        tail = new_node;
        cursor = node.next;
    }

    return head;
}

fn buildDataPairListFromConstantList(
    m: *Machine,
    list: List,
    comptime type_error_msg: []const u8,
    comptime null_payload_msg: []const u8,
) ?*DataPairNode {
    ensureListHoldsDataPairs(list, type_error_msg);

    var head: ?*DataPairNode = null;
    var tail: ?*DataPairNode = null;
    var cursor = list.items;

    while (cursor) |node| {
        if (node.value == 0) {
            utils.printlnString(null_payload_msg);
            utils.exit(std.math.maxInt(u32));
        }

        const pair_payload: *const PairPayload = @ptrFromInt(node.value);
        const key_data = materializeDataElement(m, pair_payload.first, null_payload_msg);
        const value_data = materializeDataElement(m, pair_payload.second, null_payload_msg);

        const new_node = m.heap.create(DataPairNode, &DataPairNode{
            .key = key_data,
            .value = value_data,
            .next = null,
        });

        if (tail) |t| {
            t.next = new_node;
        } else {
            head = new_node;
        }
        tail = new_node;
        cursor = node.next;
    }

    return head;
}

fn ensureListHoldsData(list: List, comptime type_error_msg: []const u8) void {
    if (list.type_length == 0) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }

    const inner = list.inner_type;
    if (@intFromPtr(inner) == 0) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }

    if (inner[0] != ConstantType.data) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }
}

fn ensureListHoldsDataPairs(list: List, comptime type_error_msg: []const u8) void {
    if (list.type_length < 3) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }

    const inner = list.inner_type;
    if (@intFromPtr(inner) == 0) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }

    if (inner[0] != ConstantType.pair or inner[1] != ConstantType.data or inner[2] != ConstantType.data) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }
}

fn materializeDataElement(
    m: *Machine,
    payload_addr: u32,
    comptime null_payload_msg: []const u8,
) *const Data {
    _ = null_payload_msg; // Builtins signal failure uniformly, so just abort evaluation.
    if (payload_addr == 0) {
        builtinEvaluationFailure();
    }

    if (serializedPayloadWordCount(payload_addr)) |word_count| {
        const payload_ptr: [*]const u8 = @ptrFromInt(payload_addr);
        var reader = SerializedDataReader.init(payload_ptr, word_count);
        const data_ptr = decodeSerializedDataPayload(m.heap, &reader);
        reader.ensureFullyConsumed();
        return data_ptr;
    }

    return @ptrFromInt(payload_addr);
}

const SerializedDataReader = struct {
    bytes: [*]const u8,
    len: u32,
    offset: u32,

    fn init(ptr: [*]const u8, word_count: u32) SerializedDataReader {
        const total_bytes = wordCountToByteLen(word_count);
        return .{ .bytes = ptr, .len = total_bytes, .offset = 0 };
    }

    fn readU32(self: *SerializedDataReader) u32 {
        if (self.len - self.offset < 4) {
            invalidSerializedData();
        }

        var result: u32 = 0;
        var i: u32 = 0;
        while (i < 4) : (i += 1) {
            const idx = self.offset + i;
            const shift: u5 = @intCast(i * 8);
            result |= (@as(u32, self.bytes[@intCast(idx)])) << shift;
        }

        self.offset += 4;
        return result;
    }

    fn readU8(self: *SerializedDataReader) u8 {
        if (self.offset >= self.len) {
            invalidSerializedData();
        }

        const byte = self.bytes[@intCast(self.offset)];
        self.offset += 1;
        return byte;
    }

    fn readBytes(self: *SerializedDataReader, byte_len: u32) [*]const u8 {
        if (self.len - self.offset < byte_len) {
            invalidSerializedData();
        }

        const start = self.offset;
        self.offset += byte_len;

        const base_addr = @intFromPtr(self.bytes);
        const start_offset: usize = @intCast(start);
        return @ptrFromInt(base_addr + start_offset);
    }

    fn alignToWord(self: *SerializedDataReader) void {
        const rem = self.offset & 3;
        if (rem == 0) return;

        const skip = 4 - rem;
        if (self.len - self.offset < skip) {
            invalidSerializedData();
        }
        self.offset += skip;
    }

    fn sliceWords(self: *SerializedDataReader, word_count: u32) SerializedDataReader {
        const byte_len = wordCountToByteLen(word_count);
        if (self.len - self.offset < byte_len) {
            invalidSerializedData();
        }

        const start = self.offset;
        self.offset += byte_len;

        const base_addr = @intFromPtr(self.bytes);
        const start_offset: usize = @intCast(start);

        return .{
            .bytes = @ptrFromInt(base_addr + start_offset),
            .len = byte_len,
            .offset = 0,
        };
    }

    fn ensureFullyConsumed(self: *SerializedDataReader) void {
        if (self.offset != self.len) {
            invalidSerializedData();
        }
    }
};

fn wordCountToByteLen(word_count: u32) u32 {
    if (word_count == 0) {
        invalidSerializedData();
    }

    if (word_count > std.math.maxInt(u32) / 4) {
        invalidSerializedData();
    }

    return word_count * 4;
}

const max_serialized_payload_words: u32 = 0x10000000;

fn serializedPayloadWordCount(payload_addr: u32) ?u32 {
    const header_size = @as(u32, @sizeOf(u32) * 2);
    if (payload_addr < header_size) {
        return null;
    }

    const header_ptr: [*]const u32 = @ptrFromInt(payload_addr - header_size);
    const tag = header_ptr[0];
    const words = header_ptr[1];

    if (tag == serialized_data_const_tag and words > 0 and words < max_serialized_payload_words) {
        return words;
    }

    return null;
}

fn decodeSerializedDataPayload(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const tag = reader.readU32();

    return switch (tag) {
        data_tag_constr => decodeSerializedConstr(heap, reader),
        data_tag_map => decodeSerializedMap(heap, reader),
        data_tag_list => decodeSerializedList(heap, reader),
        data_tag_integer => decodeSerializedInteger(heap, reader),
        data_tag_bytes => decodeSerializedBytes(heap, reader),
        else => {
            invalidSerializedData();
        },
    };
}

fn decodeSerializedConstr(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const encoded_tag = reader.readU32();
    const field_count = reader.readU32();

    var head: ?*DataListNode = null;
    var tail: ?*DataListNode = null;

    var i: u32 = 0;
    while (i < field_count) : (i += 1) {
        const field_words = reader.readU32();
        var field_reader = reader.sliceWords(field_words);
        const field_data = decodeSerializedDataPayload(heap, &field_reader);
        field_reader.ensureFullyConsumed();

        const node = heap.create(DataListNode, &DataListNode{
            .value = field_data,
            .next = null,
        });

        if (tail) |t| {
            t.next = node;
        } else {
            head = node;
        }
        tail = node;
    }

    const payload = ConstrData{
        .tag = decodeConstrTag(encoded_tag),
        .fields = head,
    };

    return heap.create(Data, &.{ .constr = payload });
}

fn decodeSerializedList(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const elem_count = reader.readU32();

    var head: ?*DataListNode = null;
    var tail: ?*DataListNode = null;

    var i: u32 = 0;
    while (i < elem_count) : (i += 1) {
        const elem_words = reader.readU32();
        var elem_reader = reader.sliceWords(elem_words);
        const elem_data = decodeSerializedDataPayload(heap, &elem_reader);
        elem_reader.ensureFullyConsumed();

        const node = heap.create(DataListNode, &DataListNode{
            .value = elem_data,
            .next = null,
        });

        if (tail) |t| {
            t.next = node;
        } else {
            head = node;
        }
        tail = node;
    }

    return heap.create(Data, &.{ .list = head });
}

fn decodeSerializedMap(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const pair_count = reader.readU32();

    var head: ?*DataPairNode = null;
    var tail: ?*DataPairNode = null;

    var i: u32 = 0;
    while (i < pair_count) : (i += 1) {
        const key_words = reader.readU32();
        var key_reader = reader.sliceWords(key_words);
        const key_data = decodeSerializedDataPayload(heap, &key_reader);
        key_reader.ensureFullyConsumed();

        const value_words = reader.readU32();
        var value_reader = reader.sliceWords(value_words);
        const value_data = decodeSerializedDataPayload(heap, &value_reader);
        value_reader.ensureFullyConsumed();

        const node = heap.create(DataPairNode, &DataPairNode{
            .key = key_data,
            .value = value_data,
            .next = null,
        });

        if (tail) |t| {
            t.next = node;
        } else {
            head = node;
        }
        tail = node;
    }

    return heap.create(Data, &.{ .map = head });
}

fn decodeSerializedInteger(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const sign = reader.readU8();
    const word_count = reader.readU32();

    const words_buf = heap.createArray(u32, word_count);
    var i: u32 = 0;
    while (i < word_count) : (i += 1) {
        words_buf[i] = reader.readU32();
    }
    reader.alignToWord();

    const words_view: [*]const u32 = words_buf;

    const big = BigInt{
        .sign = @intCast(sign),
        .length = word_count,
        .words = words_view,
    };

    return heap.create(Data, &.{ .integer = big });
}

fn decodeSerializedBytes(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const byte_len = reader.readU32();
    const src = reader.readBytes(byte_len);
    reader.alignToWord();

    const words_buf = heap.createArray(u32, byte_len);
    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        words_buf[i] = @intCast(src[@intCast(i)]);
    }

    const bytes_words: [*]const u32 = words_buf;

    const bytes_view = Bytes{
        .length = byte_len,
        .bytes = bytes_words,
    };

    return heap.create(Data, &.{ .bytes = bytes_view });
}

fn decodeConstrTag(encoded: u32) u32 {
    if ((encoded & large_constr_tag_flag) != 0) {
        return encoded & ~large_constr_tag_flag;
    }

    if (encoded >= 1280) {
        return (encoded - 1280) + 7;
    }

    if (encoded >= 121 and encoded <= 127) {
        return encoded - 121;
    }

    return encoded;
}

fn invalidSerializedData() noreturn {
    builtinEvaluationFailure();
}

pub fn constrData(m: *Machine, args: *LinkedValues) *const Value {
    // First arg: list of Data (fields)
    const fields_list = args.value.unwrapList();

    const tag_int = args.next.?.value.unwrapInteger();

    if (tag_int.sign == 1 or tag_int.length > 1) {
        utils.printlnString("constrData: tag must be a non-negative integer that fits in u32");
        utils.exit(std.math.maxInt(u32));
    }
    const tag: u32 = tag_int.words[0];

    const field_nodes = buildDataListFromConstantList(
        m,
        fields_list,
        "constrData expects a list of Data",
        "constrData: null Data constant payload",
    );

    const constr_payload = ConstrData{
        .tag = tag,
        .fields = field_nodes,
    };

    const data_ptr = m.heap.create(Data, &.{ .constr = constr_payload });

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn mapData(m: *Machine, args: *LinkedValues) *const Value {
    const pairs_list = args.value.unwrapList();

    const pair_nodes = buildDataPairListFromConstantList(
        m,
        pairs_list,
        "mapData expects a list of (Data, Data) pairs",
        "mapData: null Data constant payload",
    );

    const data_ptr = m.heap.create(Data, &.{ .map = pair_nodes });

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn listData(m: *Machine, args: *LinkedValues) *const Value {
    const elements_list = args.value.unwrapList();

    const data_nodes = buildDataListFromConstantList(
        m,
        elements_list,
        "listData expects a list of Data",
        "listData: null Data constant payload",
    );

    const data_ptr = m.heap.create(Data, &.{ .list = data_nodes });

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn iData(m: *Machine, args: *LinkedValues) *const Value {
    const int_arg = args.value.unwrapInteger();

    const data_ptr = m.heap.create(Data, &.{ .integer = int_arg });

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
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

pub fn unConstrData(m: *Machine, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unConstrData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unConstrData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .constr => |constr_payload| {
            const tag_limb_count: u32 = if (constr_payload.tag == 0) 0 else 1;
            const tag_payload = m.heap.createArray(u32, tag_limb_count + 2);
            tag_payload[0] = 0; // positive sign
            tag_payload[1] = tag_limb_count;
            if (tag_limb_count == 1) {
                tag_payload[2] = constr_payload.tag;
            }

            var list_head: ?*ListNode = null;
            var list_tail: ?*ListNode = null;
            var list_len: u32 = 0;
            var cursor = constr_payload.fields;

            // Rebuild a UPLC list of Data constants pointing at the decoded fields.
            while (cursor) |field| {
                const node = m.heap.create(ListNode, &ListNode{
                    .value = @intFromPtr(field.value),
                    .next = null,
                });

                if (list_tail) |tail| {
                    tail.next = node;
                } else {
                    list_head = node;
                }
                list_tail = node;
                list_len += 1;
                cursor = field.next;
            }

            const list_payload = m.heap.createArray(u32, 2);
            list_payload[0] = list_len;
            list_payload[1] = if (list_head) |head| @intFromPtr(head) else 0;

            const pair_payload = m.heap.create(PairPayload, &PairPayload{
                .first = @intFromPtr(tag_payload),
                .second = @intFromPtr(list_payload),
            });

            const pair_type: [*]const ConstantType = @ptrCast(&UnConstrReturnTypeDescriptor);
            const pair_const = Constant{
                .length = @intCast(UnConstrReturnTypeDescriptor.len),
                .type_list = pair_type,
                .value = @intFromPtr(pair_payload),
            };

            return createConst(m.heap, m.heap.create(Constant, &pair_const));
        },
        else => builtinEvaluationFailure(),
    }
}

pub fn unMapData(m: *Machine, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unMapData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unMapData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .map => |pairs_head| {
            var uplc_head: ?*ListNode = null;
            var uplc_tail: ?*ListNode = null;
            var length: u32 = 0;
            var cursor = pairs_head;

            while (cursor) |pair_node| {
                // Rebuild a `(Data, Data)` payload whose components point at the decoded entries.
                const pair_payload = m.heap.create(PairPayload, &PairPayload{
                    .first = @intFromPtr(pair_node.key),
                    .second = @intFromPtr(pair_node.value),
                });

                const node = m.heap.create(ListNode, &ListNode{
                    .value = @intFromPtr(pair_payload),
                    .next = null,
                });

                if (uplc_tail) |tail| {
                    tail.next = node;
                } else {
                    uplc_head = node;
                }
                uplc_tail = node;
                length += 1;
                cursor = pair_node.next;
            }

            const list_payload = m.heap.createArray(u32, 2);
            list_payload[0] = length;
            list_payload[1] = if (uplc_head) |head| @intFromPtr(head) else 0;

            const list_const = Constant{
                .length = @intCast(DataPairListTypeDescriptor.len),
                .type_list = @ptrCast(&DataPairListTypeDescriptor),
                .value = @intFromPtr(list_payload),
            };

            return createConst(m.heap, m.heap.create(Constant, &list_const));
        },
        else => builtinEvaluationFailure(),
    }
}

pub fn unListData(m: *Machine, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unListData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unListData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .list => |list_head| {
            var uplc_head: ?*ListNode = null;
            var uplc_tail: ?*ListNode = null;
            var length: u32 = 0;
            var cursor = list_head;

            // Rebuild a UPLC list whose nodes reference the decoded Data elements.
            while (cursor) |elem| {
                const node = m.heap.create(ListNode, &ListNode{
                    .value = @intFromPtr(elem.value),
                    .next = null,
                });

                if (uplc_tail) |tail| {
                    tail.next = node;
                } else {
                    uplc_head = node;
                }
                uplc_tail = node;
                length += 1;
                cursor = elem.next;
            }

            const list_payload = m.heap.createArray(u32, 2);
            list_payload[0] = length;
            list_payload[1] = if (uplc_head) |head| @intFromPtr(head) else 0;

            const list_const = Constant{
                .length = 2,
                .type_list = @ptrCast(ConstantType.listDataType()),
                .value = @intFromPtr(list_payload),
            };

            return createConst(m.heap, m.heap.create(Constant, &list_const));
        },
        else => builtinEvaluationFailure(),
    }
}

pub fn unIData(m: *Machine, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unIData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unIData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .integer => |int_payload| {
            const int_const = int_payload.createConstant(ConstantType.integerType(), m.heap);
            return createConst(m.heap, int_const);
        },
        else => builtinEvaluationFailure(),
    }
}

pub fn unBData(m: *Machine, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unBData expects a Data constant");

    // Serialized Data constants embed their payload inline, so materialize them
    // into the heap to reuse the runtime layout for both representations.
    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unBData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .bytes => |bytes_payload| {
            const bytes_const = bytes_payload.createConstant(ConstantType.bytesType(), m.heap);
            return createConst(m.heap, bytes_const);
        },
        else => builtinEvaluationFailure(),
    }
}


const SerializedView = struct {
    words: [*]const u32,
    len: u32,
};

const SerializedOwnedView = struct {
    words: [*]u32,
    len: u32,
};

const DataView = union(enum) {
    runtime: *const Data,
    serialized: SerializedView,
};

pub fn equalsData(m: *Machine, args: *LinkedValues) *const Value {
    const rhs_const = args.value.unwrapConstant();
    const lhs_const = args.next.?.value.unwrapConstant();

    const result = dataConstantsEqual(m.heap, lhs_const, rhs_const);

    var bool_result = m.heap.createArray(u32, 4);
    bool_result[0] = 1;
    bool_result[1] = @intFromPtr(ConstantType.booleanType());
    bool_result[2] = @intFromPtr(bool_result + 3);
    bool_result[3] = @intFromBool(result);

    return createConst(m.heap, @ptrCast(bool_result));
}

fn dataConstantsEqual(heap: *Heap, lhs: *const Constant, rhs: *const Constant) bool {
    const lhs_view = classifyDataConstant(lhs);
    const rhs_view = classifyDataConstant(rhs);

    return switch (lhs_view) {
        .runtime => |lhs_data| switch (rhs_view) {
            .runtime => |rhs_data| heapDataEqual(lhs_data, rhs_data),
            .serialized => |rhs_ser| serializedEqualsRuntime(heap, rhs_ser, lhs_data),
        },
        .serialized => |lhs_ser| switch (rhs_view) {
            .runtime => |rhs_data| serializedEqualsRuntime(heap, lhs_ser, rhs_data),
            .serialized => |rhs_ser| serializedViewsEqual(lhs_ser, rhs_ser),
        },
    };
}

fn classifyDataConstant(con: *const Constant) DataView {
    const uses_runtime_layout = @intFromPtr(con.type_list) == runtimeDataTypeAddr();
    if (uses_runtime_layout) {
        if (con.constType().* != .data) {
            utils.printlnString("equalsData expects Data constants");
            utils.exit(std.math.maxInt(u32));
        }

        const ptr = con.rawValue();
        if (ptr == 0) {
            utils.printlnString("equalsData received null Data pointer");
            utils.exit(std.math.maxInt(u32));
        }
        return .{ .runtime = @ptrFromInt(ptr) };
    }

    if (con.length != serialized_data_const_tag) {
        utils.printlnString("equalsData expects Data constants");
        utils.exit(std.math.maxInt(u32));
    }

    const len_words: u32 = @intCast(@intFromPtr(con.type_list));
    const data_ptr = con.rawValue();
    if (data_ptr == 0) {
        utils.printlnString("equalsData received null serialized Data payload");
        utils.exit(std.math.maxInt(u32));
    }

    return .{
        .serialized = .{
            .words = @ptrFromInt(data_ptr),
            .len = len_words,
        },
    };
}

fn serializedViewsEqual(a: SerializedView, b: SerializedView) bool {
    if (a.len != b.len) return false;

    var i: u32 = 0;
    while (i < a.len) : (i += 1) {
        if (a.words[i] != b.words[i]) return false;
    }
    return true;
}

fn serializedEqualsRuntime(heap: *Heap, serialized: SerializedView, runtime_data: *const Data) bool {
    const encoded = serializeRuntimeData(heap, runtime_data);
    defer heap.reclaimHeap(u32, encoded.len);
    return serializedViewsEqual(
        .{ .words = encoded.words, .len = encoded.len },
        serialized,
    );
}

fn serializeRuntimeData(heap: *Heap, data: *const Data) SerializedOwnedView {
    const word_count = dataSerializedWordCount(data);
    const buffer = heap.createArray(u32, word_count);
    var writer = PayloadWriter.init(buffer, word_count);

    writeSerializedData(&writer, data);

    return .{ .words = buffer, .len = word_count };
}

fn writeSerializedData(writer: *PayloadWriter, data: *const Data) void {
    switch (data.*) {
        .constr => |payload| {
            writer.writeU32(data_tag_constr);
            writer.writeU32(encodeConstrTag(payload.tag));
            writer.writeU32(countDataList(payload.fields));

            var node = payload.fields;
            while (node) |field| {
                const nested_words = dataSerializedWordCount(field.value);
                writer.writeU32(nested_words);
                writeSerializedData(writer, field.value);
                node = field.next;
            }
        },
        .map => |pairs| {
            writer.writeU32(data_tag_map);
            writer.writeU32(countDataPairs(pairs));

            var node = pairs;
            while (node) |pair| {
                const key_words = dataSerializedWordCount(pair.key);
                writer.writeU32(key_words);
                writeSerializedData(writer, pair.key);

                const value_words = dataSerializedWordCount(pair.value);
                writer.writeU32(value_words);
                writeSerializedData(writer, pair.value);

                node = pair.next;
            }
        },
        .list => |list_head| {
            writer.writeU32(data_tag_list);
            writer.writeU32(countDataList(list_head));

            var node = list_head;
            while (node) |elem| {
                const nested_words = dataSerializedWordCount(elem.value);
                writer.writeU32(nested_words);
                writeSerializedData(writer, elem.value);
                node = elem.next;
            }
        },
        .integer => |int_val| {
            writer.writeU32(data_tag_integer);
            encodeBigInt(writer, int_val);
        },
        .bytes => |bytes_val| {
            writer.writeU32(data_tag_bytes);
            encodeBytes(writer, bytes_val);
        },
    }
}

const data_tag_constr: u32 = 0;
const data_tag_map: u32 = 1;
const data_tag_list: u32 = 2;
const data_tag_integer: u32 = 3;
const data_tag_bytes: u32 = 4;

fn encodeConstrTag(tag: u32) u32 {
    if (tag < 7) return 121 + tag;
    return 1280 + (tag - 7);
}

fn dataSerializedWordCount(data: *const Data) u32 {
    return switch (data.*) {
        .constr => blk: {
            var total: u32 = 3; // variant tag, encoded tag, field count
            var node = data.constr.fields;
            while (node) |field| {
                total += 1; // field length prefix
                total += dataSerializedWordCount(field.value);
                node = field.next;
            }
            break :blk total;
        },
        .map => blk: {
            var total: u32 = 2; // variant tag + pair count
            var node = data.map;
            while (node) |pair| {
                total += 1;
                total += dataSerializedWordCount(pair.key);
                total += 1;
                total += dataSerializedWordCount(pair.value);
                node = pair.next;
            }
            break :blk total;
        },
        .list => blk: {
            var total: u32 = 2; // variant tag + element count
            var node = data.list;
            while (node) |elem| {
                total += 1;
                total += dataSerializedWordCount(elem.value);
                node = elem.next;
            }
            break :blk total;
        },
        .integer => blk: {
            const byte_len: u32 = data.integer.length * 4;
            const total_bytes = 4 + 1 + 4 + byte_len;
            break :blk bytesToWords(total_bytes);
        },
        .bytes => blk: {
            const byte_len = data.bytes.length;
            const total_bytes = 4 + 4 + byte_len;
            break :blk bytesToWords(total_bytes);
        },
    };
}

fn bytesToWords(byte_len: u32) u32 {
    if (byte_len == 0) return 0;
    return (byte_len + 3) / 4;
}

const PayloadWriter = struct {
    bytes: [*]u8,
    total_bytes: u32,
    offset: u32,

    fn init(dst_words: [*]u32, len_words: u32) PayloadWriter {
        var i: u32 = 0;
        while (i < len_words) : (i += 1) {
            dst_words[i] = 0;
        }
        return .{
            .bytes = @ptrCast(dst_words),
            .total_bytes = len_words * 4,
            .offset = 0,
        };
    }

    fn writeU32(self: *PayloadWriter, value: u32) void {
        var buf = value;
        self.writeBytes(std.mem.asBytes(&buf));
    }

    fn writeByte(self: *PayloadWriter, value: u8) void {
        if (self.offset == self.total_bytes) {
            builtinEvaluationFailure();
        }
        const idx: usize = @intCast(self.offset);
        self.bytes[idx] = value;
        self.offset += 1;
    }

    fn writeBytes(self: *PayloadWriter, data: []const u8) void {
        if (data.len == 0) return;
        const seg_len: u32 = @intCast(data.len);
        if (self.offset + seg_len > self.total_bytes) {
            builtinEvaluationFailure();
        }
        const start: usize = @intCast(self.offset);
        const dst = self.bytes[start .. start + data.len];
        @memcpy(dst, data);
        self.offset += seg_len;
    }
};

fn encodeBigInt(writer: *PayloadWriter, int_val: BigInt) void {
    writer.writeByte(@intFromBool(int_val.sign != 0));
    const byte_len: u32 = int_val.length * 4;
    writer.writeU32(byte_len);

    var i: u32 = 0;
    while (i < int_val.length) : (i += 1) {
        writer.writeU32(int_val.words[i]);
    }
}

fn encodeBytes(writer: *PayloadWriter, bytes_val: Bytes) void {
    writer.writeU32(bytes_val.length);
    var i: u32 = 0;
    while (i < bytes_val.length) : (i += 1) {
        writer.writeByte(@truncate(bytes_val.bytes[i]));
    }
}

fn heapDataEqual(lhs: *const Data, rhs: *const Data) bool {
    const tag_lhs = std.meta.activeTag(lhs.*);
    const tag_rhs = std.meta.activeTag(rhs.*);
    if (tag_lhs != tag_rhs) return false;

    return switch (tag_lhs) {
        .constr => lhs.constr.tag == rhs.constr.tag and
            dataListEqual(lhs.constr.fields, rhs.constr.fields),
        .map => dataPairEqual(lhs.map, rhs.map),
        .list => dataListEqual(lhs.list, rhs.list),
        .integer => bigIntEqual(lhs.integer, rhs.integer),
        .bytes => bytesEqual(lhs.bytes, rhs.bytes),
    };
}

fn dataListEqual(a: ?*DataListNode, b: ?*DataListNode) bool {
    var left = a;
    var right = b;

    while (true) {
        if (left == null and right == null) return true;
        if (left == null or right == null) return false;

        const lhs_node = left.?;
        const rhs_node = right.?;
        if (!heapDataEqual(lhs_node.value, rhs_node.value)) return false;

        left = lhs_node.next;
        right = rhs_node.next;
    }
}

fn dataPairEqual(a: ?*DataPairNode, b: ?*DataPairNode) bool {
    var left = a;
    var right = b;

    while (true) {
        if (left == null and right == null) return true;
        if (left == null or right == null) return false;

        const lhs_node = left.?;
        const rhs_node = right.?;
        if (!heapDataEqual(lhs_node.key, rhs_node.key)) return false;
        if (!heapDataEqual(lhs_node.value, rhs_node.value)) return false;

        left = lhs_node.next;
        right = rhs_node.next;
    }
}

fn bigIntEqual(a: BigInt, b: BigInt) bool {
    if (a.sign != b.sign) return false;
    if (a.length != b.length) return false;

    var i: u32 = 0;
    while (i < a.length) : (i += 1) {
        if (a.words[i] != b.words[i]) return false;
    }
    return true;
}

fn bytesEqual(a: Bytes, b: Bytes) bool {
    if (a.length != b.length) return false;

    var i: u32 = 0;
    while (i < a.length) : (i += 1) {
        if (a.bytes[i] != b.bytes[i]) return false;
    }
    return true;
}

fn countDataList(head: ?*DataListNode) u32 {
    var count: u32 = 0;
    var cursor = head;
    while (cursor) |node| {
        count += 1;
        cursor = node.next;
    }
    return count;
}

fn countDataPairs(head: ?*DataPairNode) u32 {
    var count: u32 = 0;
    var cursor = head;
    while (cursor) |node| {
        count += 1;
        cursor = node.next;
    }
    return count;
}

// Misc constructors
pub fn mkPairData(m: *Machine, args: *LinkedValues) *const Value {
    // LinkedValues stores the most recently applied argument first, so the
    // first component lives in `args.next`.
    const second = args.value.unwrapConstant();
    ensureDataConstant(second, "mkPairData expects Data constants");

    const first = args.next.?.value.unwrapConstant();
    ensureDataConstant(first, "mkPairData expects Data constants");

    const payload = m.heap.create(PairPayload, &PairPayload{
        .first = first.rawValue(),
        .second = second.rawValue(),
    });

    const con = Constant{
        .length = @intCast(DataPairTypeDescriptor.len),
        .type_list = @ptrCast(&DataPairTypeDescriptor),
        .value = @intFromPtr(payload),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
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

pub fn mkNilPairData(m: *Machine, args: *LinkedValues) *const Value {
    const unit_arg = args.value;
    unit_arg.unwrapUnit();

    const payload = m.heap.createArray(u32, 2);
    payload[0] = 0;
    payload[1] = 0;

    const list_const = Constant{
        .length = @intCast(DataPairListTypeDescriptor.len),
        .type_list = @ptrCast(&DataPairListTypeDescriptor),
        .value = @intFromPtr(payload),
    };

    return createConst(m.heap, m.heap.create(Constant, &list_const));
}

pub fn serialiseData(m: *Machine, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "serialiseData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "serialiseData: null Data constant payload",
    );

    return serialiseRuntimeData(m, data_ptr);
}

fn serialiseRuntimeData(m: *Machine, data_ptr: *const Data) *const Value {
    const max_byte_len = cborEncodedByteUpperBound(data_ptr);
    const word_capacity = bytesToWords(max_byte_len);
    if (word_capacity == 0) builtinEvaluationFailure();

    const allocation = initByteStringAllocation(m.heap, max_byte_len);

    const packed_words = m.heap.createArray(u32, word_capacity);
    var writer = CborWriter.init(packed_words, word_capacity);
    encodeDataAsCbor(&writer, m.heap, data_ptr);

    const actual_bytes = writer.offset;
    if (actual_bytes == 0 or actual_bytes > max_byte_len) {
        builtinEvaluationFailure();
    }

    allocation.constant_words[3] = actual_bytes;
    copyPackedWordsToUnpacked(allocation.data_words, packed_words, actual_bytes);

    m.heap.reclaimHeap(u32, word_capacity);

    return createConst(m.heap, @ptrCast(allocation.constant_words));
}

// Estimate the number of CBOR bytes serialiseData will emit so we can size the
// scratch buffer without risking overflow when chunking long byte strings.
fn cborEncodedByteUpperBound(data: *const Data) u32 {
    const total = dataCborByteUpperBound(data);
    if (total == 0 or total > std.math.maxInt(u32)) {
        builtinEvaluationFailure();
    }
    return @intCast(total);
}

fn dataCborByteUpperBound(data: *const Data) u64 {
    return switch (data.*) {
        .constr => constrCborByteUpperBound(data.constr),
        .map => mapCborByteUpperBound(data.map),
        .list => listCborByteUpperBound(data.list),
        .integer => integerCborByteUpperBound(data.integer),
        .bytes => bytesValueCborByteUpperBound(data.bytes),
    };
}

fn constrCborByteUpperBound(payload: @FieldType(Data, "constr")) u64 {
    const info = computeConstrTag(payload.tag);
    var total: u64 = @as(u64, cborMajorLen(info.cbor_tag));
    if (info.use_tuple) {
        total = addOrFail64(total, @as(u64, cborMajorLen(2)));
        total = addOrFail64(total, @as(u64, cborMajorLen(info.constructor_index)));
    }
    total = addOrFail64(total, listCborByteUpperBound(payload.fields));
    return total;
}

fn listCborByteUpperBound(head: ?*DataListNode) u64 {
    const len = countDataList(head);
    // Include the definite-length array header emitted by encodeListAsCbor.
    var total: u64 = @as(u64, cborMajorLen(@as(u64, len)));
    var node = head;
    while (node) |entry| {
        total = addOrFail64(total, dataCborByteUpperBound(entry.value));
        node = entry.next;
    }
    return total;
}

fn mapCborByteUpperBound(head: ?*DataPairNode) u64 {
    const len = countDataPairs(head);
    var total: u64 = @as(u64, cborMajorLen(@as(u64, len)));
    var node = head;
    while (node) |pair| {
        total = addOrFail64(total, dataCborByteUpperBound(pair.key));
        total = addOrFail64(total, dataCborByteUpperBound(pair.value));
        node = pair.next;
    }
    return total;
}

fn integerCborByteUpperBound(int_val: BigInt) u64 {
    const used_words = significantWordCount(int_val.words, int_val.length);

    if (int_val.sign == 0) {
        if (used_words == 0) {
            return @as(u64, cborMajorLen(0));
        }

        if (wordsToU64(int_val.words, used_words)) |value| {
            return @as(u64, cborMajorLen(value));
        }

        const byte_len = magnitudeByteLen(int_val.words, used_words);
        return addOrFail64(@as(u64, cborMajorLen(2)), byteStringEncodedLen(byte_len));
    }

    if (used_words == 0) {
        builtinEvaluationFailure();
    }

    if (wordsToU64(int_val.words, used_words)) |value| {
        if (value == 0) builtinEvaluationFailure();
        return @as(u64, cborMajorLen(value - 1));
    }

    const byte_len = magnitudeByteLen(int_val.words, used_words);
    return addOrFail64(@as(u64, cborMajorLen(3)), byteStringEncodedLen(byte_len));
}

fn bytesValueCborByteUpperBound(bytes_val: Bytes) u64 {
    return dataByteStringEncodedLen(bytes_val.length);
}

fn byteStringEncodedLen(byte_len: u32) u64 {
    const len: u64 = byte_len;
    return len + @as(u64, cborMajorLen(len));
}

const byte_string_chunk_limit: u32 = 64;

fn dataByteStringEncodedLen(byte_len: u32) u64 {
    if (byte_len <= byte_string_chunk_limit) {
        return byteStringEncodedLen(byte_len);
    }

    const chunk_len: u32 = byte_string_chunk_limit;
    const chunk_header_len: u64 = 2; // 64 >= 24 so requires 2-byte header.
    const chunk_total: u64 = chunk_header_len + chunk_len;

    const full_chunks: u32 = byte_len / chunk_len;
    const remainder: u32 = byte_len % chunk_len;

    var total: u64 = 1; // Begin indefinite byte string.

    if (full_chunks > 0) {
        const full_total: u128 = @as(u128, full_chunks) * @as(u128, chunk_total);
        if (full_total > std.math.maxInt(u64)) builtinEvaluationFailure();
        total = addOrFail64(total, @intCast(full_total));
    }

    if (remainder > 0) {
        const header_len: u64 = if (remainder < 24) 1 else 2;
        total = addOrFail64(total, header_len + @as(u64, remainder));
    }

    // Account for the CBOR break code.
    return addOrFail64(total, 1);
}

fn cborMajorLen(value: u64) u32 {
    if (value < 24) return 1;
    if (value <= 0xFF) return 2;
    if (value <= 0xFFFF) return 3;
    if (value <= 0xFFFF_FFFF) return 5;
    return 9;
}

fn addOrFail64(a: u64, b: u64) u64 {
    if (a > std.math.maxInt(u64) - b) builtinEvaluationFailure();
    return a + b;
}

// Encode runtime Data into the CBOR form returned by serialiseData.
fn encodeDataAsCbor(writer: *CborWriter, heap: *Heap, data: *const Data) void {
    switch (data.*) {
        .constr => |payload| encodeConstrAsCbor(writer, heap, payload),
        .map => |pairs| encodeMapAsCbor(writer, heap, pairs),
        .list => |elements| encodeListAsCbor(writer, heap, elements),
        .integer => |int_val| encodeIntegerAsCbor(writer, heap, int_val),
        .bytes => |bytes_val| encodeBytesAsCbor(writer, bytes_val),
    }
}

fn encodeConstrAsCbor(writer: *CborWriter, heap: *Heap, payload: @FieldType(Data, "constr")) void {
    const info = computeConstrTag(payload.tag);
    writer.writeTag(info.cbor_tag);

    if (info.use_tuple) {
        writer.writeArray(2);
        writer.writeUnsigned(info.constructor_index);
    }

    encodeListAsCbor(writer, heap, payload.fields);
}

// Mirrors the Plutus ledger encoding for constructor tags.
fn computeConstrTag(tag: u32) struct { cbor_tag: u64, constructor_index: u64, use_tuple: bool } {
    const idx: u64 = tag;
    if (idx < 7) {
        return .{ .cbor_tag = 121 + idx, .constructor_index = idx, .use_tuple = false };
    }
    if (idx < 128) {
        return .{ .cbor_tag = 1280 + (idx - 7), .constructor_index = idx, .use_tuple = false };
    }
    return .{ .cbor_tag = 102, .constructor_index = idx, .use_tuple = true };
}

fn encodeListAsCbor(writer: *CborWriter, heap: *Heap, head: ?*DataListNode) void {
    const len = countDataList(head);
    // Ledger serialiseData uses definite-length arrays for lists, so mirror that shape here.
    writer.writeArray(len);

    var node = head;
    while (node) |entry| {
        encodeDataAsCbor(writer, heap, entry.value);
        node = entry.next;
    }
}

fn encodeMapAsCbor(writer: *CborWriter, heap: *Heap, head: ?*DataPairNode) void {
    const len = countDataPairs(head);
    writer.writeMap(len);
    if (len <= 1) {
        var node = head;
        while (node) |pair| {
            encodeDataAsCbor(writer, heap, pair.key);
            encodeDataAsCbor(writer, heap, pair.value);
            node = pair.next;
        }
        return;
    }

    // Ledger serialiseData follows canonical CBOR, which requires map keys to be
    // ordered by their encoded byte representation.
    const entries = heap.createArray(MapSortEntry, len);
    const order = heap.createArray(u32, len);

    var node = head;
    var idx: u32 = 0;
    while (node) |pair| {
        const key_bound = dataCborByteUpperBound(pair.key);
        if (key_bound == 0 or key_bound > std.math.maxInt(u32)) {
            builtinEvaluationFailure();
        }
        const byte_cap: u32 = @intCast(key_bound);
        const buffer_words = bytesToWords(byte_cap);
        if (buffer_words == 0) builtinEvaluationFailure();

        const buffer = heap.createArray(u32, buffer_words);
        var key_writer = CborWriter.init(buffer, buffer_words);
        encodeDataAsCbor(&key_writer, heap, pair.key);

        entries[@intCast(idx)] = .{
            .pair = pair,
            .key_buf_words = buffer,
            .buffer_words = buffer_words,
            .key_len = key_writer.offset,
        };
        order[@intCast(idx)] = idx;
        idx += 1;
        node = pair.next;
    }

    sortCanonicalMapOrder(entries, order, len);
    ensureCanonicalMapKeysUnique(entries, order, len);

    var ord_idx: u32 = 0;
    while (ord_idx < len) : (ord_idx += 1) {
        const entry_idx = order[@intCast(ord_idx)];
        const entry = entries[@intCast(entry_idx)];
        writeEncodedKeyBytes(writer, entry.key_buf_words, entry.key_len);
        encodeDataAsCbor(writer, heap, entry.pair.value);
    }

    reclaimMapKeyBuffers(heap, entries, len);
    heap.reclaimHeap(u32, len);
    heap.reclaimHeap(MapSortEntry, len);
}

const MapSortEntry = struct {
    pair: *DataPairNode,
    key_buf_words: [*]u32,
    buffer_words: u32,
    key_len: u32,
};

fn writeEncodedKeyBytes(writer: *CborWriter, words: [*]const u32, byte_len: u32) void {
    const bytes: [*]const u8 = @ptrCast(words);
    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        writer.writeByte(bytes[@intCast(i)]);
    }
}

fn sortCanonicalMapOrder(entries: [*]const MapSortEntry, order: [*]u32, len: u32) void {
    if (len <= 1) return;

    var i: u32 = 1;
    while (i < len) : (i += 1) {
        const current = order[@intCast(i)];
        var j = i;
        while (j > 0 and mapKeyLessThan(entries, current, order[@intCast(j - 1)])) {
            order[@intCast(j)] = order[@intCast(j - 1)];
            j -= 1;
        }
        order[@intCast(j)] = current;
    }
}

fn mapKeyLessThan(entries: [*]const MapSortEntry, lhs_idx: u32, rhs_idx: u32) bool {
    const lhs = entries[@intCast(lhs_idx)];
    const rhs = entries[@intCast(rhs_idx)];

    // Canonical CBOR order sorts by encoded byte-length before doing a
    // lexicographic comparison of the actual bytes.
    if (lhs.key_len != rhs.key_len) {
        return lhs.key_len < rhs.key_len;
    }

    const lhs_bytes: [*]const u8 = @ptrCast(lhs.key_buf_words);
    const rhs_bytes: [*]const u8 = @ptrCast(rhs.key_buf_words);

    var offset: u32 = 0;
    while (offset < lhs.key_len) : (offset += 1) {
        const left_byte = lhs_bytes[@intCast(offset)];
        const right_byte = rhs_bytes[@intCast(offset)];
        if (left_byte == right_byte) continue;
        return left_byte < right_byte;
    }
    return false;
}

fn mapKeyBytesEqual(lhs: MapSortEntry, rhs: MapSortEntry) bool {
    if (lhs.key_len != rhs.key_len) return false;

    const lhs_bytes: [*]const u8 = @ptrCast(lhs.key_buf_words);
    const rhs_bytes: [*]const u8 = @ptrCast(rhs.key_buf_words);

    var offset: u32 = 0;
    while (offset < lhs.key_len) : (offset += 1) {
        if (lhs_bytes[@intCast(offset)] != rhs_bytes[@intCast(offset)]) return false;
    }

    return true;
}

fn ensureCanonicalMapKeysUnique(entries: [*]const MapSortEntry, order: [*]const u32, len: u32) void {
    if (len <= 1) return;

    var idx: u32 = 1;
    while (idx < len) : (idx += 1) {
        const current = entries[@intCast(order[@intCast(idx)])];
        const previous = entries[@intCast(order[@intCast(idx - 1)])];

        // Canonical CBOR forbids duplicate keys, so fail when two encoded keys compare
        // equal after sorting.
        if (mapKeyBytesEqual(current, previous)) {
            builtinEvaluationFailure();
        }
    }
}

fn reclaimMapKeyBuffers(heap: *Heap, entries: [*]const MapSortEntry, len: u32) void {
    var idx = len;
    while (idx > 0) {
        idx -= 1;
        const entry = entries[@intCast(idx)];
        heap.reclaimHeap(u32, entry.buffer_words);
    }
}

// Ledger chunks byte strings longer than 64 bytes into an indefinite-length CBOR item.
fn encodeBytesAsCbor(writer: *CborWriter, bytes_val: Bytes) void {
    if (bytes_val.length <= byte_string_chunk_limit) {
        writer.writeByteStringPrefix(bytes_val.length);

        var i: u32 = 0;
        while (i < bytes_val.length) : (i += 1) {
            writer.writeByte(@intCast(bytes_val.bytes[i]));
        }
        return;
    }

    writer.beginIndefiniteByteString();

    var offset: u32 = 0;
    while (offset < bytes_val.length) {
        const remaining = bytes_val.length - offset;
        const chunk_len: u32 = if (remaining > byte_string_chunk_limit) byte_string_chunk_limit else remaining;

        writer.writeByteStringPrefix(chunk_len);

        var i: u32 = 0;
        while (i < chunk_len) : (i += 1) {
            const idx = offset + i;
            writer.writeByte(@intCast(bytes_val.bytes[idx]));
        }

        offset += chunk_len;
    }

    writer.writeBreak();
}

fn encodeIntegerAsCbor(writer: *CborWriter, heap: *Heap, int_val: BigInt) void {
    const used_words = significantWordCount(int_val.words, int_val.length);

    if (int_val.sign == 0) {
        if (used_words == 0) {
            writer.writeUnsigned(0);
            return;
        }

        if (wordsToU64(int_val.words, used_words)) |value| {
            writer.writeUnsigned(value);
            return;
        }

        writeBigPositive(writer, int_val.words, used_words);
        return;
    }

    if (used_words == 0) {
        builtinEvaluationFailure();
    }

    if (wordsToU64(int_val.words, used_words)) |value| {
        if (value == 0) builtinEvaluationFailure();
        writer.writeNegative(value - 1);
        return;
    }

    writeBigNegative(writer, heap, int_val.words, used_words);
}

fn writeBigPositive(writer: *CborWriter, words: [*]const u32, used_words: u32) void {
    const byte_len = magnitudeByteLen(words, used_words);
    if (byte_len == 0) {
        writer.writeUnsigned(0);
        return;
    }

    writer.writeTag(2);
    writeMagnitudeByteString(writer, words, used_words, byte_len);
}

fn writeBigNegative(
    writer: *CborWriter,
    heap: *Heap,
    words: [*]const u32,
    used_words: u32,
) void {
    const tmp = heap.createArray(u32, used_words);
    var i: u32 = 0;
    while (i < used_words) : (i += 1) {
        tmp[i] = words[i];
    }

    subtractOneLittleEndian(tmp, used_words);
    const trimmed = significantWordCount(tmp, used_words);
    if (trimmed == 0) builtinEvaluationFailure();

    writer.writeTag(3);
    const byte_len = magnitudeByteLen(tmp, trimmed);
    writeMagnitudeByteString(writer, tmp, trimmed, byte_len);
    heap.reclaimHeap(u32, used_words);
}

fn subtractOneLittleEndian(words: [*]u32, len: u32) void {
    var idx: u32 = 0;
    while (idx < len) : (idx += 1) {
        if (words[idx] > 0) {
            words[idx] -= 1;
            return;
        }
        words[idx] = 0xFFFF_FFFF;
    }
}

fn significantWordCount(words: [*]const u32, len: u32) u32 {
    var count = len;
    while (count > 0) {
        if (words[count - 1] != 0) {
            return count;
        }
        count -= 1;
    }
    return 0;
}

fn wordsToU64(words: [*]const u32, used_words: u32) ?u64 {
    if (used_words == 0) return 0;
    if (used_words > 2) return null;

    var value: u64 = 0;
    var idx = used_words;
    while (idx > 0) {
        idx -= 1;
        value <<= 32;
        value |= @as(u64, words[idx]);
    }
    return value;
}

fn magnitudeByteLen(words: [*]const u32, used_words: u32) u32 {
    if (used_words == 0) return 0;

    const ms_word = words[used_words - 1];
    var bytes: u32 = 4;
    if ((ms_word & 0xFF000000) == 0) {
        bytes -= 1;
        if ((ms_word & 0x00FF0000) == 0) {
            bytes -= 1;
            if ((ms_word & 0x0000FF00) == 0) {
                bytes -= 1;
            }
        }
    }

    return (used_words - 1) * 4 + bytes;
}

fn writeMagnitudeByteString(
    writer: *CborWriter,
    words: [*]const u32,
    used_words: u32,
    byte_len: u32,
) void {
    if (used_words == 0 or byte_len == 0) builtinEvaluationFailure();

    var iter = MagnitudeByteIterator.init(words, used_words);

    writer.writeByteStringPrefix(byte_len);
    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        if (iter.next()) |byte_value| {
            writer.writeByte(byte_value);
        } else {
            builtinEvaluationFailure();
        }
    }
}

const MagnitudeByteIterator = struct {
    words: [*]const u32,
    word_index: i32,
    byte_index: i32,
    started: bool,

    fn init(words: [*]const u32, used_words: u32) MagnitudeByteIterator {
        return .{
            .words = words,
            .word_index = @as(i32, @intCast(used_words)) - 1,
            .byte_index = 3,
            .started = false,
        };
    }

    fn next(self: *MagnitudeByteIterator) ?u8 {
        while (self.word_index >= 0) {
            if (self.byte_index < 0) {
                self.word_index -= 1;
                self.byte_index = 3;
                continue;
            }

            const word = self.words[@intCast(self.word_index)];
            const shift_amt: u5 = @intCast(self.byte_index * 8);
            const byte_value: u8 = @intCast((word >> shift_amt) & 0xFF);
            self.byte_index -= 1;

            if (!self.started) {
                if (byte_value == 0) continue;
                self.started = true;
            }

            if (self.started) {
                return byte_value;
            }
        }
        return null;
    }
};

const CborWriter = struct {
    bytes: [*]u8,
    capacity: u32,
    offset: u32,

    fn init(buffer_words: [*]u32, len_words: u32) CborWriter {
        return .{
            .bytes = @ptrCast(buffer_words),
            .capacity = len_words * 4,
            .offset = 0,
        };
    }

    fn writeByte(self: *CborWriter, value: u8) void {
        if (self.offset >= self.capacity) {
            builtinEvaluationFailure();
        }

        const idx: usize = @intCast(self.offset);
        self.bytes[idx] = value;
        self.offset += 1;
    }

    fn writeByteStringPrefix(self: *CborWriter, len: u32) void {
        self.writeMajorType(2, @intCast(len));
    }

    fn writeBreak(self: *CborWriter) void {
        self.writeByte(0xff);
    }

    fn writeArray(self: *CborWriter, len: u32) void {
        self.writeMajorType(4, @intCast(len));
    }

    fn beginIndefiniteArray(self: *CborWriter) void {
        self.writeByte(0x9f);
    }

    fn beginIndefiniteByteString(self: *CborWriter) void {
        self.writeByte(0x5f);
    }

    fn writeMap(self: *CborWriter, len: u32) void {
        self.writeMajorType(5, @intCast(len));
    }

    fn writeTag(self: *CborWriter, tag: u64) void {
        self.writeMajorType(6, tag);
    }

    fn writeUnsigned(self: *CborWriter, value: u64) void {
        self.writeMajorType(0, value);
    }

    fn writeNegative(self: *CborWriter, encoded: u64) void {
        self.writeMajorType(1, encoded);
    }

    fn writeMajorType(self: *CborWriter, major: u8, value: u64) void {
        const prefix: u8 = (major << 5);

        if (value < 24) {
            const small: u8 = @intCast(value);
            self.writeByte(prefix | small);
        } else if (value <= 0xFF) {
            self.writeByte(prefix | 24);
            self.writeByte(@intCast(value));
        } else if (value <= 0xFFFF) {
            self.writeByte(prefix | 25);
            self.writeByte(@intCast((value >> 8) & 0xFF));
            self.writeByte(@intCast(value & 0xFF));
        } else if (value <= 0xFFFF_FFFF) {
            self.writeByte(prefix | 26);
            self.writeU32(@intCast(value));
        } else {
            self.writeByte(prefix | 27);
            self.writeU64(value);
        }
    }

    fn writeU32(self: *CborWriter, value: u32) void {
        var shift: i32 = 24;
        while (shift >= 0) : (shift -= 8) {
            const shift_amt: u5 = @intCast(shift);
            self.writeByte(@intCast((value >> shift_amt) & 0xFF));
        }
    }

    fn writeU64(self: *CborWriter, value: u64) void {
        var shift: i32 = 56;
        while (shift >= 0) : (shift -= 8) {
            const shift_amt: u6 = @intCast(shift);
            self.writeByte(@intCast((value >> shift_amt) & 0xFF));
        }
    }
};

const ByteStringAllocation = struct {
    constant_words: [*]u32,
    data_words: [*]u32,
};

fn initByteStringAllocation(heap: *Heap, byte_len: u32) ByteStringAllocation {
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

fn copyPackedWordsToUnpacked(dst: [*]u32, src_words: [*]const u32, byte_len: u32) void {
    if (byte_len == 0) return;
    const src_bytes: [*]const u8 = @ptrCast(src_words);

    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        const idx: usize = @intCast(i);
        dst[idx] = src_bytes[idx];
    }
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
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);
    var aff_q: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    var aff_q: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var aff_q: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var aff_q: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_q, &q_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
    }
    var chk_g1: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&chk_g1, &aff_g1);
    if (!blst.blst_p1_in_g1(&chk_g1)) {
        builtinEvaluationFailure();
    }

    var aff_g2: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_g2, &g2_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
    }
    var chk_g2: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&chk_g2, &aff_g2);
    if (!blst.blst_p2_in_g2(&chk_g2)) {
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
    }

    var fp_b: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_b, &b_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
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
        builtinEvaluationFailure();
    }

    var fp_ml2: blst.blst_fp12 = undefined;
    if (blst_fp12_from_bendian(&fp_ml2, &ml2_bytes) != blst.BLST_SUCCESS) {
        builtinEvaluationFailure();
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

pub fn orByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn xorByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn complementByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn readBit(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn writeBits(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn replicateByte(m: *Machine, args: *LinkedValues) *const Value {
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
pub fn shiftByteString(m: *Machine, args: *LinkedValues) *const Value {
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
            const step = divRemWideStep(rem, value.words[idx], divisor);
            rem = step.remainder;
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

pub fn rotateByteString(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn countSetBits(m: *Machine, args: *LinkedValues) *const Value {
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

pub fn findFirstSetBit(m: *Machine, args: *LinkedValues) *const Value {
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
                        builtinEvaluationFailure();
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
            .tvar => {
                if (env) |bound_env| {
                    return State{
                        .ret = .{
                            .value = bound_env.lookupVar(t.debruijnIndex()),
                        },
                    };
                }
                builtinEvaluationFailure();
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
                builtinEvaluationFailure();
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
                            builtinEvaluationFailure();
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
                        builtinEvaluationFailure();
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
                    builtinEvaluationFailure();
                }

                return State{
                    .ret = .{
                        .value = forceBuiltin(self.heap, &b),
                    },
                };
            },
            else => {
                builtinEvaluationFailure();
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
                    builtinEvaluationFailure();
                }

                if (b.arity == 0) {
                    builtinEvaluationFailure();
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
                builtinEvaluationFailure();
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

test "sha2_256 bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const input_words = [_]u32{
        0x2e, 0x7e, 0xa8, 0x4d, 0xa4, 0xbc, 0x4d, 0x7c, 0xfb, 0x46, 0x3e, 0x3f, 0x2c,
        0x86, 0x47, 0x05, 0x7a, 0xff, 0xf3, 0xfb, 0xec, 0xec, 0xa1, 0xd2, 0x00,
    };
    const input = expr.Bytes{
        .length = input_words.len,
        .bytes = input_words[0..].ptr,
    };

    const expected_words = [_]u32{
        0x76, 0xe3, 0xac, 0xbc, 0x71, 0x88, 0x36, 0xf2, 0xdf, 0x8a, 0xd2, 0xd0, 0xd2,
        0xd7, 0x6f, 0x0c, 0xfa, 0x5f, 0xea, 0x09, 0x86, 0xbe, 0x91, 0x8f, 0x10, 0xbc,
        0xee, 0x73, 0x0d, 0xf4, 0x41, 0xb9,
    };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &input, ConstantType.bytesType());

    const result = sha2_256(&machine, args);
    const digest = result.unwrapBytestring();

    try testing.expectEqual(@as(u32, expected_words.len), digest.length);
    var i: usize = 0;
    while (i < expected_words.len) : (i += 1) {
        try testing.expectEqual(expected_words[i], digest.bytes[i]);
    }
}

test "sha3_256 bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    var repeated_words: [200]u32 = undefined;
    var fill_idx: usize = 0;
    while (fill_idx < repeated_words.len) : (fill_idx += 1) {
        repeated_words[fill_idx] = 0xA3;
    }

    const len200_input = expr.Bytes{
        .length = repeated_words.len,
        .bytes = repeated_words[0..].ptr,
    };

    const expected_len200 = [_]u32{
        0x79, 0xF3, 0x8A, 0xDE, 0xC5, 0xC2, 0x03, 0x07, 0xA9, 0x8E, 0xF7, 0x6E, 0x83,
        0x24, 0xAF, 0xBF, 0xD4, 0x6C, 0xFD, 0x81, 0xB2, 0x2E, 0x39, 0x73, 0xC6, 0x5F,
        0xA1, 0xBD, 0x9D, 0xE3, 0x17, 0x87,
    };

    const len200_args = LinkedValues.create(&heap, *const expr.Bytes, &len200_input, ConstantType.bytesType());
    const len200_digest = sha3_256(&machine, len200_args).unwrapBytestring();

    try testing.expectEqual(@as(u32, expected_len200.len), len200_digest.length);
    var idx: usize = 0;
    while (idx < expected_len200.len) : (idx += 1) {
        try testing.expectEqual(expected_len200[idx], len200_digest.bytes[idx]);
    }

    const zero_buf = [_]u32{0};
    const empty_input = expr.Bytes{
        .length = 0,
        .bytes = zero_buf[0..].ptr,
    };

    const expected_empty = [_]u32{
        0xA7, 0xFF, 0xC6, 0xF8, 0xBF, 0x1E, 0xD7, 0x66, 0x51, 0xC1, 0x47, 0x56, 0xA0,
        0x61, 0xD6, 0x62, 0xF5, 0x80, 0xFF, 0x4D, 0xE4, 0x3B, 0x49, 0xFA, 0x82, 0xD8,
        0x0A, 0x4B, 0x80, 0xF8, 0x43, 0x4A,
    };

    const empty_args = LinkedValues.create(&heap, *const expr.Bytes, &empty_input, ConstantType.bytesType());
    const empty_digest = sha3_256(&machine, empty_args).unwrapBytestring();

    try testing.expectEqual(@as(u32, expected_empty.len), empty_digest.length);
    idx = 0;
    while (idx < expected_empty.len) : (idx += 1) {
        try testing.expectEqual(expected_empty[idx], empty_digest.bytes[idx]);
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
