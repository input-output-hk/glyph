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
const utils = @import("utils.zig");

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
    }

    pub fn popFrame(self: *Self) Frame {
        self.frame_ptr = @ptrFromInt(@intFromPtr(self.frame_ptr) - @sizeOf(Frame));

        const frame = std.mem.bytesToValue(Frame, self.frame_ptr);

        return frame;
    }
};

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

pub fn divideInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn quotientInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn remainderInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn modInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
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
        utils.printString("Integer larger than byte or negative");
        utils.exit(std.math.maxInt(u32));
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

// String functions
pub fn appendString(m: *Machine, args: *LinkedValues) *const Value {
    const y = args.value.unwrapString();
    const x = args.next.?.value.unwrapString();

    const length = x.length + y.length;

    // type_length 4 bytes, integer 4 bytes,  length 4 bytes, list of words 4 * (x length + y length)
    var result = m.heap.createArray(u32, length + 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.stringType());
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

pub fn encodeUtf8(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn decodeUtf8(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
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
pub fn chooseUnit(_: *Machine, args: *LinkedValues) *const Value {
    const then = args.value;
    const unit = args.next.?.value;

    if (!unit.isUnit()) {
        utils.printString("Passed in non-unit value");
        utils.exit(std.math.maxInt(u32));
    }

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

    if (list.length > 0) {
        const c = Constant{
            .length = list.type_length,
            .type_list = list.inner_type,
            .value = list.items.?.value,
        };

        const con = m.heap.create(Constant, &c);

        return createConst(m.heap, con);
    } else {
        utils.printlnString("called headList on an empty list");
        utils.exit(std.math.maxInt(u32));
    }
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

pub fn nullList(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Data functions
pub fn chooseData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn constrData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
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

pub fn bData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
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

pub fn mkNilData(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
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
pub fn bls12_381_G1_Add(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G1_Neg(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G1_ScalarMul(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G1_Equal(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G1_Compress(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G1_Uncompress(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G1_HashToGroup(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G2_Add(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G2_Neg(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G2_ScalarMul(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G2_Equal(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G2_Compress(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G2_Uncompress(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_G2_HashToGroup(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_MillerLoop(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_MulMlResult(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn bls12_381_FinalVerify(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn keccak_256(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn blake2b_224(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Conversions
pub fn integerToByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn byteStringToInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Logical
pub fn andByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
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
                return self.compute(env, cs.constr);
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
        .ret => |r| blk: {
            break :blk machine.ret(r.value);
        },
        else => {
            @panic("HERE???2");
        },
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
            @panic("HERE???2");
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
