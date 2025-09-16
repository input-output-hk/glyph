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
const utils = @import("utils.zig");

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

pub fn divideInteger(m: *Machine, args: *LinkedValues) *const Value {
    // 1. Parse inputs and handle special cases
    const n = args.value.unwrapInteger(); // Numerator BigInt
    const d = args.next.?.value.unwrapInteger(); // Denominator BigInt

    // Fast path: 0 ÷ anything = 0
    if (n.length == 1 and n.words[0] == 0) {
        // Allocate result for integer 0 (sign=0, length=1, word=0)
        var z = m.heap.createArray(u32, 5);
        z[0] = @intFromPtr(ConstantType.integerType());
        z[1] = 0; // sign = 0 (non-negative)
        z[2] = 1; // length = 1
        z[3] = 0; // limb value = 0
        return createConst(m.heap, @ptrCast(z));
    }
    // Denominator 0 -> panic (division by zero)
    if (d.length == 1 and d.words[0] == 0) {
        @panic("divideInteger: division by zero");
    }

    // Determine signs and absolute values
    const numer_neg = n.sign == 1;
    const denom_neg = d.sign == 1;
    const numer_abs = BigInt{ .sign = 0, .length = n.length, .words = n.words };
    const denom_abs = BigInt{ .sign = 0, .length = d.length, .words = d.words };

    // Compare magnitudes to see if |n| < |d|
    const cmp = numer_abs.compareMagnitude(&denom_abs);
    const numerLess = cmp[2] == &numer_abs;
    if (numerLess) {
        const needMinusOne = (numer_neg != denom_neg);
        if (!needMinusOne) {
            // If signs are same, floor division yields 0
            var z = m.heap.createArray(u32, 5);
            z[0] = @intFromPtr(ConstantType.integerType());
            z[1] = 0;
            z[2] = 1;
            z[3] = 0;
            return createConst(m.heap, @ptrCast(z));
        } else {
            // Signs differ and |n|<|d|, result is -1
            var negOne = m.heap.createArray(u32, 5);
            negOne[0] = @intFromPtr(ConstantType.integerType());
            negOne[1] = 1; // sign = 1 (negative)
            negOne[2] = 1; // length = 1
            negOne[3] = 1; // magnitude = 1
            return createConst(m.heap, @ptrCast(negOne));
        }
    }

    // 2. Unsigned magnitude division. Allocate quotient buffer.
    const q_buf_len: u32 = numer_abs.length - denom_abs.length + 1;
    var q_buf = m.heap.createArray(u32, q_buf_len);
    var rem_zero: bool = true; // will set to false if remainder is non-zero

    if (denom_abs.length == 1) {
        // Single-limb divisor fast path
        const dv: u32 = denom_abs.words[0];
        var carry: u64 = 0;
        // iterate from most significant limb to least
        var ii: u32 = numer_abs.length;
        while (ii > 0) : (ii -= 1) {
            const i = ii - 1;
            const cur: u64 = (carry << 32) | @as(u64, numer_abs.words[i]);
            const q_digit: u64 = cur / dv;
            carry = cur % dv;
            q_buf[i] = @truncate(q_digit);
        }
        rem_zero = (carry == 0);
    } else {
        // Multi-limb divisor: Knuth's Algorithm D
        const beta: u64 = 0x1_0000_0000; // β = 2^32
        const n_u = numer_abs.length;
        const m_d = denom_abs.length;

        // D1: normalization (shift left so v_norm[m_d-1] >= β/2)
        const vTop: u32 = denom_abs.words[m_d - 1];
        // Compute shift count s as u5
        const s_full = @clz(vTop);
        const s: u5 = if (s_full == 32) 0 else @truncate(s_full);
        const oneMinusS: u5 = if (s == 0) @as(u5, 0) else (@as(u5, 31) - (s - 1));
        // oneMinusS is effectively (32 - s) but expressed in u5 without overflow

        // Allocate and build u_norm (length = n_u + 1)
        var u_norm = m.heap.createArray(u32, n_u + 1);
        var carry_u: u32 = 0;
        var ui: u32 = 0;
        while (ui < n_u) : (ui += 1) {
            const w = numer_abs.words[ui];
            u_norm[ui] = @as(u32, (w << s) | carry_u);
            carry_u = if (s == 0) 0 else w >> oneMinusS;
        }
        u_norm[n_u] = carry_u;

        // Allocate and build v_norm (length = m_d)
        var v_norm = m.heap.createArray(u32, m_d);
        var carry_v: u32 = 0;
        var vi: u32 = 0;
        while (vi < m_d) : (vi += 1) {
            const w = denom_abs.words[vi];
            v_norm[vi] = @as(u32, (w << s) | carry_v);
            carry_v = if (s == 0) 0 else w >> oneMinusS;
        }

        // D2: initialize quotient loop (j from n_u - m_d down to 0)
        var jj: u32 = n_u - m_d + 1;
        var j_usize: u32 = n_u - m_d;
        while (jj > 0) : (jj -= 1) {

            // D3: Compute trial qhat and rhat
            const u_hi: u32 = u_norm[j_usize + m_d];
            const u_lo: u32 = u_norm[j_usize + m_d - 1];
            const num64: u64 = (@as(u64, u_hi) << 32) | @as(u64, u_lo);
            var qhat: u64 = num64 / @as(u64, v_norm[m_d - 1]);
            var rhat: u64 = num64 % @as(u64, v_norm[m_d - 1]);
            if (qhat == beta) {
                qhat -= 1;
                rhat += @as(u64, v_norm[m_d - 1]);
            }
            // Adjust qhat downward if necessary
            const v2: u32 = if (m_d > 1) v_norm[m_d - 2] else 0;
            const uv2: u32 = u_norm[j_usize + m_d - 2];
            while (qhat != 0) {
                if (qhat * @as(u64, v2) <= (rhat << 32) + @as(u64, uv2)) {
                    break;
                }
                qhat -= 1;
                rhat += @as(u64, v_norm[m_d - 1]);
                if (rhat >= beta) {
                    break;
                }
            }

            // D4: Multiply v_norm by qhat and subtract from u_norm segment
            var borrow: u64 = 0;
            var carry_mul: u64 = 0;
            var k: usize = 0;
            while (k < m_d) : (k += 1) {
                const p = qhat * @as(u64, v_norm[k]) + carry_mul;
                carry_mul = p >> 32;
                const sub_val = u_norm[j_usize + k];
                const diff = @as(u64, sub_val) - (p & 0xFFFF_FFFF) - borrow;
                u_norm[j_usize + k] = @truncate(diff);
                borrow = (diff >> 63) & 1;
            }
            const sub_hi = u_norm[j_usize + m_d];
            const diff_hi = @as(u64, sub_hi) - carry_mul - borrow;
            u_norm[j_usize + m_d] = @truncate(diff_hi);

            // D5: Check if we subtracted "too much" (negative remainder)
            var q_digit: u32 = @truncate(qhat);
            if ((diff_hi >> 63) & 1 != 0) {
                // If underflow, add v_norm back and decrement q_digit
                q_digit -= 1;
                var carry_back: u64 = 0;
                var t: usize = 0;
                while (t < m_d) : (t += 1) {
                    const sum = @as(u64, u_norm[j_usize + t]) + @as(u64, v_norm[t]) + carry_back;
                    u_norm[j_usize + t] = @truncate(sum);
                    carry_back = sum >> 32;
                }
                u_norm[j_usize + m_d] = @truncate(@as(u64, u_norm[j_usize + m_d]) + carry_back);
            }
            // Store quotient digit
            q_buf[j_usize] = q_digit;

            if (jj > 1) {
                j_usize -= 1;
            }
        }

        // Determine if remainder is zero (all lower m_d limbs are 0)
        rem_zero = true;
        var r_index: u32 = 0;
        while (r_index < m_d) : (r_index += 1) {
            if (u_norm[r_index] != 0) {
                rem_zero = false;
                break;
            }
        }
    }

    // Trim leading zero limbs from quotient buffer
    var q_len: u32 = q_buf_len;
    while (q_len > 1 and q_buf[q_len - 1] == 0) {
        q_len -= 1;
    }

    // 3. Post-processing for floor division semantics
    const signsDiffer = (numer_neg != denom_neg);
    const res_sign: u32 = if (signsDiffer) 1 else 0;
    // Allocate result array (with space for header + quotient limbs, possibly +1 limb for carry)
    var res = m.heap.createArray(u32, q_len + 4);
    res[0] = @intFromPtr(ConstantType.integerType());
    res[1] = res_sign;
    // Copy quotient magnitude
    var i_cpy: u32 = 0;
    while (i_cpy < q_len) : (i_cpy += 1) {
        res[3 + i_cpy] = q_buf[i_cpy];
    }

    if (signsDiffer and !rem_zero) {
        // If signs differ and remainder != 0, add 1 to magnitude (floor toward -∞)
        var carry_neg: u64 = 1;
        var idx: u32 = 0;
        while (carry_neg != 0 and idx < q_len) : (idx += 1) {
            const tmp = @as(u64, res[3 + idx]) + carry_neg;
            res[3 + idx] = @truncate(tmp);
            carry_neg = tmp >> 32;
        }
        if (carry_neg != 0) {
            res[4 + q_len] = @truncate(carry_neg);
            q_len += 1;
        }
    }
    res[2] = q_len; // set final length
    return createConst(m.heap, @ptrCast(res));
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

fn integerToScalarBytes(bi: BigInt) [32]u8 {
    if (bi.sign != 0) {
        utils.printString("Negative scalar in BLS mul\n");
        utils.exit(std.math.maxInt(u32));
    }
    if (bi.length > 8) {
        utils.printString("Scalar too large for BLS mul\n");
        utils.exit(std.math.maxInt(u32));
    }

    var scalar_bytes: [32]u8 = [_]u8{0} ** 32;
    var offset: usize = 32 - bi.length * 4;
    var k: usize = 0;
    while (k < bi.length) : (k += 1) {
        const j = bi.length - 1 - k;
        const word = bi.words[j];
        scalar_bytes[offset] = @truncate(word >> 24);
        scalar_bytes[offset + 1] = @truncate(word >> 16);
        scalar_bytes[offset + 2] = @truncate(word >> 8);
        scalar_bytes[offset + 3] = @truncate(word);
        offset += 4;
    }
    return scalar_bytes;
}

fn blst_fp12_from_bendian(ret: *blst.blst_fp12, bytes: [*]const u8) c_int {
    blst.blst_fp_from_bendian(&ret.fp6[0].fp2[0].fp[0], bytes + 0 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[0].fp2[0].fp[1], bytes + 1 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[0].fp2[1].fp[0], bytes + 2 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[0].fp2[1].fp[1], bytes + 3 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[0].fp2[2].fp[0], bytes + 4 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[0].fp2[2].fp[1], bytes + 5 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[1].fp2[0].fp[0], bytes + 6 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[1].fp2[0].fp[1], bytes + 7 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[1].fp2[1].fp[0], bytes + 8 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[1].fp2[1].fp[1], bytes + 9 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[1].fp2[2].fp[0], bytes + 10 * 48);
    blst.blst_fp_from_bendian(&ret.fp6[1].fp2[2].fp[1], bytes + 11 * 48);
    if (blst.blst_fp12_in_group(ret)) {
        return blst.BLST_SUCCESS;
    } else {
        return blst.BLST_BAD_ENCODING;
    }
}

pub fn bls12_381_G1_Add(m: *Machine, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG1();
    const p = args.next.?.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);
    var q_bytes: [48]u8 = undefined;
    @memcpy(&q_bytes, q.bytes);

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

    var buf = m.heap.createArray(u32, 13);
    buf[0] = @intFromPtr(ConstantType.g1ElementType());
    const out_bytes: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p1_compress(out_bytes, &point_r);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G1_Neg(m: *Machine, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    blst.blst_p1_cneg(&point_p, true);

    var buf = m.heap.createArray(u32, 13);
    buf[0] = @intFromPtr(ConstantType.g1ElementType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p1_compress(out_ptr, &point_p);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G1_ScalarMul(m: *Machine, args: *LinkedValues) *const Value {
    const scalar = args.value.unwrapInteger();
    const p = args.next.?.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    var scalar_bytes = integerToScalarBytes(scalar);

    var point_r: blst.blst_p1 = undefined;
    blst.blst_p1_mult(&point_r, &point_p, &scalar_bytes, 256);

    var buf = m.heap.createArray(u32, 13);
    buf[0] = @intFromPtr(ConstantType.g1ElementType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p1_compress(out_ptr, &point_r);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G1_Equal(m: *Machine, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG1();
    const p = args.next.?.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);
    var q_bytes: [48]u8 = undefined;
    @memcpy(&q_bytes, q.bytes);

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

    var result = m.heap.createArray(u32, 2);
    result[0] = @intFromPtr(ConstantType.booleanType());
    result[1] = @intFromBool(equal);

    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G1_Compress(m: *Machine, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG1();

    var p_bytes: [48]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    var out_bytes: [48]u8 = undefined;
    blst.blst_p1_compress(&out_bytes, &point_p);

    var buf = m.heap.createArray(u32, 50);
    buf[0] = @intFromPtr(ConstantType.bytesType());
    buf[1] = 48;
    for (0..48) |i| {
        buf[2 + i] = out_bytes[i];
    }

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G1_Uncompress(m: *Machine, args: *LinkedValues) *const Value {
    const bs = args.value.unwrapBytestring();
    if (bs.length != 48) {
        utils.printString("Invalid compressed G1 length\n");
        utils.exit(std.math.maxInt(u32));
    }

    var in_bytes: [48]u8 = undefined;
    for (0..48) |i| {
        in_bytes[i] = @truncate(bs.bytes[i]);
    }

    var aff_p: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_p, &in_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid compressed G1\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p1 = undefined;
    blst.blst_p1_from_affine(&point_p, &aff_p);

    var buf = m.heap.createArray(u32, 13);
    buf[0] = @intFromPtr(ConstantType.g1ElementType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p1_compress(out_ptr, &point_p);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G1_HashToGroup(m: *Machine, args: *LinkedValues) *const Value {
    const dst = args.value.unwrapBytestring();
    const msg = args.next.?.value.unwrapBytestring();

    const msg_bytes: [*]u8 = @ptrFromInt(@intFromPtr(msg.bytes));
    const dst_bytes: [*]u8 = @ptrFromInt(@intFromPtr(dst.bytes));

    var point_r: blst.blst_p1 = undefined;
    blst.blst_hash_to_g1(&point_r, msg_bytes, msg.length * 4, dst_bytes, dst.length * 4, null, 0);

    var buf = m.heap.createArray(u32, 13);
    buf[0] = @intFromPtr(ConstantType.g1ElementType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p1_compress(out_ptr, &point_r);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G2_Add(m: *Machine, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG2();
    const p = args.next.?.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);
    var q_bytes: [96]u8 = undefined;
    @memcpy(&q_bytes, q.bytes);

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

    var buf = m.heap.createArray(u32, 25);
    buf[0] = @intFromPtr(ConstantType.g2ElementType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p2_compress(out_ptr, &point_r);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G2_Neg(m: *Machine, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    blst.blst_p2_cneg(&point_p, true);

    var buf = m.heap.createArray(u32, 25);
    buf[0] = @intFromPtr(ConstantType.g2ElementType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p2_compress(out_ptr, &point_p);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G2_ScalarMul(m: *Machine, args: *LinkedValues) *const Value {
    const scalar = args.value.unwrapInteger();
    const p = args.next.?.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var scalar_bytes = integerToScalarBytes(scalar);

    var point_r: blst.blst_p2 = undefined;
    blst.blst_p2_mult(&point_r, &point_p, &scalar_bytes, 256);

    var buf = m.heap.createArray(u32, 25);
    buf[0] = @intFromPtr(ConstantType.g2ElementType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p2_compress(out_ptr, &point_r);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G2_Equal(m: *Machine, args: *LinkedValues) *const Value {
    const q = args.value.unwrapG2();
    const p = args.next.?.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);
    var q_bytes: [96]u8 = undefined;
    @memcpy(&q_bytes, q.bytes);

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

    var result = m.heap.createArray(u32, 2);
    result[0] = @intFromPtr(ConstantType.booleanType());
    result[1] = @intFromBool(equal);

    return createConst(m.heap, @ptrCast(result));
}

pub fn bls12_381_G2_Compress(m: *Machine, args: *LinkedValues) *const Value {
    const p = args.value.unwrapG2();

    var p_bytes: [96]u8 = undefined;
    @memcpy(&p_bytes, p.bytes);

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &p_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var out_bytes: [96]u8 = undefined;
    blst.blst_p2_compress(&out_bytes, &point_p);

    var buf = m.heap.createArray(u32, 98);
    buf[0] = @intFromPtr(ConstantType.bytesType());
    buf[1] = 96;
    for (0..96) |i| {
        buf[2 + i] = out_bytes[i];
    }

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G2_Uncompress(m: *Machine, args: *LinkedValues) *const Value {
    const bs = args.value.unwrapBytestring();
    if (bs.length != 96) {
        utils.printString("Invalid compressed G2 length\n");
        utils.exit(std.math.maxInt(u32));
    }

    var in_bytes: [96]u8 = undefined;
    for (0..96) |i| {
        in_bytes[i] = @truncate(bs.bytes[i]);
    }

    var aff_p: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_p, &in_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid compressed G2\n");
        utils.exit(std.math.maxInt(u32));
    }
    var point_p: blst.blst_p2 = undefined;
    blst.blst_p2_from_affine(&point_p, &aff_p);

    var buf = m.heap.createArray(u32, 25);
    buf[0] = @intFromPtr(ConstantType.g2ElementType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p2_compress(out_ptr, &point_p);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_G2_HashToGroup(m: *Machine, args: *LinkedValues) *const Value {
    const dst = args.value.unwrapBytestring();
    const msg = args.next.?.value.unwrapBytestring();

    const msg_bytes: [*]u8 = @ptrFromInt(@intFromPtr(msg.bytes));
    const dst_bytes: [*]u8 = @ptrFromInt(@intFromPtr(dst.bytes));

    var point_r: blst.blst_p2 = undefined;
    blst.blst_hash_to_g2(&point_r, msg_bytes, msg.length * 4, dst_bytes, dst.length * 4, null, 0);

    var buf = m.heap.createArray(u32, 25);
    buf[0] = @intFromPtr(ConstantType.g2ElementType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_p2_compress(out_ptr, &point_r);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_MillerLoop(m: *Machine, args: *LinkedValues) *const Value {
    const g2 = args.value.unwrapG2();
    const g1 = args.next.?.value.unwrapG1();

    var g1_bytes: [48]u8 = undefined;
    @memcpy(&g1_bytes, g1.bytes);
    var g2_bytes: [96]u8 = undefined;
    @memcpy(&g2_bytes, g2.bytes);

    var aff_g1: blst.blst_p1_affine = undefined;
    if (blst.blst_p1_uncompress(&aff_g1, &g1_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G1 point\n");
        utils.exit(std.math.maxInt(u32));
    }

    var aff_g2: blst.blst_p2_affine = undefined;
    if (blst.blst_p2_uncompress(&aff_g2, &g2_bytes) != blst.BLST_SUCCESS) {
        utils.printString("Invalid G2 point\n");
        utils.exit(std.math.maxInt(u32));
    }

    var ml: blst.blst_fp12 = undefined;
    blst.blst_miller_loop(&ml, &aff_g2, &aff_g1);

    var buf = m.heap.createArray(u32, 145);
    buf[0] = @intFromPtr(ConstantType.mlResultType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_bendian_from_fp12(out_ptr, &ml);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_MulMlResult(m: *Machine, args: *LinkedValues) *const Value {
    const b = args.value.unwrapMlResult();
    const a = args.next.?.value.unwrapMlResult();

    var a_bytes: [576]u8 = undefined;
    @memcpy(&a_bytes, a.bytes);
    var b_bytes: [576]u8 = undefined;
    @memcpy(&b_bytes, b.bytes);

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

    var buf = m.heap.createArray(u32, 145);
    buf[0] = @intFromPtr(ConstantType.mlResultType());
    const out_ptr: [*]u8 = @ptrCast(buf + 1);
    blst.blst_bendian_from_fp12(out_ptr, &fp_r);

    return createConst(m.heap, @ptrCast(buf));
}

pub fn bls12_381_FinalVerify(m: *Machine, args: *LinkedValues) *const Value {
    const ml2 = args.value.unwrapMlResult();
    const ml1 = args.next.?.value.unwrapMlResult();

    var ml1_bytes: [576]u8 = undefined;
    @memcpy(&ml1_bytes, ml1.bytes);
    var ml2_bytes: [576]u8 = undefined;
    @memcpy(&ml2_bytes, ml2.bytes);

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

    var result = m.heap.createArray(u32, 2);
    result[0] = @intFromPtr(ConstantType.booleanType());
    result[1] = @intFromBool(res);

    return createConst(m.heap, @ptrCast(result));
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

    const args = LinkedValues.create(&heap, *const expr.BigInt, &d, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &n, ConstantType.integerType());

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

    const args = LinkedValues.create(&heap, expr.BigInt, d, ConstantType.integerType())
        .extend(&heap, expr.BigInt, n, ConstantType.integerType());

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

    const args = LinkedValues.create(&heap, expr.BigInt, d, ConstantType.integerType())
        .extend(&heap, expr.BigInt, n, ConstantType.integerType());

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

    const args = LinkedValues.create(&heap, expr.BigInt, d, ConstantType.integerType())
        .extend(&heap, expr.BigInt, n, ConstantType.integerType());

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

    const args = LinkedValues.create(&heap, expr.BigInt, d, ConstantType.integerType())
        .extend(&heap, expr.BigInt, n, ConstantType.integerType());

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

    const args = LinkedValues.create(&heap, expr.BigInt, d, ConstantType.integerType())
        .extend(&heap, expr.BigInt, n, ConstantType.integerType());

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

    const args = LinkedValues.create(&heap, expr.BigInt, d, ConstantType.integerType())
        .extend(&heap, expr.BigInt, n, ConstantType.integerType());

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

    const args = LinkedValues.create(&heap, expr.BigInt, d, ConstantType.integerType())
        .extend(&heap, expr.BigInt, n, ConstantType.integerType());

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

test "bls12_381_G1_add" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };
    // const p_bytes_hex = "abd61864f519748032551e42e0ac417fd828f079454e3e3c9891c5c29ed7f10bdecc046854e3931cb7002779bd76d71f";
    // const q_bytes_hex = "950dfd33da2682260c76038dfb8bad6e84ae9d599a3c151815945ac1e6ef6b1027cd917f3907479d20d636ce437a41f5";
    // const expected_hex = "a4870e983a149bb1e7cc70fde907a2aa52302833bce4d62f679819022924e9caab52e3631d376d36d9692664b4cfbc22";
    const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };
    const q_bytes: [*]const u8 = &[_]u8{ 0x95, 0x0d, 0xfd, 0x33, 0xda, 0x26, 0x82, 0x26, 0x0c, 0x76, 0x03, 0x8d, 0xfb, 0x8b, 0xad, 0x6e, 0x84, 0xae, 0x9d, 0x59, 0x9a, 0x3c, 0x15, 0x18, 0x15, 0x94, 0x5a, 0xc1, 0xe6, 0xef, 0x6b, 0x10, 0x27, 0xcd, 0x91, 0x7f, 0x39, 0x07, 0x47, 0x9d, 0x20, 0xd6, 0x36, 0xce, 0x43, 0x7a, 0x41, 0xf5 };

    const expected_bytes: [*]const u8 = &[_]u8{ 0xa4, 0x87, 0x0e, 0x98, 0x3a, 0x14, 0x9b, 0xb1, 0xe7, 0xcc, 0x70, 0xfd, 0xe9, 0x07, 0xa2, 0xaa, 0x52, 0x30, 0x28, 0x33, 0xbc, 0xe4, 0xd6, 0x2f, 0x67, 0x98, 0x19, 0x02, 0x29, 0x24, 0xe9, 0xca, 0xab, 0x52, 0xe3, 0x63, 0x1d, 0x37, 0x6d, 0x36, 0xd9, 0x69, 0x26, 0x64, 0xb4, 0xcf, 0xbc, 0x22 };

    const args = LinkedValues.create(&heap, expr.G1Element, expr.G1Element{ .bytes = q_bytes }, ConstantType.g1ElementType())
        .extend(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType());

    const result_val = bls12_381_G1_Add(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .bls12_381_g1_element => {
                    const r = c.g1Element();
                    for (0..48) |i| {
                        try testing.expectEqual(expected_bytes[i], r.bytes[i]);
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

test "bls12_381_G1_neg" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

    const expected_bytes: [*]const u8 = &[_]u8{ 0x8b, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

    const args = LinkedValues.create(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType());

    const result_val = bls12_381_G1_Neg(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .bls12_381_g1_element => {
                    const r = c.g1Element();
                    for (0..48) |i| {
                        try testing.expectEqual(expected_bytes[i], r.bytes[i]);
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

test "bls12_381_G1_equal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

    const args = LinkedValues.create(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType())
        .extend(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType());

    const result_val = bls12_381_G1_Equal(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .boolean => {
                    const r = c.bln();
                    try testing.expect(r);
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

test "bls12_381_G1_compress" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

    const expected_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

    const args = LinkedValues.create(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType());

    const result_val = bls12_381_G1_Compress(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .bytes => {
                    const r = c.innerBytes();
                    try testing.expectEqual(r.length, 48);
                    for (0..48) |i| {
                        try testing.expectEqual(expected_bytes[i], r.bytes[i]);
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

// test "bls12_381_G1_uncompress" { // TODO FIX
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const args = LinkedValues.create(&heap, expr.Bytes, expr.Bytes{ .length = 48, .bytes = p_bytes }, ConstantType.bytesType());

//     const result_val = bls12_381_G1_Uncompress(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g1_element => {
//                     const r = c.g1Element();
//                     for (0..48) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G1_hashToGroup" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const message_bytes: [*]const u8 = &[_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20 };

//     const dst_bytes: [*]const u8 = &[_]u8{ 0x42, 0x4c, 0x53, 0x5f, 0x47, 0x31, 0x32, 0x5f, 0x47, 0x31, 0x5f, 0x58, 0x4d, 0x44, 0x3a, 0x53, 0x48, 0x41, 0x32, 0x35, 0x36, 0x5f, 0x52, 0x4f, 0x5f };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const args = LinkedValues.create(&heap, expr.Bytes, expr.Bytes{ .length = 32, .bytes = message_bytes }, ConstantType.bytesType())
//         .extend(&heap, expr.Bytes, expr.Bytes{ .length = 25, .bytes = dst_bytes }, ConstantType.bytesType());

//     const result_val = bls12_381_G1_HashToGroup(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g1_element => {
//                     const r = c.g1Element();
//                     for (0..48) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G1_scalarMul" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const scalar_words: [*]const u32 = &.{2};

//     const scalar = expr.BigInt{ .sign = 0, .length = 1, .words = scalar_words };

//     const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0xa4, 0x87, 0x0e, 0x98, 0x3a, 0x14, 0x9b, 0xb1, 0xe7, 0xcc, 0x70, 0xfd, 0xe9, 0x07, 0xa2, 0xaa, 0x52, 0x30, 0x28, 0x33, 0xbc, 0xe4, 0xd6, 0x2f, 0x67, 0x98, 0x19, 0x02, 0x29, 0x24, 0xe9, 0xca, 0xab, 0x52, 0xe3, 0x63, 0x1d, 0x37, 0x6d, 0x36, 0xd9, 0x69, 0x26, 0x64, 0xb4, 0xcf, 0xbc, 0x22 };

//     const args = LinkedValues.create(&heap, expr.BigInt, scalar, ConstantType.integerType())
//         .extend(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType());

//     const result_val = bls12_381_G1_ScalarMul(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g1_element => {
//                     const r = c.g1Element();
//                     for (0..48) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

test "bls12_381_G2_add" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const p_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

    const q_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

    const expected_bytes: [*]const u8 = &[_]u8{ 0xa4, 0x87, 0x0e, 0x98, 0x3a, 0x14, 0x9b, 0xb1, 0xe7, 0xcc, 0x70, 0xfd, 0xe9, 0x07, 0xa2, 0xaa, 0x52, 0x30, 0x28, 0x33, 0xbc, 0xe4, 0xd6, 0x2f, 0x67, 0x98, 0x19, 0x02, 0x29, 0x24, 0xe9, 0xca, 0xab, 0x52, 0xe3, 0x63, 0x1d, 0x37, 0x6d, 0x36, 0xd9, 0x69, 0x26, 0x64, 0xb4, 0xcf, 0xbc, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

    const args = LinkedValues.create(&heap, expr.G2Element, expr.G2Element{ .bytes = q_bytes }, ConstantType.g2ElementType())
        .extend(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

    const result_val = bls12_381_G2_Add(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .bls12_381_g2_element => {
                    const r = c.g2Element();
                    for (0..96) |i| {
                        try testing.expectEqual(expected_bytes[i], r.bytes[i]);
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

test "bls12_381_G2_neg" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const p_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

    const expected_bytes: [*]const u8 = &[_]u8{ 0x94, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

    const args = LinkedValues.create(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

    const result_val = bls12_381_G2_Neg(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .bls12_381_g2_element => {
                    const r = c.g2Element();
                    for (0..96) |i| {
                        try testing.expectEqual(expected_bytes[i], r.bytes[i]);
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

test "bls12_381_G2_equal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const p_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

    const args = LinkedValues.create(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType())
        .extend(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

    const result_val = bls12_381_G2_Equal(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .boolean => {
                    const r = c.bln();
                    try testing.expect(r);
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

test "bls12_381_G2_compress" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const p_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

    const expected_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

    const args = LinkedValues.create(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

    const result_val = bls12_381_G2_Compress(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .bytes => {
                    const r = c.innerBytes();
                    try testing.expectEqual(r.length, 96);
                    for (0..96) |i| {
                        try testing.expectEqual(expected_bytes[i], r.bytes[i]);
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

// test "bls12_381_G2_uncompress" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const args = LinkedValues.create(&heap, expr.Bytes, expr.Bytes{ .length = 96, .bytes = p_bytes }, ConstantType.bytesType());

//     const result_val = bls12_381_G2_Uncompress(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g2_element => {
//                     const r = c.g2Element();
//                     for (0..96) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G2_hashToGroup" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const message_bytes: [*]const u8 = &[_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20 };

//     const dst_bytes: [*]const u8 = &[_]u8{ 0x42, 0x4c, 0x53, 0x5f, 0x47, 0x31, 0x32, 0x5f, 0x47, 0x32, 0x5f, 0x58, 0x4d, 0x44, 0x3a, 0x53, 0x48, 0x41, 0x32, 0x35, 0x36, 0x5f, 0x52, 0x4f, 0x5f };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const args = LinkedValues.create(&heap, expr.Bytes, expr.Bytes{ .length = 32, .bytes = message_bytes }, ConstantType.bytesType())
//         .extend(&heap, expr.Bytes, expr.Bytes{ .length = 25, .bytes = dst_bytes }, ConstantType.bytesType());

//     const result_val = bls12_381_G2_HashToGroup(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g2_element => {
//                     const r = c.g2Element();
//                     for (0..96) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G2_scalarMul" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const scalar_words: [*]const u32 = &.{2};

//     const scalar = expr.BigInt{ .sign = 0, .length = 1, .words = scalar_words };

//     const p_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0xa4, 0x87, 0x0e, 0x98, 0x3a, 0x14, 0x9b, 0xb1, 0xe7, 0xcc, 0x70, 0xfd, 0xe9, 0x07, 0xa2, 0xaa, 0x52, 0x30, 0x28, 0x33, 0xbc, 0xe4, 0xd6, 0x2f, 0x67, 0x98, 0x19, 0x02, 0x29, 0x24, 0xe9, 0xca, 0xab, 0x52, 0xe3, 0x63, 0x1d, 0x37, 0x6d, 0x36, 0xd9, 0x69, 0x26, 0x64, 0xb4, 0xcf, 0xbc, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

//     const args = LinkedValues.create(&heap, expr.BigInt, scalar, ConstantType.integerType())
//         .extend(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

//     const result_val = bls12_381_G2_ScalarMul(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g2_element => {
//                     const r = c.g2Element();
//                     for (0..96) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

test "bls12_381_millerLoop" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const g1_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

    const g2_bytes: [*]const u8 = &[_]u8{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

    const expected_bytes: [*]const u8 = &[_]u8{0} ** 576;

    const args = LinkedValues.create(&heap, expr.G1Element, expr.G1Element{ .bytes = g1_bytes }, ConstantType.g1ElementType())
        .extend(&heap, expr.G2Element, expr.G2Element{ .bytes = g2_bytes }, ConstantType.g2ElementType());

    const result_val = bls12_381_MillerLoop(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .bls12_381_mlresult => {
                    const r = c.mlResult();
                    for (0..576) |i| {
                        try testing.expectEqual(expected_bytes[i], r.bytes[i]);
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

test "bls12_381_mulMlResult" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const ml1_bytes: [*]const u8 = &[_]u8{0} ** 576;

    const ml2_bytes: [*]const u8 = &[_]u8{0} ** 576;

    const expected_bytes: [*]const u8 = &[_]u8{0} ** 576;

    const args = LinkedValues.create(&heap, expr.MlResult, expr.MlResult{ .bytes = ml1_bytes }, ConstantType.mlResultType())
        .extend(&heap, expr.MlResult, expr.MlResult{ .bytes = ml2_bytes }, ConstantType.mlResultType());

    const result_val = bls12_381_MulMlResult(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .bls12_381_mlresult => {
                    const r = c.mlResult();
                    for (0..576) |i| {
                        try testing.expectEqual(expected_bytes[i], r.bytes[i]);
                    }
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

test "bls12_381_finalVerify" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var heap = try Heap.createTestHeap(&arena);
    var frames = try Frames.createTestFrames(&arena);
    var machine = Machine{ .heap = &heap, .frames = &frames };

    const gt1_bytes: [*]const u8 = &[_]u8{0} ** 576;

    const gt2_bytes: [*]const u8 = &[_]u8{0} ** 576;

    const args = LinkedValues.create(&heap, expr.MlResult, expr.MlResult{ .bytes = gt1_bytes }, ConstantType.mlResultType())
        .extend(&heap, expr.MlResult, expr.MlResult{ .bytes = gt2_bytes }, ConstantType.mlResultType());

    const result_val = bls12_381_FinalVerify(&machine, args);

    switch (result_val.*) {
        .constant => |c| {
            switch (c.constType().*) {
                .boolean => {
                    const r = c.bln();
                    try testing.expect(r);
                },
                else => unreachable,
            }
        },
        else => unreachable,
    }
}

// test "bls12_381_G1_neg" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0x8b, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const args = LinkedValues.create(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType());

//     const result_val = bls12_381_G1_Neg(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g1_element => {
//                     const r = c.g1Element();
//                     for (0..48) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G1_equal" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const args = LinkedValues.create(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType())
//         .extend(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType());

//     const result_val = bls12_381_G1_Equal(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .boolean => {
//                     const r = c.bln();
//                     try testing.expect(r);
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G1_compress" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const args = LinkedValues.create(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType());

//     const result_val = bls12_381_G1_Compress(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bytes => {
//                     const r = c.innerBytes();
//                     try testing.expectEqual(r.length, 48);
//                     for (0..48) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G1_uncompress" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const args = LinkedValues.create(&heap, expr.Bytes, expr.Bytes{ .length = 48, .bytes = p_bytes }, ConstantType.bytesType());

//     const result_val = bls12_381_G1_Uncompress(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g1_element => {
//                     const r = c.g1Element();
//                     for (0..48) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G1_hashToGroup" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const message_bytes: [*]const u8 = &[_]u8{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20 };

//     const dst_bytes: [*]const u8 = &[_]u8{ 0x42, 0x4c, 0x53, 0x5f, 0x47, 0x31, 0x32, 0x5f, 0x47, 0x31, 0x5f, 0x58, 0x4d, 0x44, 0x3a, 0x53, 0x48, 0x41, 0x32, 0x35, 0x36, 0x5f, 0x52, 0x4f, 0x5f };

//     const expected_bytes: [*]const u8 = &[_]u8{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const args = LinkedValues.create(&heap, expr.Bytes, expr.Bytes{ .length = 32, .bytes = message_bytes }, ConstantType.bytesType())
//         .extend(&heap, expr.Bytes, expr.Bytes{ .length = 25, .bytes = dst_bytes }, ConstantType.bytesType());

//     const result_val = bls12_381_G1_HashToGroup(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g1_element => {
//                     const r = c.g1Element();
//                     for (0..48) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G1_scalarMul" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const scalar_words: [*]const u32 = &.{2};

//     const scalar = expr.BigInt{ .sign = 0, .length = 1, .words = scalar_words };

//     const p_bytes: [*]const u32 = &[_]u32{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const expected_bytes: [*]const u32 = &[_]u32{ 0xa4, 0x87, 0x0e, 0x98, 0x3a, 0x14, 0x9b, 0xb1, 0xe7, 0xcc, 0x70, 0xfd, 0xe9, 0x07, 0xa2, 0xaa, 0x52, 0x30, 0x28, 0x33, 0xbc, 0xe4, 0xd6, 0x2f, 0x67, 0x98, 0x19, 0x02, 0x29, 0x24, 0xe9, 0xca, 0xab, 0x52, 0xe3, 0x63, 0x1d, 0x37, 0x6d, 0x36, 0xd9, 0x69, 0x26, 0x64, 0xb4, 0xcf, 0xbc, 0x22 };

//     const args = LinkedValues.create(&heap, expr.BigInt, scalar, ConstantType.integerType())
//         .extend(&heap, expr.G1Element, expr.G1Element{ .bytes = p_bytes }, ConstantType.g1ElementType());

//     const result_val = bls12_381_G1_ScalarMul(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g1_element => {
//                     const r = c.g1Element();
//                     for (0..48) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G2_add" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const q_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const expected_bytes: [*]const u32 = &[_]u32{ 0xa4, 0x87, 0x0e, 0x98, 0x3a, 0x14, 0x9b, 0xb1, 0xe7, 0xcc, 0x70, 0xfd, 0xe9, 0x07, 0xa2, 0xaa, 0x52, 0x30, 0x28, 0x33, 0xbc, 0xe4, 0xd6, 0x2f, 0x67, 0x98, 0x19, 0x02, 0x29, 0x24, 0xe9, 0xca, 0xab, 0x52, 0xe3, 0x63, 0x1d, 0x37, 0x6d, 0x36, 0xd9, 0x69, 0x26, 0x64, 0xb4, 0xcf, 0xbc, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

//     const args = LinkedValues.create(&heap, expr.G2Element, expr.G2Element{ .bytes = q_bytes }, ConstantType.g2ElementType())
//         .extend(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

//     const result_val = bls12_381_G2_Add(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g2_element => {
//                     const r = c.g2Element();
//                     for (0..96) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G2_neg" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const expected_bytes: [*]const u32 = &[_]u32{ 0x94, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const args = LinkedValues.create(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

//     const result_val = bls12_381_G2_Neg(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g2_element => {
//                     const r = c.g2Element();
//                     for (0..96) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G2_equal" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const args = LinkedValues.create(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType())
//         .extend(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

//     const result_val = bls12_381_G2_Equal(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .boolean => {
//                     const r = c.bln();
//                     try testing.expect(r);
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G2_compress" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xc3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const expected_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const args = LinkedValues.create(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

//     const result_val = bls12_381_G2_Compress(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bytes => {
//                     const r = c.innerBytes();
//                     try testing.expectEqual(r.length, 96);
//                     for (0..96) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G2_uncompress" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const p_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const expected_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const args = LinkedValues.create(&heap, expr.Bytes, expr.Bytes{ .length = 96, .bytes = p_bytes }, ConstantType.bytesType());

//     const result_val = bls12_381_G2_Uncompress(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g2_element => {
//                     const r = c.g2Element();
//                     for (0..96) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G2_hashToGroup" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const message_bytes: [*]const u32 = &[_]u32{ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20 };

//     const dst_bytes: [*]const u32 = &[_]u32{ 0x42, 0x4c, 0x53, 0x5f, 0x47, 0x31, 0x32, 0x5f, 0x47, 0x32, 0x5f, 0x58, 0x4d, 0x44, 0x3a, 0x53, 0x48, 0x41, 0x32, 0x35, 0x36, 0x5f, 0x52, 0x4f, 0x5f };

//     const expected_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const args = LinkedValues.create(&heap, expr.Bytes, expr.Bytes{ .length = 32, .bytes = message_bytes }, ConstantType.bytesType())
//         .extend(&heap, expr.Bytes, expr.Bytes{ .length = 25, .bytes = dst_bytes }, ConstantType.bytesType());

//     const result_val = bls12_381_G2_HashToGroup(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g2_element => {
//                     const r = c.g2Element();
//                     for (0..96) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_G2_scalarMul" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const scalar_words: [*]const u32 = &.{2};

//     const scalar = expr.BigInt{ .sign = 0, .length = 1, .words = scalar_words };

//     const p_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const expected_bytes: [*]const u32 = &[_]u32{ 0xa4, 0x87, 0x0e, 0x98, 0x3a, 0x14, 0x9b, 0xb1, 0xe7, 0xcc, 0x70, 0xfd, 0xe9, 0x07, 0xa2, 0xaa, 0x52, 0x30, 0x28, 0x33, 0xbc, 0xe4, 0xd6, 0x2f, 0x67, 0x98, 0x19, 0x02, 0x29, 0x24, 0xe9, 0xca, 0xab, 0x52, 0xe3, 0x63, 0x1d, 0x37, 0x6d, 0x36, 0xd9, 0x69, 0x26, 0x64, 0xb4, 0xcf, 0xbc, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

//     const args = LinkedValues.create(&heap, expr.BigInt, scalar, ConstantType.integerType())
//         .extend(&heap, expr.G2Element, expr.G2Element{ .bytes = p_bytes }, ConstantType.g2ElementType());

//     const result_val = bls12_381_G2_ScalarMul(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_g2_element => {
//                     const r = c.g2Element();
//                     for (0..96) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_millerLoop" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const g1_bytes: [*]const u32 = &[_]u32{ 0xab, 0xd6, 0x18, 0x64, 0xf5, 0x19, 0x74, 0x80, 0x32, 0x55, 0x1e, 0x42, 0xe0, 0xac, 0x41, 0x7f, 0xd8, 0x28, 0xf0, 0x79, 0x45, 0x4e, 0x3e, 0x3c, 0x98, 0x91, 0xc5, 0xc2, 0x9e, 0xd7, 0xf1, 0x0b, 0xde, 0xcc, 0x04, 0x68, 0x54, 0xe3, 0x93, 0x1c, 0xb7, 0x00, 0x27, 0x79, 0xbd, 0x76, 0xd7, 0x1f };

//     const g2_bytes: [*]const u32 = &[_]u32{ 0xb4, 0x95, 0x3c, 0x4b, 0xa1, 0x0c, 0x4d, 0x41, 0x96, 0xf9, 0x01, 0x69, 0xe7, 0x6f, 0xaf, 0x15, 0x4c, 0x26, 0x0e, 0xd7, 0x3f, 0xc7, 0x7b, 0xb6, 0x5d, 0xd3, 0xbe, 0x31, 0xe0, 0xce, 0xc6, 0x14, 0xa7, 0x28, 0x7c, 0xda, 0x94, 0x19, 0x53, 0x43, 0x67, 0x6c, 0x2c, 0x57, 0x49, 0x4f, 0x0e, 0x65, 0x15, 0x27, 0xe6, 0x50, 0x4c, 0x98, 0x40, 0x8e, 0x59, 0x9a, 0x4e, 0xb9, 0x6f, 0x7c, 0x5a, 0x8c, 0xfb, 0x85, 0xd2, 0xfd, 0xc7, 0x72, 0xf2, 0x85, 0x04, 0x58, 0x00, 0x84, 0xef, 0x55, 0x9b, 0x9b, 0x62, 0x3b, 0xc8, 0x4c, 0xe3, 0x05, 0x62, 0xed, 0x32, 0x0f, 0x6b, 0x7f, 0x65, 0x24, 0x5a, 0xd4 };

//     const expected_bytes: [*]const u32 = &[_]u32{0} ** 576;

//     const args = LinkedValues.create(&heap, expr.G1Element, expr.G1Element{ .bytes = g1_bytes }, ConstantType.g1ElementType())
//         .extend(&heap, expr.G2Element, expr.G2Element{ .bytes = g2_bytes }, ConstantType.g2ElementType());

//     const result_val = bls12_381_MillerLoop(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_mlresult => {
//                     const r = c.mlResult();
//                     for (0..576) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_mulMlResult" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const ml1_bytes: [*]const u32 = &[_]u32{0} ** 576;

//     const ml2_bytes: [*]const u32 = &[_]u32{0} ** 576;

//     const expected_bytes: [*]const u32 = &[_]u32{0} ** 576;

//     const args = LinkedValues.create(&heap, expr.MlResult, expr.MlResult{ .length = 576, .bytes = ml1_bytes }, ConstantType.mlResultType())
//         .extend(&heap, expr.MlResult, expr.MlResult{ .length = 576, .bytes = ml2_bytes }, ConstantType.mlResultType());

//     const result_val = bls12_381_MulMlResult(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .bls12_381_mlresult => {
//                     const r = c.mlResult();
//                     for (0..576) |i| {
//                         try testing.expectEqual(expected_bytes[i], r.bytes[i]);
//                     }
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }

// test "bls12_381_finalVerify" {
//     var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//     defer arena.deinit();

//     var heap = try Heap.createTestHeap(&arena);
//     var frames = try Frames.createTestFrames(&arena);
//     var machine = Machine{ .heap = &heap, .frames = &frames };

//     const gt1_bytes: [*]const u32 = &[_]u32{0} ** 576;

//     const gt2_bytes: [*]const u32 = &[_]u32{0} ** 576;

//     const args = LinkedValues.create(&heap, expr.MlResult, expr.MlResult{ .bytes = gt1_bytes }, ConstantType.mlResultType())
//         .extend(&heap, expr.MlResult, expr.MlResult{ .bytes = gt2_bytes }, ConstantType.mlResultType());

//     const result_val = bls12_381_FinalVerify(&machine, args);

//     switch (result_val.*) {
//         .constant => |c| {
//             switch (c.constType().*) {
//                 .boolean => {
//                     const r = c.bln();
//                     try testing.expect(r);
//                 },
//                 else => unreachable,
//             }
//         },
//         else => unreachable,
//     }
// }
