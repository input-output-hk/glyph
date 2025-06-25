const Heap = @import("Heap.zig");
const expr = @import("./expr.zig");
const std = @import("std");
const testing = std.testing;
const Term = expr.Term;
const TermList = expr.TermList;
const Constant = expr.Constant;
const DefaultFunction = expr.DefaultFunction;

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
const LinkedValues = struct { value: *const Value, next: ?*const LinkedValues };

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
                switch (c.*) {
                    .unit => return true,
                    else => return false,
                }
            },
            else => return false,
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
        @panic("open term during evaluation");
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

pub fn addInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn subInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn multiplyInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
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

pub fn equalsInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn lessThanInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn lessThanEqualsInteger(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// ByteString functions
pub fn appendByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn consByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn sliceByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn lengthOfByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn indexByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn equalsByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn lessThanByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn lessThanEqualsByteString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
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
pub fn appendString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn equalsString(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn encodeUtf8(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn decodeUtf8(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Bool function
pub fn ifThenElse(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Unit function
pub fn chooseUnit(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Tracing function
pub fn trace(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// Pairs functions
pub fn fstPair(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn sndPair(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

// List functions
pub fn chooseList(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn mkCons(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn headList(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
}

pub fn tailList(_: *Machine, _: *LinkedValues) *const Value {
    @panic("TODO");
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

pub fn addSignedIntegers() void {}
pub fn subSignedIntegers() void {}

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
                        @panic("Returned term other than unit");
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
                        t.constantValue(),
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

            .terror => @panic("evaluation failure"),

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
                const next_resolved = self.heap.create(
                    LinkedValues,
                    &LinkedValues{ .value = v, .next = f.resolved_fields },
                );

                if (f.fields.length == 0) {
                    return State{
                        .ret = .{
                            .value = createConstr(
                                self.heap,
                                f.tag,
                                next_resolved,
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
                                .resolved_fields = next_resolved,
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
                            @panic("constructor tag out of range");
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
                    else => @panic("case on non-constructor"),
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
                if (b.force_count == 0) @panic("builtin term argument expected");

                return State{
                    .ret = .{
                        .value = forceBuiltin(self.heap, &b),
                    },
                };
            },
            else => @panic("non-polymorphic instantiation"),
        }
    }

    fn applyEval(
        self: *Self,
        funVal: *const Value,
        argVal: *const Value,
    ) State {
        switch (funVal.*) {
            .lambda => |lam| {
                const new_env = if (lam.env) |env| blk: {
                    break :blk env.preprend(argVal, self.heap);
                } else blk: {
                    break :blk Env.init(argVal, self.heap);
                };

                return State{
                    .compute = .{
                        .env = new_env,
                        .term = lam.body,
                    },
                };
            },

            .builtin => |b| {
                if (b.force_count != 0)
                    @panic("unexpected built-in term argument");

                if (b.arity == 0) {
                    @panic("unexpected built-in term argument");
                }

                const next_arity = b.arity - 1;

                const nextArg = self.heap.create(
                    LinkedValues,
                    &LinkedValues{
                        .value = argVal,
                        .next = b.args,
                    },
                );

                const builtinValue = blk: {
                    if (next_arity == 0) {
                        break :blk self.callBuiltin(b.fun, nextArg);
                    } else {
                        break :blk self.heap.create(
                            Value,
                            &Value{
                                .builtin = .{
                                    .fun = b.fun,
                                    .force_count = b.force_count,
                                    .arity = next_arity,
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

            else => @panic("apply on non-callable"),
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

    const next_state = switch (state) {
        .compute => |c| blk: {
            break :blk machine.compute(c.env, c.term);
        },
        else => @panic("HERE???"),
    };

    const final = switch (next_state) {
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

    var next_state = switch (state) {
        .compute => |c| blk: {
            break :blk machine.compute(c.env, c.term);
        },
        else => @panic("HERE???1"),
    };

    next_state = switch (next_state) {
        .ret => |r| blk: {
            break :blk machine.ret(r.value);
        },
        else => {
            std.debug.print("{}", .{next_state});
            @panic("HERE???2");
        },
    };

    next_state = switch (next_state) {
        .compute => |c| blk: {
            break :blk machine.compute(c.env, c.term);
        },
        else => @panic("HERE???4"),
    };

    const final = switch (next_state) {
        .ret => |r| blk: {
            break :blk machine.ret(r.value);
        },
        else => {
            std.debug.print("{}", .{next_state});
            @panic("HERE???2");
        },
    };

    const first_field = LinkedValues{
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
        .next = &first_field,
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

    const term: []const u32 = &.{
        4,
        3,
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

    const term: []const u32 = &.{ 4, 0, 1, 1, 11 };

    const ptr: *const Term = @ptrCast(term);

    machine.frames.addFrame(&.no_frame);

    const state = machine.compute(null, ptr);

    switch (state) {
        .ret => |r| {
            switch (r.value.constant.*) {
                .integer => {
                    const bigInt = r.value.constant.bigInt();
                    try testing.expect(bigInt.length == 1);
                    try testing.expect(bigInt.sign == 1);
                    try testing.expect(bigInt.words[0] == 11);
                },
                else => @panic("How?"),
            }
        },
        else => @panic("How?"),
    }
}
