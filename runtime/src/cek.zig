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
const runtime_value = @import("value.zig");
const builtins_bool = @import("builtins/bool.zig");
const builtins_bytestring = @import("builtins/bytestring.zig");
const builtins_crypto = @import("builtins/crypto.zig");
const builtins_data = @import("builtins/data.zig");
const builtins_integer = @import("builtins/integer.zig");
const builtins_list = @import("builtins/list.zig");
const builtins_pair = @import("builtins/pair.zig");
const builtins_string = @import("builtins/string.zig");
usingnamespace @import("builtins/bool.zig");
usingnamespace @import("builtins/bytestring.zig");
usingnamespace @import("builtins/integer.zig");
usingnamespace @import("builtins/list.zig");
usingnamespace @import("builtins/pair.zig");
usingnamespace @import("builtins/string.zig");
const addInteger = builtins_integer.addInteger;
const subInteger = builtins_integer.subInteger;
const multiplyInteger = builtins_integer.multiplyInteger;
const divideInteger = builtins_integer.divideInteger;
const quotientInteger = builtins_integer.quotientInteger;
const remainderInteger = builtins_integer.remainderInteger;
const modInteger = builtins_integer.modInteger;
const equalsInteger = builtins_integer.equalsInteger;
const lessThanInteger = builtins_integer.lessThanInteger;
const lessThanEqualsInteger = builtins_integer.lessThanEqualsInteger;
const addSignedIntegers = builtins_integer.addSignedIntegers;
const subSignedIntegers = builtins_integer.subSignedIntegers;

const appendByteString = builtins_bytestring.appendByteString;
const consByteString = builtins_bytestring.consByteString;
const sliceByteString = builtins_bytestring.sliceByteString;
const lengthOfByteString = builtins_bytestring.lengthOfByteString;
const indexByteString = builtins_bytestring.indexByteString;
const equalsByteString = builtins_bytestring.equalsByteString;
const lessThanByteString = builtins_bytestring.lessThanByteString;
const lessThanEqualsByteString = builtins_bytestring.lessThanEqualsByteString;
const integerToByteString = builtins_bytestring.integerToByteString;
const byteStringToInteger = builtins_bytestring.byteStringToInteger;
const andByteString = builtins_bytestring.andByteString;
const orByteString = builtins_bytestring.orByteString;
const xorByteString = builtins_bytestring.xorByteString;
const complementByteString = builtins_bytestring.complementByteString;
const readBit = builtins_bytestring.readBit;
const writeBits = builtins_bytestring.writeBits;
const replicateByte = builtins_bytestring.replicateByte;
const shiftByteString = builtins_bytestring.shiftByteString;
const rotateByteString = builtins_bytestring.rotateByteString;
const countSetBits = builtins_bytestring.countSetBits;
const findFirstSetBit = builtins_bytestring.findFirstSetBit;

const appendString = builtins_string.appendString;
const equalsString = builtins_string.equalsString;
const encodeUtf8 = builtins_string.encodeUtf8;
const decodeUtf8 = builtins_string.decodeUtf8;
const chooseList = builtins_list.chooseList;
const headList = builtins_list.headList;
const tailList = builtins_list.tailList;
const nullList = builtins_list.nullList;
const ifThenElse = builtins_bool.ifThenElse;
const chooseUnit = builtins_bool.chooseUnit;
const trace = builtins_bool.trace;
const mkCons = builtins_list.mkCons;
const fstPair = builtins_pair.fstPair;
const sndPair = builtins_pair.sndPair;
const chooseData = builtins_data.chooseData;
const constrData = builtins_data.constrData;
const mapData = builtins_data.mapData;
const listData = builtins_data.listData;
const iData = builtins_data.iData;
const bData = builtins_data.bData;
const unConstrData = builtins_data.unConstrData;
const unMapData = builtins_data.unMapData;
const unListData = builtins_data.unListData;
const unIData = builtins_data.unIData;
const unBData = builtins_data.unBData;
const equalsData = builtins_data.equalsData;
const mkPairData = builtins_data.mkPairData;
const mkNilData = builtins_data.mkNilData;
const mkNilPairData = builtins_data.mkNilPairData;
const serialiseData = builtins_data.serialiseData;
const sha2_256 = builtins_crypto.sha2_256;
const sha3_256 = builtins_crypto.sha3_256;
const blake2b_256 = builtins_crypto.blake2b_256;
const verifyEd25519Signature = builtins_crypto.verifyEd25519Signature;
const verifyEcdsaSecp256k1Signature = builtins_crypto.verifyEcdsaSecp256k1Signature;
const verifySchnorrSecp256k1Signature = builtins_crypto.verifySchnorrSecp256k1Signature;
const bls12_381_G1_Add = builtins_crypto.bls12_381_G1_Add;
const bls12_381_G1_Neg = builtins_crypto.bls12_381_G1_Neg;
const bls12_381_G1_ScalarMul = builtins_crypto.bls12_381_G1_ScalarMul;
const bls12_381_G1_Equal = builtins_crypto.bls12_381_G1_Equal;
const bls12_381_G1_Compress = builtins_crypto.bls12_381_G1_Compress;
const bls12_381_G1_Uncompress = builtins_crypto.bls12_381_G1_Uncompress;
const bls12_381_G1_HashToGroup = builtins_crypto.bls12_381_G1_HashToGroup;
const bls12_381_G2_Add = builtins_crypto.bls12_381_G2_Add;
const bls12_381_G2_Neg = builtins_crypto.bls12_381_G2_Neg;
const bls12_381_G2_ScalarMul = builtins_crypto.bls12_381_G2_ScalarMul;
const bls12_381_G2_Equal = builtins_crypto.bls12_381_G2_Equal;
const bls12_381_G2_Compress = builtins_crypto.bls12_381_G2_Compress;
const bls12_381_G2_Uncompress = builtins_crypto.bls12_381_G2_Uncompress;
const bls12_381_G2_HashToGroup = builtins_crypto.bls12_381_G2_HashToGroup;
const bls12_381_MillerLoop = builtins_crypto.bls12_381_MillerLoop;
const bls12_381_MulMlResult = builtins_crypto.bls12_381_MulMlResult;
const bls12_381_FinalVerify = builtins_crypto.bls12_381_FinalVerify;
const keccak_256 = builtins_crypto.keccak_256;
const blake2b_224 = builtins_crypto.blake2b_224;
const ripemd_160 = builtins_crypto.ripemd_160;
const Builtin = runtime_value.Builtin;
const BuiltinContext = runtime_value.BuiltinContext;
const Env = runtime_value.Env;
const LinkedValues = runtime_value.LinkedValues;
const Value = runtime_value.Value;
const createConst = runtime_value.createConst;
const createDelay = runtime_value.createDelay;
const createLambda = runtime_value.createLambda;
const createBuiltin = runtime_value.createBuiltin;
const forceBuiltin = runtime_value.forceBuiltin;
const createConstr = runtime_value.createConstr;
inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
}

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

const ValueList = struct { length: u32, list: [*]*const Value };


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
    if (constant.length == builtins_data.serialized_data_const_tag) {
        if (builtins_data.serializedPayloadWordCount(constant.value) != null) {
            builtins_data.normalizeSerializedDataConstantForHost(constant);
        }
    }
    // Other constant kinds already match the host layout consumed by the host tests.
}

pub const builtinFunctions = [_]*const fn (*BuiltinContext, *LinkedValues) *const Value{
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

// integer helper builtins moved to runtime/src/builtins
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
        var ctx = BuiltinContext{ .heap = self.heap };
        return builtinFunctions[index](&ctx, args);
    }
};

test "lambda compute" {
    var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer allocator.deinit();

    var heap = try Heap.createTestHeap(&allocator);
    var frames = try Frames.createTestFrames(&allocator);
    const machine = Machine{
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
    const machine = Machine{
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
    const machine = Machine{
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
    const machine = Machine{
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
    const machine = Machine{
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
    const machine = Machine{
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
    const machine = Machine{
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
    const machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const ctx = BuiltinContext{ .heap = machine.heap };
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

    const newVal = addSignedIntegers(&ctx, x, y);

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
    const machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const ctx = BuiltinContext{ .heap = machine.heap };
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

    const newVal = addSignedIntegers(&ctx, x, y);

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
    const machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const ctx = BuiltinContext{ .heap = machine.heap };
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

    const newVal = subSignedIntegers(&ctx, x, y);

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
    const machine = Machine{
        .heap = &heap,
        .frames = &frames,
    };

    const ctx = BuiltinContext{ .heap = machine.heap };
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

    const newVal = subSignedIntegers(&ctx, x, y);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const a_words: [*]const u32 = &.{3};
    const b_words: [*]const u32 = &.{4};

    const a = expr.BigInt{ .sign = 0, .length = 1, .words = a_words };
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = b_words };

    const args = LinkedValues.create(&heap, *const expr.BigInt, &a, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &b, ConstantType.integerType());

    const newVal = multiplyInteger(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const a_words: [*]const u32 = &.{std.math.maxInt(u32)}; // 0xFFFFFFFF
    const b_words: [*]const u32 = &.{2};

    const a = expr.BigInt{ .sign = 0, .length = 1, .words = a_words };
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = b_words };

    const args = LinkedValues.create(&heap, *const expr.BigInt, &a, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &b, ConstantType.integerType());

    const newVal = multiplyInteger(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const a_words: [*]const u32 = &.{3};
    const b_words: [*]const u32 = &.{4};

    const a = expr.BigInt{ .sign = 1, .length = 1, .words = a_words }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = b_words }; // positive

    const args = LinkedValues.create(&heap, *const expr.BigInt, &a, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &b, ConstantType.integerType());

    const newVal = multiplyInteger(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aWords: [*]const u32 = &.{ 705032704, 1 };
    const bWords: [*]const u32 = &.{5};
    const cWords: [*]const u32 = &.{ 705032699, 1 };

    const a = expr.BigInt{ .sign = 1, .length = 2, .words = aWords }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = bWords }; // positive
    const c = expr.BigInt{ .sign = 1, .length = 2, .words = cWords };

    const resultWords = subSignedIntegers(&ctx, a, b);

    const args = LinkedValues.create(&heap, *const expr.BigInt, &c, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &resultWords.constant.bigInt(), ConstantType.integerType());

    const newVal = equalsInteger(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aWords: [*]const u32 = &.{ 705032704, 1 };
    const bWords: [*]const u32 = &.{5};
    const cWords: [*]const u32 = &.{ 705032698, 1 };

    const a = expr.BigInt{ .sign = 1, .length = 2, .words = aWords }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = bWords }; // positive
    const c = expr.BigInt{ .sign = 1, .length = 2, .words = cWords };

    const resultWords = subSignedIntegers(&ctx, a, b);

    const args = LinkedValues.create(&heap, *const expr.BigInt, &c, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &resultWords.constant.bigInt(), ConstantType.integerType());

    const newVal = lessThanInteger(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aWords: [*]const u32 = &.{ 705032704, 1 };
    const bWords: [*]const u32 = &.{5};
    const cWords: [*]const u32 = &.{ 705032698, 1 };

    const a = expr.BigInt{ .sign = 1, .length = 2, .words = aWords }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = bWords }; // positive
    const c = expr.BigInt{ .sign = 1, .length = 2, .words = cWords };

    const resultWords = subSignedIntegers(&ctx, a, b);

    const args = LinkedValues
        .create(&heap, *const expr.BigInt, &c, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &resultWords.constant.bigInt(), ConstantType.integerType());

    const newVal = lessThanEqualsInteger(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aWords: [*]const u32 = &.{ 705032704, 1 };
    const bWords: [*]const u32 = &.{5};
    const cWords: [*]const u32 = &.{ 705032699, 1 };

    const a = expr.BigInt{ .sign = 1, .length = 2, .words = aWords }; // negative
    const b = expr.BigInt{ .sign = 0, .length = 1, .words = bWords }; // positive
    const c = expr.BigInt{ .sign = 1, .length = 2, .words = cWords };

    const resultWords = subSignedIntegers(&ctx, a, b);

    const args = LinkedValues
        .create(&heap, *const expr.BigInt, &c, ConstantType.integerType())
        .extend(&heap, *const expr.BigInt, &resultWords.constant.bigInt(), ConstantType.integerType());

    const newVal = lessThanEqualsInteger(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aBytes: [*]const u32 = &.{ 255, 254 };
    const bBytes: [*]const u32 = &.{ 0, 255, 1 };
    const resultBytes: [*]const u32 = &.{ 255, 254, 0, 255, 1 };

    const a = expr.Bytes{ .length = 2, .bytes = aBytes };
    const b = expr.Bytes{ .length = 3, .bytes = bBytes };

    const result = expr.Bytes{ .length = 5, .bytes = resultBytes };

    const args = LinkedValues
        .create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = appendByteString(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aByte: [*]const u32 = &.{37};
    const bBytes: [*]const u32 = &.{ 0, 255, 1 };
    const resultBytes: [*]const u32 = &.{ 37, 0, 255, 1 };

    const a = expr.BigInt{ .length = 1, .sign = 0, .words = aByte };
    const b = expr.Bytes{ .length = 3, .bytes = bBytes };

    const result = expr.Bytes{ .length = 4, .bytes = resultBytes };

    const args = LinkedValues
        .create(&heap, *const expr.BigInt, &a, ConstantType.integerType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = consByteString(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
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

    const newVal = sliceByteString(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aBytes: [*]const u32 = &.{ 0, 255, 1, 4 };
    const resultBytes: [*]const u32 = &.{4};

    const a = expr.Bytes{ .length = 4, .bytes = aBytes };

    const result = expr.BigInt{ .length = 1, .sign = 0, .words = resultBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType());

    const newVal = lengthOfByteString(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aBytes: [*]const u32 = &.{ 0, 255, 1, 72, 6 };
    const index: [*]const u32 = &.{3};
    const resultBytes: [*]const u32 = &.{72};

    const a = expr.Bytes{ .length = 5, .bytes = aBytes };
    const b = expr.BigInt{ .length = 1, .sign = 0, .words = index };

    const result = expr.BigInt{ .length = 1, .sign = 0, .words = resultBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.BigInt, &b, ConstantType.integerType());

    const newVal = indexByteString(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aBytes: [*]const u32 = &.{ 0, 255, 1, 72, 6 };

    const a = expr.Bytes{ .length = 5, .bytes = aBytes };
    const b = expr.Bytes{ .length = 5, .bytes = aBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = equalsByteString(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aBytes: [*]const u32 = &.{ 0, 254, 1, 72, 6, 99 };
    const bBytes: [*]const u32 = &.{ 0, 255, 1, 72, 6 };

    const a = expr.Bytes{ .length = 6, .bytes = aBytes };
    const b = expr.Bytes{ .length = 5, .bytes = bBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = lessThanByteString(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aBytes: [*]const u32 = &.{ 0, 254, 1, 72, 6, 99 };
    const bBytes: [*]const u32 = &.{ 0, 255, 1, 72, 6 };

    const a = expr.Bytes{ .length = 6, .bytes = aBytes };
    const b = expr.Bytes{ .length = 5, .bytes = bBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = lessThanEqualsByteString(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aBytes: [*]const u32 = &.{ 0, 254, 1, 72, 6, 99 };

    const a = expr.Bytes{ .length = 6, .bytes = aBytes };
    const b = expr.Bytes{ .length = 6, .bytes = aBytes };

    const args = LinkedValues.create(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = lessThanEqualsByteString(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aBytes: [*]const u32 = &.{ 255, 254, 0, 0 };
    const bBytes: [*]const u32 = &.{ 0, 255, 1, 0 };
    const resultBytes: [*]const u32 = &.{ 255, 254, 0, 255, 1 };

    const a = expr.String{ .length = 2, .bytes = aBytes };
    const b = expr.String{ .length = 3, .bytes = bBytes };

    const result = expr.String{ .length = 5, .bytes = resultBytes };

    const args = LinkedValues
        .create(&heap, *const expr.String, &a, ConstantType.stringType())
        .extend(&heap, *const expr.String, &b, ConstantType.stringType());

    const newVal = appendString(&ctx, args);

    switch (newVal.*) {
        .constant => |con| {
            switch (con.constType().*) {
                .string => {
                    const val = con.string();
                    const view = builtins_string.analyzeString(val);
                    try testing.expectEqual(view.byte_len, result.length);

                    try testing.expectEqual(builtins_string.extractStringByte(val, view, 0), 255);
                    try testing.expectEqual(builtins_string.extractStringByte(val, view, 1), 254);
                    try testing.expectEqual(builtins_string.extractStringByte(val, view, 2), 0);
                    try testing.expectEqual(builtins_string.extractStringByte(val, view, 3), 255);
                    try testing.expectEqual(builtins_string.extractStringByte(val, view, 4), 1);
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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
    const aBytes: [*]const u32 = &.{ 0, 254 };
    const bBytes: [*]const u32 = &.{ 1, 253, 3 };

    const a = expr.Bytes{ .length = 2, .bytes = aBytes };
    const b = expr.Bytes{ .length = 3, .bytes = bBytes };
    const x = expr.Bool{ .val = @intFromBool(true) };

    const args = LinkedValues
        .create(&heap, *const expr.Bool, &x, ConstantType.booleanType())
        .extend(&heap, *const expr.Bytes, &a, ConstantType.bytesType())
        .extend(&heap, *const expr.Bytes, &b, ConstantType.bytesType());

    const newVal = ifThenElse(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
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

    const newVal = chooseList(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
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

    const newVal = chooseList(&ctx, args);

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
    const machine = Machine{ .heap = &heap, .frames = &frames };

    const ctx = BuiltinContext{ .heap = machine.heap };
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

    const newVal = mkCons(&ctx, args);

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
