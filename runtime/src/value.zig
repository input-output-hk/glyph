const std = @import("std");
const Heap = @import("Heap.zig");
const expr = @import("expr.zig");
const utils = @import("utils.zig");

const Term = expr.Term;
const ConstantType = expr.ConstantType;
const Constant = expr.Constant;
const DefaultFunction = expr.DefaultFunction;
const BigInt = expr.BigInt;
const Bytes = expr.Bytes;
const String = expr.String;
const List = expr.List;
const G1Element = expr.G1Element;
const G2Element = expr.G2Element;
const MlResult = expr.MlResult;

inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
}

pub const BuiltinContext = struct {
    heap: *Heap,
};

pub export var numer_len_debug: u32 = 0;
pub export var denom_len_debug: u32 = 0;

pub const LinkedValues = struct {
    value: *const Value,
    next: ?*const LinkedValues,

    pub fn create(heap: *Heap, comptime T: type, arg: T, types: *const ConstantType) *LinkedValues {
        const val = createConst(heap, arg.createConstant(types, heap));

        return heap.create(LinkedValues, &.{ .value = val, .next = null });
    }

    pub fn extend(
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
pub const PairConstantView = struct {
    first_value: u32,
    second_value: u32,
    first_type: [*]const ConstantType,
    second_type: [*]const ConstantType,
    first_type_len: u32,
    second_type_len: u32,
};

pub const PairPayload = extern struct {
    first: u32,
    second: u32,
};

pub const Builtin = struct {
    fun: DefaultFunction,
    force_count: u8,
    arity: u8,
    args: ?*LinkedValues,
};

pub const Value = union(enum(u32)) {
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

        try std.testing.expectEqualDeep(env.value, v);
        try std.testing.expect(env.next == null);
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

        try std.testing.expectEqualDeep(value, v);
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
