const Heap = @import("Heap.zig");
const expr = @import("./expr.zig");
const std = @import("std");

const Term = expr.Term;
const TermList = expr.TermList;
const Constant = expr.Constant;
const DefaultFunction = expr.DefaultFunction;

const ValueList = struct { length: u32, list: [*]*Value };

const Value = union(enum) {
    constant: *Constant,
    delay: struct {
        env: *Env,
        body: *const Term,
    },
    lambda: struct {
        env: *Env,
        body: *const Term,
    },
    builtin: struct {
        fun: DefaultFunction,
    },
};

const Frame = union(enum) {
    frameAwaitArg: struct {
        function: *const Value,
    },
    frameAwaitFunTerm: struct {
        env: *Env,
        argument: *const Term,
    },
    frameAwaitFunValue: struct {
        argument: *const Value,
    },
    frameForce: void,
    frameConstr: struct {
        env: *Env,
        tag: u32,
        fields: TermList,
        resolved_fields: ValueList,
    },
    frameCases: struct {
        env: *Env,
        branches: TermList,
    },
    noFrame: void,
};

pub const Env = struct {
    heap: *Heap,
    env_list: ?*EnvList,

    const Self = @This();

    const EnvList = struct { value: *Value, next: ?*EnvList };

    pub fn init(heap: *Heap) Self {
        return Env{
            .heap = heap,
            .env_list = null,
        };
    }

    // Creates Delay on the heap and returns a pointer to it
    pub fn createDelay(env: *Self, body: *const Term) *Value {
        var val = Value{ .delay = .{ .env = env, .body = body } };

        const valData: *align(4) [@sizeOf(Value)]u8 = std.mem.asBytes(&val);

        @memcpy(env.heap.heap_ptr, valData);

        const heapVal: [*]align(4) u8 = @alignCast(env.heap.heap_ptr);
        const heapVal2: *Value = @ptrCast(heapVal);

        env.heap.heap_ptr = env.heap.heap_ptr + @sizeOf(Value);

        return heapVal2;
    }

    // Creates Lambda on the heap and returns a pointer to it
    pub fn createLambda(env: *Self, body: *const Term) *const Value {
        var val = Value{ .lambda = .{ .env = env, .body = body } };

        const valData: *align(4) [@sizeOf(Value)]u8 = std.mem.asBytes(&val);

        @memcpy(env.heap.heap_ptr, valData);

        const heapVal: [*]align(4) u8 = @alignCast(env.heap.heap_ptr);
        const heapVal2: *Value = @ptrCast(heapVal);

        env.heap.heap_ptr = env.heap.heap_ptr + @sizeOf(Value);

        return heapVal2;
    }
};
