const Heap = @import("Heap.zig");
const expr = @import("./expr.zig");
const std = @import("std");
const testing = std.testing;
const Term = expr.Term;
const TermList = expr.TermList;
const Constant = expr.Constant;
const DefaultFunction = expr.DefaultFunction;

const Frame = union(enum) {
    no_frame,
    frame_await_arg: struct { function: *Value, ctx: *Frame },
    frame_await_fun_term: struct { env: ?*Env, argument: *const Term, ctx: *Frame },
    frame_await_fun_value: struct { argument: *Value, ctx: *Frame },
    frame_force: struct { ctx: *Frame },
    frame_constr: struct {
        env: ?*Env,
        tag: u32,
        fields: TermList,
        resolved_fields: ValueList,
        ctx: *Frame,
    },
    frame_cases: struct { env: ?*Env, branches: TermList, ctx: *Frame },
};

const ValueList = struct { length: u32, list: [*]*Value };

const Builtin = struct {
    fun: DefaultFunction,
    force_count: u8,
    args_count: u8,
    args: ValueList,
};

const Value = union(enum) {
    constant: *Constant,
    delay: struct {
        env: ?*Env,
        body: *const Term,
    },
    lambda: struct {
        env: ?*Env,
        body: *const Term,
    },
    builtin: Builtin,
    constr: struct { tag: u32, fields: ValueList },
};

pub const Env = struct {
    value: *const Value,
    next: ?*Env,

    const Self = @This();

    pub fn init(v: *const Value, m: *Machine) *Self {
        return m.heap.create(Env, &.{ .value = v, .next = null });
    }

    pub fn preprend(self: *Self, v: *const Value, m: *Machine) *Self {
        return m.heap.create(Env, &.{ .value = v, .next = self });
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
        var machine = Machine{
            .heap = &heap,
        };

        const t = Term.terror;
        const v = createDelay(&heap, null, &t);

        const env = Env.init(v, &machine);

        try testing.expectEqualDeep(env.value, v);
        try testing.expect(env.next == null);
    }

    test "lookup" {
        var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer allocator.deinit();

        var heap = try Heap.createTestHeap(&allocator);
        var machine = Machine{
            .heap = &heap,
        };

        const t = Term.terror;
        const v = createDelay(&heap, null, &t);

        const v2 = createLambda(&heap, null, &t);

        var env = Env.init(v, &machine);
        env = env.preprend(v2, &machine);

        const value = env.lookupVar(2);

        try testing.expectEqualDeep(value, v);
    }
};

pub fn createConst(heap: *Heap, c: *Constant) *Value {
    return heap.create(Value, &.{ .constant = c });
}

pub fn createDelay(heap: *Heap, env: ?*Env, b: *const Term) *Value {
    return heap.create(Value, &.{ .delay = .{ .env = env, .body = b } });
}

pub fn createLambda(heap: *Heap, env: ?*Env, b: *const Term) *Value {
    return heap.create(Value, &.{ .lambda = .{ .env = env, .body = b } });
}

pub fn createBuiltin(heap: *Heap, f: DefaultFunction) *Value {
    return heap.create(Value, &.{
        .builtin = .{
            .fun = f,
            .force_count = f.forceCount(),
            .args_count = f.arity(),
            .args = ValueList{ .length = 0, .list = undefined },
        },
    });
}

pub fn createConstr(heap: *Heap, tag: u32, vls: ValueList) *Value {
    return heap.create(Value, &.{ .constr = .{ .tag = tag, .fields = vls } });
}

pub const State = union(enum) {
    compute: struct {
        frame: *Frame,
        env: ?*Env,
        term: *const Term,
    },
    ret: struct {
        frame: *Frame,
        value: *const Value,
    },
    done: *Term,
};

pub const Machine = struct {
    heap: *Heap,

    const Self = @This();

    pub fn init(heap: *Heap) Self {
        return .{ .heap = heap };
    }

    pub fn run(self: *Self, t: *const Term) void {
        const ctx = self.heap.create(Frame, &.no_frame);

        var state = State{
            .compute = .{
                .frame = ctx,
                .env = null,
                .term = t,
            },
        };

        while (true) {
            switch (state) {
                .compute => |c| {
                    state = self.compute(c.frame, c.env, c.term);
                },
                .ret => |r| {
                    state = self.ret(r.frame, r.value);
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

    fn compute(self: *Self, ctx: *Frame, env: ?*Env, t: *const Term) State {
        switch (t.*) {
            .tvar => return State{
                .ret = .{
                    .frame = ctx,
                    .value = env.?.lookupVar(t.debruijnIndex()),
                },
            },
            // .constant => return self.ret(ctx, self.makeConst(t.constantValue())),
            .lambda => return State{
                .ret = .{
                    .frame = ctx,
                    .value = createLambda(self.heap, env, t.termBody()),
                },
            },
            .delay => return State{
                .ret = .{
                    .frame = ctx,
                    .value = createDelay(self.heap, env, t.termBody()),
                },
            },
            .force => {
                const nctx = self.heap.create(Frame, &.{ .frame_force = .{ .ctx = ctx } });
                return self.compute(nctx, env, t.termBody());
            },

            // .apply => {
            //     const p = t.appliedTerms();
            //     const nctx = self.makeFrame(.{ .frameAwaitFunTerm = .{
            //         .env = self,
            //         .argument = p.argument,
            //         .ctx = ctx,
            //     } });
            //     return self.compute(nctx, p.function);
            // },

            // .builtin => return self.ret(ctx, self.makeBuilt(t.defaultFunction())),

            // .constr => {
            //     const c = t.constrValues();
            //     if (c.fields.length == 0)
            //         return self.ret(ctx, self.makeConstr(c.tag, ValueList{ .length = 0, .list = undefined }));

            //     const rest = TermList{
            //         .length = c.fields.length - 1,
            //         .list = c.fields.list + 1,
            //     };
            //     const fr = Frame{ .frameConstr = .{
            //         .env = self,
            //         .tag = c.tag,
            //         .fields = rest,
            //         .resolved_fields = ValueList{ .length = 0, .list = undefined },
            //         .ctx = ctx,
            //     } };
            //     const nctx = self.makeFrame(fr);
            //     return self.compute(nctx, c.fields.list[0]);
            // },

            // .case => {
            //     const cs = t.caseValues();
            //     const fr = Frame{ .frameCases = .{
            //         .env = self,
            //         .branches = cs.branches,
            //         .ctx = ctx,
            //     } };
            //     const nctx = self.makeFrame(fr);
            //     return self.compute(nctx, cs.constr);
            // },

            .terror => @panic("evaluation failure"),
            else => @panic("Impossible"),
        }
    }

    fn ret(self: *Self, ctx: *Frame, v: *Value) State {
        switch (ctx.*) {
            .noFrame => return v,

            .frameForce => |f| return self.forceEval(f.ctx, v),

            .frameAwaitArg => |f| return self.applyEval(f.ctx, f.function, v),
            .frameAwaitFunValue => |f| return self.applyEval(f.ctx, v, f.argument),

            .frameAwaitFunTerm => |f| {
                const nctx = self.makeFrame(.{ .frameAwaitArg = .{ .function = v, .ctx = f.ctx } });
                return self.compute(nctx, f.argument);
            },

            .frameConstr => |f| {
                const new_len = f.resolved_fields.length + 1;
                const dst = self.heap.allocArray(*Value, new_len);
                dst[0] = v;
                var i: u32 = 0;
                while (i < f.resolved_fields.length) : (i += 1)
                    dst[i + 1] = f.resolved_fields.list[i];

                const done = ValueList{ .length = new_len, .list = dst };

                if (f.fields.length == 0) {
                    return self.ret(f.ctx, self.makeConstr(f.tag, done));
                }

                const next = f.fields.list[0];
                const rest = TermList{
                    .length = f.fields.length - 1,
                    .list = f.fields.list + 1,
                };
                const fr2 = Frame{ .frameConstr = .{
                    .env = f.env,
                    .tag = f.tag,
                    .fields = rest,
                    .resolved_fields = done,
                    .ctx = f.ctx,
                } };
                const nctx = self.makeFrame(fr2);
                return self.compute(nctx, next);
            },

            .frameCases => |f| {
                switch (v.*) {
                    .constr => |c| {
                        if (c.tag >= f.branches.length)
                            @panic("constructor tag out of range");
                        // push constructor fields in reverse order
                        var nctx = f.ctx;
                        var idx: i32 = @as(i32, @intCast(c.fields.length)) - 1;
                        while (idx >= 0) : (idx -= 1)
                            nctx = self.makeFrame(.{ .frameAwaitFunValue = .{
                                .argument = c.fields.list[@as(usize, @intCast(idx))],
                                .ctx = nctx,
                            } });
                        return self.compute(nctx, f.branches.list[c.tag]);
                    },
                    else => @panic("case on non-constructor"),
                }
            },
        }
    }

    fn forceEval(self: *Self, ctx: *Frame, v: *Value) *Value {
        switch (v.*) {
            .delay => |d| return d.env.compute(ctx, d.body),
            .builtin => |*b| {
                if (b.force_count == 0) @panic("builtin term argument expected");
                b.force_count -= 1;
                return self.ret(ctx, v);
            },
            else => @panic("non-polymorphic instantiation"),
        }
    }

    fn applyEval(
        self: *Self,
        ctx: *Frame,
        funVal: *Value,
        argVal: *Value,
    ) *Value {
        switch (funVal.*) {
            .lambda => |lam| {
                const new_env = lam.env.extend(argVal);
                return new_env.compute(ctx, lam.body);
            },

            .builtin => |b| {
                if (b.force_count != 0)
                    @panic("unexpected built-in term argument");

                const res = self.evalBuiltin(&b, argVal);
                return self.ret(ctx, res);
            },

            else => @panic("apply on non-callable"),
        }
    }

    fn evalBuiltin(self: *Self, b: *const Builtin, next: *Value) *Value {
        const new_len = b.args.length + 1;
        const dst = self.heap.allocArray(*Value, new_len);
        var i: u32 = 0;
        while (i < b.args.length) : (i += 1) dst[i] = b.args.list[i];
        dst[b.args.length] = next;

        if (b.args_count == 1) {
            return callBuiltin(self, b.fun, ValueList{ .length = new_len, .list = dst });
        }

        // Create a new builtin value with updated args
        const new_builtin = self.heap.create(.{ .builtin = .{
            .fun = b.fun,
            .force_count = b.force_count,
            .args_count = b.args_count - 1,
            .args = ValueList{ .length = new_len, .list = dst },
        } });
        return new_builtin;
    }

    fn callBuiltin(
        self: *Self,
        fun: DefaultFunction,
        args: ValueList,
    ) *Value {
        switch (fun) {
            .add_integer => {
                if (args.length != 2) @panic("arity error for add_integer");
                const a = self.unwrapInt(args.list[0]);
                const b = self.unwrapInt(args.list[1]);
                return self.wrapInt(a + b);
            },
            .subtract_integer => {
                if (args.length != 2) @panic("arity error for subtract_integer");
                const a = self.unwrapInt(args.list[0]);
                const b = self.unwrapInt(args.list[1]);
                return self.wrapInt(a - b);
            },
        }
    }

    fn unwrapInt(_: *Self, v: *Value) i128 {
        if (v.* != .constant) @panic("expected integer constant");
        return v.constant.*.integer;
    }
    fn wrapInt(self: *Self, i: i128) *Value {
        const c = self.heap.allocType(Constant);
        c.* = Constant{ .integer = i };
        return self.makeConst(c);
    }

    test "lambda compute" {
        var allocator = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer allocator.deinit();

        var heap = try Heap.createTestHeap(&allocator);
        var machine = Self{
            .heap = &heap,
        };

        const v = Term.tvar;

        const expected = createLambda(&heap, null, &v);

        const memory: []const u32 = &.{ 2, 0, 1 };
        const ptr: *const Term = @ptrCast(memory);

        const ctx = machine.heap.create(Frame, &.no_frame);

        const state = machine.compute(ctx, null, ptr);

        switch (state) {
            .ret => |r| {
                try testing.expectEqualDeep(r.frame.*, .no_frame);
                try testing.expectEqualDeep(r.value, expected);
            },
            else => {
                @panic("HOW");
            },
        }
    }
};
