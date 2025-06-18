const Heap = @import("Heap.zig");
const expr = @import("./expr.zig");
const std = @import("std");

const Term = expr.Term;
const TermList = expr.TermList;
const Constant = expr.Constant;
const DefaultFunction = expr.DefaultFunction;

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
        env: *Env,
        body: *const Term,
    },
    lambda: struct {
        env: *Env,
        body: *const Term,
    },
    builtin: Builtin,
    constr: struct { tag: u32, fields: ValueList },
};

const Frame = union(enum) {
    frameAwaitArg: struct { function: *Value, ctx: *Frame },
    frameAwaitFunTerm: struct { env: *Env, argument: *const Term, ctx: *Frame },
    frameAwaitFunValue: struct { argument: *Value, ctx: *Frame },
    frameForce: struct { ctx: *Frame },
    frameConstr: struct {
        env: *Env,
        tag: u32,
        fields: TermList,
        resolved_fields: ValueList,
        ctx: *Frame,
    },
    frameCases: struct { env: *Env, branches: TermList, ctx: *Frame },
    noFrame: void,
};

pub const Env = struct {
    heap: *Heap,
    env_list: ?*EnvList,

    const Self = @This();
    const EnvList = struct { value: *Value, next: ?*EnvList };

    pub fn init(heap: *Heap) Self {
        return .{ .heap = heap, .env_list = null };
    }
    fn clone(self: *Self) *Self {
        const e = self.heap.allocType(Env);
        e.* = .{ .heap = self.heap, .env_list = self.env_list };
        return e;
    }
    pub fn extend(self: *Self, v: *Value) *Self {
        const node = self.heap.allocType(EnvList);
        node.* = .{ .value = v, .next = self.env_list };
        const e = self.heap.allocType(Env);
        e.* = .{ .heap = self.heap, .env_list = node };
        return e;
    }

    pub fn mkValue(self: *Self, v: Value) *Value {
        const out = self.heap.allocType(Value);
        out.* = v;
        return out;
    }
    fn mkConst(self: *Self, c: *Constant) *Value {
        return self.mkValue(.{ .constant = c });
    }
    fn mkDelay(self: *Self, b: *const Term) *Value {
        return self.mkValue(.{ .delay = .{ .env = self.clone(), .body = b } });
    }
    pub fn createDelay(self: *Self, b: *const Term) *Value {
        return self.mkDelay(b);
    }
    fn mkLam(self: *Self, b: *const Term) *Value {
        return self.mkValue(.{ .lambda = .{ .env = self.clone(), .body = b } });
    }

    fn mkBuilt(self: *Self, f: DefaultFunction) *Value {
        return self.mkValue(.{ .builtin = .{
            .fun = f,
            .force_count = getDefaultForceCount(f),
            .args_count = getDefaultArgCount(f),
            .args = ValueList{ .length = 0, .list = undefined },
        } });
    }
    fn mkConstr(self: *Self, tag: u32, vls: ValueList) *Value {
        return self.mkValue(.{ .constr = .{ .tag = tag, .fields = vls } });
    }

    fn lookupVar(self: *Self, idx: u32) *Value {
        if (idx == 0) @panic("invalid de Bruijn index 0");
        var cur = self.env_list;
        var i: u32 = 1;
        while (cur) |n| : (i += 1) {
            if (i == idx) {
                return n.value;
            }
            cur = n.next;
        }
        @panic("open term during evaluation");
    }

    pub fn run(self: *Self, t: *const Term) *Value {
        const root = self.mkFrame(.{ .noFrame = {} });
        return self.compute(root, t);
    }

    fn mkFrame(self: *Self, fr: Frame) *Frame {
        const slot = self.heap.allocType(Frame);
        slot.* = fr;
        return slot;
    }

    fn compute(self: *Self, ctx: *Frame, t: *const Term) *Value {
        switch (t.*) {
            .tvar => return self.ret(ctx, self.lookupVar(t.debruijnIndex())),
            .constant => return self.ret(ctx, self.mkConst(t.constantValue())),
            .lambda => return self.ret(ctx, self.mkLam(t.termBody())),
            .delay => return self.ret(ctx, self.mkDelay(t.termBody())),

            .force => {
                const nctx = self.mkFrame(.{ .frameForce = .{ .ctx = ctx } });
                return self.compute(nctx, t.termBody());
            },

            .apply => {
                const p = t.appliedTerms();
                const nctx = self.mkFrame(.{ .frameAwaitFunTerm = .{
                    .env = self,
                    .argument = p.argument,
                    .ctx = ctx,
                } });
                return self.compute(nctx, p.function);
            },

            .builtin => return self.ret(ctx, self.mkBuilt(t.defaultFunction())),

            .constr => {
                const c = t.constrValues();
                if (c.fields.length == 0)
                    return self.ret(ctx, self.mkConstr(c.tag, ValueList{ .length = 0, .list = undefined }));

                const rest = TermList{
                    .length = c.fields.length - 1,
                    .list = c.fields.list + 1,
                };
                const fr = Frame{ .frameConstr = .{
                    .env = self,
                    .tag = c.tag,
                    .fields = rest,
                    .resolved_fields = ValueList{ .length = 0, .list = undefined },
                    .ctx = ctx,
                } };
                const nctx = self.mkFrame(fr);
                return self.compute(nctx, c.fields.list[0]);
            },

            .case => {
                const cs = t.caseValues();
                const fr = Frame{ .frameCases = .{
                    .env = self,
                    .branches = cs.branches,
                    .ctx = ctx,
                } };
                const nctx = self.mkFrame(fr);
                return self.compute(nctx, cs.constr);
            },

            .terror => @panic("evaluation failure"),
        }
    }

    fn ret(self: *Self, ctx: *Frame, v: *Value) *Value {
        switch (ctx.*) {
            .noFrame => return v,

            .frameForce => |f| return self.forceEval(f.ctx, v),

            .frameAwaitArg => |f| return self.applyEval(f.ctx, f.function, v),
            .frameAwaitFunValue => |f| return self.applyEval(f.ctx, v, f.argument),

            .frameAwaitFunTerm => |f| {
                const nctx = self.mkFrame(.{ .frameAwaitArg = .{ .function = v, .ctx = f.ctx } });
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
                    return self.ret(f.ctx, self.mkConstr(f.tag, done));
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
                const nctx = self.mkFrame(fr2);
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
                            nctx = self.mkFrame(.{ .frameAwaitFunValue = .{
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
        const new_builtin = self.mkValue(.{ .builtin = .{
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
        return self.mkConst(c);
    }
};

fn getDefaultForceCount(f: DefaultFunction) u8 {
    return switch (f) {
        .add_integer => 0,
        .subtract_integer => 0,
    };
}
fn getDefaultArgCount(f: DefaultFunction) u8 {
    return switch (f) {
        .add_integer => 2,
        .subtract_integer => 2,
    };
}
