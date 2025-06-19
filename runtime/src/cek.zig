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
    frame_await_arg: struct { function: *const Value },
    frame_await_fun_term: struct { env: ?*Env, argument: *const Term },
    frame_await_fun_value: struct { argument: *const Value },
    frame_force,
    frame_constr: struct {
        env: ?*Env,
        tag: u32,
        fields: TermList,
        resolved_fields: ValueList,
    },
    frame_cases: struct {
        env: ?*Env,
        branches: TermList,
    },
};

const Frames = struct {
    frame_ptr: [*]u8,

    const Self = @This();

    pub fn createTestFrames(arena: *std.heap.ArenaAllocator) !Frames {
        const frameMemory = try arena.allocator().alloc(Frame, 1000);
        const framePointer: [*]u8 = @ptrCast(frameMemory);

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

const Builtin = struct {
    fun: DefaultFunction,
    force_count: u8,
    arity: u8,
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
    constr: struct { tag: u32, values: ValueList },
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
            .args = ValueList{
                .length = 0,
                .list = undefined,
            },
        },
    });
}

pub fn forceBuiltin(heap: *Heap, b: *const Builtin) *Value {
    return heap.create(Value, &.{
        .builtin = .{
            .fun = b.fun,
            .force_count = b.force_count - 1,
            .arity = b.arity,
            .args = ValueList{
                .length = 0,
                .list = undefined,
            },
        },
    });
}

pub fn createConstr(heap: *Heap, tag: u32, vls: ValueList) *Value {
    return heap.create(
        Value,
        &Value{
            .constr = .{ .tag = tag, .values = vls },
        },
    );
}

pub const State = union(enum) {
    compute: struct {
        env: ?*Env,
        term: *const Term,
    },
    ret: struct {
        value: *const Value,
    },
    done: *Term,
};

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

    pub fn run(self: *Self, t: *const Term) void {
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
            // constant
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
                                ValueList{
                                    .length = 0,
                                    .list = undefined,
                                },
                            ),
                        },
                    };
                }

                // TODO: TEST THIS
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
                            .resolved_fields = ValueList{ .length = 0, .list = undefined },
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
                    .frame_cases = .{
                        .env = env,
                        .branches = cs.branches,
                    },
                });
                return self.compute(env, cs.constr);
            },

            else => @panic("TODO"),
        }
    }

    fn ret(self: *Self, v: *const Value) State {
        const frame = self.frames.popFrame();

        switch (frame) {
            .no_frame => @panic("TODO"),

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

            // .frameConstr => |f| {
            //     const new_len = f.resolved_fields.length + 1;
            //     const dst = self.heap.allocArray(*Value, new_len);
            //     dst[0] = v;
            //     var i: u32 = 0;
            //     while (i < f.resolved_fields.length) : (i += 1)
            //         dst[i + 1] = f.resolved_fields.list[i];

            //     const done = ValueList{ .length = new_len, .list = dst };

            //     if (f.fields.length == 0) {
            //         return self.ret(f.ctx, self.makeConstr(f.tag, done));
            //     }

            //     const next = f.fields.list[0];
            //     const rest = TermList{
            //         .length = f.fields.length - 1,
            //         .list = f.fields.list + 1,
            //     };
            //     const fr2 = Frame{ .frameConstr = .{
            //         .env = f.env,
            //         .tag = f.tag,
            //         .fields = rest,
            //         .resolved_fields = done,
            //         .ctx = f.ctx,
            //     } };
            //     const nctx = self.makeFrame(fr2);
            //     return self.compute(nctx, next);
            // },

            // .frameCases => |f| {
            //     switch (v.*) {
            //         .constr => |c| {
            //             if (c.tag >= f.branches.length)
            //                 @panic("constructor tag out of range");
            //             // push constructor fields in reverse order
            //             var nctx = f.ctx;
            //             var idx: i32 = @as(i32, @intCast(c.fields.length)) - 1;
            //             while (idx >= 0) : (idx -= 1)
            //                 nctx = self.makeFrame(.{ .frameAwaitFunValue = .{
            //                     .argument = c.fields.list[@as(usize, @intCast(idx))],
            //                     .ctx = nctx,
            //                 } });
            //             return self.compute(nctx, f.branches.list[c.tag]);
            //         },
            //         else => @panic("case on non-constructor"),
            //     }
            // },
            else => @panic("TODO"),
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

                @panic("TODO");
                // const res = self.evalBuiltin(&b, argVal);
                // return self.ret(ctx, res);
            },

            else => @panic("apply on non-callable"),
        }
    }

    // fn evalBuiltin(self: *Self, b: *const Builtin, next: *Value) *Value {
    //     const new_len = b.args.length + 1;
    //     const dst = self.heap.allocArray(*Value, new_len);
    //     var i: u32 = 0;
    //     while (i < b.args.length) : (i += 1) dst[i] = b.args.list[i];
    //     dst[b.args.length] = next;

    //     if (b.args_count == 1) {
    //         return callBuiltin(self, b.fun, ValueList{ .length = new_len, .list = dst });
    //     }

    //     // Create a new builtin value with updated args
    //     const new_builtin = self.heap.create(.{ .builtin = .{
    //         .fun = b.fun,
    //         .force_count = b.force_count,
    //         .args_count = b.args_count - 1,
    //         .args = ValueList{ .length = new_len, .list = dst },
    //     } });
    //     return new_builtin;
    // }

    // fn callBuiltin(
    //     self: *Self,
    //     fun: DefaultFunction,
    //     args: ValueList,
    // ) *Value {
    //     switch (fun) {
    //         .add_integer => {
    //             if (args.length != 2) @panic("arity error for add_integer");
    //             const a = self.unwrapInt(args.list[0]);
    //             const b = self.unwrapInt(args.list[1]);
    //             return self.wrapInt(a + b);
    //         },
    //         .subtract_integer => {
    //             if (args.length != 2) @panic("arity error for subtract_integer");
    //             const a = self.unwrapInt(args.list[0]);
    //             const b = self.unwrapInt(args.list[1]);
    //             return self.wrapInt(a - b);
    //         },
    //     }
    // }

    // fn unwrapInt(_: *Self, v: *Value) i128 {
    //     if (v.* != .constant) @panic("expected integer constant");
    //     return v.constant.*.integer;
    // }
    // fn wrapInt(self: *Self, i: i128) *Value {
    //     const c = self.heap.allocType(Constant);
    //     c.* = Constant{ .integer = i };
    //     return self.makeConst(c);
    // }

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
};

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
