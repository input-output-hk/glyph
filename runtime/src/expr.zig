pub const Term = enum(u32) {
    tvar,
    delay,
    lambda,
    apply,
    constant,
    force,
    terror,
    builtin,
    constr,
    case,

    // For Var
    pub fn debruijnIndex(ptr: *const Term) u32 {
        const dbIndex: *u32 = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        return dbIndex.*;
    }

    // For lambda, delay, force
    pub fn termBody(ptr: *const Term) *const Term {
        const nextTerm: *const Term = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        return nextTerm;
    }

    // For Apply
    pub fn appliedTerms(ptr: *const Term) Apply {
        const argTerm: **const Term = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        const funcTerm: *const Term = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32) * 2);

        return .{
            .function = funcTerm,
            .argument = argTerm.*,
        };
    }

    // For Builtin
    pub fn defaultFunction(ptr: *const Term) DefaultFunction {
        const func: *const DefaultFunction = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        return func.*;
    }

    // For constr
    pub fn constrValues(ptr: *const Term) Constr {
        const tag: *const u32 = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        const field_length: *const u32 = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32) * 2);

        const fields: [*]*const Term = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32) * 3);

        return .{
            .tag = tag.*,
            .fields = TermList{
                .length = field_length.*,
                .list = fields,
            },
        };
    }

    // For case
    pub fn caseValues(ptr: *const Term) Case {
        const constr: **const Term = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        const branch_length: *const u32 = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32) * 2);

        const branches: [*]*const u32 = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32) * 3);

        return .{
            .constr = constr.*,
            .branches = TermList{
                .length = branch_length.*,
                .list = branches,
            },
        };
    }

    // For constant
    pub fn constantValue(ptr: *const Term) *Constant {
        const value: *Constant = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));
        return value;
    }
};

pub const Apply = struct { function: *const Term, argument: *const Term };
pub const Constr = struct { tag: u32, fields: TermList };
pub const Case = struct { constr: *const Term, branches: TermList };

pub const TermList = extern struct { length: u32, list: [*]*const Term };

pub const DefaultFunction = enum(u32) {
    add_integer,
    subtract_integer,

    pub fn forceCount(f: DefaultFunction) u8 {
        return switch (f) {
            .add_integer => 0,
            .subtract_integer => 0,
        };
    }

    pub fn arity(f: DefaultFunction) u8 {
        return switch (f) {
            .add_integer => 2,
            .subtract_integer => 2,
        };
    }
};

pub const Constant = struct {
    integer: i128,
};
