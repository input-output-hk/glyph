const Heap = @import("Heap.zig");

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

pub const BigInt = extern struct {
    sign: u32,
    length: u32,
    words: [*]const u32,

    pub fn compareMagnitude(x: *const BigInt, y: *const BigInt) struct { bool, *const BigInt, *const BigInt } {
        if (x.length > y.length) {
            return .{ false, x, y };
        }

        if (y.length > x.length) {
            return .{ false, y, x };
        }

        var i: u32 = x.length - 1;
        while (true) : (i -= 1) {
            if (x.words[i] > y.words[i]) {
                return .{ false, x, y };
            }

            if (y.words[i] > x.words[i]) {
                return .{ false, y, x };
            }

            if (i == 0) {
                break;
            }
        }

        return .{ true, x, y };
    }

    /// Allocate a Constant.integer in the bump‑heap from an `expr.BigInt` that
    /// already lives elsewhere in memory.  The layout exactly matches what
    /// `Constant.bigInt()` expects.
    ///
    /// returns: pointer to the freshly‑allocated `Constant`
    pub fn createConstant(
        self: BigInt,
        heap: *Heap,
    ) *Constant {
        const total_words: u32 = self.length + 4; // len of type | tag | sign | length | words…
        var buf = heap.createArray(u32, total_words);

        buf[0] = 1;
        buf[1] = @intFromEnum(ConstantType.integer);
        buf[2] = self.sign;
        buf[3] = self.length;

        var i: u32 = 0;
        while (i < self.length) : (i += 1) {
            buf[i + 4] = self.words[i];
        }

        return @ptrCast(buf);
    }
};

pub const Bytes = extern struct {
    length: u32,
    bytes: [*]const u32,

    pub fn compareBytes(x: *const Bytes, y: *const Bytes) struct { bool, *const Bytes, *const Bytes } {
        const lenCompare: struct { greater: *const Bytes, less: *const Bytes } = if (x.length >= y.length) blk: {
            break :blk .{ .greater = x, .less = y };
        } else blk: {
            break :blk .{ .greater = y, .less = x };
        };

        var i: u32 = 0;
        while (i < lenCompare.greater.length) : (i += 1) {
            if (i >= lenCompare.less.length) {
                return .{ false, lenCompare.greater, lenCompare.less };
            }

            if (lenCompare.greater.bytes[i] > lenCompare.less.bytes[i]) {
                return .{ false, lenCompare.greater, lenCompare.less };
            }

            if (lenCompare.less.bytes[i] > lenCompare.greater.bytes[i]) {
                return .{ false, lenCompare.less, lenCompare.greater };
            }
        }

        return .{ true, x, y };
    }

    pub fn createConstant(
        self: Bytes,
        heap: *Heap,
    ) *Constant {
        const total_words: u32 = self.length + 3; // len of type | tag | sign | length | words…
        var buf = heap.createArray(u32, total_words);

        buf[0] = 1;
        buf[1] = @intFromEnum(ConstantType.bytes);
        buf[2] = self.length;

        var i: u32 = 0;
        while (i < self.length) : (i += 1) {
            buf[i + 3] = self.bytes[i];
        }

        return @ptrCast(buf);
    }
};

pub const Constant = extern struct {
    length: u32,

    const Self = @This();

    pub fn constType(self: *const Self) *const ConstantType {
        const cType: *const ConstantType = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));

        return cType;
    }

    pub fn canListHoldType(self: *const Self, listInnerType: *const ConstantType, len: u32) bool {
        const selfTypes: [*]const ConstantType = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));
        const otherTypes: [*]const ConstantType = @ptrCast(listInnerType);

        if (self.length != len) {
            return false;
        }

        var i: u32 = 0;
        while (i < self.length) : (i += 1) {
            if (selfTypes[i] != otherTypes[i]) {
                return false;
            }
        }
        return true;
    }

    pub fn bigInt(self: *const Self) BigInt {
        const offset: *const u32 = @ptrFromInt(@intFromPtr(self) + self.length * @sizeOf(u32));

        const sign: *const u32 = @ptrFromInt(@intFromPtr(offset) + @sizeOf(u32));
        const length: *const u32 = @ptrFromInt(@intFromPtr(offset) + @sizeOf(u32) * 2);

        const words: [*]const u32 = @ptrFromInt(@intFromPtr(offset) + @sizeOf(u32) * 3);

        return BigInt{
            .sign = sign.*,
            .length = length.*,
            .words = words,
        };
    }

    pub fn innerBytes(self: *const Self) Bytes {
        const offset: *const u32 = @ptrFromInt(@intFromPtr(self) + self.length * @sizeOf(u32));

        const length: *const u32 = @ptrFromInt(@intFromPtr(offset) + @sizeOf(u32));

        const bytes: [*]const u32 = @ptrFromInt(@intFromPtr(offset) + @sizeOf(u32) * 2);

        return Bytes{
            .length = length.*,
            .bytes = bytes,
        };
    }

    pub fn bln(self: *const Self) bool {
        const offset: *const u32 = @ptrFromInt(@intFromPtr(self) + self.length * @sizeOf(u32));

        const b: *const u32 = @ptrFromInt(@intFromPtr(offset) + @sizeOf(u32));

        return b.* == 1;
    }
};

pub const ConstantType = enum(u32) {
    integer,
    bytes,
    string,
    unit,
    boolean,
    list,
    pair,
    data,

    pub fn listData() *const ConstantType {
        return @ptrCast(
            [2]ConstantType{
                ConstantType.list,
                ConstantType.data,
            },
        );
    }
};
