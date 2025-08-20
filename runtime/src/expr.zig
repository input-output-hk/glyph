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
    pub fn constantValue(ptr: *const Term, heap: *Heap) *Constant {
        const constant = heap.createArray(u32, 3);
        const consValue: [*]u32 = @ptrFromInt(@intFromPtr(ptr) + @sizeOf(u32));

        constant[0] = consValue[0];
        constant[1] = consValue[1];
        constant[2] = @intFromPtr(consValue + 2);

        return @ptrCast(constant);
    }
};

pub const Apply = struct { function: *const Term, argument: *const Term };
pub const Constr = struct { tag: u32, fields: TermList };
pub const Case = struct { constr: *const Term, branches: TermList };

pub const TermList = extern struct { length: u32, list: [*]*const Term };

pub const DefaultFunction = enum(u32) {
    add_integer,
    subtract_integer,
    multiply_integer,
    divide_integer,
    quotient_integer,
    remainder_integer,
    mod_integer,
    equals_integer,
    less_than_integer,
    less_than_equals_integer,
    append_byte_string,
    cons_byte_string,
    slice_byte_string,
    length_of_byte_string,
    index_byte_string,
    equals_byte_string,
    less_than_byte_string,
    less_than_equals_byte_string,
    sha2_256,
    sha3_256,
    blake2b_256,
    verify_ed25519_signature,
    append_string,
    equals_string,
    encode_utf8,
    decode_utf8,
    if_then_else,
    choose_unit,
    trace,
    fst_pair,
    snd_pair,
    choose_list,
    mk_cons,
    head_list,
    tail_list,
    null_list,
    choose_data,
    constr_data,
    map_data,
    list_data,
    i_data,
    b_data,
    un_constr_data,
    un_map_data,
    un_list_data,
    un_i_data,
    un_b_data,
    equals_data,
    mk_pair_data,
    mk_nil_data,
    mk_nil_pair_data,
    serialise_data,
    verify_ecdsa_secp256k1_signature,
    verify_schnorr_secp256k1_signature,
    bls12_381_g1_add,
    bls12_381_g1_neg,
    bls12_381_g1_scalar_mul,
    bls12_381_g1_equal,
    bls12_381_g1_compress,
    bls12_381_g1_uncompress,
    bls12_381_g1_hash_to_group,
    bls12_381_g2_add,
    bls12_381_g2_neg,
    bls12_381_g2_scalar_mul,
    bls12_381_g2_equal,
    bls12_381_g2_compress,
    bls12_381_g2_uncompress,
    bls12_381_g2_hash_to_group,
    bls12_381_miller_loop,
    bls12_381_mul_ml_result,
    bls12_381_final_verify,
    keccak_256,
    blake2b_224,
    integer_to_byte_string,
    byte_string_to_integer,
    and_byte_string,
    or_byte_string,
    xor_byte_string,
    complement_byte_string,
    read_bit,
    write_bits,
    replicate_byte,
    shift_byte_string,
    rotate_byte_string,
    count_set_bits,
    find_first_set_bit,
    ripemd_160,

    pub fn forceCount(f: DefaultFunction) u8 {
        return switch (f) {
            .add_integer => 0,
            .subtract_integer => 0,
            .multiply_integer => 0,
            .divide_integer => 0,
            .quotient_integer => 0,
            .remainder_integer => 0,
            .mod_integer => 0,
            .equals_integer => 0,
            .less_than_integer => 0,
            .less_than_equals_integer => 0,
            .append_byte_string => 0,
            .cons_byte_string => 0,
            .slice_byte_string => 0,
            .length_of_byte_string => 0,
            .index_byte_string => 0,
            .equals_byte_string => 0,
            .less_than_byte_string => 0,
            .less_than_equals_byte_string => 0,
            .sha2_256 => 0,
            .sha3_256 => 0,
            .blake2b_224 => 0,
            .blake2b_256 => 0,
            .keccak_256 => 0,
            .verify_ed25519_signature => 0,
            .verify_ecdsa_secp256k1_signature => 0,
            .verify_schnorr_secp256k1_signature => 0,
            .append_string => 0,
            .equals_string => 0,
            .encode_utf8 => 0,
            .decode_utf8 => 0,
            .if_then_else => 1,
            .choose_unit => 1,
            .trace => 1,
            .fst_pair => 2,
            .snd_pair => 2,
            .choose_list => 2,
            .mk_cons => 1,
            .head_list => 1,
            .tail_list => 1,
            .null_list => 1,
            .choose_data => 1,
            .constr_data => 0,
            .map_data => 0,
            .list_data => 0,
            .i_data => 0,
            .b_data => 0,
            .un_constr_data => 0,
            .un_map_data => 0,
            .un_list_data => 0,
            .un_i_data => 0,
            .un_b_data => 0,
            .equals_data => 0,
            .serialise_data => 0,
            .mk_pair_data => 0,
            .mk_nil_data => 0,
            .mk_nil_pair_data => 0,
            .bls12_381_g1_add => 0,
            .bls12_381_g1_neg => 0,
            .bls12_381_g1_scalar_mul => 0,
            .bls12_381_g1_equal => 0,
            .bls12_381_g1_compress => 0,
            .bls12_381_g1_uncompress => 0,
            .bls12_381_g1_hash_to_group => 0,
            .bls12_381_g2_add => 0,
            .bls12_381_g2_neg => 0,
            .bls12_381_g2_scalar_mul => 0,
            .bls12_381_g2_equal => 0,
            .bls12_381_g2_compress => 0,
            .bls12_381_g2_uncompress => 0,
            .bls12_381_g2_hash_to_group => 0,
            .bls12_381_miller_loop => 0,
            .bls12_381_mul_ml_result => 0,
            .bls12_381_final_verify => 0,
            .integer_to_byte_string => 0,
            .byte_string_to_integer => 0,
            .and_byte_string => 0,
            .or_byte_string => 0,
            .xor_byte_string => 0,
            .complement_byte_string => 0,
            .read_bit => 0,
            .write_bits => 0,
            .replicate_byte => 0,
            .shift_byte_string => 0,
            .rotate_byte_string => 0,
            .count_set_bits => 0,
            .find_first_set_bit => 0,
            .ripemd_160 => 0,
        };
    }

    pub fn arity(f: DefaultFunction) u8 {
        return switch (f) {
            .add_integer => 2,
            .subtract_integer => 2,
            .multiply_integer => 2,
            .divide_integer => 2,
            .quotient_integer => 2,
            .remainder_integer => 2,
            .mod_integer => 2,
            .equals_integer => 2,
            .less_than_integer => 2,
            .less_than_equals_integer => 2,
            .append_byte_string => 2,
            .cons_byte_string => 2,
            .slice_byte_string => 3,
            .length_of_byte_string => 1,
            .index_byte_string => 2,
            .equals_byte_string => 2,
            .less_than_byte_string => 2,
            .less_than_equals_byte_string => 2,
            .sha2_256 => 1,
            .sha3_256 => 1,
            .blake2b_224 => 1,
            .blake2b_256 => 1,
            .keccak_256 => 1,
            .verify_ed25519_signature => 3,
            .verify_ecdsa_secp256k1_signature => 3,
            .verify_schnorr_secp256k1_signature => 3,
            .append_string => 2,
            .equals_string => 2,
            .encode_utf8 => 1,
            .decode_utf8 => 1,
            .if_then_else => 3,
            .choose_unit => 2,
            .trace => 2,
            .fst_pair => 1,
            .snd_pair => 1,
            .choose_list => 3,
            .mk_cons => 2,
            .head_list => 1,
            .tail_list => 1,
            .null_list => 1,
            .choose_data => 6,
            .constr_data => 2,
            .map_data => 1,
            .list_data => 1,
            .i_data => 1,
            .b_data => 1,
            .un_constr_data => 1,
            .un_map_data => 1,
            .un_list_data => 1,
            .un_i_data => 1,
            .un_b_data => 1,
            .equals_data => 2,
            .serialise_data => 1,
            .mk_pair_data => 2,
            .mk_nil_data => 1,
            .mk_nil_pair_data => 1,
            .bls12_381_g1_add => 2,
            .bls12_381_g1_neg => 1,
            .bls12_381_g1_scalar_mul => 2,
            .bls12_381_g1_equal => 2,
            .bls12_381_g1_compress => 1,
            .bls12_381_g1_uncompress => 1,
            .bls12_381_g1_hash_to_group => 2,
            .bls12_381_g2_add => 2,
            .bls12_381_g2_neg => 1,
            .bls12_381_g2_scalar_mul => 2,
            .bls12_381_g2_equal => 2,
            .bls12_381_g2_compress => 1,
            .bls12_381_g2_uncompress => 1,
            .bls12_381_g2_hash_to_group => 2,
            .bls12_381_miller_loop => 2,
            .bls12_381_mul_ml_result => 2,
            .bls12_381_final_verify => 2,
            .integer_to_byte_string => 3,
            .byte_string_to_integer => 2,
            .and_byte_string => 3,
            .or_byte_string => 3,
            .xor_byte_string => 3,
            .complement_byte_string => 1,
            .read_bit => 2,
            .write_bits => 3,
            .replicate_byte => 2,
            .shift_byte_string => 2,
            .rotate_byte_string => 2,
            .count_set_bits => 1,
            .find_first_set_bit => 1,
            .ripemd_160 => 1,
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
        self: *const BigInt,
        _: *const ConstantType,
        heap: *Heap,
    ) *Constant {
        const buf = heap.createArray(u32, self.length + 2);
        buf[0] = self.sign;
        buf[1] = self.length;
        var i: u32 = 0;
        while (i < self.length) : (i += 1) {
            buf[i + 2] = self.words[i];
        }

        const con = Constant{
            .length = 1,
            .type_list = @ptrCast(ConstantType.integerType()),
            .value = @intFromPtr(buf),
        };

        return heap.create(Constant, &con);
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
        self: *const Bytes,
        _: *const ConstantType,
        heap: *Heap,
    ) *Constant {
        const buf = heap.createArray(u32, self.length + 1);
        buf[0] = self.length;
        var i: u32 = 0;
        while (i < self.length) : (i += 1) {
            buf[i + 1] = self.bytes[i];
        }

        const con = Constant{
            .length = 1,
            .type_list = @ptrCast(ConstantType.bytesType()),
            .value = @intFromPtr(buf),
        };

        return heap.create(Constant, &con);
    }
};

pub const String = extern struct {
    length: u32,
    bytes: [*]const u32,

    pub fn createConstant(
        self: *const String,
        _: *const ConstantType,
        heap: *Heap,
    ) *Constant {
        const buf = heap.createArray(u32, self.length + 1);
        buf[0] = self.length;
        var i: u32 = 0;
        while (i < self.length) : (i += 1) {
            buf[i + 1] = self.bytes[i];
        }

        const con = Constant{
            .length = 1,
            .type_list = @ptrCast(ConstantType.stringType()),
            .value = @intFromPtr(buf),
        };

        return heap.create(Constant, &con);
    }

    pub fn equals(x: *const String, y: *const String) bool {
        const lenCompare: struct { greater: *const String, less: *const String } = if (x.length >= y.length) blk: {
            break :blk .{ .greater = x, .less = y };
        } else blk: {
            break :blk .{ .greater = y, .less = x };
        };

        var i: u32 = 0;
        while (i < lenCompare.greater.length) : (i += 1) {
            if (i >= lenCompare.less.length) {
                return false;
            }

            if (lenCompare.greater.bytes[i] > lenCompare.less.bytes[i]) {
                return false;
            }

            if (lenCompare.less.bytes[i] > lenCompare.greater.bytes[i]) {
                return false;
            }
        }

        return true;
    }
};

pub const Bool = extern struct {
    val: u32,

    pub fn createConstant(
        self: *const Bool,
        _: *const ConstantType,
        heap: *Heap,
    ) *Constant {
        const con = Constant{
            .length = 1,
            .type_list = @ptrCast(ConstantType.booleanType()),
            .value = @intFromPtr(&self.val),
        };

        return heap.create(Constant, &con);
    }
};

pub const ListNode = extern struct {
    value: u32,
    next: ?*ListNode,
};

pub const List = struct {
    type_length: u32,
    inner_type: [*]const ConstantType,
    length: u32,
    items: ?*ListNode,

    pub fn createConstant(
        self: *const List,
        types: *const ConstantType,
        heap: *Heap,
    ) *Constant {
        const con = Constant{
            .length = self.type_length + 1,
            .type_list = @ptrCast(types),
            .value = @intFromPtr(self) + @sizeOf(u32) * 2,
        };

        return heap.create(Constant, &con);
    }
};

pub const G1Element = extern struct {
    bytes: [*]const u8,

    pub fn createConstant(
        self: G1Element,
        types: *const ConstantTypeList,
        heap: *Heap,
    ) *Constant {
        const total_words: u32 = 13;
        var buf = heap.createArray(u32, total_words);

        buf[0] = @intFromPtr(types);

        // var i: u32 = 0;
        // while (i < 12) : (i += 1) {
        //     buf[i + 1] = self.bytes[i];
        // }
        var i: u32 = 0;
        while (i < 24) : (i += 1) {
            const byte_offset = i * 4;
            buf[i + 1] = (@as(u32, self.bytes[byte_offset])) |
                (@as(u32, self.bytes[byte_offset + 1]) << 8) |
                (@as(u32, self.bytes[byte_offset + 2]) << 16) |
                (@as(u32, self.bytes[byte_offset + 3]) << 24);
        }

        return @ptrCast(buf);
    }
};

pub const G2Element = extern struct {
    bytes: [*]const u8,

    pub fn createConstant(
        self: G2Element,
        types: *const ConstantTypeList,
        heap: *Heap,
    ) *Constant {
        const total_words: u32 = 25;
        var buf = heap.createArray(u32, total_words);

        buf[0] = @intFromPtr(types);

        // var i: u32 = 0;
        // while (i < 24) : (i += 1) {
        //     buf[i + 1] = self.bytes[i];
        // }
        var i: u32 = 0;
        while (i < 24) : (i += 1) {
            const byte_offset = i * 4;
            buf[i + 1] = (@as(u32, self.bytes[byte_offset])) |
                (@as(u32, self.bytes[byte_offset + 1]) << 8) |
                (@as(u32, self.bytes[byte_offset + 2]) << 16) |
                (@as(u32, self.bytes[byte_offset + 3]) << 24);
        }

        return @ptrCast(buf);
    }
};

pub const MlResult = extern struct {
    length: u32,
    bytes: [*]const u8,

    pub fn createConstant(
        self: MlResult,
        types: *const ConstantTypeList,
        heap: *Heap,
    ) *Constant {
        const total_words: u32 = self.length + 2;
        var buf = heap.createArray(u32, total_words);

        buf[0] = @intFromPtr(types);
        buf[1] = self.length;

        // var i: u32 = 0;
        // while (i < self.length) : (i += 1) {
        //     buf[i + 2] = self.bytes[i];
        // }
        var i: u32 = 0;
        while (i < 24) : (i += 1) {
            const byte_offset = i * 4;
            buf[i + 1] = (@as(u32, self.bytes[byte_offset])) |
                (@as(u32, self.bytes[byte_offset + 1]) << 8) |
                (@as(u32, self.bytes[byte_offset + 2]) << 16) |
                (@as(u32, self.bytes[byte_offset + 3]) << 24);
        }

        return @ptrCast(buf);
    }
};

pub const Constant = extern struct {
    type_list: *ConstantTypeList,

    const Self = @This();

    pub fn rawValue(self: *const Self) u32 {
        const value = @intFromPtr(self) + @sizeOf(u32);

        return value;
    }

    pub fn bigInt(self: *const Self) BigInt {
        const sign: *const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));
        const length: *const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32) * 2);

        const words: [*]const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32) * 3);

        return BigInt{
            .sign = sign.*,
            .length = length.*,
            .words = words,
        };
    }

    pub fn innerBytes(self: *const Self) Bytes {
        const length: *const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));

        const bytes: [*]const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32) * 2);

        return Bytes{
            .length = length.*,
            .bytes = bytes,
        };
    }

    pub fn string(self: *const Self) String {
        const length: *const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));

        const bytes: [*]const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32) * 2);

        return String{
            .length = length.*,
            .bytes = bytes,
        };
    }

    pub fn bln(self: *const Self) bool {
        const b: *const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));

        return b.* == 1;
    }

    pub fn g1Element(self: *const Self) G1Element {
        const bytes: [*]const u8 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));

        return G1Element{
            .bytes = bytes,
        };
    }

    pub fn g2Element(self: *const Self) G2Element {
        const bytes: [*]const u8 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));

        return G2Element{
            .bytes = bytes,
        };
    }

    pub fn mlResult(self: *const Self) MlResult {
        const length: *const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));

        const bytes: [*]const u8 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32) * 2);

        return MlResult{
            .length = length.*,
            .bytes = bytes,
        };
    }

    pub fn list(self: *const Self) List {
        const length: *const u32 = @ptrFromInt(@intFromPtr(self) + @sizeOf(u32));

        const items: ?*ListNode = if (length.* > 0) blk: {
            break :blk @ptrFromInt(@intFromPtr(self) + @sizeOf(u32) * 2);
        } else blk: {
            break :blk null;
        };

        return List{
            .type_length = self.type_list.length - 1,
            .inner_type = self.type_list.innerListType(),
            .length = length.*,
            .items = items,
        };
    }
};

pub const ConstantTypeList = extern struct {
    length: u32,
    type_list: [*]const ConstantType,
    value: u32,

    const Self = @This();

    pub fn constType(self: *const Self) *const ConstantType {
        return @ptrCast(self.type_list);
    }

    pub fn innerListType(self: *const Self) [*]const ConstantType {
        return @ptrCast(self.type_list + 1);
    }

    pub fn matchingTypes(self: *const Self, listInnerType: [*]const ConstantType, len: u32) bool {
        if (self.length != len) {
            return false;
        }

        const selfTypes: [*]const ConstantType = @ptrCast(self.constType());
        const otherTypes: [*]const ConstantType = @ptrCast(listInnerType);

        var i: u32 = 0;
        while (i < len) : (i += 1) {
            if (selfTypes[i] != otherTypes[i]) {
                return false;
            }
        }
        return true;
    }

    pub fn rawValue(self: *const Self) u32 {
        return self.value;
    }

    pub fn bigInt(self: *const Self) BigInt {
        const ptr = self.value;
        const sign: *const u32 = @ptrFromInt(ptr);
        const length: *const u32 = @ptrFromInt(ptr + @sizeOf(u32));

        const words: [*]const u32 = @ptrFromInt(ptr + @sizeOf(u32) * 2);

        return BigInt{
            .sign = sign.*,
            .length = length.*,
            .words = words,
        };
    }

    pub fn innerBytes(self: *const Self) Bytes {
        const ptr = self.value;
        const length: *const u32 = @ptrFromInt(ptr);

        const bytes: [*]const u32 = @ptrFromInt(ptr + @sizeOf(u32));

        return Bytes{
            .length = length.*,
            .bytes = bytes,
        };
    }

    pub fn string(self: *const Self) String {
        const ptr = self.value;
        const length: *const u32 = @ptrFromInt(ptr);

        const bytes: [*]const u32 = @ptrFromInt(ptr + @sizeOf(u32));

        return String{
            .length = length.*,
            .bytes = bytes,
        };
    }

    pub fn bln(self: *const Self) bool {
        const ptr = self.value;
        const b: *const u32 = @ptrFromInt(ptr);

        return b.* == 1;
    }

    pub fn list(self: *const Self) List {
        const ptr = self.value;
        const length: *const u32 = @ptrFromInt(ptr);

        const items: *?*ListNode = @ptrFromInt(ptr + @sizeOf(u32));

        return List{
            .type_length = self.length - 1,
            .inner_type = self.innerListType(),
            .length = length.*,
            .items = items.*,
        };
    }

    pub fn bls12_381_g1_element() *const ConstantTypeList {
        const types: *const ConstantTypeList = @ptrCast(&UplcG1Element);
        return types;
    }

    pub fn bls12_381_g2_element() *const ConstantTypeList {
        const types: *const ConstantTypeList = @ptrCast(&UplcG2Element);
        return types;
    }

    pub fn bls12_381_mlresult() *const ConstantTypeList {
        const types: *const ConstantTypeList = @ptrCast(&UplcMlResult);
        return types;
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
    bls12_381_g1_element,
    bls12_381_g2_element,
    bls12_381_mlresult,

    pub fn listDataType() *const ConstantType {
        const types: *const ConstantType = @ptrCast(&UplcListData);
        return types;
    }

    pub fn integerType() *const ConstantType {
        const types: *const ConstantType = @ptrCast(&UplcInteger);

        return types;
    }

    pub fn bytesType() *const ConstantType {
        const types: *const ConstantType = @ptrCast(&UplcBytes);
        return types;
    }

    pub fn stringType() *const ConstantType {
        const types: *const ConstantType = @ptrCast(&UplcString);
        return types;
    }

    pub fn unitType() *const ConstantType {
        const types: *const ConstantType = @ptrCast(&UplcUnit);
        return types;
    }

    pub fn booleanType() *const ConstantType {
        const types: *const ConstantType = @ptrCast(&UplcBoolean);
        return types;
    }
};

const UplcInteger = [1]u32{
    @intFromEnum(ConstantType.integer),
};
const UplcBytes = [1]u32{
    @intFromEnum(ConstantType.bytes),
};
const UplcString = [1]u32{
    @intFromEnum(ConstantType.string),
};
const UplcUnit = [1]u32{
    @intFromEnum(ConstantType.unit),
};
const UplcBoolean =
    [1]u32{
        @intFromEnum(ConstantType.boolean),
    };
const UplcListData = [2]u32{
    @intFromEnum(ConstantType.list),
    @intFromEnum(ConstantType.data),
};
const UplcG1Element = [2]u32{
    1,
    @intFromEnum(ConstantType.bls12_381_g1_element),
};
const UplcG2Element = [2]u32{
    1,
    @intFromEnum(ConstantType.bls12_381_g2_element),
};
const UplcMlResult = [2]u32{
    1,
    @intFromEnum(ConstantType.bls12_381_mlresult),
};
