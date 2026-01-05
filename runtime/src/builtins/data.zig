const std = @import("std");
const expr = @import("../expr.zig");
const Heap = @import("../Heap.zig");
const utils = @import("../utils.zig");
const runtime_value = @import("../value.zig");
const builtins_bytestring = @import("bytestring.zig");

const ConstantType = expr.ConstantType;
const Constant = expr.Constant;
const BigInt = expr.BigInt;
const Bytes = expr.Bytes;
const List = expr.List;
const ListNode = expr.ListNode;
const Data = expr.Data;
const DataListNode = expr.DataListNode;
const DataPairNode = expr.DataPairNode;
const ConstrData = expr.ConstrData;

const BuiltinContext = runtime_value.BuiltinContext;
const LinkedValues = runtime_value.LinkedValues;
const PairPayload = runtime_value.PairPayload;
const Value = runtime_value.Value;
const createConst = runtime_value.createConst;

inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
}

const DataTag = std.meta.Tag(Data);

const UplcDataType = [1]u32{
    @intFromEnum(ConstantType.data),
};

inline fn dataTypePtr() [*]const ConstantType {
    return @ptrCast(&UplcDataType);
}

// Runtime-built Data constants reuse this shared type descriptor.
inline fn runtimeDataTypeAddr() usize {
    return @intFromPtr(dataTypePtr());
}

const UnConstrReturnTypeDescriptor = [4]u32{
    @intFromEnum(ConstantType.pair),
    @intFromEnum(ConstantType.integer),
    @intFromEnum(ConstantType.list),
    @intFromEnum(ConstantType.data),
};

// Reusable descriptors for [(Data, Data)] results (unMapData et al.).
const DataPairTypeDescriptor = [3]u32{
    @intFromEnum(ConstantType.pair),
    @intFromEnum(ConstantType.data),
    @intFromEnum(ConstantType.data),
};

const DataPairListTypeDescriptor = [4]u32{
    @intFromEnum(ConstantType.list),
    @intFromEnum(ConstantType.pair),
    @intFromEnum(ConstantType.data),
    @intFromEnum(ConstantType.data),
};

pub fn normalizeSerializedDataConstantForHost(constant: *const Constant) void {
    // Serialized Data constants (length == 0x05) reuse the same header fields as every
    // other constant even though their payload format differs. The host side only needs
    // to read the type descriptor to decide how to compare the result, so repoint the
    // descriptor to the shared Data type and trim the reported type length to 1.
    const mutable: *Constant = @constCast(constant);
    mutable.length = 1;
    mutable.type_list = dataTypePtr();
}

pub fn chooseData(_: *BuiltinContext, args: *LinkedValues) *const Value {
    const bytes_branch = args.value;
    const int_branch = args.next.?.value;
    const list_branch = args.next.?.next.?.value;
    const map_branch = args.next.?.next.?.next.?.value;
    const constr_branch = args.next.?.next.?.next.?.next.?.value;
    const data_value = args.next.?.next.?.next.?.next.?.next.?.value.unwrapConstant();

    const variant = decodeDataVariant(data_value);

    return switch (variant) {
        .constr => constr_branch,
        .map => map_branch,
        .list => list_branch,
        .integer => int_branch,
        .bytes => bytes_branch,
    };
}

pub const serialized_data_const_tag: u32 = 0x05; // Mirrors serializer/constants.rs::const_tag::DATA
const large_constr_tag_flag: u32 = 0x80000000; // Mirrors serializer LARGE_CONSTR_TAG_FLAG

// Data constants are either runtime pointers (type list matches `dataTypePtr`) or
// serialized payloads tagged with `serialized_data_const_tag`. Reject everything else
// so callers can safely materialize the payload.
fn ensureDataConstant(con: *const Constant, comptime type_error_msg: []const u8) void {
    _ = type_error_msg; // Preserve signature for clearer call sites; failure always bubbles up.
    const uses_runtime_layout = @intFromPtr(con.type_list) == runtimeDataTypeAddr();
    if (uses_runtime_layout) {
        if (con.constType().* == .data) {
            return;
        }

        builtinEvaluationFailure();
    }

    if (con.length == serialized_data_const_tag) {
        return;
    }

    builtinEvaluationFailure();
}

fn decodeDataVariant(con: *const Constant) DataTag {
    const raw_ptr = con.rawValue();
    if (raw_ptr == 0) {
        utils.printlnString("null Data pointer");
        utils.exit(std.math.maxInt(u32));
    }

    const uses_runtime_layout = @intFromPtr(con.type_list) == runtimeDataTypeAddr();
    // Serialized constants keep an embedded payload instead of referencing the shared type.
    if (!uses_runtime_layout) {
        if (con.length != serialized_data_const_tag) {
            utils.printlnString("chooseData expects a Data constant");
            utils.exit(std.math.maxInt(u32));
        }

        // Serialized constants store the Data payload inline after the header.
        return readSerializedDataTag(raw_ptr);
    }

    const ty = con.constType().*;
    if (ty != .data) {
        utils.printlnString("chooseData expects a Data constant");
        utils.exit(std.math.maxInt(u32));
    }

    return readHeapDataTag(raw_ptr);
}

fn readSerializedDataTag(raw_ptr: u32) DataTag {
    if (raw_ptr == 0) {
        utils.printlnString("null Data pointer");
        utils.exit(std.math.maxInt(u32));
    }

    // Serialized constants place the variant tag at the start of the payload the
    // Constant points to, so we can read it directly.
    const words: [*]const u32 = @ptrFromInt(raw_ptr);
    return tagFromWord(words[0]);
}

fn readHeapDataTag(raw_ptr: u32) DataTag {
    if (raw_ptr == 0) {
        utils.printlnString("null Data pointer");
        utils.exit(std.math.maxInt(u32));
    }

    const heap_data: *const Data = @ptrFromInt(raw_ptr);
    return std.meta.activeTag(heap_data.*);
}

fn tagFromWord(raw: u32) DataTag {
    return std.meta.intToEnum(DataTag, raw) catch {
        utils.printlnString("invalid Data tag");
        utils.exit(std.math.maxInt(u32));
    };
}

fn buildDataListFromConstantList(
    m: *BuiltinContext,
    list: List,
    comptime type_error_msg: []const u8,
    comptime null_payload_msg: []const u8,
) ?*DataListNode {
    ensureListHoldsData(list, type_error_msg);

    var head: ?*DataListNode = null;
    var tail: ?*DataListNode = null;
    var cursor = list.items;

    while (cursor) |node| {
        const data_ptr = materializeDataElement(m, node.value, null_payload_msg);
        const new_node = m.heap.create(DataListNode, &DataListNode{
            .value = data_ptr,
            .next = null,
        });

        if (tail) |t| {
            t.next = new_node;
        } else {
            head = new_node;
        }
        tail = new_node;
        cursor = node.next;
    }

    return head;
}

fn buildDataPairListFromConstantList(
    m: *BuiltinContext,
    list: List,
    comptime type_error_msg: []const u8,
    comptime null_payload_msg: []const u8,
) ?*DataPairNode {
    ensureListHoldsDataPairs(list, type_error_msg);

    var head: ?*DataPairNode = null;
    var tail: ?*DataPairNode = null;
    var cursor = list.items;

    while (cursor) |node| {
        if (node.value == 0) {
            utils.printlnString(null_payload_msg);
            utils.exit(std.math.maxInt(u32));
        }

        const pair_payload: *const PairPayload = @ptrFromInt(node.value);
        const key_data = materializeDataElement(m, pair_payload.first, null_payload_msg);
        const value_data = materializeDataElement(m, pair_payload.second, null_payload_msg);

        const new_node = m.heap.create(DataPairNode, &DataPairNode{
            .key = key_data,
            .value = value_data,
            .next = null,
        });

        if (tail) |t| {
            t.next = new_node;
        } else {
            head = new_node;
        }
        tail = new_node;
        cursor = node.next;
    }

    return head;
}

fn ensureListHoldsData(list: List, comptime type_error_msg: []const u8) void {
    if (list.type_length == 0) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }

    const inner = list.inner_type;
    if (@intFromPtr(inner) == 0) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }

    if (inner[0] != ConstantType.data) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }
}

fn ensureListHoldsDataPairs(list: List, comptime type_error_msg: []const u8) void {
    if (list.type_length < 3) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }

    const inner = list.inner_type;
    if (@intFromPtr(inner) == 0) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }

    if (inner[0] != ConstantType.pair or inner[1] != ConstantType.data or inner[2] != ConstantType.data) {
        utils.printlnString(type_error_msg);
        utils.exit(std.math.maxInt(u32));
    }
}

fn materializeDataElement(
    m: *BuiltinContext,
    payload_addr: u32,
    comptime null_payload_msg: []const u8,
) *const Data {
    _ = null_payload_msg; // Builtins signal failure uniformly, so just abort evaluation.
    if (payload_addr == 0) {
        builtinEvaluationFailure();
    }

    if (serializedPayloadWordCount(payload_addr)) |word_count| {
        const payload_ptr: [*]const u8 = @ptrFromInt(payload_addr);
        var reader = SerializedDataReader.init(payload_ptr, word_count);
        const data_ptr = decodeSerializedDataPayload(m.heap, &reader);
        reader.ensureFullyConsumed();
        return data_ptr;
    }

    return @ptrFromInt(payload_addr);
}

const SerializedDataReader = struct {
    bytes: [*]const u8,
    len: u32,
    offset: u32,

    fn init(ptr: [*]const u8, word_count: u32) SerializedDataReader {
        const total_bytes = wordCountToByteLen(word_count);
        return .{ .bytes = ptr, .len = total_bytes, .offset = 0 };
    }

    fn readU32(self: *SerializedDataReader) u32 {
        if (self.len - self.offset < 4) {
            invalidSerializedData();
        }

        var result: u32 = 0;
        var i: u32 = 0;
        while (i < 4) : (i += 1) {
            const idx = self.offset + i;
            const shift: u5 = @intCast(i * 8);
            result |= (@as(u32, self.bytes[@intCast(idx)])) << shift;
        }

        self.offset += 4;
        return result;
    }

    fn readU8(self: *SerializedDataReader) u8 {
        if (self.offset >= self.len) {
            invalidSerializedData();
        }

        const byte = self.bytes[@intCast(self.offset)];
        self.offset += 1;
        return byte;
    }

    fn readBytes(self: *SerializedDataReader, byte_len: u32) [*]const u8 {
        if (self.len - self.offset < byte_len) {
            invalidSerializedData();
        }

        const start = self.offset;
        self.offset += byte_len;

        const base_addr = @intFromPtr(self.bytes);
        const start_offset: usize = @intCast(start);
        return @ptrFromInt(base_addr + start_offset);
    }

    fn alignToWord(self: *SerializedDataReader) void {
        const rem = self.offset & 3;
        if (rem == 0) return;

        const skip = 4 - rem;
        if (self.len - self.offset < skip) {
            invalidSerializedData();
        }
        self.offset += skip;
    }

    fn sliceWords(self: *SerializedDataReader, word_count: u32) SerializedDataReader {
        const byte_len = wordCountToByteLen(word_count);
        if (self.len - self.offset < byte_len) {
            invalidSerializedData();
        }

        const start = self.offset;
        self.offset += byte_len;

        const base_addr = @intFromPtr(self.bytes);
        const start_offset: usize = @intCast(start);

        return .{
            .bytes = @ptrFromInt(base_addr + start_offset),
            .len = byte_len,
            .offset = 0,
        };
    }

    fn ensureFullyConsumed(self: *SerializedDataReader) void {
        if (self.offset != self.len) {
            invalidSerializedData();
        }
    }
};

fn wordCountToByteLen(word_count: u32) u32 {
    if (word_count == 0) {
        invalidSerializedData();
    }

    if (word_count > std.math.maxInt(u32) / 4) {
        invalidSerializedData();
    }

    return word_count * 4;
}

const max_serialized_payload_words: u32 = 0x10000000;

pub fn serializedPayloadWordCount(payload_addr: u32) ?u32 {
    const header_size = @as(u32, @sizeOf(u32) * 2);
    if (payload_addr < header_size) {
        return null;
    }

    const header_ptr: [*]const u32 = @ptrFromInt(payload_addr - header_size);
    const tag = header_ptr[0];
    const words = header_ptr[1];

    if (tag == serialized_data_const_tag and words > 0 and words < max_serialized_payload_words) {
        return words;
    }

    return null;
}

fn decodeSerializedDataPayload(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const tag = reader.readU32();

    return switch (tag) {
        data_tag_constr => decodeSerializedConstr(heap, reader),
        data_tag_map => decodeSerializedMap(heap, reader),
        data_tag_list => decodeSerializedList(heap, reader),
        data_tag_integer => decodeSerializedInteger(heap, reader),
        data_tag_bytes => decodeSerializedBytes(heap, reader),
        else => {
            invalidSerializedData();
        },
    };
}

fn decodeSerializedConstr(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const encoded_tag = reader.readU32();
    const field_count = reader.readU32();

    var head: ?*DataListNode = null;
    var tail: ?*DataListNode = null;

    var i: u32 = 0;
    while (i < field_count) : (i += 1) {
        const field_words = reader.readU32();
        var field_reader = reader.sliceWords(field_words);
        const field_data = decodeSerializedDataPayload(heap, &field_reader);
        field_reader.ensureFullyConsumed();

        const node = heap.create(DataListNode, &DataListNode{
            .value = field_data,
            .next = null,
        });

        if (tail) |t| {
            t.next = node;
        } else {
            head = node;
        }
        tail = node;
    }

    const payload = ConstrData{
        .tag = decodeConstrTag(encoded_tag),
        .fields = head,
    };

    return heap.create(Data, &.{ .constr = payload });
}

fn decodeSerializedList(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const elem_count = reader.readU32();

    var head: ?*DataListNode = null;
    var tail: ?*DataListNode = null;

    var i: u32 = 0;
    while (i < elem_count) : (i += 1) {
        const elem_words = reader.readU32();
        var elem_reader = reader.sliceWords(elem_words);
        const elem_data = decodeSerializedDataPayload(heap, &elem_reader);
        elem_reader.ensureFullyConsumed();

        const node = heap.create(DataListNode, &DataListNode{
            .value = elem_data,
            .next = null,
        });

        if (tail) |t| {
            t.next = node;
        } else {
            head = node;
        }
        tail = node;
    }

    return heap.create(Data, &.{ .list = head });
}

fn decodeSerializedMap(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const pair_count = reader.readU32();

    var head: ?*DataPairNode = null;
    var tail: ?*DataPairNode = null;

    var i: u32 = 0;
    while (i < pair_count) : (i += 1) {
        const key_words = reader.readU32();
        var key_reader = reader.sliceWords(key_words);
        const key_data = decodeSerializedDataPayload(heap, &key_reader);
        key_reader.ensureFullyConsumed();

        const value_words = reader.readU32();
        var value_reader = reader.sliceWords(value_words);
        const value_data = decodeSerializedDataPayload(heap, &value_reader);
        value_reader.ensureFullyConsumed();

        const node = heap.create(DataPairNode, &DataPairNode{
            .key = key_data,
            .value = value_data,
            .next = null,
        });

        if (tail) |t| {
            t.next = node;
        } else {
            head = node;
        }
        tail = node;
    }

    return heap.create(Data, &.{ .map = head });
}

fn decodeSerializedInteger(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const sign = reader.readU8();
    const word_count = reader.readU32();

    const words_buf = heap.createArray(u32, word_count);
    var i: u32 = 0;
    while (i < word_count) : (i += 1) {
        words_buf[i] = reader.readU32();
    }
    reader.alignToWord();

    const words_view: [*]const u32 = words_buf;

    const big = BigInt{
        .sign = @intCast(sign),
        .length = word_count,
        .words = words_view,
    };

    return heap.create(Data, &.{ .integer = big });
}

fn decodeSerializedBytes(heap: *Heap, reader: *SerializedDataReader) *const Data {
    const byte_len = reader.readU32();
    const src = reader.readBytes(byte_len);
    reader.alignToWord();

    const words_buf = heap.createArray(u32, byte_len);
    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        words_buf[i] = @intCast(src[@intCast(i)]);
    }

    const bytes_words: [*]const u32 = words_buf;

    const bytes_view = Bytes{
        .length = byte_len,
        .bytes = bytes_words,
    };

    return heap.create(Data, &.{ .bytes = bytes_view });
}

fn decodeConstrTag(encoded: u32) u32 {
    if ((encoded & large_constr_tag_flag) != 0) {
        return encoded & ~large_constr_tag_flag;
    }

    if (encoded >= 1280) {
        return (encoded - 1280) + 7;
    }

    if (encoded >= 121 and encoded <= 127) {
        return encoded - 121;
    }

    return encoded;
}

fn invalidSerializedData() noreturn {
    builtinEvaluationFailure();
}

pub fn constrData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    // First arg: list of Data (fields)
    const fields_list = args.value.unwrapList();

    const tag_int = args.next.?.value.unwrapInteger();

    if (tag_int.sign == 1 or tag_int.length > 1) {
        utils.printlnString("constrData: tag must be a non-negative integer that fits in u32");
        utils.exit(std.math.maxInt(u32));
    }
    const tag: u32 = tag_int.words[0];

    const field_nodes = buildDataListFromConstantList(
        m,
        fields_list,
        "constrData expects a list of Data",
        "constrData: null Data constant payload",
    );

    const constr_payload = ConstrData{
        .tag = tag,
        .fields = field_nodes,
    };

    const data_ptr = m.heap.create(Data, &.{ .constr = constr_payload });

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn mapData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const pairs_list = args.value.unwrapList();

    const pair_nodes = buildDataPairListFromConstantList(
        m,
        pairs_list,
        "mapData expects a list of (Data, Data) pairs",
        "mapData: null Data constant payload",
    );

    const data_ptr = m.heap.create(Data, &.{ .map = pair_nodes });

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn listData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const elements_list = args.value.unwrapList();

    const data_nodes = buildDataListFromConstantList(
        m,
        elements_list,
        "listData expects a list of Data",
        "listData: null Data constant payload",
    );

    const data_ptr = m.heap.create(Data, &.{ .list = data_nodes });

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn iData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const int_arg = args.value.unwrapInteger();

    const data_ptr = m.heap.create(Data, &.{ .integer = int_arg });

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn bData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const bytes = args.value.unwrapBytestring();

    // ByteStrings are immutable, so it is safe to share their backing buffer with Data.
    const payload = Data{ .bytes = bytes };
    const data_ptr = m.heap.create(Data, &payload);

    const con = Constant{
        .length = 1,
        .type_list = dataTypePtr(),
        .value = @intFromPtr(data_ptr),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn unConstrData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unConstrData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unConstrData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .constr => |constr_payload| {
            const tag_limb_count: u32 = if (constr_payload.tag == 0) 0 else 1;
            const tag_payload = m.heap.createArray(u32, tag_limb_count + 2);
            tag_payload[0] = 0; // positive sign
            tag_payload[1] = tag_limb_count;
            if (tag_limb_count == 1) {
                tag_payload[2] = constr_payload.tag;
            }

            var list_head: ?*ListNode = null;
            var list_tail: ?*ListNode = null;
            var list_len: u32 = 0;
            var cursor = constr_payload.fields;

            // Rebuild a UPLC list of Data constants pointing at the decoded fields.
            while (cursor) |field| {
                const node = m.heap.create(ListNode, &ListNode{
                    .value = @intFromPtr(field.value),
                    .next = null,
                });

                if (list_tail) |tail| {
                    tail.next = node;
                } else {
                    list_head = node;
                }
                list_tail = node;
                list_len += 1;
                cursor = field.next;
            }

            const list_payload = m.heap.createArray(u32, 2);
            list_payload[0] = list_len;
            list_payload[1] = if (list_head) |head| @intFromPtr(head) else 0;

            const pair_payload = m.heap.create(PairPayload, &PairPayload{
                .first = @intFromPtr(tag_payload),
                .second = @intFromPtr(list_payload),
            });

            const pair_type: [*]const ConstantType = @ptrCast(&UnConstrReturnTypeDescriptor);
            const pair_const = Constant{
                .length = @intCast(UnConstrReturnTypeDescriptor.len),
                .type_list = pair_type,
                .value = @intFromPtr(pair_payload),
            };

            return createConst(m.heap, m.heap.create(Constant, &pair_const));
        },
        else => builtinEvaluationFailure(),
    }
}

pub fn unMapData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unMapData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unMapData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .map => |pairs_head| {
            var uplc_head: ?*ListNode = null;
            var uplc_tail: ?*ListNode = null;
            var length: u32 = 0;
            var cursor = pairs_head;

            while (cursor) |pair_node| {
                // Rebuild a `(Data, Data)` payload whose components point at the decoded entries.
                const pair_payload = m.heap.create(PairPayload, &PairPayload{
                    .first = @intFromPtr(pair_node.key),
                    .second = @intFromPtr(pair_node.value),
                });

                const node = m.heap.create(ListNode, &ListNode{
                    .value = @intFromPtr(pair_payload),
                    .next = null,
                });

                if (uplc_tail) |tail| {
                    tail.next = node;
                } else {
                    uplc_head = node;
                }
                uplc_tail = node;
                length += 1;
                cursor = pair_node.next;
            }

            const list_payload = m.heap.createArray(u32, 2);
            list_payload[0] = length;
            list_payload[1] = if (uplc_head) |head| @intFromPtr(head) else 0;

            const list_const = Constant{
                .length = @intCast(DataPairListTypeDescriptor.len),
                .type_list = @ptrCast(&DataPairListTypeDescriptor),
                .value = @intFromPtr(list_payload),
            };

            return createConst(m.heap, m.heap.create(Constant, &list_const));
        },
        else => builtinEvaluationFailure(),
    }
}

pub fn unListData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unListData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unListData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .list => |list_head| {
            var uplc_head: ?*ListNode = null;
            var uplc_tail: ?*ListNode = null;
            var length: u32 = 0;
            var cursor = list_head;

            // Rebuild a UPLC list whose nodes reference the decoded Data elements.
            while (cursor) |elem| {
                const node = m.heap.create(ListNode, &ListNode{
                    .value = @intFromPtr(elem.value),
                    .next = null,
                });

                if (uplc_tail) |tail| {
                    tail.next = node;
                } else {
                    uplc_head = node;
                }
                uplc_tail = node;
                length += 1;
                cursor = elem.next;
            }

            const list_payload = m.heap.createArray(u32, 2);
            list_payload[0] = length;
            list_payload[1] = if (uplc_head) |head| @intFromPtr(head) else 0;

            const list_const = Constant{
                .length = 2,
                .type_list = @ptrCast(ConstantType.listDataType()),
                .value = @intFromPtr(list_payload),
            };

            return createConst(m.heap, m.heap.create(Constant, &list_const));
        },
        else => builtinEvaluationFailure(),
    }
}

pub fn unIData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unIData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unIData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .integer => |int_payload| {
            const int_const = int_payload.createConstant(ConstantType.integerType(), m.heap);
            return createConst(m.heap, int_const);
        },
        else => builtinEvaluationFailure(),
    }
}

pub fn unBData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "unBData expects a Data constant");

    // Serialized Data constants embed their payload inline, so materialize them
    // into the heap to reuse the runtime layout for both representations.
    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "unBData: null Data constant payload",
    );

    switch (data_ptr.*) {
        .bytes => |bytes_payload| {
            const bytes_const = bytes_payload.createConstant(ConstantType.bytesType(), m.heap);
            return createConst(m.heap, bytes_const);
        },
        else => builtinEvaluationFailure(),
    }
}

const SerializedView = struct {
    words: [*]const u32,
    len: u32,
};

const SerializedOwnedView = struct {
    words: [*]u32,
    len: u32,
};

const DataView = union(enum) {
    runtime: *const Data,
    serialized: SerializedView,
};

pub fn equalsData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const rhs_const = args.value.unwrapConstant();
    const lhs_const = args.next.?.value.unwrapConstant();

    const result = dataConstantsEqual(m.heap, lhs_const, rhs_const);

    var bool_result = m.heap.createArray(u32, 4);
    bool_result[0] = 1;
    bool_result[1] = @intFromPtr(ConstantType.booleanType());
    bool_result[2] = @intFromPtr(bool_result + 3);
    bool_result[3] = @intFromBool(result);

    return createConst(m.heap, @ptrCast(bool_result));
}

fn dataConstantsEqual(heap: *Heap, lhs: *const Constant, rhs: *const Constant) bool {
    const lhs_view = classifyDataConstant(lhs);
    const rhs_view = classifyDataConstant(rhs);

    return switch (lhs_view) {
        .runtime => |lhs_data| switch (rhs_view) {
            .runtime => |rhs_data| heapDataEqual(lhs_data, rhs_data),
            .serialized => |rhs_ser| serializedEqualsRuntime(heap, rhs_ser, lhs_data),
        },
        .serialized => |lhs_ser| switch (rhs_view) {
            .runtime => |rhs_data| serializedEqualsRuntime(heap, lhs_ser, rhs_data),
            .serialized => |rhs_ser| serializedViewsEqual(lhs_ser, rhs_ser),
        },
    };
}

fn classifyDataConstant(con: *const Constant) DataView {
    const uses_runtime_layout = @intFromPtr(con.type_list) == runtimeDataTypeAddr();
    if (uses_runtime_layout) {
        if (con.constType().* != .data) {
            utils.printlnString("equalsData expects Data constants");
            utils.exit(std.math.maxInt(u32));
        }

        const ptr = con.rawValue();
        if (ptr == 0) {
            utils.printlnString("equalsData received null Data pointer");
            utils.exit(std.math.maxInt(u32));
        }
        return .{ .runtime = @ptrFromInt(ptr) };
    }

    if (con.length != serialized_data_const_tag) {
        utils.printlnString("equalsData expects Data constants");
        utils.exit(std.math.maxInt(u32));
    }

    const len_words: u32 = @intCast(@intFromPtr(con.type_list));
    const data_ptr = con.rawValue();
    if (data_ptr == 0) {
        utils.printlnString("equalsData received null serialized Data payload");
        utils.exit(std.math.maxInt(u32));
    }

    return .{
        .serialized = .{
            .words = @ptrFromInt(data_ptr),
            .len = len_words,
        },
    };
}

fn serializedViewsEqual(a: SerializedView, b: SerializedView) bool {
    if (a.len != b.len) return false;

    var i: u32 = 0;
    while (i < a.len) : (i += 1) {
        if (a.words[i] != b.words[i]) return false;
    }
    return true;
}

fn serializedEqualsRuntime(heap: *Heap, serialized: SerializedView, runtime_data: *const Data) bool {
    const encoded = serializeRuntimeData(heap, runtime_data);
    defer heap.reclaimHeap(u32, encoded.len);
    return serializedViewsEqual(
        .{ .words = encoded.words, .len = encoded.len },
        serialized,
    );
}

fn serializeRuntimeData(heap: *Heap, data: *const Data) SerializedOwnedView {
    const word_count = dataSerializedWordCount(data);
    const buffer = heap.createArray(u32, word_count);
    var writer = PayloadWriter.init(buffer, word_count);

    writeSerializedData(&writer, data);

    return .{ .words = buffer, .len = word_count };
}

fn writeSerializedData(writer: *PayloadWriter, data: *const Data) void {
    switch (data.*) {
        .constr => |payload| {
            writer.writeU32(data_tag_constr);
            writer.writeU32(encodeConstrTag(payload.tag));
            writer.writeU32(countDataList(payload.fields));

            var node = payload.fields;
            while (node) |field| {
                const nested_words = dataSerializedWordCount(field.value);
                writer.writeU32(nested_words);
                writeSerializedData(writer, field.value);
                node = field.next;
            }
        },
        .map => |pairs| {
            writer.writeU32(data_tag_map);
            writer.writeU32(countDataPairs(pairs));

            var node = pairs;
            while (node) |pair| {
                const key_words = dataSerializedWordCount(pair.key);
                writer.writeU32(key_words);
                writeSerializedData(writer, pair.key);

                const value_words = dataSerializedWordCount(pair.value);
                writer.writeU32(value_words);
                writeSerializedData(writer, pair.value);

                node = pair.next;
            }
        },
        .list => |list_head| {
            writer.writeU32(data_tag_list);
            writer.writeU32(countDataList(list_head));

            var node = list_head;
            while (node) |elem| {
                const nested_words = dataSerializedWordCount(elem.value);
                writer.writeU32(nested_words);
                writeSerializedData(writer, elem.value);
                node = elem.next;
            }
        },
        .integer => |int_val| {
            writer.writeU32(data_tag_integer);
            encodeBigInt(writer, int_val);
        },
        .bytes => |bytes_val| {
            writer.writeU32(data_tag_bytes);
            encodeBytes(writer, bytes_val);
        },
    }
}

const data_tag_constr: u32 = 0;
const data_tag_map: u32 = 1;
const data_tag_list: u32 = 2;
const data_tag_integer: u32 = 3;
const data_tag_bytes: u32 = 4;

fn encodeConstrTag(tag: u32) u32 {
    if (tag < 7) return 121 + tag;
    return 1280 + (tag - 7);
}

fn dataSerializedWordCount(data: *const Data) u32 {
    return switch (data.*) {
        .constr => blk: {
            var total: u32 = 3; // variant tag, encoded tag, field count
            var node = data.constr.fields;
            while (node) |field| {
                total += 1; // field length prefix
                total += dataSerializedWordCount(field.value);
                node = field.next;
            }
            break :blk total;
        },
        .map => blk: {
            var total: u32 = 2; // variant tag + pair count
            var node = data.map;
            while (node) |pair| {
                total += 1;
                total += dataSerializedWordCount(pair.key);
                total += 1;
                total += dataSerializedWordCount(pair.value);
                node = pair.next;
            }
            break :blk total;
        },
        .list => blk: {
            var total: u32 = 2; // variant tag + element count
            var node = data.list;
            while (node) |elem| {
                total += 1;
                total += dataSerializedWordCount(elem.value);
                node = elem.next;
            }
            break :blk total;
        },
        .integer => blk: {
            const byte_len: u32 = data.integer.length * 4;
            const total_bytes = 4 + 1 + 4 + byte_len;
            break :blk bytesToWords(total_bytes);
        },
        .bytes => blk: {
            const byte_len = data.bytes.length;
            const total_bytes = 4 + 4 + byte_len;
            break :blk bytesToWords(total_bytes);
        },
    };
}

fn bytesToWords(byte_len: u32) u32 {
    if (byte_len == 0) return 0;
    return (byte_len + 3) / 4;
}

const PayloadWriter = struct {
    bytes: [*]u8,
    total_bytes: u32,
    offset: u32,

    fn init(dst_words: [*]u32, len_words: u32) PayloadWriter {
        var i: u32 = 0;
        while (i < len_words) : (i += 1) {
            dst_words[i] = 0;
        }
        return .{
            .bytes = @ptrCast(dst_words),
            .total_bytes = len_words * 4,
            .offset = 0,
        };
    }

    fn writeU32(self: *PayloadWriter, value: u32) void {
        var buf = value;
        self.writeBytes(std.mem.asBytes(&buf));
    }

    fn writeByte(self: *PayloadWriter, value: u8) void {
        if (self.offset == self.total_bytes) {
            builtinEvaluationFailure();
        }
        const idx: usize = @intCast(self.offset);
        self.bytes[idx] = value;
        self.offset += 1;
    }

    fn writeBytes(self: *PayloadWriter, data: []const u8) void {
        if (data.len == 0) return;
        const seg_len: u32 = @intCast(data.len);
        if (self.offset + seg_len > self.total_bytes) {
            builtinEvaluationFailure();
        }
        const start: usize = @intCast(self.offset);
        const dst = self.bytes[start .. start + data.len];
        @memcpy(dst, data);
        self.offset += seg_len;
    }
};

fn encodeBigInt(writer: *PayloadWriter, int_val: BigInt) void {
    writer.writeByte(@intFromBool(int_val.sign != 0));
    const byte_len: u32 = int_val.length * 4;
    writer.writeU32(byte_len);

    var i: u32 = 0;
    while (i < int_val.length) : (i += 1) {
        writer.writeU32(int_val.words[i]);
    }
}

fn encodeBytes(writer: *PayloadWriter, bytes_val: Bytes) void {
    writer.writeU32(bytes_val.length);
    var i: u32 = 0;
    while (i < bytes_val.length) : (i += 1) {
        writer.writeByte(@truncate(bytes_val.bytes[i]));
    }
}

fn heapDataEqual(lhs: *const Data, rhs: *const Data) bool {
    const tag_lhs = std.meta.activeTag(lhs.*);
    const tag_rhs = std.meta.activeTag(rhs.*);
    if (tag_lhs != tag_rhs) return false;

    return switch (tag_lhs) {
        .constr => lhs.constr.tag == rhs.constr.tag and
            dataListEqual(lhs.constr.fields, rhs.constr.fields),
        .map => dataPairEqual(lhs.map, rhs.map),
        .list => dataListEqual(lhs.list, rhs.list),
        .integer => bigIntEqual(lhs.integer, rhs.integer),
        .bytes => bytesEqual(lhs.bytes, rhs.bytes),
    };
}

fn dataListEqual(a: ?*DataListNode, b: ?*DataListNode) bool {
    var left = a;
    var right = b;

    while (true) {
        if (left == null and right == null) return true;
        if (left == null or right == null) return false;

        const lhs_node = left.?;
        const rhs_node = right.?;
        if (!heapDataEqual(lhs_node.value, rhs_node.value)) return false;

        left = lhs_node.next;
        right = rhs_node.next;
    }
}

fn dataPairEqual(a: ?*DataPairNode, b: ?*DataPairNode) bool {
    var left = a;
    var right = b;

    while (true) {
        if (left == null and right == null) return true;
        if (left == null or right == null) return false;

        const lhs_node = left.?;
        const rhs_node = right.?;
        if (!heapDataEqual(lhs_node.key, rhs_node.key)) return false;
        if (!heapDataEqual(lhs_node.value, rhs_node.value)) return false;

        left = lhs_node.next;
        right = rhs_node.next;
    }
}

fn bigIntEqual(a: BigInt, b: BigInt) bool {
    if (a.sign != b.sign) return false;
    if (a.length != b.length) return false;

    var i: u32 = 0;
    while (i < a.length) : (i += 1) {
        if (a.words[i] != b.words[i]) return false;
    }
    return true;
}

fn bytesEqual(a: Bytes, b: Bytes) bool {
    if (a.length != b.length) return false;

    var i: u32 = 0;
    while (i < a.length) : (i += 1) {
        if (a.bytes[i] != b.bytes[i]) return false;
    }
    return true;
}

fn countDataList(head: ?*DataListNode) u32 {
    var count: u32 = 0;
    var cursor = head;
    while (cursor) |node| {
        count += 1;
        cursor = node.next;
    }
    return count;
}

fn countDataPairs(head: ?*DataPairNode) u32 {
    var count: u32 = 0;
    var cursor = head;
    while (cursor) |node| {
        count += 1;
        cursor = node.next;
    }
    return count;
}

// Misc constructors
pub fn mkPairData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    // LinkedValues stores the most recently applied argument first, so the
    // first component lives in `args.next`.
    const second = args.value.unwrapConstant();
    ensureDataConstant(second, "mkPairData expects Data constants");

    const first = args.next.?.value.unwrapConstant();
    ensureDataConstant(first, "mkPairData expects Data constants");

    const payload = m.heap.create(PairPayload, &PairPayload{
        .first = first.rawValue(),
        .second = second.rawValue(),
    });

    const con = Constant{
        .length = @intCast(DataPairTypeDescriptor.len),
        .type_list = @ptrCast(&DataPairTypeDescriptor),
        .value = @intFromPtr(payload),
    };

    return createConst(m.heap, m.heap.create(Constant, &con));
}

pub fn mkNilData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const unit_arg = args.value;

    // mkNilData takes a unit argument to fix its polymorphic type.
    unit_arg.unwrapUnit();

    var result = m.heap.createArray(u32, 5);
    result[0] = 2;
    result[1] = @intFromPtr(ConstantType.listDataType());
    result[2] = @intFromPtr(result + 3);
    result[3] = 0;
    result[4] = 0;

    return createConst(m.heap, @ptrCast(result));
}

pub fn mkNilPairData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const unit_arg = args.value;
    unit_arg.unwrapUnit();

    const payload = m.heap.createArray(u32, 2);
    payload[0] = 0;
    payload[1] = 0;

    const list_const = Constant{
        .length = @intCast(DataPairListTypeDescriptor.len),
        .type_list = @ptrCast(&DataPairListTypeDescriptor),
        .value = @intFromPtr(payload),
    };

    return createConst(m.heap, m.heap.create(Constant, &list_const));
}

pub fn serialiseData(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const data_const = args.value.unwrapConstant();
    ensureDataConstant(data_const, "serialiseData expects a Data constant");

    const data_ptr = materializeDataElement(
        m,
        data_const.rawValue(),
        "serialiseData: null Data constant payload",
    );

    return serialiseRuntimeData(m, data_ptr);
}

fn serialiseRuntimeData(m: *BuiltinContext, data_ptr: *const Data) *const Value {
    const max_byte_len = cborEncodedByteUpperBound(data_ptr);
    const word_capacity = bytesToWords(max_byte_len);
    if (word_capacity == 0) builtinEvaluationFailure();

    const allocation = builtins_bytestring.initByteStringAllocation(m.heap, max_byte_len);

    const packed_words = m.heap.createArray(u32, word_capacity);
    var writer = CborWriter.init(packed_words, word_capacity);
    encodeDataAsCbor(&writer, m.heap, data_ptr);

    const actual_bytes = writer.offset;
    if (actual_bytes == 0 or actual_bytes > max_byte_len) {
        builtinEvaluationFailure();
    }

    allocation.constant_words[3] = actual_bytes;
    copyPackedWordsToUnpacked(allocation.data_words, packed_words, actual_bytes);

    m.heap.reclaimHeap(u32, word_capacity);

    return createConst(m.heap, @ptrCast(allocation.constant_words));
}

// Estimate the number of CBOR bytes serialiseData will emit so we can size the
// scratch buffer without risking overflow when chunking long byte strings.
fn cborEncodedByteUpperBound(data: *const Data) u32 {
    const total = dataCborByteUpperBound(data);
    if (total == 0 or total > std.math.maxInt(u32)) {
        builtinEvaluationFailure();
    }
    return @intCast(total);
}

fn dataCborByteUpperBound(data: *const Data) u64 {
    return switch (data.*) {
        .constr => constrCborByteUpperBound(data.constr),
        .map => mapCborByteUpperBound(data.map),
        .list => listCborByteUpperBound(data.list),
        .integer => integerCborByteUpperBound(data.integer),
        .bytes => bytesValueCborByteUpperBound(data.bytes),
    };
}

fn constrCborByteUpperBound(payload: @FieldType(Data, "constr")) u64 {
    const info = computeConstrTag(payload.tag);
    var total: u64 = @as(u64, cborMajorLen(info.cbor_tag));
    if (info.use_tuple) {
        total = addOrFail64(total, @as(u64, cborMajorLen(2)));
        total = addOrFail64(total, @as(u64, cborMajorLen(info.constructor_index)));
    }
    total = addOrFail64(total, listCborByteUpperBound(payload.fields));
    return total;
}

fn listCborByteUpperBound(head: ?*DataListNode) u64 {
    const len = countDataList(head);
    // Include the definite-length array header emitted by encodeListAsCbor.
    var total: u64 = @as(u64, cborMajorLen(@as(u64, len)));
    var node = head;
    while (node) |entry| {
        total = addOrFail64(total, dataCborByteUpperBound(entry.value));
        node = entry.next;
    }
    return total;
}

fn mapCborByteUpperBound(head: ?*DataPairNode) u64 {
    const len = countDataPairs(head);
    var total: u64 = @as(u64, cborMajorLen(@as(u64, len)));
    var node = head;
    while (node) |pair| {
        total = addOrFail64(total, dataCborByteUpperBound(pair.key));
        total = addOrFail64(total, dataCborByteUpperBound(pair.value));
        node = pair.next;
    }
    return total;
}

fn integerCborByteUpperBound(int_val: BigInt) u64 {
    const used_words = significantWordCount(int_val.words, int_val.length);

    if (int_val.sign == 0) {
        if (used_words == 0) {
            return @as(u64, cborMajorLen(0));
        }

        if (wordsToU64(int_val.words, used_words)) |value| {
            return @as(u64, cborMajorLen(value));
        }

        const byte_len = magnitudeByteLen(int_val.words, used_words);
        return addOrFail64(@as(u64, cborMajorLen(2)), byteStringEncodedLen(byte_len));
    }

    if (used_words == 0) {
        builtinEvaluationFailure();
    }

    if (wordsToU64(int_val.words, used_words)) |value| {
        if (value == 0) builtinEvaluationFailure();
        return @as(u64, cborMajorLen(value - 1));
    }

    const byte_len = magnitudeByteLen(int_val.words, used_words);
    return addOrFail64(@as(u64, cborMajorLen(3)), byteStringEncodedLen(byte_len));
}

fn bytesValueCborByteUpperBound(bytes_val: Bytes) u64 {
    return dataByteStringEncodedLen(bytes_val.length);
}

fn byteStringEncodedLen(byte_len: u32) u64 {
    const len: u64 = byte_len;
    return len + @as(u64, cborMajorLen(len));
}

const byte_string_chunk_limit: u32 = 64;

fn dataByteStringEncodedLen(byte_len: u32) u64 {
    if (byte_len <= byte_string_chunk_limit) {
        return byteStringEncodedLen(byte_len);
    }

    const chunk_len: u32 = byte_string_chunk_limit;
    const chunk_header_len: u64 = 2; // 64 >= 24 so requires 2-byte header.
    const chunk_total: u64 = chunk_header_len + chunk_len;

    const full_chunks: u32 = byte_len / chunk_len;
    const remainder: u32 = byte_len % chunk_len;

    var total: u64 = 1; // Begin indefinite byte string.

    if (full_chunks > 0) {
        const full_total: u128 = @as(u128, full_chunks) * @as(u128, chunk_total);
        if (full_total > std.math.maxInt(u64)) builtinEvaluationFailure();
        total = addOrFail64(total, @intCast(full_total));
    }

    if (remainder > 0) {
        const header_len: u64 = if (remainder < 24) 1 else 2;
        total = addOrFail64(total, header_len + @as(u64, remainder));
    }

    // Account for the CBOR break code.
    return addOrFail64(total, 1);
}

fn cborMajorLen(value: u64) u32 {
    if (value < 24) return 1;
    if (value <= 0xFF) return 2;
    if (value <= 0xFFFF) return 3;
    if (value <= 0xFFFF_FFFF) return 5;
    return 9;
}

fn addOrFail64(a: u64, b: u64) u64 {
    if (a > std.math.maxInt(u64) - b) builtinEvaluationFailure();
    return a + b;
}

// Encode runtime Data into the CBOR form returned by serialiseData.
fn encodeDataAsCbor(writer: *CborWriter, heap: *Heap, data: *const Data) void {
    switch (data.*) {
        .constr => |payload| encodeConstrAsCbor(writer, heap, payload),
        .map => |pairs| encodeMapAsCbor(writer, heap, pairs),
        .list => |elements| encodeListAsCbor(writer, heap, elements),
        .integer => |int_val| encodeIntegerAsCbor(writer, heap, int_val),
        .bytes => |bytes_val| encodeBytesAsCbor(writer, bytes_val),
    }
}

fn encodeConstrAsCbor(writer: *CborWriter, heap: *Heap, payload: @FieldType(Data, "constr")) void {
    const info = computeConstrTag(payload.tag);
    writer.writeTag(info.cbor_tag);

    if (info.use_tuple) {
        writer.writeArray(2);
        writer.writeUnsigned(info.constructor_index);
    }

    encodeListAsCbor(writer, heap, payload.fields);
}

// Mirrors the Plutus ledger encoding for constructor tags.
fn computeConstrTag(tag: u32) struct { cbor_tag: u64, constructor_index: u64, use_tuple: bool } {
    const idx: u64 = tag;
    if (idx < 7) {
        return .{ .cbor_tag = 121 + idx, .constructor_index = idx, .use_tuple = false };
    }
    if (idx < 128) {
        return .{ .cbor_tag = 1280 + (idx - 7), .constructor_index = idx, .use_tuple = false };
    }
    return .{ .cbor_tag = 102, .constructor_index = idx, .use_tuple = true };
}

fn encodeListAsCbor(writer: *CborWriter, heap: *Heap, head: ?*DataListNode) void {
    const len = countDataList(head);
    // Ledger serialiseData uses definite-length arrays for lists, so mirror that shape here.
    writer.writeArray(len);

    var node = head;
    while (node) |entry| {
        encodeDataAsCbor(writer, heap, entry.value);
        node = entry.next;
    }
}

fn encodeMapAsCbor(writer: *CborWriter, heap: *Heap, head: ?*DataPairNode) void {
    const len = countDataPairs(head);
    writer.writeMap(len);
    if (len <= 1) {
        var node = head;
        while (node) |pair| {
            encodeDataAsCbor(writer, heap, pair.key);
            encodeDataAsCbor(writer, heap, pair.value);
            node = pair.next;
        }
        return;
    }

    // Ledger serialiseData follows canonical CBOR, which requires map keys to be
    // ordered by their encoded byte representation.
    const entries = heap.createArray(MapSortEntry, len);
    const order = heap.createArray(u32, len);

    var node = head;
    var idx: u32 = 0;
    while (node) |pair| {
        const key_bound = dataCborByteUpperBound(pair.key);
        if (key_bound == 0 or key_bound > std.math.maxInt(u32)) {
            builtinEvaluationFailure();
        }
        const byte_cap: u32 = @intCast(key_bound);
        const buffer_words = bytesToWords(byte_cap);
        if (buffer_words == 0) builtinEvaluationFailure();

        const buffer = heap.createArray(u32, buffer_words);
        var key_writer = CborWriter.init(buffer, buffer_words);
        encodeDataAsCbor(&key_writer, heap, pair.key);

        entries[@intCast(idx)] = .{
            .pair = pair,
            .key_buf_words = buffer,
            .buffer_words = buffer_words,
            .key_len = key_writer.offset,
        };
        order[@intCast(idx)] = idx;
        idx += 1;
        node = pair.next;
    }

    sortCanonicalMapOrder(entries, order, len);
    ensureCanonicalMapKeysUnique(entries, order, len);

    var ord_idx: u32 = 0;
    while (ord_idx < len) : (ord_idx += 1) {
        const entry_idx = order[@intCast(ord_idx)];
        const entry = entries[@intCast(entry_idx)];
        writeEncodedKeyBytes(writer, entry.key_buf_words, entry.key_len);
        encodeDataAsCbor(writer, heap, entry.pair.value);
    }

    reclaimMapKeyBuffers(heap, entries, len);
    heap.reclaimHeap(u32, len);
    heap.reclaimHeap(MapSortEntry, len);
}

const MapSortEntry = struct {
    pair: *DataPairNode,
    key_buf_words: [*]u32,
    buffer_words: u32,
    key_len: u32,
};

fn writeEncodedKeyBytes(writer: *CborWriter, words: [*]const u32, byte_len: u32) void {
    const bytes: [*]const u8 = @ptrCast(words);
    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        writer.writeByte(bytes[@intCast(i)]);
    }
}

fn sortCanonicalMapOrder(entries: [*]const MapSortEntry, order: [*]u32, len: u32) void {
    if (len <= 1) return;

    var i: u32 = 1;
    while (i < len) : (i += 1) {
        const current = order[@intCast(i)];
        var j = i;
        while (j > 0 and mapKeyLessThan(entries, current, order[@intCast(j - 1)])) {
            order[@intCast(j)] = order[@intCast(j - 1)];
            j -= 1;
        }
        order[@intCast(j)] = current;
    }
}

fn mapKeyLessThan(entries: [*]const MapSortEntry, lhs_idx: u32, rhs_idx: u32) bool {
    const lhs = entries[@intCast(lhs_idx)];
    const rhs = entries[@intCast(rhs_idx)];

    // Canonical CBOR order sorts by encoded byte-length before doing a
    // lexicographic comparison of the actual bytes.
    if (lhs.key_len != rhs.key_len) {
        return lhs.key_len < rhs.key_len;
    }

    const lhs_bytes: [*]const u8 = @ptrCast(lhs.key_buf_words);
    const rhs_bytes: [*]const u8 = @ptrCast(rhs.key_buf_words);

    var offset: u32 = 0;
    while (offset < lhs.key_len) : (offset += 1) {
        const left_byte = lhs_bytes[@intCast(offset)];
        const right_byte = rhs_bytes[@intCast(offset)];
        if (left_byte == right_byte) continue;
        return left_byte < right_byte;
    }
    return false;
}

fn mapKeyBytesEqual(lhs: MapSortEntry, rhs: MapSortEntry) bool {
    if (lhs.key_len != rhs.key_len) return false;

    const lhs_bytes: [*]const u8 = @ptrCast(lhs.key_buf_words);
    const rhs_bytes: [*]const u8 = @ptrCast(rhs.key_buf_words);

    var offset: u32 = 0;
    while (offset < lhs.key_len) : (offset += 1) {
        if (lhs_bytes[@intCast(offset)] != rhs_bytes[@intCast(offset)]) return false;
    }

    return true;
}

fn ensureCanonicalMapKeysUnique(entries: [*]const MapSortEntry, order: [*]const u32, len: u32) void {
    if (len <= 1) return;

    var idx: u32 = 1;
    while (idx < len) : (idx += 1) {
        const current = entries[@intCast(order[@intCast(idx)])];
        const previous = entries[@intCast(order[@intCast(idx - 1)])];

        // Canonical CBOR forbids duplicate keys, so fail when two encoded keys compare
        // equal after sorting.
        if (mapKeyBytesEqual(current, previous)) {
            builtinEvaluationFailure();
        }
    }
}

fn reclaimMapKeyBuffers(heap: *Heap, entries: [*]const MapSortEntry, len: u32) void {
    var idx = len;
    while (idx > 0) {
        idx -= 1;
        const entry = entries[@intCast(idx)];
        heap.reclaimHeap(u32, entry.buffer_words);
    }
}

// Ledger chunks byte strings longer than 64 bytes into an indefinite-length CBOR item.
fn encodeBytesAsCbor(writer: *CborWriter, bytes_val: Bytes) void {
    if (bytes_val.length <= byte_string_chunk_limit) {
        writer.writeByteStringPrefix(bytes_val.length);

        var i: u32 = 0;
        while (i < bytes_val.length) : (i += 1) {
            writer.writeByte(@intCast(bytes_val.bytes[i]));
        }
        return;
    }

    writer.beginIndefiniteByteString();

    var offset: u32 = 0;
    while (offset < bytes_val.length) {
        const remaining = bytes_val.length - offset;
        const chunk_len: u32 = if (remaining > byte_string_chunk_limit) byte_string_chunk_limit else remaining;

        writer.writeByteStringPrefix(chunk_len);

        var i: u32 = 0;
        while (i < chunk_len) : (i += 1) {
            const idx = offset + i;
            writer.writeByte(@intCast(bytes_val.bytes[idx]));
        }

        offset += chunk_len;
    }

    writer.writeBreak();
}

fn encodeIntegerAsCbor(writer: *CborWriter, heap: *Heap, int_val: BigInt) void {
    const used_words = significantWordCount(int_val.words, int_val.length);

    if (int_val.sign == 0) {
        if (used_words == 0) {
            writer.writeUnsigned(0);
            return;
        }

        if (wordsToU64(int_val.words, used_words)) |value| {
            writer.writeUnsigned(value);
            return;
        }

        writeBigPositive(writer, int_val.words, used_words);
        return;
    }

    if (used_words == 0) {
        builtinEvaluationFailure();
    }

    if (wordsToU64(int_val.words, used_words)) |value| {
        if (value == 0) builtinEvaluationFailure();
        writer.writeNegative(value - 1);
        return;
    }

    writeBigNegative(writer, heap, int_val.words, used_words);
}

fn writeBigPositive(writer: *CborWriter, words: [*]const u32, used_words: u32) void {
    const byte_len = magnitudeByteLen(words, used_words);
    if (byte_len == 0) {
        writer.writeUnsigned(0);
        return;
    }

    writer.writeTag(2);
    writeMagnitudeByteString(writer, words, used_words, byte_len);
}

fn writeBigNegative(
    writer: *CborWriter,
    heap: *Heap,
    words: [*]const u32,
    used_words: u32,
) void {
    const tmp = heap.createArray(u32, used_words);
    var i: u32 = 0;
    while (i < used_words) : (i += 1) {
        tmp[i] = words[i];
    }

    subtractOneLittleEndian(tmp, used_words);
    const trimmed = significantWordCount(tmp, used_words);
    if (trimmed == 0) builtinEvaluationFailure();

    writer.writeTag(3);
    const byte_len = magnitudeByteLen(tmp, trimmed);
    writeMagnitudeByteString(writer, tmp, trimmed, byte_len);
    heap.reclaimHeap(u32, used_words);
}

fn subtractOneLittleEndian(words: [*]u32, len: u32) void {
    var idx: u32 = 0;
    while (idx < len) : (idx += 1) {
        if (words[idx] > 0) {
            words[idx] -= 1;
            return;
        }
        words[idx] = 0xFFFF_FFFF;
    }
}

fn significantWordCount(words: [*]const u32, len: u32) u32 {
    var count = len;
    while (count > 0) {
        if (words[count - 1] != 0) {
            return count;
        }
        count -= 1;
    }
    return 0;
}

fn wordsToU64(words: [*]const u32, used_words: u32) ?u64 {
    if (used_words == 0) return 0;
    if (used_words > 2) return null;

    var value: u64 = 0;
    var idx = used_words;
    while (idx > 0) {
        idx -= 1;
        value <<= 32;
        value |= @as(u64, words[idx]);
    }
    return value;
}

fn magnitudeByteLen(words: [*]const u32, used_words: u32) u32 {
    if (used_words == 0) return 0;

    const ms_word = words[used_words - 1];
    var bytes: u32 = 4;
    if ((ms_word & 0xFF000000) == 0) {
        bytes -= 1;
        if ((ms_word & 0x00FF0000) == 0) {
            bytes -= 1;
            if ((ms_word & 0x0000FF00) == 0) {
                bytes -= 1;
            }
        }
    }

    return (used_words - 1) * 4 + bytes;
}

fn writeMagnitudeByteString(
    writer: *CborWriter,
    words: [*]const u32,
    used_words: u32,
    byte_len: u32,
) void {
    if (used_words == 0 or byte_len == 0) builtinEvaluationFailure();

    var iter = MagnitudeByteIterator.init(words, used_words);

    writer.writeByteStringPrefix(byte_len);
    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        if (iter.next()) |byte_value| {
            writer.writeByte(byte_value);
        } else {
            builtinEvaluationFailure();
        }
    }
}

const MagnitudeByteIterator = struct {
    words: [*]const u32,
    word_index: i32,
    byte_index: i32,
    started: bool,

    fn init(words: [*]const u32, used_words: u32) MagnitudeByteIterator {
        return .{
            .words = words,
            .word_index = @as(i32, @intCast(used_words)) - 1,
            .byte_index = 3,
            .started = false,
        };
    }

    fn next(self: *MagnitudeByteIterator) ?u8 {
        while (self.word_index >= 0) {
            if (self.byte_index < 0) {
                self.word_index -= 1;
                self.byte_index = 3;
                continue;
            }

            const word = self.words[@intCast(self.word_index)];
            const shift_amt: u5 = @intCast(self.byte_index * 8);
            const byte_value: u8 = @intCast((word >> shift_amt) & 0xFF);
            self.byte_index -= 1;

            if (!self.started) {
                if (byte_value == 0) continue;
                self.started = true;
            }

            if (self.started) {
                return byte_value;
            }
        }
        return null;
    }
};

const CborWriter = struct {
    bytes: [*]u8,
    capacity: u32,
    offset: u32,

    fn init(buffer_words: [*]u32, len_words: u32) CborWriter {
        return .{
            .bytes = @ptrCast(buffer_words),
            .capacity = len_words * 4,
            .offset = 0,
        };
    }

    fn writeByte(self: *CborWriter, value: u8) void {
        if (self.offset >= self.capacity) {
            builtinEvaluationFailure();
        }

        const idx: usize = @intCast(self.offset);
        self.bytes[idx] = value;
        self.offset += 1;
    }

    fn writeByteStringPrefix(self: *CborWriter, len: u32) void {
        self.writeMajorType(2, @intCast(len));
    }

    fn writeBreak(self: *CborWriter) void {
        self.writeByte(0xff);
    }

    fn writeArray(self: *CborWriter, len: u32) void {
        self.writeMajorType(4, @intCast(len));
    }

    fn beginIndefiniteArray(self: *CborWriter) void {
        self.writeByte(0x9f);
    }

    fn beginIndefiniteByteString(self: *CborWriter) void {
        self.writeByte(0x5f);
    }

    fn writeMap(self: *CborWriter, len: u32) void {
        self.writeMajorType(5, @intCast(len));
    }

    fn writeTag(self: *CborWriter, tag: u64) void {
        self.writeMajorType(6, tag);
    }

    fn writeUnsigned(self: *CborWriter, value: u64) void {
        self.writeMajorType(0, value);
    }

    fn writeNegative(self: *CborWriter, encoded: u64) void {
        self.writeMajorType(1, encoded);
    }

    fn writeMajorType(self: *CborWriter, major: u8, value: u64) void {
        const prefix: u8 = (major << 5);

        if (value < 24) {
            const small: u8 = @intCast(value);
            self.writeByte(prefix | small);
        } else if (value <= 0xFF) {
            self.writeByte(prefix | 24);
            self.writeByte(@intCast(value));
        } else if (value <= 0xFFFF) {
            self.writeByte(prefix | 25);
            self.writeByte(@intCast((value >> 8) & 0xFF));
            self.writeByte(@intCast(value & 0xFF));
        } else if (value <= 0xFFFF_FFFF) {
            self.writeByte(prefix | 26);
            self.writeU32(@intCast(value));
        } else {
            self.writeByte(prefix | 27);
            self.writeU64(value);
        }
    }

    fn writeU32(self: *CborWriter, value: u32) void {
        var shift: i32 = 24;
        while (shift >= 0) : (shift -= 8) {
            const shift_amt: u5 = @intCast(shift);
            self.writeByte(@intCast((value >> shift_amt) & 0xFF));
        }
    }

    fn writeU64(self: *CborWriter, value: u64) void {
        var shift: i32 = 56;
        while (shift >= 0) : (shift -= 8) {
            const shift_amt: u6 = @intCast(shift);
            self.writeByte(@intCast((value >> shift_amt) & 0xFF));
        }
    }
};

fn copyPackedWordsToUnpacked(dst: [*]u32, src_words: [*]const u32, byte_len: u32) void {
    if (byte_len == 0) return;
    const src_bytes: [*]const u8 = @ptrCast(src_words);

    var i: u32 = 0;
    while (i < byte_len) : (i += 1) {
        const idx: usize = @intCast(i);
        dst[idx] = src_bytes[idx];
    }
}

