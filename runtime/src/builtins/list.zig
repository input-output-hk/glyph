const std = @import("std");
const expr = @import("../expr.zig");
const utils = @import("../utils.zig");
const runtime_value = @import("../value.zig");
const ConstantType = expr.ConstantType;
const Constant = expr.Constant;
const ListNode = expr.ListNode;

const BuiltinContext = runtime_value.BuiltinContext;
const LinkedValues = runtime_value.LinkedValues;
const Value = runtime_value.Value;
const createConst = runtime_value.createConst;

inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
}

// List functions
// In Plutus the builtin receives arguments in the order (list, nil_branch, cons_branch),
// but our LinkedValues list is reversed (last applied argument first), so `args.value`
// corresponds to the cons branch and `args.next?.value` to the nil branch.
pub fn chooseList(_: *BuiltinContext, args: *LinkedValues) *const Value {
    const otherwise = args.value;
    const then = args.next.?.value;
    const list = args.next.?.next.?.value.unwrapList();

    if (list.length > 0) {
        return otherwise;
    } else {
        return then;
    }
}

pub fn mkCons(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value;
    const list = y.unwrapList();
    const yType = y.constant;
    const newItem = args.next.?.value.unwrapConstant();

    if (newItem.matchingTypes(list.inner_type, list.type_length)) {
        const prev = if (list.length > 0) blk: {
            break :blk list.items.?;
        } else blk: {
            break :blk null;
        };

        const node = m.heap.create(ListNode, &ListNode{ .value = newItem.rawValue(), .next = prev });

        // (Type length + 1) * 4 bytes + 4 byte to hold type length + 4 byte for list length + 4 byte for pointer to first list item (or null)
        var result = m.heap.createArray(u32, 5);

        result[0] = yType.length;
        result[1] = @intFromPtr(yType.type_list);
        result[2] = @intFromPtr(result + 3);
        result[3] = list.length + 1;
        result[4] = @intFromPtr(node);

        return createConst(m.heap, @ptrCast(result));
    }

    // mkCons is partial: mismatched element types must signal an evaluation failure.
    builtinEvaluationFailure();
}

pub fn headList(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value;
    const list = y.unwrapList();

    if (list.length == 0) {
        // Per the spec, headList is partial and must signal an evaluation failure on empty lists.
        builtinEvaluationFailure();
    }

    const c = Constant{
        .length = list.type_length,
        .type_list = list.inner_type,
        .value = list.items.?.value,
    };

    const con = m.heap.create(Constant, &c);

    return createConst(m.heap, con);
}

pub fn tailList(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const y = args.value;
    const list_const = y.unwrapConstant();
    const list = list_const.list();

    if (list.length == 0) {
        // tailList is partial per the spec; empty lists must fail.
        builtinEvaluationFailure();
    }

    const result = m.heap.createArray(u32, 2);
    result[0] = list.length - 1;
    result[1] = @intFromPtr(list.items.?.next);

    const c = Constant{
        .length = list_const.length,
        .type_list = list_const.type_list,
        .value = @intFromPtr(result),
    };

    const con = m.heap.create(Constant, &c);

    return createConst(m.heap, con);
}

pub fn nullList(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const list = args.value.unwrapList();

    var result = m.heap.createArray(u32, 4);

    result[0] = 1;
    result[1] = @intFromPtr(ConstantType.booleanType());
    result[2] = @intFromPtr(result + 3);
    result[3] = @intFromBool(list.length == 0);

    return createConst(m.heap, @ptrCast(result));
}
