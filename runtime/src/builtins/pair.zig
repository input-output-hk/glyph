const std = @import("std");
const expr = @import("../expr.zig");
const Constant = expr.Constant;
const runtime_value = @import("../value.zig");

const BuiltinContext = runtime_value.BuiltinContext;
const LinkedValues = runtime_value.LinkedValues;
const Value = runtime_value.Value;
const createConst = runtime_value.createConst;

// Pairs functions
pub fn fstPair(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const pair = args.value.unwrapPair();
    const c = Constant{
        .length = pair.first_type_len,
        .type_list = pair.first_type,
        .value = pair.first_value,
    };
    const result = m.heap.create(Constant, &c);
    return createConst(m.heap, result);
}

pub fn sndPair(m: *BuiltinContext, args: *LinkedValues) *const Value {
    const pair = args.value.unwrapPair();
    const c = Constant{
        .length = pair.second_type_len,
        .type_list = pair.second_type,
        .value = pair.second_value,
    };
    const result = m.heap.create(Constant, &c);
    return createConst(m.heap, result);
}
