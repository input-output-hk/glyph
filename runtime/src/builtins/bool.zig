const std = @import("std");
const utils = @import("../utils.zig");
const runtime_value = @import("../value.zig");
const BuiltinContext = runtime_value.BuiltinContext;
const LinkedValues = runtime_value.LinkedValues;
const Value = runtime_value.Value;

inline fn builtinEvaluationFailure() noreturn {
    utils.exit(std.math.maxInt(u32));
}

// Bool function
pub fn ifThenElse(_: *BuiltinContext, args: *LinkedValues) *const Value {
    const otherwise = args.value;
    const then = args.next.?.value;
    const cond_value = args.next.?.next.?.value;

    // Avoid unwrapBool so invalid inputs trigger a silent evaluation failure
    // instead of touching the (unimplemented) debug console.
    const cond = switch (cond_value.*) {
        .constant => |c| switch (c.constType().*) {
            .boolean => c.bln(),
            else => builtinEvaluationFailure(),
        },
        else => builtinEvaluationFailure(),
    };

    return if (cond) then else otherwise;
}

// Unit function
// Our LinkedValues list stores the most-recent argument at the head, so for
// `chooseUnit` the branch argument sits in `args.value` and the unit value is
// at `args.next?.value`.
pub fn chooseUnit(_: *BuiltinContext, args: *LinkedValues) *const Value {
    const then = args.value;
    const unit_node = args.next orelse {
        utils.printlnString("chooseUnit expects a unit argument");
        utils.exit(std.math.maxInt(u32));
    };

    // The scrutinee must still be an actual unit value, even though the branch
    // result is returned unchanged.  `unwrapUnit` enforces that without forcing
    // any structure on the branch argument itself (which may be non-constant).
    _ = unit_node.value.unwrapUnit();

    return then;
}

// Tracing function
pub fn trace(_: *BuiltinContext, args: *LinkedValues) *const Value {
    const then = args.value;
    const msg = args.next.?.value.unwrapString();

    // The Plutus trace builtin is effectful only through the surrounding
    // execution environment, so on the CEK we simply force the string argument
    // (to preserve error behaviour) and return the runtime_value.  Writing to the
    // original memory-mapped console at 0xA000_1000 causes `SectionNotFound`
    // errors under the conformance runner, which doesn't emulate that device.
    _ = msg;

    return then;
}
