//! IR Optimization
//!
//! This module provides optimization passes for the IR.

use crate::ir::IRInstr;
use uplc::builtins::DefaultFunction as BuiltinFn;

/// Optimization level for the compiler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Default optimization level
    Default,
    /// Maximum optimization level
    Maximum,
}

/// Optimize IR instructions
#[allow(dead_code)]
pub fn optimize_ir(instructions: &[IRInstr]) -> Vec<IRInstr> {
    let mut optimized = instructions.to_vec();
            
    // Apply BitVMX-compatible optimizations
    optimized = constant_folding(&optimized);
            
    optimized
}

/// Constant folding optimization
///
/// This optimization evaluates constant expressions at compile time.
#[allow(dead_code)]
fn constant_folding(instructions: &[IRInstr]) -> Vec<IRInstr> {
    let mut result = Vec::new();
    let mut i = 0;
    
    while i < instructions.len() {
        // Check for patterns that can be constant-folded
        if i + 2 < instructions.len() {
            match (&instructions[i], &instructions[i+1], &instructions[i+2]) {
                (IRInstr::PushConst(a), IRInstr::PushConst(b), IRInstr::CallBuiltin(builtin)) => {
                    // Constant fold binary operations on integers
                    match builtin {
                        BuiltinFn::AddInteger => {
                            result.push(IRInstr::PushConst(a + b));
                            i += 3;
                            continue;
                        }
                        BuiltinFn::SubtractInteger => {
                            result.push(IRInstr::PushConst(a - b));
                            i += 3;
                            continue;
                        }
                        BuiltinFn::MultiplyInteger => {
                            result.push(IRInstr::PushConst(a * b));
                            i += 3;
                            continue;
                        }
                        BuiltinFn::DivideInteger => {
                            if *b != 0 {
                                result.push(IRInstr::PushConst(a / b));
                                i += 3;
                                continue;
                            }
                        }
                        BuiltinFn::RemainderInteger => {
                            if *b != 0 {
                                result.push(IRInstr::PushConst(a % b));
                                i += 3;
                                continue;
                            }
                        }
                        BuiltinFn::LessThanInteger => {
                            result.push(IRInstr::PushBool(a < b));
                            i += 3;
                            continue;
                        }
                        BuiltinFn::LessThanEqualsInteger => {
                            result.push(IRInstr::PushBool(a <= b));
                            i += 3;
                            continue;
                        }
                        BuiltinFn::EqualsInteger => {
                            result.push(IRInstr::PushBool(a == b));
                            i += 3;
                            continue;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        
        // If no pattern matched, keep the instruction as is
        result.push(instructions[i].clone());
        i += 1;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constant_folding() {
        let instructions = vec![
            IRInstr::PushConst(5),
            IRInstr::PushConst(3),
            IRInstr::CallBuiltin(BuiltinFn::AddInteger),
        ];
        
        let optimized = constant_folding(&instructions);
        
        assert_eq!(optimized.len(), 1);
        match &optimized[0] {
            IRInstr::PushConst(value) => assert_eq!(*value, 8),
            _ => panic!("Expected PushConst instruction"),
        }
    }
} 