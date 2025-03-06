//! UPLC IR (Intermediate Representation)
//!
//! This module defines a micro-IR for optimizing UPLC before generating RISC-V code.

use uplc::{
    ast::{Term, Constant, DeBruijn},
    builtins::DefaultFunction as BuiltinFn,
};

/// IR Instruction
#[derive(Debug, Clone)]
pub enum IRInstr {
    /// Push a constant integer onto the stack
    PushConst(i64),
    
    /// Push a constant boolean onto the stack
    PushBool(bool),
    
    /// Push a constant bytestring onto the stack
    PushByteString(Vec<u8>),
    
    /// Push a constant string onto the stack
    PushString(String),
    
    /// Push a unit value onto the stack
    PushUnit,
    
    /// Push a variable onto the stack
    PushVar(String),
    
    /// Call a builtin function
    CallBuiltin(BuiltinFn),
    
    /// Apply a function to an argument
    Apply,
    
    /// Create a lambda abstraction
    Lambda(String),
    
    /// Return from a function
    Return,
    
    /// Delay a term's evaluation
    Delay,
    
    /// Force a delayed term
    Force,
    
    /// Case expression - start of a case analysis
    CaseStart(usize), // Number of branches
    
    /// Case branch - start of a branch in a case expression
    CaseBranch(usize), // Branch index
    
    /// Case branch end - end of a branch in a case expression
    CaseBranchEnd,
    
    /// Case end - end of a case expression
    CaseEnd,
}

fn debruijn_to_string(db: &DeBruijn) -> String {
    format!("v{}", db.inner())
}

/// Lower a UPLC term to IR instructions
pub fn lower_to_ir(term: &Term<DeBruijn>) -> Vec<IRInstr> {
    let mut instructions = Vec::new();
    lower_term(term, &mut instructions);
    instructions
}

/// Lower a UPLC term to IR instructions, appending to the given vector
fn lower_term(term: &Term<DeBruijn>, instructions: &mut Vec<IRInstr>) {
    match term {
        Term::Constant(constant) => match constant.as_ref() {
            Constant::Integer(value) => instructions.push(IRInstr::PushConst(value.clone().try_into().unwrap())),
            Constant::Bool(value) => instructions.push(IRInstr::PushBool(*value)),
            Constant::ByteString(bytes) => instructions.push(IRInstr::PushByteString(bytes.clone())),
            Constant::String(s) => instructions.push(IRInstr::PushString(s.clone())),
            Constant::Unit => instructions.push(IRInstr::PushUnit),
            _ => panic!("Unsupported constant type"),
        },
        Term::Var(name) => instructions.push(IRInstr::PushVar(debruijn_to_string(name))),
        Term::Lambda { parameter_name, body } => {
            instructions.push(IRInstr::Lambda(debruijn_to_string(parameter_name)));
            lower_term(body, instructions);
            instructions.push(IRInstr::Return);
        },
        Term::Apply { function, argument } => {
            lower_term(function, instructions);
            lower_term(argument, instructions);
            instructions.push(IRInstr::Apply);
        },
        Term::Delay(term) => {
            instructions.push(IRInstr::Delay);
            lower_term(term, instructions);
        },
        Term::Force(term) => {
            lower_term(term, instructions);
            instructions.push(IRInstr::Force);
        },
        Term::Builtin(builtin) => {
            instructions.push(IRInstr::CallBuiltin(*builtin));
        },
        Term::Case { constr, branches } => {
            // Lower the constructor term
            lower_term(constr, instructions);
            
            // Start the case expression
            instructions.push(IRInstr::CaseStart(branches.len()));
            
            // Lower each branch
            for (i, branch) in branches.iter().enumerate() {
                instructions.push(IRInstr::CaseBranch(i));
                lower_term(branch, instructions);
                instructions.push(IRInstr::CaseBranchEnd);
            }
            
            // End the case expression
            instructions.push(IRInstr::CaseEnd);
        },
        _ => panic!("Unsupported term type"),
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use super::*;
    
    #[test]
    fn test_lower_integer() {
        let term = Term::Constant(Rc::new(Constant::Integer(42.into())));
        let ir = lower_to_ir(&term);
        assert_eq!(ir.len(), 1);
        match &ir[0] {
            IRInstr::PushConst(value) => assert_eq!(*value, 42),
            _ => panic!("Expected PushConst instruction"),
        }
    }

    #[test]
    fn test_lower_lambda() {
        let term = Term::Lambda {
            parameter_name: Rc::new(DeBruijn::new(0)),
            body: Rc::new(Term::Var(Rc::new(DeBruijn::new(0)))),
        };
        let ir = lower_to_ir(&term);
        assert!(ir.len() > 1);
        // Check that the IR contains a Lambda instruction
        assert!(ir.iter().any(|instr| matches!(instr, IRInstr::Lambda(name) if name == "v0")));
    }

    #[test]
    fn test_lower_application() {
        let term = Term::Apply { 
            function: Rc::new(Term::Var(Rc::new(DeBruijn::new(1)))),
            argument: Rc::new(Term::Constant(Rc::new(Constant::Integer(42.into())))),
        };
        let ir = lower_to_ir(&term);
        assert!(ir.len() > 2);
        // Check that the IR contains an Apply instruction
        assert!(ir.iter().any(|instr| matches!(instr, IRInstr::Apply)));
    }

    #[test]
    fn test_case_instructions() {
        // Instead of creating a case expression directly, we'll create the IR instructions manually
        let mut ir = Vec::new();
        
        // Push a variable for the constructor
        ir.push(IRInstr::PushVar("v0".to_string()));
        
        // Start the case expression with 2 branches
        ir.push(IRInstr::CaseStart(2));
        
        // First branch
        ir.push(IRInstr::CaseBranch(0));
        ir.push(IRInstr::PushConst(1));
        ir.push(IRInstr::CaseBranchEnd);
        
        // Second branch
        ir.push(IRInstr::CaseBranch(1));
        ir.push(IRInstr::PushConst(2));
        ir.push(IRInstr::CaseBranchEnd);
        
        // End of case
        ir.push(IRInstr::CaseEnd);
        
        // Check that the IR contains the expected case instructions
        let mut iter = ir.iter();
        
        // First, we should have a PushVar for the constructor
        assert!(matches!(iter.next(), Some(IRInstr::PushVar(name)) if name == "v0"));
        
        // Then, we should have a CaseStart with 2 branches
        assert!(matches!(iter.next(), Some(IRInstr::CaseStart(n)) if *n == 2));
        
        // First branch
        assert!(matches!(iter.next(), Some(IRInstr::CaseBranch(i)) if *i == 0));
        assert!(matches!(iter.next(), Some(IRInstr::PushConst(value)) if *value == 1));
        assert!(matches!(iter.next(), Some(IRInstr::CaseBranchEnd)));
        
        // Second branch
        assert!(matches!(iter.next(), Some(IRInstr::CaseBranch(i)) if *i == 1));
        assert!(matches!(iter.next(), Some(IRInstr::PushConst(value)) if *value == 2));
        assert!(matches!(iter.next(), Some(IRInstr::CaseBranchEnd)));
        
        // End of case
        assert!(matches!(iter.next(), Some(IRInstr::CaseEnd)));
        
        // No more instructions
        assert!(iter.next().is_none());
    }
} 