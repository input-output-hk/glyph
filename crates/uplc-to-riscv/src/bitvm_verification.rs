//! BitVMX Verification Support
//!
//! This module provides functionality for generating execution traces and hash chains
//! for BitVMX's challenge-response verification mechanism.
//! It aligns with BitVMX-CPU's trace module for compatibility.

use std::collections::HashMap;
use sha2::{Digest, Sha256};
use bitvm_common::{
    verification::{
        generate_initial_hash, generate_step_hash, hash_to_hex_string,
        create_verification_script, create_verification_script_mapping
    },
    trace::{ExecutionTrace as BitVMCommonExecutionTrace, ExecutionStep as BitVMCommonExecutionStep},
};

/// Execution trace for BitVMX verification
/// 
/// This is a wrapper around bitvm_common::trace::ExecutionTrace
/// providing additional verification-specific functionality.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// The underlying trace from bitvm-common
    pub inner: BitVMCommonExecutionTrace,
}

impl ExecutionTrace {
    /// Create a new execution trace
    pub fn new() -> Self {
        Self {
            inner: BitVMCommonExecutionTrace::new(),
        }
    }
    
    /// Add an execution step to the trace
    pub fn add_step(&mut self, step: BitVMCommonExecutionStep) {
        self.inner.add_step(step);
    }
    
    /// Get the number of steps in the trace
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    
    /// Check if the trace is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    
    /// Convert the execution trace to CSV format
    pub fn to_csv(&self) -> String {
        self.inner.to_csv()
    }
    
    /// Generate a verification proof for a specific step
    pub fn generate_proof(&self, step_index: usize) -> Option<VerificationProof> {
        if step_index >= self.inner.len() {
            return None;
        }
        
        let step = self.inner.steps.get(step_index)?.clone();
        let hash = step.compute_hash();
        
        Some(VerificationProof {
            step,
            hash,
            step_index,
        })
    }
}

/// Execution step in a BitVMX trace, aligned with BitVMX-CPU's TraceRWStep
/// 
/// This is simply a re-export of bitvm_common::trace::ExecutionStep
pub type ExecutionStep = BitVMCommonExecutionStep;

/// Verification proof for a specific execution step
#[derive(Debug, Clone)]
pub struct VerificationProof {
    /// The execution step
    pub step: ExecutionStep,
    
    /// Hash of the step
    #[allow(dead_code)]
    pub hash: [u8; 32],
    
    /// Index of the step in the trace
    pub step_index: usize,
}

/// Dispute resolver for BitVMX verification
#[derive(Debug)]
pub struct DisputeResolver {
    /// The execution trace
    trace: ExecutionTrace,
}

impl DisputeResolver {
    /// Create a new dispute resolver
    pub fn new(trace: ExecutionTrace) -> Self {
        Self {
            trace,
        }
    }
    
    /// Respond to a challenge
    pub fn respond_to_challenge(&self, challenge: Challenge) -> Response {
        match challenge {
            Challenge::StepExecution { step_index } => {
                if let Some(proof) = self.trace.generate_proof(step_index) {
                    Response::StepProof(proof)
                } else {
                    Response::Error(format!("Invalid step index: {}", step_index))
                }
            }
            Challenge::MemoryValue { address, step_index } => {
                if step_index >= self.trace.len() {
                    return Response::Error(format!("Invalid step index: {}", step_index));
                }
                
                // Find the last write to this address before or at the given step
                let mut value = 0;
                let mut last_step = 0;
                
                for (i, step) in self.trace.inner.steps.iter().enumerate() {
                    if i > step_index {
                        break;
                    }
                    
                    if let (Some(addr), Some(val)) = (step.write_addr, step.write_value) {
                        if addr == address {
                            value = val;
                            last_step = i;
                        }
                    }
                }
                
                Response::MemoryValue {
                    address,
                    value,
                    step_index: last_step,
                }
            }
            Challenge::HashChain { start_index, end_index } => {
                if start_index >= self.trace.len() || end_index >= self.trace.len() || start_index > end_index {
                    return Response::Error(format!("Invalid index range: {} to {}", start_index, end_index));
                }
                
                let hashes = self.trace.inner.steps[start_index..=end_index]
                    .iter()
                    .map(|step| step.compute_hash())
                    .collect();
                
                Response::HashChain {
                    hashes,
                    start_index,
                    end_index,
                }
            }
        }
    }
}

/// Challenge for BitVMX verification
#[derive(Debug)]
pub enum Challenge {
    /// Challenge the execution of a specific step
    StepExecution {
        step_index: usize,
    },
    
    /// Challenge the value of a memory location at a specific step
    MemoryValue {
        address: u32,
        step_index: usize,
    },
    
    /// Challenge a range of the hash chain
    HashChain {
        start_index: usize,
        end_index: usize,
    },
}

/// Response to a BitVMX verification challenge
#[derive(Debug)]
pub enum Response {
    /// Proof of a specific execution step
    StepProof(VerificationProof),
    
    /// Value of a memory location at a specific step
    MemoryValue {
        address: u32,
        value: u32,
        step_index: usize,
    },
    
    /// Range of the hash chain
    HashChain {
        #[allow(dead_code)]
        hashes: Vec<[u8; 32]>,
        #[allow(dead_code)]
        start_index: usize,
        #[allow(dead_code)]
        end_index: usize,
    },
    
    /// Error response
    Error(String),
}

/// Generate the initial hash using bitvm-common (deprecated)
#[deprecated(
    since = "0.1.0",
    note = "Use bitvm_common::verification::generate_initial_hash directly"
)]
pub fn get_initial_hash() -> Vec<u8> {
    bitvm_common::verification::generate_initial_hash()
}

/// Generate a step hash using bitvm-common (deprecated)
#[deprecated(
    since = "0.1.0",
    note = "Use bitvm_common::verification::generate_step_hash directly"
)]
pub fn get_step_hash(previous_hash: &[u8], data: &[u8]) -> [u8; 32] {
    bitvm_common::verification::generate_step_hash(previous_hash, data)
}

/// Convert a hash to hex using bitvm-common (deprecated)
#[deprecated(
    since = "0.1.0",
    note = "Use bitvm_common::verification::hash_to_hex_string directly"
)]
pub fn get_hash_hex(hash: &[u8]) -> String {
    bitvm_common::verification::hash_to_hex_string(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dispute_resolver() {
        let mut trace = ExecutionTrace::new();
        
        let step1 = ExecutionStep::new(0x1000, 0x1000, 0x12345678, 0x1004, 0)
            .with_read1(0x2000, 0xAABBCCDD, 0)
            .with_write(0x3000, 0x55667788);
        
        let step2 = ExecutionStep::new(0x1004, 0x1004, 0x87654321, 0x1008, 1)
            .with_read1(0x3000, 0x55667788, 0)
            .with_read2(0x2004, 0x11223344, 0)
            .with_write(0x3004, 0x99AABBCC);
        
        trace.add_step(step1);
        trace.add_step(step2);
        
        // Create a dispute resolver
        let resolver = DisputeResolver::new(trace);
        
        // Verify that the resolver has the correct number of steps
        assert_eq!(resolver.trace.len(), 2);
        
        // Test responding to a challenge
        let challenge = Challenge::StepExecution { step_index: 0 };
        let response = resolver.respond_to_challenge(challenge);
        
        match response {
            Response::StepProof(proof) => {
                assert_eq!(proof.step_index, 0);
                assert_eq!(proof.step.pc, 0x1000);
            }
            _ => panic!("Expected StepProof response"),
        }
    }

    #[test]
    fn test_verification_script() {
        // Test creating a verification script
        let script = create_verification_script(0x1000, 0x12345678);
        assert!(script.is_some());
        
        // Check that the script is not empty
        let script_str = script.unwrap();
        assert!(!script_str.is_empty());
        
        // Test creating a verification script mapping
        let mapping = create_verification_script_mapping(0x10000000);
        assert!(!mapping.is_empty());
        
        // Check that some common instructions are in the mapping
        assert!(mapping.contains_key("add"));
        assert!(mapping.contains_key("sub"));
        assert!(mapping.contains_key("beq"));
    }
} 