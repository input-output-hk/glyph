//! BitVMX Verification Support
//!
//! This module provides functionality for generating execution traces and hash chains
//! for BitVMX's challenge-response verification mechanism.

use std::collections::HashMap;
use sha2::{Digest, Sha256};

/// Execution step in a BitVMX trace
#[derive(Debug, Clone)]
pub struct ExecutionStep {
    /// Program counter
    #[allow(dead_code)]
    pub pc: u32,
    
    /// First memory address being read
    pub read_addr1: Option<u32>,
    
    /// First memory value being read
    pub read_value1: Option<u32>,
    
    /// Last step when first memory was modified
    pub read_last_step1: Option<u64>,
    
    /// Second memory address being read
    pub read_addr2: Option<u32>,
    
    /// Second memory value being read
    pub read_value2: Option<u32>,
    
    /// Last step when second memory was modified
    pub read_last_step2: Option<u64>,
    
    /// Opcode memory address
    #[allow(dead_code)]
    pub opcode_addr: u32,
    
    /// Opcode value
    #[allow(dead_code)]
    pub opcode_value: u32,
    
    /// Memory address being written
    pub write_addr: Option<u32>,
    
    /// Memory value being written
    pub write_value: Option<u32>,
    
    /// Next program counter value
    #[allow(dead_code)]
    pub next_pc: u32,
    
    /// Micro-instruction step
    pub micro: u8,
    
    /// Hash of the previous step (for hash chain)
    pub prev_hash: Option<[u8; 32]>,
    
    /// Step number in the execution trace
    pub step: u64,
}

impl ExecutionStep {
    /// Create a new execution step
    pub fn new(pc: u32, opcode_addr: u32, opcode_value: u32, next_pc: u32, step: u64) -> Self {
        Self {
            pc,
            read_addr1: None,
            read_value1: None,
            read_last_step1: None,
            read_addr2: None,
            read_value2: None,
            read_last_step2: None,
            opcode_addr,
            opcode_value,
            write_addr: None,
            write_value: None,
            next_pc,
            micro: 0,
            prev_hash: None,
            step,
        }
    }
    
    /// Set the first memory read
    pub fn with_read1(mut self, addr: u32, value: u32, last_step: u64) -> Self {
        self.read_addr1 = Some(addr);
        self.read_value1 = Some(value);
        self.read_last_step1 = Some(last_step);
        self
    }
    
    /// Set the second memory read
    pub fn with_read2(mut self, addr: u32, value: u32, last_step: u64) -> Self {
        self.read_addr2 = Some(addr);
        self.read_value2 = Some(value);
        self.read_last_step2 = Some(last_step);
        self
    }
    
    /// Set the memory write
    pub fn with_write(mut self, addr: u32, value: u32) -> Self {
        self.write_addr = Some(addr);
        self.write_value = Some(value);
        self
    }
    
    /// Set the previous hash
    pub fn with_prev_hash(mut self, hash: [u8; 32]) -> Self {
        self.prev_hash = Some(hash);
        self
    }
    
    /// Set the micro-instruction step
    pub fn with_micro(mut self, micro: u8) -> Self {
        self.micro = micro;
        self
    }
    
    /// Convert the execution step to a byte vector for hashing
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Add write address and value
        if let Some(addr) = self.write_addr {
            bytes.extend_from_slice(&addr.to_be_bytes());
        } else {
            bytes.extend_from_slice(&0u32.to_be_bytes());
        }
        
        if let Some(value) = self.write_value {
            bytes.extend_from_slice(&value.to_be_bytes());
        } else {
            bytes.extend_from_slice(&0u32.to_be_bytes());
        }
        
        // Add next PC and micro
        bytes.extend_from_slice(&self.next_pc.to_be_bytes());
        bytes.push(self.micro);
        
        bytes
    }
    
    /// Compute the hash of this step
    pub fn compute_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        
        // If there's a previous hash, include it
        if let Some(prev_hash) = self.prev_hash {
            hasher.update(prev_hash);
        } else {
            // Use initial hash for the first step
            hasher.update(generate_initial_hash());
        }
        
        // Add the step data
        hasher.update(self.to_bytes());
        
        // Finalize and return the hash
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result[..]);
        hash
    }
    
    /// Format the execution step as a CSV row
    pub fn to_csv(&self) -> String {
        format!(
            "{};{};{};{};{};{};{};{};{};{};{};{};{}",
            self.read_addr1.unwrap_or(0),
            self.read_value1.unwrap_or(0),
            self.read_last_step1.unwrap_or(0),
            self.read_addr2.unwrap_or(0),
            self.read_value2.unwrap_or(0),
            self.read_last_step2.unwrap_or(0),
            self.pc,
            self.micro,
            self.opcode_value,
            self.write_addr.unwrap_or(0),
            self.write_value.unwrap_or(0),
            self.next_pc,
            self.micro
        )
    }
}

/// Generate the initial hash for the hash chain
pub fn generate_initial_hash() -> Vec<u8> {
    // Convert "ff" to bytes as in BitVMX
    let initial_bytes = hex::decode("ff").expect("Invalid hex string");
    
    // Compute the SHA-256 hash
    let mut hasher = Sha256::new();
    hasher.update(initial_bytes);
    hasher.finalize().to_vec()
}

/// Execution trace containing a sequence of execution steps
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Execution steps
    pub steps: Vec<ExecutionStep>,
    
    /// Memory state (address -> (value, last_modified_step))
    pub memory: HashMap<u32, (u32, u64)>,
    
    /// Current step number
    pub current_step: u64,
}

impl ExecutionTrace {
    /// Create a new execution trace
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            memory: HashMap::new(),
            current_step: 0,
        }
    }
    
    /// Add a step to the execution trace
    pub fn add_step(&mut self, mut step: ExecutionStep) {
        // Update memory state for writes
        if let Some(addr) = step.write_addr {
            if let Some(value) = step.write_value {
                self.memory.insert(addr, (value, self.current_step));
            }
        }
        
        // Compute hash of the previous step
        if !self.steps.is_empty() {
            let prev_hash = self.steps.last().unwrap().compute_hash();
            step = step.with_prev_hash(prev_hash);
        }
        
        // Add step to trace
        self.steps.push(step);
        self.current_step += 1;
    }
    
    /// Get the last modified step for a memory address
    pub fn get_last_modified_step(&self, addr: u32) -> u64 {
        self.memory.get(&addr).map(|(_, step)| *step).unwrap_or(0)
    }
    
    /// Get the value of a memory address
    pub fn get_memory_value(&self, addr: u32) -> u32 {
        self.memory.get(&addr).map(|(value, _)| *value).unwrap_or(0)
    }
    
    /// Convert the execution trace to CSV format
    pub fn to_csv(&self) -> String {
        let mut result = String::new();
        result.push_str("step,pc,opcode_addr,opcode_value,read_addr1,read_value1,read_addr2,read_value2,write_addr,write_value,next_pc,micro,hash\n");
        for step in &self.steps {
            result.push_str(&step.to_csv());
            result.push('\n');
        }
        result
    }
    
    /// Get the final hash of the execution trace
    pub fn get_final_hash(&self) -> Option<[u8; 32]> {
        self.steps.last().map(|step| step.compute_hash())
    }
    
    /// Get the length of the execution trace
    pub fn len(&self) -> usize {
        self.steps.len()
    }
    
    /// Check if the execution trace is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
    
    /// Get a step from the execution trace
    pub fn get_step(&self, index: usize) -> Option<&ExecutionStep> {
        self.steps.get(index)
    }
    
    /// Get the hash chain of the execution trace
    pub fn hash_chain(&self) -> Vec<[u8; 32]> {
        self.steps.iter().map(|step| step.compute_hash()).collect()
    }
    
    /// Generate a proof for a specific step
    pub fn generate_proof(&self, step_index: usize) -> Option<VerificationProof> {
        if step_index >= self.steps.len() {
            return None;
        }
        
        let step = self.steps[step_index].clone();
        let hash = step.compute_hash();
        
        Some(VerificationProof {
            step,
            hash,
            step_index,
        })
    }
}

/// Verification proof for a specific execution step
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VerificationProof {
    /// The execution step
    pub step: ExecutionStep,
    
    /// Hash of the step
    #[allow(dead_code)]
    pub hash: [u8; 32],
    
    /// Index of the step in the trace
    pub step_index: usize,
}

/// BitVMX dispute resolution support
#[derive(Debug)]
#[allow(dead_code)]
pub struct DisputeResolver {
    /// The execution trace
    trace: ExecutionTrace,
}

impl DisputeResolver {
    /// Create a new dispute resolver with the given execution trace
    #[allow(dead_code)]
    pub fn new(trace: ExecutionTrace) -> Self {
        Self { trace }
    }
    
    /// Respond to a challenge
    #[allow(dead_code)]
    pub fn respond_to_challenge(&self, challenge: Challenge) -> Response {
        match challenge {
            Challenge::StepExecution { step_index } => {
                if let Some(proof) = self.trace.generate_proof(step_index) {
                    Response::StepProof(proof)
                } else {
                    Response::Error("Invalid step index".to_string())
                }
            }
            Challenge::MemoryValue { address, step_index } => {
                if step_index >= self.trace.len() {
                    return Response::Error("Invalid step index".to_string());
                }
                
                let step = &self.trace.steps[step_index];
                
                // Check if the address matches any of the addresses in the step
                if let Some(addr1) = step.read_addr1 {
                    if addr1 == address {
                        return Response::MemoryValue {
                            address,
                            value: step.read_value1.unwrap_or(0),
                            step_index,
                        };
                    }
                }
                
                if let Some(addr2) = step.read_addr2 {
                    if addr2 == address {
                        return Response::MemoryValue {
                            address,
                            value: step.read_value2.unwrap_or(0),
                            step_index,
                        };
                    }
                }
                
                if let Some(addr) = step.write_addr {
                    if addr == address {
                        return Response::MemoryValue {
                            address,
                            value: step.write_value.unwrap_or(0),
                            step_index,
                        };
                    }
                }
                
                Response::Error(format!("Address 0x{:08x} not accessed in step {}", address, step_index))
            }
            Challenge::HashChain { start_index, end_index } => {
                if start_index >= self.trace.len() || end_index >= self.trace.len() || start_index > end_index {
                    return Response::Error("Invalid index range".to_string());
                }
                
                let hashes = self.trace.hash_chain()[start_index..=end_index].to_vec();
                Response::HashChain { hashes, start_index, end_index }
            }
        }
    }
}

/// Challenge in the BitVMX verification game
#[derive(Debug, Clone)]
#[allow(dead_code)]
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

/// Response to a challenge in the BitVMX verification game
#[derive(Debug, Clone)]
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_execution_step() {
        let step = ExecutionStep::new(0x1000, 0x1000, 0x12345678, 0x1004, 0)
            .with_read1(0x2000, 0xAABBCCDD, 0)
            .with_write(0x3000, 0x11223344);
        
        assert_eq!(step.pc, 0x1000);
        assert_eq!(step.read_addr1, Some(0x2000));
        assert_eq!(step.read_value1, Some(0xAABBCCDD));
        assert_eq!(step.read_addr2, None);
        assert_eq!(step.write_addr, Some(0x3000));
        assert_eq!(step.write_value, Some(0x11223344));
        assert_eq!(step.next_pc, 0x1004);
    }
    
    #[test]
    fn test_execution_trace() {
        let mut trace = ExecutionTrace::new();
        
        let step1 = ExecutionStep::new(0x1000, 0x1000, 0x12345678, 0x1004, 0)
            .with_read1(0x2000, 0xAABBCCDD, 0)
            .with_write(0x3000, 0x11223344);
        
        let step2 = ExecutionStep::new(0x1004, 0x1004, 0x87654321, 0x1008, 1)
            .with_read1(0x3000, 0x11223344, 0)
            .with_read2(0x4000, 0x55667788, 0)
            .with_write(0x5000, 0x99AABBCC);
        
        trace.add_step(step1);
        trace.add_step(step2);
        
        assert_eq!(trace.len(), 2);
        assert_eq!(trace.get_step(0).unwrap().pc, 0x1000);
        assert_eq!(trace.get_step(1).unwrap().pc, 0x1004);
        assert_eq!(trace.hash_chain().len(), 2);
    }
    
    #[test]
    fn test_dispute_resolver() {
        let mut trace = ExecutionTrace::new();
        
        let step1 = ExecutionStep::new(0x1000, 0x1000, 0x12345678, 0x1004, 0)
            .with_read1(0x2000, 0xAABBCCDD, 0)
            .with_write(0x3000, 0x11223344);
        
        let step2 = ExecutionStep::new(0x1004, 0x1004, 0x87654321, 0x1008, 1)
            .with_read1(0x3000, 0x11223344, 0)
            .with_read2(0x4000, 0x55667788, 0)
            .with_write(0x5000, 0x99AABBCC);
        
        trace.add_step(step1);
        trace.add_step(step2);
        
        let resolver = DisputeResolver::new(trace);
        
        let challenge = Challenge::StepExecution { step_index: 1 };
        let response = resolver.respond_to_challenge(challenge);
        
        match response {
            Response::StepProof(proof) => {
                assert_eq!(proof.step_index, 1);
                assert_eq!(proof.step.pc, 0x1004);
            }
            _ => panic!("Expected StepProof response"),
        }
        
        let challenge = Challenge::MemoryValue { address: 0x3000, step_index: 1 };
        let response = resolver.respond_to_challenge(challenge);
        
        match response {
            Response::MemoryValue { address, value, step_index } => {
                assert_eq!(address, 0x3000);
                assert_eq!(value, 0x11223344);
                assert_eq!(step_index, 1);
            }
            _ => panic!("Expected MemoryValue response"),
        }
    }
} 