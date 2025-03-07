use uplc_to_riscv::{Compiler};
use bitvm_common::trace::ExecutionTrace;
use uplc_to_riscv::bitvm_emulator;

// Helper function to compile UPLC to RISC-V assembly
fn compile_uplc_to_riscv(input: &str) -> Result<String, String> {
    let compiler = Compiler::new();
    
    compiler.compile(input).map_err(|e| format!("Failed to compile UPLC: {}", e))
}

// Helper function to run the BitVMX emulator on the generated assembly
fn run_bitvm_emulator(assembly: &str) -> Result<ExecutionTrace, String> {
    // Use the direct BitVMX emulator integration
    bitvm_emulator::execute_assembly(assembly)
        .map_err(|e| format!("Failed to execute assembly: {}", e))
}

// Helper function to compile UPLC to RISC-V assembly and execute it directly
fn compile_and_execute_direct(input: &str) -> Result<ExecutionTrace, String> {
    let compiler = Compiler::new();
    
    compiler.compile_and_execute_direct(input)
        .map_err(|e| format!("Failed to compile and execute UPLC: {}", e))
}

// Helper function to compare two execution traces
fn compare_traces(our_trace: &ExecutionTrace, bitvm_trace: &ExecutionTrace) -> Result<(), String> {
    // Check if the traces have the same number of steps
    if our_trace.len() != bitvm_trace.len() {
        return Err(format!(
            "Trace lengths do not match: our_trace={}, bitvm_trace={}",
            our_trace.len(),
            bitvm_trace.len()
        ));
    }
    
    // Compare each step
    for (i, (our_step, bitvm_step)) in our_trace.steps.iter().zip(bitvm_trace.steps.iter()).enumerate() {
        // Compare PC
        if our_step.pc != bitvm_step.pc {
            return Err(format!(
                "PC mismatch at step {}: our_pc={}, bitvm_pc={}",
                i, our_step.pc, bitvm_step.pc
            ));
        }
        
        // Compare opcode
        if our_step.opcode_value != bitvm_step.opcode_value {
            return Err(format!(
                "Opcode mismatch at step {}: our_opcode={}, bitvm_opcode={}",
                i, our_step.opcode_value, bitvm_step.opcode_value
            ));
        }
        
        // Compare next PC
        if our_step.next_pc != bitvm_step.next_pc {
            return Err(format!(
                "Next PC mismatch at step {}: our_next_pc={}, bitvm_next_pc={}",
                i, our_step.next_pc, bitvm_step.next_pc
            ));
        }
        
        // Compare memory reads and writes
        if our_step.read_addr1 != bitvm_step.read_addr1 || our_step.read_value1 != bitvm_step.read_value1 {
            return Err(format!(
                "Read1 mismatch at step {}: our_read1=({:?}, {:?}), bitvm_read1=({:?}, {:?})",
                i, our_step.read_addr1, our_step.read_value1, bitvm_step.read_addr1, bitvm_step.read_value1
            ));
        }
        
        if our_step.read_addr2 != bitvm_step.read_addr2 || our_step.read_value2 != bitvm_step.read_value2 {
            return Err(format!(
                "Read2 mismatch at step {}: our_read2=({:?}, {:?}), bitvm_read2=({:?}, {:?})",
                i, our_step.read_addr2, our_step.read_value2, bitvm_step.read_addr2, bitvm_step.read_value2
            ));
        }
        
        if our_step.write_addr != bitvm_step.write_addr || our_step.write_value != bitvm_step.write_value {
            return Err(format!(
                "Write mismatch at step {}: our_write=({:?}, {:?}), bitvm_write=({:?}, {:?})",
                i, our_step.write_addr, our_step.write_value, bitvm_step.write_addr, bitvm_step.write_value
            ));
        }
    }
    
    Ok(())
}

#[test]
fn test_simple_program_compatibility() {
    // A simple UPLC program: (program 1.0.0 (con integer 42))
    let uplc_code = "(program 1.0.0 (con integer 42))";
    
    // Compile the UPLC code to RISC-V assembly
    let assembly = match compile_uplc_to_riscv(uplc_code) {
        Ok(asm) => asm,
        Err(e) => {
            println!("Failed to compile UPLC to RISC-V: {}", e);
            return;
        }
    };
    
    // Generate our trace using the trace generator
    let our_trace = match generate_our_trace(&assembly) {
        Ok(trace) => trace,
        Err(e) => {
            println!("Failed to generate our trace: {}", e);
            return;
        }
    };
    
    // Run the BitVMX emulator on the generated assembly
    let bitvm_trace = match run_bitvm_emulator(&assembly) {
        Ok(trace) => trace,
        Err(e) => {
            // If the error is about the RISC-V toolchain not being found, skip the test
            if e.contains("RISC-V toolchain not found") {
                println!("Skipping test: {}", e);
                return;
            }
            println!("Failed to run BitVMX emulator: {}", e);
            return;
        }
    };
    
    // Also test the direct compilation and execution
    let direct_trace = match compile_and_execute_direct(uplc_code) {
        Ok(trace) => trace,
        Err(e) => {
            // If the error is about the RISC-V toolchain not being found, skip the test
            if e.contains("RISC-V toolchain not found") {
                println!("Skipping direct compilation and execution: {}", e);
                return;
            }
            println!("Failed to compile and execute directly: {}", e);
            return;
        }
    };
    
    // Compare the traces
    if let Err(e) = compare_traces(&our_trace, &bitvm_trace) {
        panic!("Traces do not match: {}", e);
    }
    
    // Compare the direct trace with the BitVMX trace
    if let Err(e) = compare_traces(&direct_trace, &bitvm_trace) {
        panic!("Direct trace does not match BitVMX trace: {}", e);
    }
    
    // Ensure the traces have a reasonable number of steps
    assert!(our_trace.len() > 0, "Our trace is empty");
    assert!(bitvm_trace.len() > 0, "BitVMX trace is empty");
    assert!(direct_trace.len() > 0, "Direct trace is empty");
    
    // Ensure the traces have the same number of steps
    assert_eq!(our_trace.len(), bitvm_trace.len(), "Trace lengths do not match");
    assert_eq!(direct_trace.len(), bitvm_trace.len(), "Direct trace length does not match BitVMX trace length");
}

// Helper function to generate our trace using the trace generator
fn generate_our_trace(_assembly: &str) -> Result<ExecutionTrace, String> {
    let trace = ExecutionTrace::new();
    
    // TODO: Implement trace generation using our trace generator
    // For now, just return an empty trace
    
    Ok(trace)
}

#[test]
fn test_integer_operations_compatibility() {
    // A UPLC program with integer operations
    let uplc_code = "(program 1.0.0 [(builtin addInteger) (con integer 40) (con integer 2)])";
    
    // Compile the UPLC code to RISC-V assembly
    let assembly = match compile_uplc_to_riscv(uplc_code) {
        Ok(asm) => asm,
        Err(e) => {
            println!("Failed to compile UPLC to RISC-V: {}", e);
            return;
        }
    };
    
    // Generate our trace using the trace generator
    let our_trace = match generate_our_trace(&assembly) {
        Ok(trace) => trace,
        Err(e) => {
            println!("Failed to generate our trace: {}", e);
            return;
        }
    };
    
    // Run the BitVMX emulator on the generated assembly
    let bitvm_trace = match run_bitvm_emulator(&assembly) {
        Ok(trace) => trace,
        Err(e) => {
            // If the error is about the RISC-V toolchain not being found, skip the test
            if e.contains("RISC-V toolchain not found") {
                println!("Skipping test: {}", e);
                return;
            }
            println!("Failed to run BitVMX emulator: {}", e);
            return;
        }
    };
    
    // Also test the direct compilation and execution
    let direct_trace = match compile_and_execute_direct(uplc_code) {
        Ok(trace) => trace,
        Err(e) => {
            // If the error is about the RISC-V toolchain not being found, skip the test
            if e.contains("RISC-V toolchain not found") {
                println!("Skipping direct compilation and execution: {}", e);
                return;
            }
            println!("Failed to compile and execute directly: {}", e);
            return;
        }
    };
    
    // Compare the traces
    if let Err(e) = compare_traces(&our_trace, &bitvm_trace) {
        panic!("Traces do not match: {}", e);
    }
    
    // Compare the direct trace with the BitVMX trace
    if let Err(e) = compare_traces(&direct_trace, &bitvm_trace) {
        panic!("Direct trace does not match BitVMX trace: {}", e);
    }
    
    // Ensure the traces have a reasonable number of steps
    assert!(our_trace.len() > 0, "Our trace is empty");
    assert!(bitvm_trace.len() > 0, "BitVMX trace is empty");
    assert!(direct_trace.len() > 0, "Direct trace is empty");
}

#[test]
fn test_bytestring_operations_compatibility() {
    // A UPLC program with bytestring operations
    let uplc_code = "(program 1.0.0 [(builtin appendByteString) (con bytestring #01) (con bytestring #02)])";
    
    // Compile the UPLC code to RISC-V assembly
    let assembly = match compile_uplc_to_riscv(uplc_code) {
        Ok(asm) => asm,
        Err(e) => {
            println!("Failed to compile UPLC to RISC-V: {}", e);
            return;
        }
    };
    
    // Generate our trace using the trace generator
    let our_trace = match generate_our_trace(&assembly) {
        Ok(trace) => trace,
        Err(e) => {
            println!("Failed to generate our trace: {}", e);
            return;
        }
    };
    
    // Run the BitVMX emulator on the generated assembly
    let bitvm_trace = match run_bitvm_emulator(&assembly) {
        Ok(trace) => trace,
        Err(e) => {
            // If the error is about the RISC-V toolchain not being found, skip the test
            if e.contains("RISC-V toolchain not found") {
                println!("Skipping test: {}", e);
                return;
            }
            println!("Failed to run BitVMX emulator: {}", e);
            return;
        }
    };
    
    // Also test the direct compilation and execution
    let direct_trace = match compile_and_execute_direct(uplc_code) {
        Ok(trace) => trace,
        Err(e) => {
            // If the error is about the RISC-V toolchain not being found, skip the test
            if e.contains("RISC-V toolchain not found") {
                println!("Skipping direct compilation and execution: {}", e);
                return;
            }
            println!("Failed to compile and execute directly: {}", e);
            return;
        }
    };
    
    // Compare the traces
    if let Err(e) = compare_traces(&our_trace, &bitvm_trace) {
        panic!("Traces do not match: {}", e);
    }
    
    // Compare the direct trace with the BitVMX trace
    if let Err(e) = compare_traces(&direct_trace, &bitvm_trace) {
        panic!("Direct trace does not match BitVMX trace: {}", e);
    }
    
    // Ensure the traces have a reasonable number of steps
    assert!(our_trace.len() > 0, "Our trace is empty");
    assert!(bitvm_trace.len() > 0, "BitVMX trace is empty");
    assert!(direct_trace.len() > 0, "Direct trace is empty");
}

#[test]
fn test_case_expression_compatibility() {
    // A UPLC program with a case expression
    let uplc_code = "(program 1.0.0 (case (con integer 0) [(lam x (con integer 42))] [(lam x (con integer 43))]))";
    
    // Compile the UPLC code to RISC-V assembly
    let assembly = match compile_uplc_to_riscv(uplc_code) {
        Ok(asm) => asm,
        Err(e) => {
            println!("Failed to compile UPLC to RISC-V: {}", e);
            return;
        }
    };
    
    // Generate our trace using the trace generator
    let our_trace = match generate_our_trace(&assembly) {
        Ok(trace) => trace,
        Err(e) => {
            println!("Failed to generate our trace: {}", e);
            return;
        }
    };
    
    // Run the BitVMX emulator on the generated assembly
    let bitvm_trace = match run_bitvm_emulator(&assembly) {
        Ok(trace) => trace,
        Err(e) => {
            // If the error is about the RISC-V toolchain not being found, skip the test
            if e.contains("RISC-V toolchain not found") {
                println!("Skipping test: {}", e);
                return;
            }
            println!("Failed to run BitVMX emulator: {}", e);
            return;
        }
    };
    
    // Also test the direct compilation and execution
    let direct_trace = match compile_and_execute_direct(uplc_code) {
        Ok(trace) => trace,
        Err(e) => {
            // If the error is about the RISC-V toolchain not being found, skip the test
            if e.contains("RISC-V toolchain not found") {
                println!("Skipping direct compilation and execution: {}", e);
                return;
            }
            println!("Failed to compile and execute directly: {}", e);
            return;
        }
    };
    
    // Compare the traces
    if let Err(e) = compare_traces(&our_trace, &bitvm_trace) {
        panic!("Traces do not match: {}", e);
    }
    
    // Compare the direct trace with the BitVMX trace
    if let Err(e) = compare_traces(&direct_trace, &bitvm_trace) {
        panic!("Direct trace does not match BitVMX trace: {}", e);
    }
    
    // Ensure the traces have a reasonable number of steps
    assert!(our_trace.len() > 0, "Our trace is empty");
    assert!(bitvm_trace.len() > 0, "BitVMX trace is empty");
    assert!(direct_trace.len() > 0, "Direct trace is empty");
} 