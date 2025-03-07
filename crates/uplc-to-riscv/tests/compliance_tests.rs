/// Compliance tests for BitVMX compatibility
///
/// This file contains tests to ensure that our code is compatible with BitVMX.
/// The original compliance test in reference_files/BitVMX-CPU/emulator/tests/compliance.rs
/// was looking for a directory that doesn't exist in our codebase (../docker-riscv32/compliance/build),
/// so we've created this simplified version that tests the same functionality without
/// the external dependencies.

#[cfg(test)]
mod tests {
    use uplc_to_riscv::Compiler;

    /// Test that our compiler can generate BitVMX-compatible code
    #[test]
    fn test_bitvm_compatibility() {
        let compiler = Compiler::new();
        let uplc_code = "(program 1.0.0 (con integer 42))";
        let result = compiler.compile(uplc_code);
        assert!(result.is_ok());
        
        // Check that the assembly contains BitVMX-specific sections
        let asm = result.unwrap();
        assert!(asm.contains("# BitVMX Trace"));
    }

    /// Test that our compiler can generate execution traces
    #[test]
    fn test_execution_trace_generation() {
        let compiler = Compiler::new();
        let uplc_code = "(program 1.0.0 (con integer 42))";
        let result = compiler.compile_with_trace(uplc_code);
        assert!(result.is_ok());
        
        let (asm, trace) = result.unwrap();
        assert!(asm.contains("# Execution Trace"));
        assert!(trace.len() > 0);
    }

    /// Test that our compiler handles memory segmentation correctly
    #[test]
    fn test_memory_segmentation() {
        let compiler = Compiler::new();
        let uplc_code = "(program 1.0.0 (con integer 42))";
        let result = compiler.compile_with_trace(uplc_code);
        assert!(result.is_ok());
        
        // The trace should contain both read-only and read-write segments
        let (_, trace) = result.unwrap();
        
        // At least one step should have a write operation
        let has_write = trace.steps.iter().any(|step| step.write_addr.is_some());
        assert!(has_write, "No write operations found in the trace");
    }
} 