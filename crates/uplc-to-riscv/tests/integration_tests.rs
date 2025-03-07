use uplc_to_riscv::Compiler;

/// Test compiling a simple integer constant in different formats
#[test]
fn test_compile_integer_constants() {
    let compiler = Compiler::new();
    // Test with multi-line format
    let uplc_code = "(program\n  1.0.0\n  (con integer 42)\n)";
    let result = compiler.compile(uplc_code);
    assert!(result.is_ok());
    
    let asm = result.unwrap();
    // Check that the assembly contains the expected instructions
    assert!(asm.contains("_start:"));
    assert!(asm.contains("li a0, 42"));
    
    // Test with single-line format
    let uplc_code = "(program 1.0.0 (con integer 42))";
    let result = compiler.compile(uplc_code);
    assert!(result.is_ok());
}

/// Test compiling a builtin function
#[test]
fn test_compile_builtin() {
    let compiler = Compiler::new();
    let uplc_code = "(program\n  1.0.0\n  (builtin addInteger)\n)";
    let result = compiler.compile(uplc_code);
    assert!(result.is_ok());
    
    let asm = result.unwrap();
    // Check that the assembly contains the expected instructions
    assert!(asm.contains("_start:"));
    // In a more complete implementation, we would check for specific builtin-related instructions
} 