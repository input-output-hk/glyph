use uplc_to_riscv::Compiler;

/// Test compiling a simple integer constant
#[test]
fn test_compile_integer_constant() {
    let compiler = Compiler::new();
    let uplc_code = "(program\n  1.0.0\n  (con integer 42)\n)";
    let result = compiler.compile(uplc_code);
    assert!(result.is_ok());
    
    let asm = result.unwrap();
    // Check that the assembly contains the expected instructions
    assert!(asm.contains("_start:"));
    assert!(asm.contains("li a0, 42"));
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

/// Test compiling with optimization enabled
#[test]
fn test_compile_with_optimization() {
    let compiler = Compiler::new().with_optimization(true);
    let uplc_code = "(program\n  1.0.0\n  (con integer 42)\n)";
    let result = compiler.compile(uplc_code);
    assert!(result.is_ok());
}

/// Test compiling with optimization disabled
#[test]
fn test_compile_without_optimization() {
    let compiler = Compiler::new().with_optimization(false);
    let uplc_code = "(program\n  1.0.0\n  (con integer 42)\n)";
    let result = compiler.compile(uplc_code);
    assert!(result.is_ok());
} 