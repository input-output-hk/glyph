use std::fs;
use std::process::Command;
use tempfile::tempdir;

#[test]
fn test_cli_compile() {
    // Create a temporary directory for our test files
    let temp_dir = tempdir().unwrap();
    let input_path = temp_dir.path().join("input.uplc");
    let output_path = temp_dir.path().join("output.s");
    
    // Write a simple UPLC program to the input file
    fs::write(&input_path, "(program\n  1.0.0\n  (con integer 42)\n)").unwrap();
    
    // Run the CLI command
    let status = Command::new(env!("CARGO_BIN_EXE_uplc-to-risc-cli"))
        .args([
            "compile",
            "--input", input_path.to_str().unwrap(),
            "--output", output_path.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    
    // Check that the command succeeded
    assert!(status.success());
    
    // Check that the output file exists
    assert!(output_path.exists());
    
    // Read the output file and check that it contains expected RISC-V assembly
    let output = fs::read_to_string(&output_path).unwrap();
    assert!(output.contains("_start:"));
    assert!(output.contains("li a0, 42"));
}

#[test]
fn test_cli_compile_with_optimization() {
    // Create a temporary directory for our test files
    let temp_dir = tempdir().unwrap();
    let input_path = temp_dir.path().join("input.uplc");
    let output_path = temp_dir.path().join("output.s");
    
    // Write a simple UPLC program to the input file
    fs::write(&input_path, "(program\n  1.0.0\n  [(builtin addInteger) (con integer 1) (con integer 2)]\n)").unwrap();
    
    // Run the CLI command with optimization
    let status = Command::new(env!("CARGO_BIN_EXE_uplc-to-risc-cli"))
        .args([
            "compile",
            "--input", input_path.to_str().unwrap(),
            "--output", output_path.to_str().unwrap(),
            "--optimize", "aggressive",
        ])
        .status()
        .unwrap();
    
    // Check that the command succeeded
    assert!(status.success());
    
    // Check that the output file exists
    assert!(output_path.exists());
    
    // Read the output file and check that it contains expected RISC-V assembly
    let output = fs::read_to_string(&output_path).unwrap();
    assert!(output.contains("_start:"));
} 