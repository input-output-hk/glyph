// Tests for ELF generation capabilities
// These tests verify the functionality described in elf_plan.md

use risc_v_gen::{CodeGenerator, Instruction, Register, Result, assemble_and_link, DEFAULT_LINKER_SCRIPT};
use risc_v_gen::emulator::verify_file;
use std::path::Path;
use std::fs;

// For the tests that need temporary directories
#[cfg(test)]
mod tests_that_need_tempfile {
    use super::*;
    
    // New test for assemble_and_link function
    #[test]
    fn test_assemble_and_link_string() {
        // Create a CodeGenerator for assembly code generation
        let mut gen = CodeGenerator::new();
        
        // Add section and global directives
        gen.add_instruction(Instruction::Section("text".to_string()));
        gen.add_instruction(Instruction::Global("_start".to_string()));
        
        // Add label for entry point
        gen.add_instruction(Instruction::Label("_start".to_string()));
        
        // Add instructions for the program
        gen.add_instruction(Instruction::Li(Register::A0, 42));
        gen.add_instruction(Instruction::Li(Register::A7, 93));
        gen.add_instruction(Instruction::Ecall);
        
        // Generate the assembly code string
        let asm_code = gen.generate();
        
        // Output path
        let output_path = Path::new("test_asm_string.elf");
        
        // Assemble and link
        assemble_and_link(&asm_code, output_path, Some(DEFAULT_LINKER_SCRIPT)).unwrap();
        
        // Check that the file exists and has a non-zero size
        assert!(output_path.exists());
        let metadata = fs::metadata(&output_path).unwrap();
        assert!(metadata.len() > 0);
        
        // Execute the ELF file using the emulator
        let result = verify_file(output_path.to_str().unwrap());
        assert!(result.is_ok(), "Failed to verify ELF file: {:?}", result.err());
        
        // Check that the exit code is 42 as expected
        let (execution_result, _trace) = result.unwrap();
        match execution_result {
            emulator::ExecutionResult::Halt(exit_value, _) => {
                assert_eq!(exit_value, 42, "Program exit code should be 42");
            },
            other => panic!("Expected Halt variant but got: {:?}", other),
        }
        
        // Clean up
        let _ = fs::remove_file(output_path);
    }
    
    // Test with data section
    #[test] // Not sure if we need this: TODO
    fn test_assemble_and_link_with_data() {
        // Create a CodeGenerator for assembly code generation
        let mut gen = CodeGenerator::new();
        
        // Add text section and global directives
        gen.add_instruction(Instruction::Section("text".to_string()));
        gen.add_instruction(Instruction::Global("_start".to_string()));
        
        // Add label for entry point
        gen.add_instruction(Instruction::Label("_start".to_string()));
        
        // Load address of message
        gen.add_instruction(Instruction::La(Register::A1, "message".to_string()));
        
        // Set up parameters for write syscall
        gen.add_instruction(Instruction::Li(Register::A0, 1));       // stdout
        gen.add_instruction(Instruction::Li(Register::A2, 14));      // length of message
        gen.add_instruction(Instruction::Li(Register::A7, 64));      // write syscall
        gen.add_instruction(Instruction::Ecall);
        
        // Exit
        gen.add_instruction(Instruction::Li(Register::A0, 0));       // Exit code
        gen.add_instruction(Instruction::Li(Register::A7, 93));      // Exit syscall
        gen.add_instruction(Instruction::Ecall);
        
        // Add data section
        gen.add_instruction(Instruction::Section("data".to_string()));
        
        // Add message label and data
        gen.add_instruction(Instruction::Label("message".to_string()));
        gen.add_instruction(Instruction::Ascii("Hello, RISC-V!".to_string()));
        // TODO: Adding \n to the end of the message breaks the test.
        
        // Generate the assembly code string
        let asm_code = gen.generate();
        
        // Output path
        let output_path = Path::new("test_asm_data.elf");
        
        // Assemble and link
        assemble_and_link(&asm_code, output_path, None).unwrap(); // Use default linker script
        
        // Check that the file exists and has a non-zero size
        assert!(output_path.exists());
        let metadata = fs::metadata(&output_path).unwrap();
        assert!(metadata.len() > 0);
        
        // Clean up
        let _ = fs::remove_file(output_path);
    }
}

#[test]
fn test_basic_linker_script_parsing() {
    let script = r#"
MEMORY
{
  RAM (rwx) : ORIGIN = 0x10000, LENGTH = 0x10000
  ROM (rx)  : ORIGIN = 0x00000, LENGTH = 0x1000
}

SECTIONS
{
  .text : { *(.text) } > ROM
  .data : { *(.data) } > RAM
  .bss  : { *(.bss)  } > RAM
}

ENTRY(_start)
"#;

    println!("Basic script:\n{}", script);
    
    // Parse the linker script - function to be implemented in Phase 2
    let linker_script = parse_linker_script(script).unwrap();
    
    // Debug prints
    println!("Parsed sections:");
    for (i, section) in linker_script.sections.iter().enumerate() {
        println!("{}: {} in {} (alignment: {})", i, section.name, section.memory_region, section.alignment);
    }
    
    // Verify memory regions
    assert_eq!(linker_script.memory_regions.len(), 2);
    assert_eq!(linker_script.memory_regions[0].name, "RAM");
    assert_eq!(linker_script.memory_regions[0].origin, 0x10000);
    assert_eq!(linker_script.memory_regions[0].length, 0x10000);
    
    // Verify sections
    assert_eq!(linker_script.sections.len(), 3);
    assert_eq!(linker_script.sections[0].name, ".text");
    assert_eq!(linker_script.sections[0].memory_region, "ROM");
    
    // Verify entry point
    assert_eq!(linker_script.entry_point, Some("_start".to_string()));
}

#[test]
fn test_linker_script_edge_cases() {
    // Test more complex linker script with alignment, symbols, etc.
    let complex_script = r#"
MEMORY
{
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x4000000
}

SECTIONS
{
  .text : ALIGN(4) {
    _text_start = .;
    *(.text)
    *(.text.*)
    _text_end = .;
  } > RAM
  
  .rodata : ALIGN(8) {
    _rodata_start = .;
    *(.rodata)
    *(.rodata.*)
    _rodata_end = .;
  } > RAM
  
  .data : ALIGN(8) {
    _data_start = .;
    *(.data)
    *(.data.*)
    _data_end = .;
  } > RAM
  
  .bss : ALIGN(8) {
    _bss_start = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    _bss_end = .;
  } > RAM
}

ENTRY(_start)
"#;

    println!("Complex script:\n{}", complex_script);

    let linker_script = parse_linker_script(complex_script).unwrap();
    
    // Debug prints
    println!("Parsed complex sections:");
    for (i, section) in linker_script.sections.iter().enumerate() {
        println!("{}: {} in {} (alignment: {})", i, section.name, section.memory_region, section.alignment);
    }
    
    // Verify memory regions
    assert_eq!(linker_script.memory_regions.len(), 1);
    assert_eq!(linker_script.memory_regions[0].name, "RAM");
    assert_eq!(linker_script.memory_regions[0].origin, 0x80000000);
    
    // Verify sections with alignment
    assert_eq!(linker_script.sections.len(), 4);
    assert_eq!(linker_script.sections[0].name, ".text");
    assert_eq!(linker_script.sections[0].alignment, 4);
    assert_eq!(linker_script.sections[1].name, ".rodata");
    assert_eq!(linker_script.sections[1].alignment, 8);
}

#[test]
fn test_invalid_linker_script() {
    // Test invalid linker script scenarios
    let invalid_script = r#"
MEMORY
{
  RAM (rwx) : ORIGIN = -1, LENGTH = 0x10000 /* Invalid negative origin */
}

SECTIONS
{
  .text : { *(.text) } > FLASH /* Reference to undefined memory region */
}
"#;

    let result = parse_linker_script(invalid_script);
    assert!(result.is_err());
}

#[test]
fn test_validation_helpers() {
    // Test immediate range validation
    assert!(validate_imm_range(42, -2048, 2047));
    assert!(!validate_imm_range(4096, -2048, 2047));
    assert!(!validate_imm_range(-3000, -2048, 2047));
    
    // Test section overlap validation
    let sections = vec![
        Section {
            name: ".text".into(),
            vma: 0x10000,
            size: 0x1000,
            memory_region: "RAM".into(),
            alignment: 4,
            input_patterns: vec![],
        },
        Section {
            name: ".data".into(),
            vma: 0x11000,
            size: 0x1000,
            memory_region: "RAM".into(),
            alignment: 4,
            input_patterns: vec![],
        },
    ];
    
    assert!(!validate_section_overlap(&sections));
    
    // Test section overlap with actually overlapping sections
    let overlapping_sections = vec![
        Section {
            name: ".text".into(),
            vma: 0x10000,
            size: 0x1000,
            memory_region: "RAM".into(),
            alignment: 4,
            input_patterns: vec![],
        },
        Section {
            name: ".data".into(),
            vma: 0x10800,
            size: 0x1000,
            memory_region: "RAM".into(),
            alignment: 4,
            input_patterns: vec![],
        },
    ];
    
    assert!(validate_section_overlap(&overlapping_sections));
}

// =========== Helper types and functions for tests ===========

// Use the actual types from the crate
use risc_v_gen::elf::{LinkerScript, Section};

fn parse_linker_script(script: &str) -> Result<LinkerScript> {
    risc_v_gen::parse_linker_script(script)
}

fn validate_imm_range(imm: i32, min: i32, max: i32) -> bool {
    risc_v_gen::validate_imm_range(imm, min, max)
}

fn validate_section_overlap(sections: &[Section]) -> bool {
    risc_v_gen::validate_section_overlap(sections)
} 