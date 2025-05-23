// // Tests for ELF generation capabilities
// // These tests verify the functionality described in elf_plan.md

// use risc_v_gen::emulator::verify_file;
// use risc_v_gen::{
//     assemble_and_link, CodeGenerator, Instruction, Register, Result, DEFAULT_LINKER_SCRIPT,
// };
// use std::fs;
// use std::path::Path;

// // For the tests that need temporary directories
// #[cfg(test)]
// mod tests_that_need_tempfile {
//     use super::*;

//     // New test for assemble_and_link function
//     #[test]
//     fn test_assemble_and_link_string() {
//         // Create a simple assembly program
//         let asm_code = r#"
//         .section .text
//         .global _start
//         _start:
//             li a0, 42
//             li a7, 93
//             ecall
//         "#;

//         // Output path
//         let output_path = Path::new("test_asm_string.elf");

//         // Assemble and link
//         assemble_and_link(asm_code, output_path, Some(DEFAULT_LINKER_SCRIPT)).unwrap();

//         // Check that the file exists and has a non-zero size
//         assert!(output_path.exists());
//         let metadata = fs::metadata(output_path).unwrap();
//         assert!(metadata.len() > 0);

//         // Execute the ELF file using the emulator
//         let result = verify_file(output_path.to_str().unwrap());
//         assert!(
//             result.is_ok(),
//             "Failed to verify ELF file: {:?}",
//             result.err()
//         );

//         // Check that the exit code is 42 as expected
//         let (execution_result, _trace, _program) = result.unwrap();
//         match execution_result {
//             emulator::ExecutionResult::Halt(exit_value, _) => {
//                 assert_eq!(exit_value, 42, "Program exit code should be 42");
//             },
//             other => panic!("Expected Halt variant but got: {:?}", other),
//         }

//         // Clean up
//         let _ = fs::remove_file(output_path);
//     }

//     // Test with data section
//     #[test] // Not sure if we need this: TODO
//     fn test_assemble_and_link_with_data() {
//         // Create a program with both text and data sections
//         let asm_code = r#"
//         .section .text
//         .global _start
//         _start:
//             # Load address of message
//             la a1, message

//             # Set up parameters for write syscall
//             li a0, 1       # stdout
//             li a2, 14      # length of message
//             li a7, 64      # write syscall
//             ecall

//             # Exit
//             li a0, 0       # Exit code
//             li a7, 93      # Exit syscall
//             ecall

//         .section .data
//         message:
//             .ascii "Hello, RISC-V!\n"
//         "#;

//         // Output path
//         let output_path = Path::new("test_asm_data.elf");

//         // Assemble and link
//         assemble_and_link(asm_code, output_path, None).unwrap(); // Use default linker script

//         // Check that the file exists and has a non-zero size
//         assert!(output_path.exists());
//         let metadata = fs::metadata(output_path).unwrap();
//         assert!(metadata.len() > 0);

//         // Clean up
//         let _ = fs::remove_file(output_path);
//     }

//     // We're temporarily making these non-ignored tests run by removing the #[ignore] attribute
//     // since we added the tempfile dependency
//     #[test]
//     fn test_basic_elf_generation() {
//         // Create a minimal program with entry point
//         let mut code_generator = CodeGenerator::new();

//         // Add a simple program that sets register a0 to 42 and exits
//         code_generator.add_instruction(Instruction::Label("_start".to_string()));
//         code_generator.add_instruction(Instruction::Li(Register::A0, 42));
//         code_generator.add_instruction(Instruction::Li(Register::A7, 93)); // exit syscall
//         code_generator.add_instruction(Instruction::Ecall);

//         // Create a simple linker script
//         let linker_script = r#"
//         MEMORY
//         {
//           RAM (rwx) : ORIGIN = 0x10000, LENGTH = 0x10000
//         }

//         SECTIONS
//         {
//           .text : { *(.text) } > RAM
//         }

//         ENTRY(_start)
//         "#;

//         // Build the ELF file - function to be implemented in Phase 3
//         // Using a simple file path for test - no tempfile dependency
//         let output_path = Path::new("test_program.elf");

//         build_elf(&code_generator, linker_script, output_path).unwrap();

//         // Check that the file exists and has a non-zero size
//         assert!(output_path.exists());
//         let metadata = fs::metadata(output_path).unwrap();
//         assert!(metadata.len() > 0);

//         // Clean up
//         let _ = fs::remove_file(output_path);
//     }

//     #[test]
//     fn test_complex_elf_program() {
//         // Test a more complex program with multiple sections
//         let mut code_generator = CodeGenerator::new();

//         // Add .text section with code
//         code_generator.add_instruction(Instruction::Section(".text".to_string()));
//         code_generator.add_instruction(Instruction::Global("_start".to_string()));
//         code_generator.add_instruction(Instruction::Label("_start".to_string()));
//         code_generator.add_instruction(Instruction::La(Register::T0, "message".to_string()));
//         code_generator.add_instruction(Instruction::Li(Register::A0, 1)); // stdout
//         code_generator.add_instruction(Instruction::Mv(Register::A1, Register::T0)); // buffer
//         code_generator.add_instruction(Instruction::Li(Register::A2, 13)); // length
//         code_generator.add_instruction(Instruction::Li(Register::A7, 64)); // write syscall
//         code_generator.add_instruction(Instruction::Ecall);
//         code_generator.add_instruction(Instruction::Li(Register::A0, 0)); // exit code
//         code_generator.add_instruction(Instruction::Li(Register::A7, 93)); // exit syscall
//         code_generator.add_instruction(Instruction::Ecall);

//         // Add .data section with message
//         code_generator.add_instruction(Instruction::Section(".data".to_string()));
//         code_generator.add_instruction(Instruction::Label("message".to_string()));
//         code_generator.add_instruction(Instruction::Asciiz("Hello, World!".to_string()));

//         // Create a linker script with text and data sections
//         let linker_script = r#"
//         MEMORY
//         {
//           RAM (rwx) : ORIGIN = 0x10000, LENGTH = 0x10000
//         }

//         SECTIONS
//         {
//           .text : { *(.text) } > RAM
//           .data : { *(.data) } > RAM
//           .bss  : { *(.bss)  } > RAM
//         }

//         ENTRY(_start)
//         "#;

//         // Build the ELF file
//         let output_path = Path::new("complex_program.elf");

//         build_elf(&code_generator, linker_script, output_path).unwrap();

//         // Check that the file exists
//         assert!(output_path.exists());

//         // Clean up
//         let _ = fs::remove_file(output_path);
//     }

//     #[test]
//     fn test_elf_with_symbols() {
//         // Test ELF generation with symbols and relocation
//         let mut code_generator = CodeGenerator::new();

//         // Create a program with function calls and symbols
//         code_generator.add_instruction(Instruction::Section(".text".to_string()));
//         code_generator.add_instruction(Instruction::Global("_start".to_string()));

//         // Main entry point
//         code_generator.add_instruction(Instruction::Label("_start".to_string()));
//         code_generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, -16)); // Adjust stack
//         code_generator.add_instruction(Instruction::Sw(Register::Ra, 12, Register::Sp)); // Save return address
//         code_generator.add_instruction(Instruction::Jal(Register::Ra, "print_message".to_string())); // Call function
//         code_generator.add_instruction(Instruction::Lw(Register::Ra, 12, Register::Sp)); // Restore return address
//         code_generator.add_instruction(Instruction::Addi(Register::Sp, Register::Sp, 16)); // Restore stack
//         code_generator.add_instruction(Instruction::Li(Register::A0, 0)); // Exit code
//         code_generator.add_instruction(Instruction::Li(Register::A7, 93)); // exit syscall
//         code_generator.add_instruction(Instruction::Ecall);

//         // Function that prints a message
//         code_generator.add_instruction(Instruction::Label("print_message".to_string()));
//         code_generator.add_instruction(Instruction::La(Register::T0, "message".to_string()));
//         code_generator.add_instruction(Instruction::Li(Register::A0, 1)); // stdout
//         code_generator.add_instruction(Instruction::Mv(Register::A1, Register::T0)); // buffer
//         code_generator.add_instruction(Instruction::Li(Register::A2, 13)); // length
//         code_generator.add_instruction(Instruction::Li(Register::A7, 64)); // write syscall
//         code_generator.add_instruction(Instruction::Ecall);
//         code_generator.add_instruction(Instruction::Jalr(Register::Zero, Register::Ra, 0)); // Return

//         // Data section
//         code_generator.add_instruction(Instruction::Section(".data".to_string()));
//         code_generator.add_instruction(Instruction::Label("message".to_string()));
//         code_generator.add_instruction(Instruction::Asciiz("Hello, World!".to_string()));

//         // Create a linker script
//         let linker_script = r#"
//         MEMORY
//         {
//           RAM (rwx) : ORIGIN = 0x10000, LENGTH = 0x10000
//         }

//         SECTIONS
//         {
//           .text : { *(.text) } > RAM
//           .data : { *(.data) } > RAM
//         }

//         ENTRY(_start)
//         "#;

//         // Build the ELF file
//         let output_path = Path::new("symbol_program.elf");

//         build_elf(&code_generator, linker_script, output_path).unwrap();

//         // Check that the file exists
//         assert!(output_path.exists());

//         // Clean up
//         let _ = fs::remove_file(output_path);
//     }

//     #[test]
//     fn test_data_initializers() {
//         // Test data section initialization
//         let mut code_generator = CodeGenerator::new();

//         // Create a program with initialized data
//         code_generator.add_instruction(Instruction::Section(".data".to_string()));
//         code_generator.add_instruction(Instruction::Label("integers".to_string()));
//         code_generator.add_instruction(Instruction::Word("".to_string()));
//         code_generator.add_instruction(Instruction::Word("".to_string()));
//         code_generator.add_instruction(Instruction::Word("".to_string()));

//         code_generator.add_instruction(Instruction::Label("bytes".to_string()));
//         code_generator.add_instruction(Instruction::Byte(vec![1, 2, 3, 4, 5]));

//         code_generator.add_instruction(Instruction::Label("string".to_string()));
//         code_generator.add_instruction(Instruction::Asciiz("Hello, World!".to_string()));

//         // Create a simple linker script
//         let linker_script = r#"
//         MEMORY
//         {
//           RAM (rwx) : ORIGIN = 0x10000, LENGTH = 0x10000
//         }

//         SECTIONS
//         {
//           .data : { *(.data) } > RAM
//         }
//         "#;

//         // Build the ELF file
//         let output_path = Path::new("data_program.elf");

//         build_elf(&code_generator, linker_script, output_path).unwrap();

//         // Check that the file exists
//         assert!(output_path.exists());

//         // Clean up
//         let _ = fs::remove_file(output_path);
//     }
// }

// // =========== Phase 1: Instruction → bytes tests ===========

// #[test]
// fn test_instruction_to_bytes_conversion() {
//     // This test verifies that our instructions can be properly converted to binary machine code
//     let instructions = vec![
//         Instruction::Add(Register::A0, Register::A1, Register::A2),
//         Instruction::Sub(Register::T0, Register::T1, Register::T2),
//         Instruction::Addi(Register::S0, Register::S1, 42),
//     ];

//     // This function will be implemented as part of Phase 1
//     let bytes = assemble_instructions(&instructions).unwrap();

//     // We should get a non-empty byte vector
//     assert!(!bytes.is_empty());

//     // The size should be a multiple of 4 bytes (32-bit instructions)
//     assert_eq!(bytes.len() % 4, 0);

//     // The number of bytes should match the number of instructions × 4
//     assert_eq!(bytes.len(), instructions.len() * 4);
// }

// #[test]
// fn test_instruction_edge_cases() {
//     // Test edge cases for immediates and other instruction parameters
//     let edge_cases = vec![
//         // Test maximum positive immediate for ADDI (12 bits)
//         Instruction::Addi(Register::A0, Register::Zero, 2047),
//         // Test minimum negative immediate for ADDI
//         Instruction::Addi(Register::A0, Register::Zero, -2048),
//         // Add a label before using it
//         Instruction::Label("far_label".to_string()),
//         // Test branch with maximum offset
//         Instruction::Beq(Register::Zero, Register::Zero, "far_label".to_string()),
//         // Test LUI with large immediate
//         Instruction::Lui(Register::A0, 0xFFFFF),
//     ];

//     // Should successfully assemble without errors
//     let bytes = assemble_instructions(&edge_cases).unwrap();
//     // We expect 4 instructions (not counting the label)
//     assert_eq!(bytes.len(), (edge_cases.len() - 1) * 4);
// }

// #[test]
// fn test_all_instruction_types() {
//     // Test all instruction types to ensure they're properly encoded
//     let all_types = vec![
//         // R-type
//         Instruction::Add(Register::T0, Register::T1, Register::T2),
//         // I-type
//         Instruction::Addi(Register::S0, Register::S1, 10),
//         Instruction::Lw(Register::A0, 4, Register::Sp),
//         // S-type
//         Instruction::Sw(Register::A0, 8, Register::Sp),
//         // Add labels before using them
//         Instruction::Label("branch_label".to_string()),
//         // B-type
//         Instruction::Beq(Register::A0, Register::A1, "branch_label".to_string()),
//         // U-type
//         Instruction::Lui(Register::A0, 0x12345),
//         // Add jump label
//         Instruction::Label("jump_label".to_string()),
//         // J-type
//         Instruction::Jal(Register::Ra, "jump_label".to_string()),
//     ];

//     let bytes = assemble_instructions(&all_types).unwrap();
//     // We expect 7 instructions (not counting the 2 labels)
//     assert_eq!(bytes.len(), (all_types.len() - 2) * 4);
// }

// #[test]
// fn test_invalid_instructions() {
//     // Test that invalid instructions are properly rejected
//     let invalid_instructions = vec![
//         // Out of range immediate for ADDI
//         Instruction::Addi(Register::A0, Register::Zero, 4096), // Too large
//         // Out of range immediate for ADDI (negative)
//         Instruction::Addi(Register::A0, Register::Zero, -2049), // Too small
//     ];

//     // This should return an error
//     let result = assemble_instructions(&invalid_instructions);
//     assert!(result.is_err());
// }

// // =========== Phase 2: Linker script parser tests ===========

// #[test]
// fn test_basic_linker_script_parsing() {
//     let script = r#"
// MEMORY
// {
//   RAM (rwx) : ORIGIN = 0x10000, LENGTH = 0x10000
//   ROM (rx)  : ORIGIN = 0x00000, LENGTH = 0x1000
// }

// SECTIONS
// {
//   .text : { *(.text) } > ROM
//   .data : { *(.data) } > RAM
//   .bss  : { *(.bss)  } > RAM
// }

// ENTRY(_start)
// "#;

//     println!("Basic script:\n{}", script);

//     // Parse the linker script - function to be implemented in Phase 2
//     let linker_script = parse_linker_script(script).unwrap();

//     // Debug prints
//     println!("Parsed sections:");
//     for (i, section) in linker_script.sections.iter().enumerate() {
//         println!(
//             "{}: {} in {} (alignment: {})",
//             i, section.name, section.memory_region, section.alignment
//         );
//     }

//     // Verify memory regions
//     assert_eq!(linker_script.memory_regions.len(), 2);
//     assert_eq!(linker_script.memory_regions[0].name, "RAM");
//     assert_eq!(linker_script.memory_regions[0].origin, 0x10000);
//     assert_eq!(linker_script.memory_regions[0].length, 0x10000);

//     // Verify sections
//     assert_eq!(linker_script.sections.len(), 3);
//     assert_eq!(linker_script.sections[0].name, ".text");
//     assert_eq!(linker_script.sections[0].memory_region, "ROM");

//     // Verify entry point
//     assert_eq!(linker_script.entry_point, Some("_start".to_string()));
// }

// #[test]
// fn test_linker_script_edge_cases() {
//     // Test more complex linker script with alignment, symbols, etc.
//     let complex_script = r#"
// MEMORY
// {
//   RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x4000000
// }

// SECTIONS
// {
//   .text : ALIGN(4) {
//     _text_start = .;
//     *(.text)
//     *(.text.*)
//     _text_end = .;
//   } > RAM

//   .rodata : ALIGN(8) {
//     _rodata_start = .;
//     *(.rodata)
//     *(.rodata.*)
//     _rodata_end = .;
//   } > RAM

//   .data : ALIGN(8) {
//     _data_start = .;
//     *(.data)
//     *(.data.*)
//     _data_end = .;
//   } > RAM

//   .bss : ALIGN(8) {
//     _bss_start = .;
//     *(.bss)
//     *(.bss.*)
//     *(COMMON)
//     _bss_end = .;
//   } > RAM
// }

// ENTRY(_start)
// "#;

//     println!("Complex script:\n{}", complex_script);

//     let linker_script = parse_linker_script(complex_script).unwrap();

//     // Debug prints
//     println!("Parsed complex sections:");
//     for (i, section) in linker_script.sections.iter().enumerate() {
//         println!(
//             "{}: {} in {} (alignment: {})",
//             i, section.name, section.memory_region, section.alignment
//         );
//     }

//     // Verify memory regions
//     assert_eq!(linker_script.memory_regions.len(), 1);
//     assert_eq!(linker_script.memory_regions[0].name, "RAM");
//     assert_eq!(linker_script.memory_regions[0].origin, 0x80000000);

//     // Verify sections with alignment
//     assert_eq!(linker_script.sections.len(), 4);
//     assert_eq!(linker_script.sections[0].name, ".text");
//     assert_eq!(linker_script.sections[0].alignment, 4);
//     assert_eq!(linker_script.sections[1].name, ".rodata");
//     assert_eq!(linker_script.sections[1].alignment, 8);
// }

// #[test]
// fn test_invalid_linker_script() {
//     // Test invalid linker script scenarios
//     let invalid_script = r#"
// MEMORY
// {
//   RAM (rwx) : ORIGIN = -1, LENGTH = 0x10000 /* Invalid negative origin */
// }

// SECTIONS
// {
//   .text : { *(.text) } > FLASH /* Reference to undefined memory region */
// }
// "#;

//     let result = parse_linker_script(invalid_script);
//     assert!(result.is_err());
// }

// #[test]
// fn test_validation_helpers() {
//     // Test immediate range validation
//     assert!(validate_imm_range(42, -2048, 2047));
//     assert!(!validate_imm_range(4096, -2048, 2047));
//     assert!(!validate_imm_range(-3000, -2048, 2047));

//     // Test section overlap validation
//     let sections = vec![
//         Section {
//             name: ".text".into(),
//             vma: 0x10000,
//             size: 0x1000,
//             memory_region: "RAM".into(),
//             alignment: 4,
//             input_patterns: vec![],
//         },
//         Section {
//             name: ".data".into(),
//             vma: 0x11000,
//             size: 0x1000,
//             memory_region: "RAM".into(),
//             alignment: 4,
//             input_patterns: vec![],
//         },
//     ];

//     assert!(!validate_section_overlap(&sections));

//     // Test section overlap with actually overlapping sections
//     let overlapping_sections = vec![
//         Section {
//             name: ".text".into(),
//             vma: 0x10000,
//             size: 0x1000,
//             memory_region: "RAM".into(),
//             alignment: 4,
//             input_patterns: vec![],
//         },
//         Section {
//             name: ".data".into(),
//             vma: 0x10800,
//             size: 0x1000,
//             memory_region: "RAM".into(),
//             alignment: 4,
//             input_patterns: vec![],
//         },
//     ];

//     assert!(validate_section_overlap(&overlapping_sections));
// }

// // =========== Helper types and functions for tests ===========

// // These would be implemented in the actual code

// // Helper type for parse_linker_script tests

// // Use the actual types from the crate
// use risc_v_gen::elf::{LinkerScript, Section};

// fn assemble_instructions(instructions: &[Instruction]) -> Result<Vec<u8>> {
//     risc_v_gen::assemble_instructions(instructions)
// }

// fn parse_linker_script(script: &str) -> Result<LinkerScript> {
//     risc_v_gen::parse_linker_script(script)
// }

// fn build_elf(code_gen: &CodeGenerator, linker_script: &str, output_path: &Path) -> Result<()> {
//     risc_v_gen::build_elf(code_gen, linker_script, output_path)
// }

// fn validate_imm_range(imm: i32, min: i32, max: i32) -> bool {
//     risc_v_gen::validate_imm_range(imm, min, max)
// }

// fn validate_section_overlap(sections: &[Section]) -> bool {
//     risc_v_gen::validate_section_overlap(sections)
// }
