use emulator::{
    executor::{
        fetcher::{execute_program, FullTrace},
        utils::FailConfiguration,
    },
    loader::program::{load_elf, Program},
    ExecutionResult,
};

/// Verify a RISC-V ELF file by executing it in the BitVMX emulator
pub fn verify_file(fname: &str) -> Result<(ExecutionResult, FullTrace, Program), ExecutionResult> {
    let mut program = load_elf(fname, true).unwrap();

    // Execute the program with default settings

    let (result, trace) = execute_program(
        &mut program,
        Vec::new(),
        ".bss",
        false,
        &None,
        None,
        true,
        false,
        false,
        false,
        true,
        true,
        None,
        None,
        FailConfiguration::default(),
    );

    Ok((result, trace, program))
}

/// Test that the emulator integration works
/// Note: This requires a valid test.elf file in the uplc-to-risc directory
#[test]
#[ignore] // Ignore by default since it requires a specific test file
fn run_file() {
    let test_file = "../../test.elf";

    if !std::path::Path::new(test_file).exists() {
        println!("Test file {} does not exist, skipping test", test_file);
        return;
    }

    match verify_file(test_file) {
        Ok((_traces, result, _program)) => {
            println!("Execution result: {:?}", result);
            // for trace in traces {
            // println!("Trace: {}", trace);
            // }
        },
        Err(e) => println!("Failed to verify file: {:?}", e),
    }
}
