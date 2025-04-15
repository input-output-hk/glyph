use emulator::{
    executor::{
        fetcher::{execute_program, FullTrace},
        utils::FailConfiguration,
    },
    loader::program::{generate_rom_commitment, load_elf, RomCommitment},
    EmulatorError, ExecutionResult,
};

fn gen_com(fname: &str) -> RomCommitment {
    let program = load_elf(fname, false).unwrap();
    generate_rom_commitment(&program).unwrap()
}

fn verify_file(fname: &str) -> Result<(ExecutionResult, FullTrace), EmulatorError> {
    let mut program = load_elf(fname, true)?;

    println!("PROG IS {:#?}", program);
    Ok(execute_program(
        &mut program,
        Vec::new(),
        ".bss",
        false,
        &None,
        None,
        false,
        false,
        false,
        false,
        false,
        true,
        None,
        None,
        FailConfiguration::default(),
    ))
}

#[test]
fn run_file() {
    let g = gen_com("../../test.elf");
    // dbg!(g);
    let v = verify_file("../../test.elf");
    dbg!(v.unwrap());
    todo!();
}
