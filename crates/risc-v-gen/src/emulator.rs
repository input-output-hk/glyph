use emulator::{executor::fetcher::execute_program, loader::program::{generate_rom_commitment, load_elf, Program, RomCommitment}, ExecutionResult};

fn gen_com(fname: &str) -> RomCommitment {
    let mut program = load_elf(&fname, false).unwrap();
    return generate_rom_commitment(&program);
}

fn verify_file(
    fname: &str,
) -> Result<(Vec<String>, ExecutionResult), ExecutionResult> {
    let mut program = load_elf(&fname, false)?;
    execute_program(
        &mut program,
        Vec::new(),
        "",
        false,
        &None,
        None,
        false,
        false,
        false,
        false,
        true,
        true,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

#[test]
fn run_file() {
    let g = gen_com("./test.elf");
    dbg!(g);
    let v = verify_file("./test.elf");
    dbg!(v);
    todo!();
}