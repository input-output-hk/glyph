use crate::emulator::{Assign, CodeGenerator, Instruction, Register};
// use crate::serializer::constants::const_tag::{self, BOOL};
use emulator::ExecutionResult;
use emulator::executor::fetcher::{FullTrace, execute_program};
use emulator::executor::utils::FailConfiguration;
use emulator::loader::program::{self, load_elf};
use strum::IntoEnumIterator;

#[derive(Debug)]
pub struct RegisterMap {
    free: Vec<Register>,
    used: Vec<Register>,
}

pub struct Freed {}

impl RegisterMap {
    pub fn new() -> RegisterMap {
        RegisterMap {
            free: Register::iter().collect::<Vec<_>>(),
            used: vec![],
        }
    }

    pub fn var(&mut self, name: impl ToString, register: Register) -> Assign {
        let Some(index) = self.free.iter().position(|item| item == &register) else {
            panic!(
                "Assigning Used Register\n\nVar: {}, Register: {}",
                name.to_string(),
                register.name()
            )
        };

        self.free.remove(index);

        self.used.push(register);

        Assign {
            name: name.to_string(),
            assigned: false,
            mutable: true,
            register: Some(register),
        }
    }

    pub fn constnt(&mut self, name: impl ToString, register: Register) -> Assign {
        let mut assign = self.var(name, register);
        assign.mutable = false;
        assign
    }

    // Useful for top level functions
    pub fn free_all(&mut self) -> Freed {
        self.free = Register::iter().collect::<Vec<_>>();
        self.used = vec![];

        Freed {}
    }

    // Useful for branching in functions
    pub fn free_assigns(&mut self, assigns: Vec<Assign>) -> Freed {
        for assign in assigns {
            if let Some(register) = assign.register {
                let Some(index) = self.used.iter().position(|item| item == &register) else {
                    unreachable!("Likely you generated an Assign outside of the normal interface");
                };

                self.used.remove(index);
                self.free.push(register);
            }
        }

        Freed {}
    }
}

impl Default for RegisterMap {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct Cek {
    generator: CodeGenerator,
}

impl Cek {
    pub fn new() -> Cek {
        Cek {
            generator: CodeGenerator::default(),
        }
    }

    pub fn cek_assembly(mut self, bytes: Vec<u8>) -> CodeGenerator {
        // Generate the core CEK implementation
        self.generator
            .add_instruction(Instruction::section("text".to_string()));
        self.generator
            .add_instruction(Instruction::global("_start".to_string()));
        self.generator
            .add_instruction(Instruction::label("_start".to_string()));

        self.generator
            .add_instruction(Instruction::Lui(Register::Sp, 0xE0100));

        // Call the init function from the runtime
        self.generator
            .add_instruction(Instruction::Jal(Register::Ra, "init".to_string()));

        // After init returns, halt (this shouldn't be reached as init calls exit)
        self.generator
            .add_instruction(Instruction::Li(Register::A7, 93)); // exit syscall
        self.generator.add_instruction(Instruction::Ecall);

        self.generator
            .add_instruction(Instruction::section("data".to_string()));

        self.initial_term(bytes);

        self.generator
    }

    pub fn initial_term(&mut self, bytes: Vec<u8>) {
        self.generator
            .add_instruction(Instruction::label("initial_term".to_string()));

        self.generator.add_instruction(Instruction::byte(bytes));
    }
}

impl Default for Cek {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a RISC-V ELF file by executing it in the BitVMX emulator with an input
pub fn run_file(
    fname: &str,
    input: Vec<u8>,
) -> Result<(ExecutionResult, FullTrace, program::Program), ExecutionResult> {
    let mut program = load_elf(fname, true).unwrap();

    // Execute the program with default settings

    let (result, trace) = execute_program(
        &mut program,
        input,
        ".bss",
        false,
        &None,
        None,
        true,
        false,
        false,
        true,
        true,
        true,
        None,
        None,
        FailConfiguration::default(),
    );

    Ok((result, trace, program))
}

#[cfg(test)]
mod tests {}
