use risc_v_gen::{BitVMXCodeGenerator, Instruction, Register};

struct Cek {
    generator: Vec<Instruction>,
}

impl Cek {
    pub fn new(generator: BitVMXCodeGenerator) -> Self {
        Self {
            generator: Vec::new(),
        }
    }

    pub fn init(&mut self) {
        self.generator.push(Instruction::Label("init".to_string()));

        self.generator.push(Instruction::Li(
            Register::S2,
            0x00100000, // start at 1 MB
        ));

        self.generator
            .push(Instruction::Li(Register::S0, 0x01000000));

        self.generator
            .push(Instruction::La(Register::T0, "initial_term".to_string()));

        self.generator
            .push(Instruction::Sw(Register::T0, 0, Register::Zero));

        self.generator
            .push(Instruction::Mv(Register::S1, Register::Zero));

        self.generator
            .push(Instruction::Jal(Register::A0, "eval".to_string()));
    }
}
