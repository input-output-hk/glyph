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
            .push(Instruction::Jal(Register::T0, "eval".to_string()));
    }

    ///   s0: KP - Continuation stack pointer (points to top of K stack)
    ///   s1: E  - Environment pointer (points to current environment linked list)
    ///   s2: HP - Heap pointer (points to next free heap address)
    ///   a0: Temporary storage for C (control) pointer, updated at 0x0
    pub fn compute(&mut self) {
        self.generator
            .push(Instruction::Label("compute".to_string()));

        // Load the initial term from memory
        self.generator
            .push(Instruction::Lw(Register::A0, 0, Register::Zero));

        // Load term tag
        self.generator
            .push(Instruction::Lb(Register::T0, 0, Register::A0));

        // Var
        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::Zero,
            "handle_var".to_string(),
        ));

        // Delay
        self.generator.push(Instruction::Li(Register::T1, 1));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_delay".to_string(),
        ));

        // Lambda
        self.generator.push(Instruction::Li(Register::T1, 2));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_lambda".to_string(),
        ));

        // Apply
        self.generator.push(Instruction::Li(Register::T1, 3));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_apply".to_string(),
        ));

        // Constant
        self.generator.push(Instruction::Li(Register::T1, 4));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_constant".to_string(),
        ));

        // Force
        self.generator.push(Instruction::Li(Register::T1, 5));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_force".to_string(),
        ));

        // Error
        self.generator.push(Instruction::Li(Register::T1, 6));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_error".to_string(),
        ));

        // Builtin
        self.generator.push(Instruction::Li(Register::T1, 7));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_builtin".to_string(),
        ));

        // Constr
        self.generator.push(Instruction::Li(Register::T1, 8));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_constr".to_string(),
        ));

        // Case
        self.generator.push(Instruction::Li(Register::T1, 9));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_case".to_string(),
        ));
    }

    pub fn return_value(&mut self) {
        self.generator
            .push(Instruction::Label("return".to_string()));
    }

    pub fn handle_lambda(&mut self) {
        self.generator
            .push(Instruction::Label("handle_lambda".to_string()));

        // Load term body
        self.generator
            .push(Instruction::Lw(Register::T2, 1, Register::A0));

        // 9 bytes for LambdaValue allocation
        self.generator
            .push(Instruction::Addi(Register::S2, Register::S2, 9));

        self.generator
            .push(Instruction::Sw(Register::S2, 4, Register::Zero));

        // tag is 2 in rust
        self.generator.push(Instruction::Li(Register::T1, 2));

        // first byte is tag
        self.generator
            .push(Instruction::Sb(Register::T1, -9, Register::S2));
        // Store body
        self.generator
            .push(Instruction::Sw(Register::T2, -8, Register::S2));
        // Store environment
        self.generator
            .push(Instruction::Sw(Register::S1, -4, Register::S2));

        self.generator
            .push(Instruction::Addi(Register::A0, Register::S2, -9));

        self.generator
            .push(Instruction::Jal(Register::T0, "return".to_string()));
    }
}
