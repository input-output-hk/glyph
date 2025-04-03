use risc_v_gen::{BitVMXCodeGenerator, Instruction, Register};

// pub enum Value {
//     Con(Rc<Constant>),
//     Delay(Rc<Term<NamedDeBruijn>>, Env),
//     Lambda {
//         parameter_name: Rc<NamedDeBruijn>,
//         body: Rc<Term<NamedDeBruijn>>,
//         env: Env,
//     },
//     Builtin {
//         fun: DefaultFunction,
//         runtime: BuiltinRuntime,
//     },
//     Constr {
//         tag: usize,
//         fields: Vec<Value>,
//     },
// }

// enum Context {
//     FrameAwaitArg(Value, Box<Context>),
//     FrameAwaitFunTerm(Env, Term<NamedDeBruijn>, Box<Context>),
//     FrameAwaitFunValue(Value, Box<Context>),
//     FrameForce(Box<Context>),
//     FrameConstr(
//         Env,
//         usize,
//         Vec<Term<NamedDeBruijn>>,
//         Vec<Value>,
//         Box<Context>,
//     ),
//     FrameCases(Env, Vec<Term<NamedDeBruijn>>, Box<Context>),
//     NoFrame,
// }

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

        // 1 byte for NoFrame allocation
        self.generator
            .push(Instruction::Addi(Register::S0, Register::S0, -1));

        // Tag is 3 for NoFrame
        self.generator.push(Instruction::Li(Register::T1, 6));

        // Push NoFrame tag onto stack
        self.generator
            .push(Instruction::Sb(Register::T1, 0, Register::S0));

        // Load address of initial_term
        self.generator
            .push(Instruction::La(Register::T0, "initial_term".to_string()));

        // Environment stack pointer
        self.generator
            .push(Instruction::Mv(Register::S1, Register::Zero));

        // A0 is return register
        self.generator
            .push(Instruction::Mv(Register::A0, Register::T0));

        // Ignore link by storing in T0
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

        //Term address should be in A0
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

    pub fn return_compute(&mut self) {
        self.generator
            .push(Instruction::Label("return".to_string()));

        // Load Frame from S0
        // Frame tag is first byte of frame
        self.generator
            .push(Instruction::Lb(Register::T0, 0, Register::S0));

        // FrameAwaitArg
        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::Zero,
            "handle_frame_await_arg".to_string(),
        ));

        // FrameAwaitFunTerm
        self.generator.push(Instruction::Li(Register::T1, 1));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_frame_await_fun_term".to_string(),
        ));

        // FrameAwaitFunValue
        self.generator.push(Instruction::Li(Register::T1, 2));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_frame_await_fun_value".to_string(),
        ));

        // FrameForce
        self.generator.push(Instruction::Li(Register::T1, 3));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_frame_force".to_string(),
        ));

        // TODO: CASE CONSTR

        // NoFrame
        self.generator.push(Instruction::Li(Register::T1, 6));

        self.generator.push(Instruction::Beq(
            Register::T0,
            Register::T1,
            "handle_no_frame".to_string(),
        ));
    }

    pub fn handle_var(&mut self) {
        self.generator
            .push(Instruction::Label("handle_var".to_string()));

        // load debruijn index into temp
        self.generator
            .push(Instruction::Lw(Register::T0, 1, Register::A0));

        // Put environment into A1
        self.generator
            .push(Instruction::Mv(Register::A1, Register::S1));

        // Put debruijn index into A2
        self.generator
            .push(Instruction::Mv(Register::A2, Register::T0));

        self.generator
            .push(Instruction::Jal(Register::T0, "lookup".to_string()));
    }

    pub fn handle_delay(&mut self) {
        self.generator
            .push(Instruction::Label("handle_delay".to_string()));

        // Term body is next byte
        self.generator
            .push(Instruction::Addi(Register::T0, Register::A0, 1));

        // 9 bytes for DelayValue allocation
        self.generator
            .push(Instruction::Addi(Register::S2, Register::S2, 9));

        // tag is 1 in rust
        self.generator.push(Instruction::Li(Register::T1, 2));

        // first byte is tag
        self.generator
            .push(Instruction::Sb(Register::T1, -9, Register::S2));
        // Store body
        self.generator
            .push(Instruction::Sw(Register::T0, -8, Register::S2));
        // Store environment
        self.generator
            .push(Instruction::Sw(Register::S1, -4, Register::S2));

        // Put return value into A0
        self.generator
            .push(Instruction::Addi(Register::A0, Register::S2, -9));

        self.generator
            .push(Instruction::Jal(Register::T0, "return".to_string()));
    }

    pub fn handle_lambda(&mut self) {
        self.generator
            .push(Instruction::Label("handle_lambda".to_string()));

        // Term body is next byte
        self.generator
            .push(Instruction::Addi(Register::T0, Register::A0, 1));

        // 9 bytes for LambdaValue allocation
        self.generator
            .push(Instruction::Addi(Register::S2, Register::S2, 9));

        // tag is 2 in rust
        self.generator.push(Instruction::Li(Register::T1, 2));

        // first byte is tag
        self.generator
            .push(Instruction::Sb(Register::T1, -9, Register::S2));
        // Store body
        self.generator
            .push(Instruction::Sw(Register::T0, -8, Register::S2));
        // Store environment
        self.generator
            .push(Instruction::Sw(Register::S1, -4, Register::S2));

        // Put return value into A0
        self.generator
            .push(Instruction::Addi(Register::A0, Register::S2, -9));

        self.generator
            .push(Instruction::Jal(Register::T0, "return".to_string()));
    }

    pub fn handle_apply(&mut self) {
        self.generator
            .push(Instruction::Label("handle_apply".to_string()));

        // Apply is tag |argument address| function
        // Function is 5 bytes after tag location
        self.generator
            .push(Instruction::Addi(Register::T0, Register::A0, 5));

        // Load argument into temp
        self.generator
            .push(Instruction::Lw(Register::T1, 1, Register::A0));

        // 9 bytes for FrameAwaitFunTerm allocation
        self.generator
            .push(Instruction::Addi(Register::S0, Register::S0, -9));

        // Tag is 1 for FrameAwaitFunTerm
        self.generator.push(Instruction::Li(Register::T2, 1));

        // Push tag onto stack
        self.generator
            .push(Instruction::Sb(Register::T2, 0, Register::S0));

        // Push argument onto stack
        self.generator
            .push(Instruction::Sw(Register::T1, 1, Register::S0));

        // Push environment onto stack
        self.generator
            .push(Instruction::Sw(Register::S1, 5, Register::S0));

        // Put function address into A0
        self.generator
            .push(Instruction::Mv(Register::A0, Register::T0));

        self.generator
            .push(Instruction::Jal(Register::T0, "compute".to_string()));
    }

    pub fn handle_force(&mut self) {
        self.generator
            .push(Instruction::Label("handle_force".to_string()));

        // Load term body
        self.generator
            .push(Instruction::Addi(Register::T0, Register::A0, 1));

        // 1 byte for FrameForce allocation
        self.generator
            .push(Instruction::Addi(Register::S0, Register::S0, -1));

        // Tag is 3 for FrameForce
        self.generator.push(Instruction::Li(Register::T1, 3));

        // Push FrameForce tag onto stack
        self.generator
            .push(Instruction::Sb(Register::T1, 0, Register::S0));

        // Put term body address into A0
        self.generator
            .push(Instruction::Mv(Register::A0, Register::T0));

        self.generator
            .push(Instruction::Jal(Register::T0, "compute".to_string()));
    }

    pub fn handle_error(&mut self) {
        self.generator
            .push(Instruction::Label("handle_error".to_string()));

        // Load error tag into A0
        self.generator
            .push(Instruction::Lb(Register::A0, 0, Register::A0));

        self.generator
            .push(Instruction::Jal(Register::T0, "halt".to_string()));
    }

    pub fn handle_frame_await_arg(&mut self) {
        self.generator
            .push(Instruction::Label("handle_frame_await_arg".to_string()));

        // load function value pointer from stack
        self.generator
            .push(Instruction::Lw(Register::T0, 1, Register::S0));

        // reset stack Kontinuation pointer
        self.generator
            .push(Instruction::Addi(Register::S0, Register::S0, 5));

        self.generator
            .push(Instruction::Mv(Register::A1, Register::T0));

        self.generator
            .push(Instruction::Jal(Register::T0, "apply_evaluate".to_string()));
    }

    // Takes in a0 and passes it to apply_evaluate
    pub fn handle_frame_await_fun_term(&mut self) {
        self.generator.push(Instruction::Label(
            "handle_frame_await_fun_term".to_string(),
        ));

        // load argument pointer from stack
        self.generator
            .push(Instruction::Lw(Register::T0, 1, Register::S0));

        // load environment from stack
        self.generator
            .push(Instruction::Lw(Register::T1, 5, Register::S0));

        // reset stack Kontinuation pointer
        self.generator
            .push(Instruction::Addi(Register::S0, Register::S0, 9));

        // 5 bytes for FrameAwaitArg allocation
        self.generator
            .push(Instruction::Addi(Register::S0, Register::S0, -5));

        // Tag is 0 for FrameAwaitArg
        self.generator.push(Instruction::Li(Register::T2, 0));

        // Push tag onto stack
        self.generator
            .push(Instruction::Sb(Register::T2, 0, Register::S0));

        // Push function value pointer onto stack
        self.generator
            .push(Instruction::Sw(Register::A0, 1, Register::S0));

        // Set new environment pointer
        self.generator
            .push(Instruction::Mv(Register::S1, Register::T1));

        self.generator
            .push(Instruction::Mv(Register::A0, Register::T0));

        self.generator
            .push(Instruction::Jal(Register::T0, "compute".to_string()));
    }

    // Takes in a0 and passes it to force_evaluate
    pub fn handle_frame_force(&mut self) {
        self.generator
            .push(Instruction::Label("handle_frame_force".to_string()));

        // reset stack Kontinuation pointer
        self.generator
            .push(Instruction::Addi(Register::S0, Register::S0, 1));

        self.generator
            .push(Instruction::Jal(Register::T0, "force_evaluate".to_string()));
    }

    pub fn halt(&mut self) {
        self.generator.push(Instruction::Label("halt".to_string()));

        self.generator.push(Instruction::Ecall);
    }

    pub fn force_evaluate(&mut self) {
        self.generator
            .push(Instruction::Label("force_evaluate".to_string()));

        //Value address should be in A0
        // Load value tag
        self.generator
            .push(Instruction::Lb(Register::T0, 0, Register::A0));

        // Delay
        self.generator.push(Instruction::Li(Register::T1, 1));

        self.generator.push(Instruction::Bne(
            Register::T0,
            Register::T1,
            "force_evaluate_builtin".to_string(),
        ));

        // load body pointer from stack
        self.generator
            .push(Instruction::Lw(Register::T0, 1, Register::A0));

        // load environment from stack
        self.generator
            .push(Instruction::Lw(Register::T1, 5, Register::A0));

        self.generator
            .push(Instruction::Mv(Register::S1, Register::T1));

        self.generator
            .push(Instruction::Mv(Register::A0, Register::T0));

        self.generator
            .push(Instruction::Jal(Register::T0, "compute".to_string()));

        // Builtin
        self.generator
            .push(Instruction::Label("force_evaluate_builtin".to_string()));

        self.generator.push(Instruction::Li(Register::T1, 3));

        self.generator.push(Instruction::Bne(
            Register::T0,
            Register::T1,
            "force_evaluate_error".to_string(),
        ));

        // Error
        self.generator
            .push(Instruction::Label("force_evaluate_error".to_string()));
    }
}
