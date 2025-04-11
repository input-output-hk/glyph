use risc_v_gen::{CodeGenerator, Instruction, Register};

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

#[derive(Default, Debug)]
struct Cek {
    generator: CodeGenerator,
    frames: Register,
    env: Register,
    heap: Register,
    discard: Register,
}

impl Cek {
    pub fn init(&mut self) -> Register {
        self.generator
            .add_instruction(Instruction::Section("text".to_string()));

        self.generator
            .add_instruction(Instruction::Label("init".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("start at 10 MB".to_string()));

        self.discard = Register::T0;

        self.heap = Register::S2;

        self.generator
            .add_instruction(Instruction::Li(self.heap, 0x01000000));

        self.generator
            .add_instruction(Instruction::Comment("start at 1 MB".to_string()));

        self.frames = Register::S0;

        self.generator
            .add_instruction(Instruction::Li(self.frames, 0x00100000));

        self.generator.add_instruction(Instruction::Comment(
            "1 byte for NoFrame allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(self.frames, self.frames, -1));

        self.generator
            .add_instruction(Instruction::Comment("Tag is 6 for NoFrame".to_string()));

        let frame_tag = Register::T1;

        self.generator
            .add_instruction(Instruction::Li(frame_tag, 6));

        self.generator.add_instruction(Instruction::Comment(
            "Push NoFrame tag onto stack".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Sb(frame_tag, 0, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "Load address of initial_term".to_string(),
        ));

        let term = Register::T0;

        self.generator
            .add_instruction(Instruction::La(term, "initial_term".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Environment stack pointer".to_string(),
        ));

        self.env = Register::S1;

        self.generator
            .add_instruction(Instruction::Mv(self.env, Register::Zero));

        self.generator
            .add_instruction(Instruction::Comment("A0 is return register".to_string()));

        let ret = Register::A0;

        self.generator.add_instruction(Instruction::Mv(ret, term));

        self.generator.add_instruction(Instruction::Comment(
            "Ignore link by storing in T0".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));

        ret
    }

    // The return register
    pub fn compute(&mut self, ret: Register) {
        let term = ret;
        self.generator.add_instruction(Instruction::Comment(
            "  s0: KP - Continuation stack pointer (points to top of K stack)".to_string(),
        ));
        self.generator.add_instruction(Instruction::Comment(
            "  s1: E  - Environment pointer (points to current environment linked list)"
                .to_string(),
        ));
        self.generator.add_instruction(Instruction::Comment(
            "  s2: HP - Heap pointer (points to next free heap address)".to_string(),
        ));
        self.generator.add_instruction(Instruction::Comment(
            "  a0: Storage for C (control) pointer".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Label("compute".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Term address should be in A0".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Comment("Load term tag".to_string()));

        let term_tag = Register::T0;

        self.generator
            .add_instruction(Instruction::Lbu(term_tag, 0, term));

        self.generator
            .add_instruction(Instruction::Comment("Var".to_string()));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            Register::Zero,
            "handle_var".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("Delay".to_string()));

        let match_tag = Register::T1;

        self.generator
            .add_instruction(Instruction::Li(match_tag, 1));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            match_tag,
            "handle_delay".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("Lambda".to_string()));
        self.generator
            .add_instruction(Instruction::Li(match_tag, 2));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            match_tag,
            "handle_lambda".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("Apply".to_string()));
        self.generator
            .add_instruction(Instruction::Li(match_tag, 3));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            match_tag,
            "handle_apply".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("Constant".to_string()));
        self.generator
            .add_instruction(Instruction::Li(match_tag, 4));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            match_tag,
            "handle_constant".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("Force".to_string()));
        self.generator
            .add_instruction(Instruction::Li(match_tag, 5));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            match_tag,
            "handle_force".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("Error".to_string()));
        self.generator
            .add_instruction(Instruction::Li(match_tag, 6));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            match_tag,
            "handle_error".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("Builtin".to_string()));
        self.generator
            .add_instruction(Instruction::Li(match_tag, 7));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            match_tag,
            "handle_builtin".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("Constr".to_string()));

        self.generator
            .add_instruction(Instruction::Li(match_tag, 8));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            match_tag,
            "handle_constr".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("Case".to_string()));

        self.generator
            .add_instruction(Instruction::Li(match_tag, 9));

        self.generator.add_instruction(Instruction::Beq(
            term_tag,
            match_tag,
            "handle_case".to_string(),
        ));
    }

    pub fn return_compute(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("return".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("Load Frame from S0".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Frame tag is first byte of frame".to_string(),
        ));

        let frame_tag = Register::T0;

        self.generator
            .add_instruction(Instruction::Lbu(frame_tag, 0, self.frames));

        self.generator
            .add_instruction(Instruction::Comment("FrameAwaitArg".to_string()));

        self.generator.add_instruction(Instruction::Beq(
            frame_tag,
            Register::Zero,
            "handle_frame_await_arg".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("FrameAwaitFunTerm".to_string()));

        let match_tag = Register::T1;

        self.generator
            .add_instruction(Instruction::Li(match_tag, 1));

        self.generator.add_instruction(Instruction::Beq(
            frame_tag,
            match_tag,
            "handle_frame_await_fun_term".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("FrameAwaitFunValue".to_string()));

        self.generator
            .add_instruction(Instruction::Li(match_tag, 2));

        self.generator.add_instruction(Instruction::Beq(
            frame_tag,
            match_tag,
            "handle_frame_await_fun_value".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("FrameForce".to_string()));

        self.generator
            .add_instruction(Instruction::Li(match_tag, 3));

        self.generator.add_instruction(Instruction::Beq(
            frame_tag,
            match_tag,
            "handle_frame_force".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Li(match_tag, 4));

        self.generator.add_instruction(Instruction::Beq(
            frame_tag,
            match_tag,
            "handle_frame_constr".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Li(match_tag, 5));

        self.generator.add_instruction(Instruction::Beq(
            frame_tag,
            match_tag,
            "handle_frame_case".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Comment("NoFrame".to_string()));

        self.generator
            .add_instruction(Instruction::Li(match_tag, 6));

        self.generator.add_instruction(Instruction::Beq(
            frame_tag,
            match_tag,
            "handle_no_frame".to_string(),
        ));
    }

    pub fn handle_var(&mut self, ret: Register) -> Register {
        let var = ret;
        self.generator
            .add_instruction(Instruction::Label("handle_var".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "load debruijn index into temp".to_string(),
        ));

        let var_index = Register::T0;

        self.generator
            .add_instruction(Instruction::Lw(var_index, 1, var));

        self.generator.add_instruction(Instruction::Comment(
            "Put debruijn index into A1".to_string(),
        ));

        let lookup_index = Register::A1;

        self.generator
            .add_instruction(Instruction::Mv(lookup_index, var_index));

        self.generator
            .add_instruction(Instruction::J("lookup".to_string()));

        lookup_index
    }

    pub fn handle_delay(&mut self, ret: Register) {
        let delay_term = ret;
        self.generator
            .add_instruction(Instruction::Label("handle_delay".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("Term body is next byte".to_string()));

        let body = Register::T0;

        self.generator
            .add_instruction(Instruction::Addi(body, delay_term, 1));

        self.generator.add_instruction(Instruction::Comment(
            "9 bytes for DelayValue allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(self.heap, self.heap, 9));

        self.generator
            .add_instruction(Instruction::Comment("tag is 1 in rust".to_string()));

        let vdelay_tag = Register::T1;

        self.generator
            .add_instruction(Instruction::Li(vdelay_tag, 1));

        self.generator
            .add_instruction(Instruction::Comment("first byte is tag".to_string()));

        self.generator
            .add_instruction(Instruction::Sb(vdelay_tag, -9, self.heap));

        self.generator
            .add_instruction(Instruction::Comment("Store body pointer".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(body, -8, self.heap));

        self.generator
            .add_instruction(Instruction::Comment("Store environment".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(self.env, -4, self.heap));

        self.generator
            .add_instruction(Instruction::Comment("Put return value into A0".to_string()));

        self.generator
            .add_instruction(Instruction::Addi(ret, self.heap, -9));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn handle_lambda(&mut self, ret: Register) {
        let lambda_term = ret;

        self.generator
            .add_instruction(Instruction::Label("handle_lambda".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("Term body is next byte".to_string()));

        let body = Register::T0;

        self.generator
            .add_instruction(Instruction::Addi(body, lambda_term, 1));

        self.generator.add_instruction(Instruction::Comment(
            "9 bytes for LambdaValue allocation".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Addi(self.heap, self.heap, 9));

        self.generator
            .add_instruction(Instruction::Comment("tag is 2 in rust".to_string()));

        let vlambda_tag = Register::T1;

        self.generator
            .add_instruction(Instruction::Li(vlambda_tag, 2));

        self.generator
            .add_instruction(Instruction::Comment("first byte is tag".to_string()));

        self.generator
            .add_instruction(Instruction::Sb(vlambda_tag, -9, self.heap));

        self.generator
            .add_instruction(Instruction::Comment("Store body".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(body, -8, self.heap));

        self.generator
            .add_instruction(Instruction::Comment("Store environment".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(self.env, -4, self.heap));

        self.generator
            .add_instruction(Instruction::Comment("Put return value into A0".to_string()));

        self.generator
            .add_instruction(Instruction::Addi(ret, self.heap, -9));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn handle_apply(&mut self, ret: Register) {
        let apply_term = ret;
        self.generator
            .add_instruction(Instruction::Label("handle_apply".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Apply is tag |argument address| function".to_string(),
        ));
        self.generator.add_instruction(Instruction::Comment(
            "Function is 5 bytes after tag location".to_string(),
        ));

        let function = Register::T0;

        self.generator
            .add_instruction(Instruction::Addi(function, apply_term, 5));

        self.generator
            .add_instruction(Instruction::Comment("Load argument into temp".to_string()));

        let argument = Register::T1;
        self.generator
            .add_instruction(Instruction::Lw(argument, 1, apply_term));

        self.generator.add_instruction(Instruction::Comment(
            "9 bytes for FrameAwaitFunTerm allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(self.frames, self.frames, -9));

        self.generator.add_instruction(Instruction::Comment(
            "Tag is 1 for FrameAwaitFunTerm".to_string(),
        ));
        let frame_tag = Register::T2;

        self.generator
            .add_instruction(Instruction::Li(frame_tag, 1));

        self.generator
            .add_instruction(Instruction::Comment("Push tag onto stack".to_string()));

        self.generator
            .add_instruction(Instruction::Sb(frame_tag, 0, self.frames));

        self.generator
            .add_instruction(Instruction::Comment("Push argument onto stack".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(argument, 1, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "Push environment onto stack".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Sw(self.env, 5, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "Put function address into A0".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Mv(ret, function));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));
    }

    pub fn handle_constant(&mut self, ret: Register) {
        let constant_term = ret;

        // store pointer to constant in T0
        let constant = Register::T0;
        self.generator
            .add_instruction(Instruction::Addi(constant, constant_term, 1));
        let constant_value = Register::T2;

        self.generator
            .add_instruction(Instruction::Mv(constant_value, self.heap));

        // allocate 5 bytes on the heap
        self.generator
            .add_instruction(Instruction::Addi(self.heap, self.heap, 5));

        let constant_tag = Register::T1;
        self.generator
            .add_instruction(Instruction::Li(constant_tag, 0));

        self.generator
            .add_instruction(Instruction::Sb(constant_tag, 0, constant_value));

        self.generator
            .add_instruction(Instruction::Sw(constant, 1, constant_value));

        self.generator
            .add_instruction(Instruction::Mv(ret, constant_value));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn handle_force(&mut self, ret: Register) {
        let force_term = ret;

        self.generator
            .add_instruction(Instruction::Label("handle_force".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("Load term body".to_string()));

        let body = Register::T0;

        self.generator
            .add_instruction(Instruction::Addi(body, force_term, 1));

        self.generator.add_instruction(Instruction::Comment(
            "1 byte for FrameForce allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(self.frames, self.frames, -1));

        self.generator
            .add_instruction(Instruction::Comment("Tag is 3 for FrameForce".to_string()));

        let tag = Register::T1;
        self.generator.add_instruction(Instruction::Li(tag, 3));

        self.generator.add_instruction(Instruction::Comment(
            "Push FrameForce tag onto stack".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Sb(tag, 0, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "Put term body address into A0".to_string(),
        ));
        self.generator.add_instruction(Instruction::Mv(ret, body));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));
    }

    pub fn handle_error(&mut self, ret: Register) {
        self.generator
            .add_instruction(Instruction::Label("handle_error".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("Load -1 into A0".to_string()));

        self.generator.add_instruction(Instruction::Li(ret, -1));

        self.generator
            .add_instruction(Instruction::J("halt".to_string()));
    }

    pub fn handle_constr(&mut self, ret: Register) -> (Register, Register, Register, Register) {
        let constr = ret;

        self.generator
            .add_instruction(Instruction::Label("handle_constr".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Load the tag of the constr into T0".to_string(),
        ));

        let constr_tag = Register::T0;

        self.generator
            .add_instruction(Instruction::Lw(constr_tag, 1, constr));

        self.generator.add_instruction(Instruction::Comment(
            "Load the length of the constr fields into T1".to_string(),
        ));

        let constr_len = Register::T1;
        self.generator
            .add_instruction(Instruction::Lw(constr_len, 5, constr));

        self.generator.add_instruction(Instruction::Beq(
            constr_len,
            Register::Zero,
            "handle_constr_empty".to_string(),
        ));

        let (second_field, frames_arg, size, callback) = {
            self.generator.add_instruction(Instruction::Comment(
                "-- Fields is not empty --".to_string(),
            ));

            let constr_len_popped = Register::T1;

            self.generator
                .add_instruction(Instruction::Addi(constr_len_popped, constr_len, -1));

            self.generator.add_instruction(Instruction::Comment(
                "Minimum size for FrameConstr is 17 bytes".to_string(),
            ));

            let min_byte_size = Register::T2;
            self.generator
                .add_instruction(Instruction::Li(min_byte_size, -17));

            let shift_left_amount = Register::T3;

            self.generator
                .add_instruction(Instruction::Li(shift_left_amount, 2));

            let elements_byte_size = Register::T4;

            self.generator.add_instruction(Instruction::Sll(
                elements_byte_size,
                constr_len_popped,
                shift_left_amount,
            ));

            let total_byte_size = Register::T2;

            self.generator.add_instruction(Instruction::Sub(
                total_byte_size,
                min_byte_size,
                elements_byte_size,
            ));

            self.generator.add_instruction(Instruction::Comment(
                "Allocate 17 + 4 * constr fields length + 4 * values length".to_string(),
            ));
            self.generator.add_instruction(Instruction::Comment(
                "where values length is 0 and fields length is fields in constr - 1".to_string(),
            ));
            self.generator.add_instruction(Instruction::Comment(
                "frame tag 1 byte + constr tag 4 bytes + environment 4 bytes".to_string(),
            ));
            self.generator.add_instruction(Instruction::Comment(
                "+ fields length 4 bytes + 4 * fields length in bytes".to_string(),
            ));
            self.generator.add_instruction(Instruction::Comment(
                "+ values length 4 bytes + 4 * values length in bytes".to_string(),
            ));
            self.generator.add_instruction(Instruction::Comment(
                "Remember this is subtracting the above value".to_string(),
            ));
            self.generator.add_instruction(Instruction::Add(
                self.frames,
                self.frames,
                total_byte_size,
            ));

            let frames = Register::T3;

            self.generator
                .add_instruction(Instruction::Mv(frames, self.frames));

            let constr_frame_tag = Register::T4;

            self.generator
                .add_instruction(Instruction::Li(constr_frame_tag, 4));

            self.generator
                .add_instruction(Instruction::Comment("store frame tag".to_string()));
            self.generator
                .add_instruction(Instruction::Sb(constr_frame_tag, 0, frames));

            self.generator
                .add_instruction(Instruction::Comment("move up 1 byte".to_string()));
            self.generator
                .add_instruction(Instruction::Addi(frames, frames, 1));

            self.generator
                .add_instruction(Instruction::Comment(" store constr tag".to_string()));
            self.generator
                .add_instruction(Instruction::Sw(constr_tag, 0, frames));
            self.generator
                .add_instruction(Instruction::Comment("move up 4 bytes".to_string()));
            self.generator
                .add_instruction(Instruction::Addi(frames, frames, 4));

            self.generator
                .add_instruction(Instruction::Comment("store environment".to_string()));
            self.generator
                .add_instruction(Instruction::Sw(self.env, 0, frames));

            self.generator
                .add_instruction(Instruction::Comment("move up 4 bytes".to_string()));
            self.generator
                .add_instruction(Instruction::Addi(frames, frames, 4));

            self.generator
                .add_instruction(Instruction::Comment("store fields length -1".to_string()));

            self.generator
                .add_instruction(Instruction::Sw(constr_len_popped, 0, frames));

            self.generator
                .add_instruction(Instruction::Comment("move up 4 bytes".to_string()));
            self.generator
                .add_instruction(Instruction::Addi(frames, frames, 4));

            self.generator
                .add_instruction(Instruction::Comment("Load first field to A4".to_string()));

            let first_field = Register::A4;

            self.generator
                .add_instruction(Instruction::Lw(first_field, 9, constr));

            self.generator.add_instruction(Instruction::Comment(
                "move fields length - 1 to A2".to_string(),
            ));

            let size = Register::A2;

            self.generator
                .add_instruction(Instruction::Mv(size, constr_len_popped));

            self.generator.add_instruction(Instruction::Comment(
                "move current stack pointer to A1".to_string(),
            ));

            let frames_arg = Register::A1;

            self.generator
                .add_instruction(Instruction::Mv(frames_arg, frames));

            self.generator.add_instruction(Instruction::Comment(
                "move A0 pointer to second element in fields (regardless if there or not)"
                    .to_string(),
            ));

            let second_field = ret;

            self.generator
                .add_instruction(Instruction::Addi(second_field, constr, 13));

            // Takes in A0 - elements pointer, A1 - destination pointer, A2 - length
            // A3 - return address

            let callback = Register::A3;

            self.generator
                .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

            // New destination pointer is moved to T3
            self.generator
                .add_instruction(Instruction::Mv(frames, frames_arg));

            self.generator.add_instruction(Instruction::Comment(
                "Store 0 for values length".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::Sw(Register::Zero, 0, frames));

            // No need to move T0 since we are done storage

            self.generator.add_instruction(Instruction::Comment(
                "Mv A4 (pointer to first field term) to A0".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::Mv(ret, first_field));

            self.generator
                .add_instruction(Instruction::J("compute".to_string()));

            (second_field, frames_arg, size, callback)
        };

        {
            self.generator
                .add_instruction(Instruction::Comment("-- Empty fields --".to_string()));

            self.generator
                .add_instruction(Instruction::Label("handle_constr_empty".to_string()));

            self.generator.add_instruction(Instruction::Comment(
                "9 bytes allocated on heap".to_string(),
            ));
            self.generator.add_instruction(Instruction::Comment(
                "1 byte value tag + 4 bytes constr tag + 4 bytes constr fields length which is 0"
                    .to_string(),
            ));
            self.generator
                .add_instruction(Instruction::Addi(self.heap, self.heap, 9));

            let vconstr_tag = Register::T2;

            self.generator
                .add_instruction(Instruction::Li(vconstr_tag, 4));

            self.generator
                .add_instruction(Instruction::Sb(vconstr_tag, -9, self.heap));

            self.generator
                .add_instruction(Instruction::Sw(constr_tag, -8, self.heap));

            self.generator
                .add_instruction(Instruction::Sw(constr_len, -4, self.heap));

            self.generator
                .add_instruction(Instruction::Addi(ret, self.heap, -9));

            self.generator
                .add_instruction(Instruction::J("return".to_string()));
        }

        (second_field, frames_arg, size, callback)
    }

    pub fn handle_case(&mut self, ret: Register) -> (Register, Register, Register, Register) {
        let case = ret;
        self.generator
            .add_instruction(Instruction::Label("handle_case".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Load the term pointer of the constr of case into A4".to_string(),
        ));

        let constr = Register::A4;
        self.generator
            .add_instruction(Instruction::Lw(constr, 1, case));

        self.generator.add_instruction(Instruction::Comment(
            "Load the length of the constr fields into T1".to_string(),
        ));

        let size = Register::T1;
        self.generator
            .add_instruction(Instruction::Lw(size, 5, constr));

        self.generator.add_instruction(Instruction::Comment(
            "Minimum size for FrameCase is 9 bytes".to_string(),
        ));

        let min_bytes = Register::T2;
        self.generator
            .add_instruction(Instruction::Li(min_bytes, -9));

        let shift_left_amount = Register::T3;

        self.generator
            .add_instruction(Instruction::Li(shift_left_amount, 2));

        let elements_byte_size = Register::T4;

        self.generator.add_instruction(Instruction::Sll(
            elements_byte_size,
            size,
            shift_left_amount,
        ));

        let total_byte_size = Register::T2;

        self.generator.add_instruction(Instruction::Sub(
            total_byte_size,
            min_bytes,
            elements_byte_size,
        ));

        self.generator.add_instruction(Instruction::Comment(
            "Allocate 9 + 4 * cases length".to_string(),
        ));

        self.generator.add_instruction(Instruction::Comment(
            "frame tag 1 byte + environment 4 bytes".to_string(),
        ));
        self.generator.add_instruction(Instruction::Comment(
            "+ cases length 4 bytes + 4 * cases length in bytes".to_string(),
        ));

        self.generator.add_instruction(Instruction::Comment(
            "Remember this is subtracting the above value".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Add(self.frames, self.frames, Register::T2));

        let frames = Register::T3;

        self.generator
            .add_instruction(Instruction::Mv(frames, self.frames));

        // FrameCase tag
        let frame_case_tag = Register::T0;

        self.generator
            .add_instruction(Instruction::Li(frame_case_tag, 5));

        self.generator
            .add_instruction(Instruction::Sb(frame_case_tag, 0, frames));

        self.generator
            .add_instruction(Instruction::Addi(frames, frames, 1));

        self.generator
            .add_instruction(Instruction::Sw(self.env, 0, frames));

        self.generator
            .add_instruction(Instruction::Addi(frames, frames, 4));

        self.generator
            .add_instruction(Instruction::Sw(size, 0, frames));

        self.generator
            .add_instruction(Instruction::Addi(frames, frames, 4));

        // A0 pointer to terms array
        // A1 is new stack pointer
        // A2 is length of terms array
        // A3 holds return address

        let list_size = Register::A2;
        self.generator
            .add_instruction(Instruction::Mv(list_size, size));

        let frames_arg = Register::A1;

        self.generator
            .add_instruction(Instruction::Mv(frames_arg, frames));

        let branches = ret;
        self.generator
            .add_instruction(Instruction::Addi(branches, case, 9));

        let callback = Register::A3;

        self.generator
            .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

        // Move term pointer into A0
        self.generator.add_instruction(Instruction::Mv(ret, constr));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));

        (branches, frames_arg, list_size, callback)
    }

    pub fn handle_frame_await_arg(&mut self, ret: Register) -> (Register, Register) {
        let arg = Register::A1;

        self.generator
            .add_instruction(Instruction::Label("handle_frame_await_arg".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "load function value pointer from stack".to_string(),
        ));

        let function = Register::T0;
        self.generator
            .add_instruction(Instruction::Lw(function, 1, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "reset stack Kontinuation pointer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Addi(self.frames, self.frames, 5));

        let argument = Register::A1;

        self.generator
            .add_instruction(Instruction::Mv(argument, arg));

        self.generator
            .add_instruction(Instruction::Mv(ret, function));

        self.generator
            .add_instruction(Instruction::J("apply_evaluate".to_string()));

        (ret, argument)
    }

    // Takes in a0 and passes it to apply_evaluate
    pub fn handle_frame_await_fun_term(&mut self, ret: Register) {
        let function = ret;

        self.generator.add_instruction(Instruction::Label(
            "handle_frame_await_fun_term".to_string(),
        ));

        self.generator.add_instruction(Instruction::Comment(
            "load argument pointer from stack".to_string(),
        ));

        let argument = Register::T0;
        self.generator
            .add_instruction(Instruction::Lw(argument, 1, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "load environment from stack".to_string(),
        ));

        let environment = Register::T1;
        self.generator
            .add_instruction(Instruction::Lw(environment, 5, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "reset stack Kontinuation pointer".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(self.frames, self.frames, 9));

        self.generator.add_instruction(Instruction::Comment(
            "5 bytes for FrameAwaitArg allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(self.frames, self.frames, -5));

        self.generator.add_instruction(Instruction::Comment(
            "Tag is 0 for FrameAwaitArg".to_string(),
        ));

        let frame_tag = Register::T2;
        self.generator
            .add_instruction(Instruction::Li(frame_tag, 0));

        self.generator
            .add_instruction(Instruction::Comment("Push tag onto stack".to_string()));

        self.generator
            .add_instruction(Instruction::Sb(frame_tag, 0, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "Push function value pointer onto stack".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Sw(function, 1, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "Set new environment pointer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Mv(self.env, environment));

        self.generator
            .add_instruction(Instruction::Mv(ret, argument));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));
    }

    // Takes in a0 and passes it to force_evaluate
    pub fn handle_frame_force(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("handle_frame_force".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "reset stack Kontinuation pointer".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(self.frames, self.frames, 1));

        self.generator
            .add_instruction(Instruction::J("force_evaluate".to_string()));
    }

    pub fn handle_frame_constr(&mut self, ret: Register) {
        let computed_value = ret;
        let frame = self.frames;
        self.generator
            .add_instruction(Instruction::Label("handle_frame_constr".to_string()));

        let constr_tag = Register::T0;

        // Load the constructor tag from the frame
        self.generator
            .add_instruction(Instruction::Lw(constr_tag, 1, frame));

        let environment = Register::T1;

        // Load the environment from the frame
        self.generator
            .add_instruction(Instruction::Lw(environment, 5, frame));

        let fields_len = Register::T2;

        self.generator
            .add_instruction(Instruction::Lw(fields_len, 9, frame));

        // bytes offset from frame to values len based on fields len
        let bytes_offset = Register::T3;

        self.generator
            .add_instruction(Instruction::Mv(bytes_offset, fields_len));

        self.generator
            .add_instruction(Instruction::Slli(bytes_offset, bytes_offset, 2));

        self.generator
            .add_instruction(Instruction::Addi(bytes_offset, bytes_offset, 13));

        self.generator
            .add_instruction(Instruction::Add(bytes_offset, bytes_offset, frame));

        let values_len = Register::T4;

        self.generator
            .add_instruction(Instruction::Lw(values_len, 0, bytes_offset));

        self.generator
            .add_instruction(Instruction::Addi(values_len, values_len, 1));

        self.generator
            .add_instruction(Instruction::Sw(values_len, 0, bytes_offset));

        self.generator.add_instruction(Instruction::Beq(
            fields_len,
            Register::Zero,
            "handle_frame_constr_empty".to_string(),
        ));

        {
            let first_field = Register::A5;
            self.generator
                .add_instruction(Instruction::Lw(first_field, 13, frame));

            let current_field_len = fields_len;

            self.generator.add_instruction(Instruction::Addi(
                current_field_len,
                current_field_len,
                -1,
            ));

            let new_value = Register::A4;

            self.generator
                .add_instruction(Instruction::Mv(new_value, computed_value));

            let length_arg = Register::A2;

            self.generator
                .add_instruction(Instruction::Mv(length_arg, current_field_len));

            self.generator
                .add_instruction(Instruction::Add(length_arg, length_arg, values_len));

            let new_list = Register::A1;

            self.generator
                .add_instruction(Instruction::Addi(new_list, frame, 13));

            let src_list = ret;

            self.generator
                .add_instruction(Instruction::Addi(src_list, frame, 17));

            let callback = Register::A3;

            self.generator
                .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

            self.generator
                .add_instruction(Instruction::Sw(new_value, 0, new_list));

            self.generator
                .add_instruction(Instruction::Mv(self.env, environment));

            self.generator
                .add_instruction(Instruction::Mv(ret, first_field));

            self.generator
                .add_instruction(Instruction::J("compute".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("handle_frame_constr_empty".to_string()));

            // fields length is 0 and not needed
            // allocation amount in bytes is 4 * value length + 9
            // 1 for frame tag, 4 for constr tag, and 4 for value length
            let allocation_amount = Register::T2;
            self.generator
                .add_instruction(Instruction::Mv(allocation_amount, values_len));

            self.generator.add_instruction(Instruction::Slli(
                allocation_amount,
                allocation_amount,
                2,
            ));

            self.generator.add_instruction(Instruction::Addi(
                allocation_amount,
                allocation_amount,
                9,
            ));

            // Allocate VConstr on the heap
            // 9 + 4 * value length
            self.generator.add_instruction(Instruction::Add(
                self.heap,
                self.heap,
                allocation_amount,
            ));

            let allocator_space = Register::T2;
            self.generator.add_instruction(Instruction::Sub(
                allocator_space,
                self.heap,
                allocation_amount,
            ));

            let value_tag = Register::T5;

            self.generator
                .add_instruction(Instruction::Li(value_tag, 4));

            self.generator
                .add_instruction(Instruction::Sb(value_tag, 0, allocator_space));

            self.generator
                .add_instruction(Instruction::Sw(constr_tag, 1, allocator_space));

            self.generator
                .add_instruction(Instruction::Sw(values_len, 5, allocator_space));

            let return_value = Register::A5;

            self.generator
                .add_instruction(Instruction::Mv(return_value, allocator_space));

            self.generator
                .add_instruction(Instruction::Addi(allocator_space, allocator_space, 9));

            self.generator
                .add_instruction(Instruction::Addi(values_len, values_len, -1));

            let list_byte_length = Register::T0;

            self.generator
                .add_instruction(Instruction::Mv(list_byte_length, values_len));

            self.generator.add_instruction(Instruction::Slli(
                list_byte_length,
                list_byte_length,
                2,
            ));

            let tail_list = Register::T1;

            self.generator.add_instruction(Instruction::Add(
                tail_list,
                list_byte_length,
                bytes_offset,
            ));

            let next_frame = Register::A6;
            self.generator
                .add_instruction(Instruction::Addi(next_frame, tail_list, 4));

            let new_value = Register::A4;

            self.generator
                .add_instruction(Instruction::Mv(new_value, computed_value));

            let list_to_reverse = ret;

            self.generator
                .add_instruction(Instruction::Mv(list_to_reverse, tail_list));

            let dest = Register::A1;
            self.generator
                .add_instruction(Instruction::Mv(dest, allocator_space));

            let size = Register::A2;
            self.generator
                .add_instruction(Instruction::Mv(size, values_len));

            let callback = Register::A3;
            self.generator
                .add_instruction(Instruction::Jal(callback, "reverse_clone_list".to_string()));

            self.generator
                .add_instruction(Instruction::Sw(new_value, 0, dest));

            self.generator
                .add_instruction(Instruction::Mv(ret, return_value));

            self.generator
                .add_instruction(Instruction::Mv(self.frames, next_frame));

            self.generator
                .add_instruction(Instruction::J("return".to_string()));
        }
    }

    pub fn handle_frame_case(&mut self, ret: Register) {
        let constr = Register::T5;

        let case_frame = self.frames;

        self.generator
            .add_instruction(Instruction::Label("handle_frame_case".to_string()));

        self.generator.add_instruction(Instruction::Mv(constr, ret));

        let constr_term_tag = Register::T0;
        self.generator
            .add_instruction(Instruction::Lbu(constr_term_tag, 0, constr));

        let expected_tag = Register::T1;

        self.generator
            .add_instruction(Instruction::Li(expected_tag, 4));

        self.generator.add_instruction(Instruction::Bne(
            constr_term_tag,
            expected_tag,
            "handle_frame_case_error".to_string(),
        ));

        {
            let constr_tag = Register::T0;
            self.generator
                .add_instruction(Instruction::Lw(constr_tag, 1, constr));

            let branches_len = Register::T1;
            self.generator
                .add_instruction(Instruction::Lw(branches_len, 5, case_frame));

            self.generator.add_instruction(Instruction::Bge(
                constr_tag,
                branches_len,
                "handle_frame_case_error".to_string(),
            ));

            // Don't need scope here since handle error is already in another scope
            // {}

            // set env
            self.generator
                .add_instruction(Instruction::Lw(self.env, 1, case_frame));

            let offset_to_branch = Register::T2;
            self.generator
                .add_instruction(Instruction::Mv(offset_to_branch, constr_tag));

            self.generator.add_instruction(Instruction::Slli(
                offset_to_branch,
                offset_to_branch,
                2,
            ));

            self.generator.add_instruction(Instruction::Addi(
                offset_to_branch,
                offset_to_branch,
                9,
            ));

            self.generator.add_instruction(Instruction::Add(
                offset_to_branch,
                offset_to_branch,
                case_frame,
            ));

            // Put branch term in return register
            self.generator
                .add_instruction(Instruction::Lw(ret, 0, offset_to_branch));

            // reset frame pointer
            let claim_stack_item = Register::T4;
            self.generator
                .add_instruction(Instruction::Mv(claim_stack_item, branches_len));

            self.generator.add_instruction(Instruction::Slli(
                claim_stack_item,
                claim_stack_item,
                2,
            ));

            self.generator.add_instruction(Instruction::Addi(
                claim_stack_item,
                claim_stack_item,
                9,
            ));

            self.generator.add_instruction(Instruction::Add(
                self.frames,
                self.frames,
                claim_stack_item,
            ));

            // Don't care about the frame anymore

            let constr_fields_len = Register::T2;
            self.generator
                .add_instruction(Instruction::Lw(constr_fields_len, 5, constr));

            let current_index = Register::T3;
            self.generator
                .add_instruction(Instruction::Mv(current_index, Register::Zero));

            let current_offset = Register::T1;
            // 9 for constant offset
            // 1 for frame tag + 4 for constr tag + 4 for constr fields len
            self.generator
                .add_instruction(Instruction::Addi(current_offset, current_offset, 9));

            self.generator.add_instruction(Instruction::Add(
                current_offset,
                current_offset,
                constr,
            ));

            {
                self.generator
                    .add_instruction(Instruction::Label("transfer_fields_as_args".to_string()));

                self.generator.add_instruction(Instruction::Beq(
                    current_index,
                    constr_fields_len,
                    "compute".to_string(),
                ));

                // Allocate for FrameAwaitFunValue to stack
                // 5 bytes, 1 for frame tag, 4 for argument value pointer
                self.generator
                    .add_instruction(Instruction::Addi(self.frames, self.frames, -5));

                let frame_tag = Register::T0;

                self.generator
                    .add_instruction(Instruction::Li(frame_tag, 2));

                self.generator
                    .add_instruction(Instruction::Sb(frame_tag, 0, self.frames));

                let arg = Register::T0;

                self.generator
                    .add_instruction(Instruction::Lw(arg, 0, current_offset));

                self.generator
                    .add_instruction(Instruction::Sw(arg, 1, self.frames));

                self.generator.add_instruction(Instruction::Addi(
                    current_offset,
                    current_offset,
                    4,
                ));

                self.generator
                    .add_instruction(Instruction::Addi(current_index, current_index, 1));

                self.generator
                    .add_instruction(Instruction::J("transfer_fields_as_args".to_string()));
            }
        }

        {
            self.generator
                .add_instruction(Instruction::Label("handle_frame_case_error".to_string()));

            self.generator
                .add_instruction(Instruction::J("handle_error".to_string()));
        }
    }

    pub fn halt(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("halt".to_string()));

        self.generator
            .add_instruction(Instruction::Li(Register::A7, 93));

        self.generator.add_instruction(Instruction::Ecall);
    }

    pub fn force_evaluate(&mut self, ret: Register) {
        let value = ret;
        self.generator
            .add_instruction(Instruction::Label("force_evaluate".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Value address should be in A0".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Comment("Load value tag".to_string()));

        let tag = Register::T0;
        self.generator
            .add_instruction(Instruction::Lbu(tag, 0, value));

        self.generator
            .add_instruction(Instruction::Comment("Delay".to_string()));

        let delay_value_tag = Register::T1;
        self.generator
            .add_instruction(Instruction::Li(delay_value_tag, 1));

        self.generator.add_instruction(Instruction::Bne(
            tag,
            delay_value_tag,
            "force_evaluate_builtin".to_string(),
        ));

        {
            self.generator.add_instruction(Instruction::Comment(
                "load body pointer from a0 which is Value".to_string(),
            ));

            // We can overwrite T0 here since we can't reach the other tag cases at this point
            let body = Register::T0;
            self.generator
                .add_instruction(Instruction::Lw(body, 1, value));

            self.generator.add_instruction(Instruction::Comment(
                "load environment from a0 which is Value".to_string(),
            ));

            let environment = Register::T1;
            self.generator
                .add_instruction(Instruction::Lw(environment, 5, value));

            self.generator
                .add_instruction(Instruction::Mv(self.env, environment));

            self.generator.add_instruction(Instruction::Mv(ret, body));

            self.generator
                .add_instruction(Instruction::J("compute".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("force_evaluate_builtin".to_string()));

            // T0 is still tag here
            let builtin_value_tag = Register::T1;
            self.generator
                .add_instruction(Instruction::Li(builtin_value_tag, 3));

            self.generator.add_instruction(Instruction::Bne(
                tag,
                builtin_value_tag,
                "force_evaluate_error".to_string(),
            ));

            {
                self.generator
                    .add_instruction(Instruction::Comment("Builtin TODO".to_string()));

                self.generator.add_instruction(Instruction::Nop);
            }

            {
                self.generator
                    .add_instruction(Instruction::Label("force_evaluate_error".to_string()));

                self.generator
                    .add_instruction(Instruction::J("handle_error".to_string()));
            }
        }
    }

    pub fn apply_evaluate(&mut self, ret: Register, function: Register, argument: Register) {
        self.generator
            .add_instruction(Instruction::Label("apply_evaluate".to_string()));

        //Value address should be in A0
        self.generator
            .add_instruction(Instruction::Comment("Load function value tag".to_string()));

        let function_tag = Register::T0;
        self.generator
            .add_instruction(Instruction::Lbu(function_tag, 0, function));

        self.generator
            .add_instruction(Instruction::Comment("Lambda".to_string()));

        let lambda_value_tag = Register::T1;

        self.generator
            .add_instruction(Instruction::Li(lambda_value_tag, 2));

        self.generator.add_instruction(Instruction::Bne(
            function_tag,
            lambda_value_tag,
            "apply_evaluate_builtin".to_string(),
        ));

        {
            self.generator.add_instruction(Instruction::Comment(
                "load body pointer from a0 which is function Value".to_string(),
            ));

            let body = Register::T0;

            self.generator
                .add_instruction(Instruction::Lw(body, 1, function));

            self.generator.add_instruction(Instruction::Comment(
                "load environment from a0 which is function Value".to_string(),
            ));

            let environment = Register::T1;

            self.generator
                .add_instruction(Instruction::Lw(environment, 5, function));

            self.generator
                .add_instruction(Instruction::Mv(self.env, environment));

            self.generator.add_instruction(Instruction::Comment(
                "Important this is the only place we modify environment".to_string(),
            ));

            self.generator.add_instruction(Instruction::Comment(
                "Allocate 8 bytes on the heap".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::Addi(self.heap, self.heap, 8));

            self.generator.add_instruction(Instruction::Comment(
                "pointer to argument value".to_string(),
            ));
            self.generator
                .add_instruction(Instruction::Sw(argument, -8, self.heap));

            self.generator.add_instruction(Instruction::Comment(
                "pointer to previous environment".to_string(),
            ));
            self.generator
                .add_instruction(Instruction::Sw(self.env, -4, self.heap));

            self.generator.add_instruction(Instruction::Comment(
                "Save allocated heap location in environment pointer".to_string(),
            ));
            self.generator
                .add_instruction(Instruction::Addi(self.env, self.heap, -8));

            self.generator.add_instruction(Instruction::Mv(ret, body));

            self.generator
                .add_instruction(Instruction::J("compute".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("apply_evaluate_builtin".to_string()));

            self.generator
                .add_instruction(Instruction::Li(Register::T1, 3));

            self.generator.add_instruction(Instruction::Bne(
                function_tag,
                Register::T1,
                "apply_evaluate_error".to_string(),
            ));

            {
                self.generator
                    .add_instruction(Instruction::Comment("Builtin TODO".to_string()));

                self.generator.add_instruction(Instruction::Nop);
            }

            {
                self.generator
                    .add_instruction(Instruction::Label("apply_evaluate_error".to_string()));

                self.generator
                    .add_instruction(Instruction::J("handle_error".to_string()));
            }
        }
    }

    pub fn lookup(&mut self, ret: Register, index: Register) {
        self.generator
            .add_instruction(Instruction::Label("lookup".to_string()));

        let current_index = Register::T0;

        self.generator
            .add_instruction(Instruction::Mv(current_index, index));

        self.generator
            .add_instruction(Instruction::Addi(current_index, current_index, -1));

        self.generator.add_instruction(Instruction::Beq(
            current_index,
            Register::Zero,
            "lookup_return".to_string(),
        ));

        {
            self.generator.add_instruction(Instruction::Comment(
                "pointer to next environment node".to_string(),
            ));

            let current_env = Register::T1;
            self.generator
                .add_instruction(Instruction::Lw(current_env, 4, self.env));

            self.generator
                .add_instruction(Instruction::Mv(self.env, current_env));

            self.generator
                .add_instruction(Instruction::Mv(index, current_index));

            self.generator
                .add_instruction(Instruction::J("lookup".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("lookup_return".to_string()));

            self.generator
                .add_instruction(Instruction::Lw(ret, 0, self.env));

            self.generator
                .add_instruction(Instruction::J("return".to_string()));
        }
    }

    // A0 pointer to terms array
    // A1 is new stack pointer
    // A2 is length of terms array
    // A3 is the return address
    pub fn clone_list(
        &mut self,
        list: Register,
        dest_list: Register,
        length: Register,
        callback: Register,
    ) {
        self.generator
            .add_instruction(Instruction::Label("clone_list".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("A2 contains terms length".to_string()));

        self.generator.add_instruction(Instruction::Beq(
            length,
            Register::Zero,
            "clone_list_return".to_string(),
        ));

        {
            let list_item = Register::T0;
            self.generator
                .add_instruction(Instruction::Lw(list_item, 0, list));

            // Store term in new storage
            self.generator
                .add_instruction(Instruction::Sw(list_item, 0, dest_list));

            // move fields up by 4 bytes
            self.generator
                .add_instruction(Instruction::Addi(list, list, 4));

            // move pointer up by 4 bytes
            self.generator
                .add_instruction(Instruction::Addi(dest_list, dest_list, 4));

            // decrement terms length
            self.generator
                .add_instruction(Instruction::Addi(length, length, -1));

            self.generator
                .add_instruction(Instruction::J("clone_list".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("clone_list_return".to_string()));

            self.generator.add_instruction(Instruction::Comment(
                "A3 contains return address".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::Jalr(self.discard, callback, 0));
        }
    }

    // A0 pointer to terms array to decrement from
    // A1 is new stack pointer
    // A2 is length of terms array
    // A3 is the return address
    pub fn reverse_clone_list(
        &mut self,
        list: Register,
        dest_list: Register,
        length: Register,
        callback: Register,
    ) {
        self.generator
            .add_instruction(Instruction::Label("reverse_clone_list".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("A2 contains terms length".to_string()));

        self.generator.add_instruction(Instruction::Beq(
            length,
            Register::Zero,
            "reverse_clone_list_return".to_string(),
        ));

        {
            let list_item = Register::T0;
            self.generator
                .add_instruction(Instruction::Lw(list_item, 0, list));

            // Store term in new storage
            self.generator
                .add_instruction(Instruction::Sw(list_item, 0, dest_list));

            // move backwards by 4 bytes
            self.generator
                .add_instruction(Instruction::Addi(list, list, -4));

            // move pointer up by 4 bytes
            self.generator
                .add_instruction(Instruction::Addi(dest_list, dest_list, 4));

            // decrement terms length
            self.generator
                .add_instruction(Instruction::Addi(length, length, -1));

            self.generator
                .add_instruction(Instruction::J("reverse_clone_list".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("reverse_clone_list_return".to_string()));

            self.generator.add_instruction(Instruction::Comment(
                "A3 contains return address".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::Jalr(self.discard, callback, 0));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cek::Cek;

    #[test]
    fn test_cek_machine() {
        // Initialize the CEK machine
        let mut cek = Cek::default();

        // Generate the core CEK implementation
        let ret = cek.init();
        cek.compute(ret);
        cek.return_compute();
        let index = cek.handle_var(ret);
        cek.handle_delay(ret);
        cek.handle_lambda(ret);
        cek.handle_apply(ret);
        cek.handle_constant(ret);
        cek.handle_force(ret);
        cek.handle_error(ret);
        let (second_field, frames_arg, size, callback1) = cek.handle_constr(ret);
        let (list, list_dest, length, callback) = cek.handle_case(ret);

        assert!(
            second_field == list
                && list_dest == frames_arg
                && size == length
                && callback1 == callback
        );
        let (function, argument) = cek.handle_frame_await_arg(ret);
        cek.handle_frame_await_fun_term(ret);
        cek.handle_frame_force();
        cek.handle_frame_constr(ret);

        cek.halt();
        cek.force_evaluate(ret);
        cek.apply_evaluate(ret, function, argument);
        cek.lookup(ret, index);
        cek.clone_list(list, list_dest, length, callback);
        cek.reverse_clone_list(list, list_dest, length, callback);

        // println!("CEK Debug printed is {:#?}", cek);

        let code_gen = cek.generator;

        println!("{}", code_gen.generate());
    }
}
