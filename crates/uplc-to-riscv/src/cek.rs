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
//
// #[derive(Clone, Debug, PartialEq)]
// pub struct BuiltinRuntime {
//     pub(super) args: Vec<Value>,
//     pub fun: DefaultFunction,
//     pub(super) forces: u32,
// }
//
//
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

//AddInteger
//SubtractInteger
//MultiplyInteger
//DivideInteger
//QuotientInteger
//RemainderInteger
//ModInteger
//EqualsInteger
//LessThanInteger
//LessThanEqualsInteger
//AppendByteString
//ConsByteString
//SliceByteString
//LengthOfByteString
//IndexByteString
//EqualsByteString
//LessThanByteString
//LessThanEqualsByteString
//Sha2_256
//Sha3_256
//Blake2b_224
//Blake2b_256
//Keccak_256
//VerifyEd25519Signature
//VerifyEcdsaSecp256k1Signature
//VerifySchnorrSecp256k1Signature
//AppendString
//EqualsString
//EncodeUtf8
//DecodeUtf8
//IfThenElse
//ChooseUnit
//Trace
//FstPair
//SndPair
//ChooseList
//MkCons
//HeadList
//TailList
//NullList
//ChooseData
//ConstrData
//MapData
//ListData
//IData
//BData
//UnConstrData
//UnMapData
//UnListData
//UnIData
//UnBData
//EqualsData
//SerialiseData
//MkPairData
//MkNilData
//MkNilPairData
//Bls12_381_G1_Add
//Bls12_381_G1_Neg
//Bls12_381_G1_ScalarMul
//Bls12_381_G1_Equal
//Bls12_381_G1_Compress
//Bls12_381_G1_Uncompress
//Bls12_381_G1_HashToGroup
//Bls12_381_G2_Add
//Bls12_381_G2_Neg
//Bls12_381_G2_ScalarMul
//Bls12_381_G2_Equal
//Bls12_381_G2_Compress
//Bls12_381_G2_Uncompress
//Bls12_381_G2_HashToGroup
//Bls12_381_MillerLoop
//Bls12_381_MulMlResult
//Bls12_381_FinalVerify
//IntegerToByteString
//ByteStringToInteger
//AndByteString
//OrByteString
//XorByteString
//ComplementByteString
//ReadBit
//WriteBits
//ReplicateByte
//ShiftByteString
//RotateByteString
//CountSetBits
//FindFirstSetBit
//Ripemd_160
//ExpModInteger

// Must be exactly divisible by 4
const force_counts: [u8; 88] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

const arities: [u8; 88] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 1, 1, 3, 2,
    2, 1, 1, 3, 2, 1, 1, 1, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2,
    1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 2, 3, 3, 3, 1, 2, 3, 2, 2, 2, 1, 1, 1, 3,
];

#[derive(Default, Debug)]
pub struct Cek {
    generator: CodeGenerator,
    frames: Register,
    env: Register,
    heap: Register,
    discard: Register,
    return_reg: Register,
    first_arg: Register,
    second_arg: Register,
    third_arg: Register,
    fourth_arg: Register,
    fifth_arg: Register,
    sixth_arg: Register,
    seventh_arg: Register,
    eighth_arg: Register,
    first_temp: Register,
    second_temp: Register,
    third_temp: Register,
    fourth_temp: Register,
    fifth_temp: Register,
    sixth_temp: Register,
    seventh_temp: Register,
}

impl Cek {
    pub fn new() -> Cek {
        Cek {
            generator: CodeGenerator::default(),
            frames: Register::Sp,
            env: Register::S1,
            heap: Register::S2,
            discard: Register::T0,
            return_reg: Register::A0,
            first_arg: Register::A0,
            second_arg: Register::A1,
            third_arg: Register::A2,
            fourth_arg: Register::A3,
            fifth_arg: Register::A4,
            sixth_arg: Register::A5,
            seventh_arg: Register::A6,
            eighth_arg: Register::A7,
            first_temp: Register::T0,
            second_temp: Register::T1,
            third_temp: Register::T2,
            fourth_temp: Register::T3,
            fifth_temp: Register::T4,
            sixth_temp: Register::T5,
            seventh_temp: Register::T6,
        }
    }

    pub fn cek_assembly(mut self, bytes: Vec<u8>) -> CodeGenerator {
        // Generate the core CEK implementation
        self.generator
            .add_instruction(Instruction::Global("_start".to_string()));
        self.generator
            .add_instruction(Instruction::Label("_start".to_string()));
        let ret = self.init();
        self.compute();
        self.return_compute();
        self.handle_var();
        self.handle_delay();
        self.handle_lambda();
        self.handle_apply();
        self.handle_constant();
        self.handle_force();
        self.handle_error();
        self.handle_builtin();
        self.handle_constr(ret);
        let (list, list_dest, length, callback) = self.handle_case(ret);
        self.handle_frame_await_fun_term(ret);
        let (function, argument) = self.handle_frame_await_arg(ret);
        self.handle_frame_await_fun_value(ret);
        self.handle_frame_force();
        self.handle_frame_constr(ret);
        self.handle_frame_case(ret);
        self.handle_no_frame(ret);
        self.halt();
        self.force_evaluate(ret);
        self.apply_evaluate(ret, function, argument);
        self.lookup();
        self.clone_list(list, list_dest, length, callback);
        self.reverse_clone_list(list, list_dest, length, callback);
        self.initial_term(bytes);

        // self.generator
        //     .add_instruction(Instruction::Section("heap".to_string()));

        // self.generator
        //     .add_instruction(Instruction::Label("heap".to_string()));

        // self.generator
        //     .add_instruction(Instruction::Byte(vec![0, 0, 0, 0]));

        self.generator
    }

    pub fn init(&mut self) -> Register {
        self.generator
            .add_instruction(Instruction::Section("text".to_string()));

        self.generator
            .add_instruction(Instruction::Label("init".to_string()));

        self.generator
            .add_instruction(Instruction::Lui(self.heap, 0xc0000));

        // self.generator
        //     .add_instruction(Instruction::Lui(self.frames, 0xe0000));

        self.generator.add_instruction(Instruction::Comment(
            "1 byte for NoFrame allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(self.frames, self.frames, -1));

        self.generator
            .add_instruction(Instruction::Comment("Tag is 6 for NoFrame".to_string()));

        let frame_tag = self.first_temp;

        self.generator
            .add_instruction(Instruction::Li(frame_tag, 6));

        self.generator.add_instruction(Instruction::Comment(
            "Push NoFrame tag onto stack".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Sb(frame_tag, 0, self.frames));

        self.generator.add_instruction(Instruction::Comment(
            "Environment stack pointer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Mv(self.env, Register::Zero));

        self.generator
            .add_instruction(Instruction::Comment("A0 is return register".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Load address of initial_term".to_string(),
        ));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::La(ret, "initial_term".to_string()));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));

        ret
    }

    // The return register
    pub fn compute(&mut self) {
        let term = self.first_arg;
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

        let term_tag = self.first_temp;

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

        let match_tag = self.second_temp;

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

        let frame_tag = self.first_temp;

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

        let match_tag = self.second_temp;

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

    pub fn handle_var(&mut self) {
        let var = self.first_arg;
        self.generator
            .add_instruction(Instruction::Label("handle_var".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "load debruijn index into temp".to_string(),
        ));

        let var_index = self.first_temp;

        self.generator
            .add_instruction(Instruction::Lw(var_index, 1, var));

        self.generator.add_instruction(Instruction::Comment(
            "Put debruijn index into A0".to_string(),
        ));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Mv(ret, var_index));

        self.generator
            .add_instruction(Instruction::J("lookup".to_string()));
    }

    pub fn handle_delay(&mut self) {
        let delay_term = self.first_arg;
        let heap = self.heap;
        let env = self.env;
        self.generator
            .add_instruction(Instruction::Label("handle_delay".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("Term body is next byte".to_string()));

        let body = self.first_temp;

        self.generator
            .add_instruction(Instruction::Addi(body, delay_term, 1));

        self.generator.add_instruction(Instruction::Comment(
            "9 bytes for DelayValue allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(heap, heap, 9));

        self.generator
            .add_instruction(Instruction::Comment("tag is 1 in rust".to_string()));

        let vdelay_tag = self.second_temp;

        self.generator
            .add_instruction(Instruction::Li(vdelay_tag, 1));

        self.generator
            .add_instruction(Instruction::Comment("first byte is tag".to_string()));

        self.generator
            .add_instruction(Instruction::Sb(vdelay_tag, -9, heap));

        self.generator
            .add_instruction(Instruction::Comment("Store body pointer".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(body, -8, heap));

        self.generator
            .add_instruction(Instruction::Comment("Store environment".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(env, -4, heap));

        self.generator
            .add_instruction(Instruction::Comment("Put return value into A0".to_string()));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Addi(ret, heap, -9));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn handle_lambda(&mut self) {
        let lambda_term = self.first_arg;
        let heap = self.heap;
        let env = self.env;
        self.generator
            .add_instruction(Instruction::Label("handle_lambda".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("Term body is next byte".to_string()));

        let body = self.first_temp;

        self.generator
            .add_instruction(Instruction::Addi(body, lambda_term, 1));

        self.generator.add_instruction(Instruction::Comment(
            "9 bytes for LambdaValue allocation".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Addi(heap, heap, 9));

        self.generator
            .add_instruction(Instruction::Comment("tag is 2 in rust".to_string()));

        let vlambda_tag = self.second_temp;

        self.generator
            .add_instruction(Instruction::Li(vlambda_tag, 2));

        self.generator
            .add_instruction(Instruction::Comment("first byte is tag".to_string()));

        self.generator
            .add_instruction(Instruction::Sb(vlambda_tag, -9, heap));

        self.generator
            .add_instruction(Instruction::Comment("Store body".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(body, -8, heap));

        self.generator
            .add_instruction(Instruction::Comment("Store environment".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(env, -4, heap));

        self.generator
            .add_instruction(Instruction::Comment("Put return value into A0".to_string()));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Addi(ret, heap, -9));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn handle_apply(&mut self) {
        let apply_term = self.first_arg;
        let frames = self.frames;
        let env = self.env;
        self.generator
            .add_instruction(Instruction::Label("handle_apply".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Apply is tag |argument address| function".to_string(),
        ));
        self.generator.add_instruction(Instruction::Comment(
            "Function is 5 bytes after tag location".to_string(),
        ));

        let function = self.first_temp;

        self.generator
            .add_instruction(Instruction::Addi(function, apply_term, 5));

        self.generator
            .add_instruction(Instruction::Comment("Load argument into temp".to_string()));

        let argument = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(argument, 1, apply_term));

        self.generator.add_instruction(Instruction::Comment(
            "9 bytes for FrameAwaitFunTerm allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(frames, frames, -9));

        self.generator.add_instruction(Instruction::Comment(
            "Tag is 1 for FrameAwaitFunTerm".to_string(),
        ));
        let frame_tag = self.third_temp;

        self.generator
            .add_instruction(Instruction::Li(frame_tag, 1));

        self.generator
            .add_instruction(Instruction::Comment("Push tag onto stack".to_string()));

        self.generator
            .add_instruction(Instruction::Sb(frame_tag, 0, frames));

        self.generator
            .add_instruction(Instruction::Comment("Push argument onto stack".to_string()));

        self.generator
            .add_instruction(Instruction::Sw(argument, 1, frames));

        self.generator.add_instruction(Instruction::Comment(
            "Push environment onto stack".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Sw(env, 5, frames));

        self.generator.add_instruction(Instruction::Comment(
            "Put function address into A0".to_string(),
        ));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Mv(ret, function));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));
    }

    pub fn handle_constant(&mut self) {
        let constant_term = self.first_arg;
        let heap = self.heap;

        self.generator
            .add_instruction(Instruction::Label("handle_constant".to_string()));

        // store pointer to constant in T0
        let constant = self.first_temp;
        self.generator
            .add_instruction(Instruction::Addi(constant, constant_term, 1));

        // allocate 5 bytes on the heap
        self.generator
            .add_instruction(Instruction::Addi(heap, heap, 5));

        let constant_tag = self.second_temp;
        self.generator
            .add_instruction(Instruction::Li(constant_tag, 0));

        self.generator
            .add_instruction(Instruction::Sb(constant_tag, -5, heap));

        self.generator
            .add_instruction(Instruction::Sw(constant, -4, heap));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Addi(ret, heap, -5));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn handle_force(&mut self) {
        let force_term = self.first_arg;
        let frames = self.frames;

        self.generator
            .add_instruction(Instruction::Label("handle_force".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("Load term body".to_string()));

        let body = self.first_temp;

        self.generator
            .add_instruction(Instruction::Addi(body, force_term, 1));

        self.generator.add_instruction(Instruction::Comment(
            "1 byte for FrameForce allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(frames, frames, -1));

        self.generator
            .add_instruction(Instruction::Comment("Tag is 3 for FrameForce".to_string()));

        let tag = self.second_temp;
        self.generator.add_instruction(Instruction::Li(tag, 3));

        self.generator.add_instruction(Instruction::Comment(
            "Push FrameForce tag onto stack".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Sb(tag, 0, frames));

        self.generator.add_instruction(Instruction::Comment(
            "Put term body address into A0".to_string(),
        ));

        let ret = self.return_reg;
        self.generator.add_instruction(Instruction::Mv(ret, body));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));
    }

    pub fn handle_error(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("handle_error".to_string()));

        self.generator
            .add_instruction(Instruction::Comment("Load -1 into A0".to_string()));

        let ret = self.return_reg;
        self.generator.add_instruction(Instruction::Li(ret, -1));

        self.generator
            .add_instruction(Instruction::J("halt".to_string()));
    }

    pub fn handle_builtin(&mut self) {
        let builtin = self.first_arg;
        let heap = self.heap;

        self.generator
            .add_instruction(Instruction::Label("handle_builtin".to_string()));

        let builtin_func_index = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lbu(builtin_func_index, 1, builtin));

        // 1 byte for value tag, 1 byte for func index, 1 byte for forces, 4 bytes for args length 0
        self.generator.add_instruction(Instruction::Comment(
            "7 bytes for VBuiltin allocation".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Addi(heap, heap, 7));

        let value_tag = self.second_temp;
        self.generator
            .add_instruction(Instruction::Li(value_tag, 3));

        self.generator
            .add_instruction(Instruction::Sb(value_tag, -7, heap));

        self.generator
            .add_instruction(Instruction::Sb(builtin_func_index, -6, heap));

        let force_lookup = self.third_temp;
        self.generator
            .add_instruction(Instruction::La(force_lookup, "force_counts".to_string()));

        self.generator.add_instruction(Instruction::Add(
            force_lookup,
            force_lookup,
            builtin_func_index,
        ));

        let forces = self.fourth_temp;
        self.generator
            .add_instruction(Instruction::Lbu(forces, 0, force_lookup));

        self.generator
            .add_instruction(Instruction::Sb(forces, -5, heap));

        self.generator
            .add_instruction(Instruction::Sw(Register::Zero, -4, heap));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Addi(ret, heap, -7));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
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

            let elements_byte_size = Register::T4;

            self.generator.add_instruction(Instruction::Slli(
                elements_byte_size,
                constr_len_popped,
                2,
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
        let arg = ret;

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

    pub fn handle_frame_await_fun_value(&mut self, ret: Register) {
        let function = ret;

        self.generator.add_instruction(Instruction::Label(
            "handle_frame_await_fun_value".to_string(),
        ));

        self.generator.add_instruction(Instruction::Comment(
            "load function value pointer from stack".to_string(),
        ));

        let arg = Register::T0;
        self.generator
            .add_instruction(Instruction::Lw(arg, 1, self.frames));

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
            // Mutate fields length to be 1 less since we are popping from the front
            let current_field_len = fields_len;

            self.generator.add_instruction(Instruction::Addi(
                current_field_len,
                current_field_len,
                -1,
            ));

            self.generator
                .add_instruction(Instruction::Sw(current_field_len, 9, frame));

            let first_field = Register::A5;
            self.generator
                .add_instruction(Instruction::Lw(first_field, 13, frame));

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

    pub fn handle_no_frame(&mut self, ret: Register) {
        self.generator
            .add_instruction(Instruction::Label("handle_no_frame".to_string()));

        let temp = Register::T0;
        self.generator
            .add_instruction(Instruction::Lw(temp, 1, ret));

        self.generator
            .add_instruction(Instruction::Lw(ret, 0, temp));

        self.generator
            .add_instruction(Instruction::J("halt".to_string()));
    }

    pub fn halt(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("halt".to_string()));

        self.generator
            .add_instruction(Instruction::Li(Register::A7, 93));

        self.generator.add_instruction(Instruction::Ecall);
    }

    pub fn force_evaluate(&mut self, ret: Register) {
        let function = ret;
        self.generator
            .add_instruction(Instruction::Label("force_evaluate".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Value address should be in A0".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Comment("Load value tag".to_string()));

        let tag = Register::T0;
        self.generator
            .add_instruction(Instruction::Lbu(tag, 0, function));

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
                .add_instruction(Instruction::Lw(body, 1, function));

            self.generator.add_instruction(Instruction::Comment(
                "load environment from a0 which is Value".to_string(),
            ));

            let environment = Register::T1;
            self.generator
                .add_instruction(Instruction::Lw(environment, 5, function));

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
                let force_count = Register::T1;
                self.generator
                    .add_instruction(Instruction::Lbu(force_count, 2, function));

                self.generator.add_instruction(Instruction::Beq(
                    Register::Zero,
                    force_count,
                    "force_evaluate_error".to_string(),
                ));

                self.generator
                    .add_instruction(Instruction::Addi(force_count, force_count, -1));

                // Create clone of current value with number of forces changed
                self.generator
                    .add_instruction(Instruction::Addi(self.heap, self.heap, 7));

                self.generator
                    .add_instruction(Instruction::Sb(tag, -7, self.heap));

                let builtin_func_index = Register::T2;
                self.generator
                    .add_instruction(Instruction::Lbu(builtin_func_index, 1, function));

                self.generator
                    .add_instruction(Instruction::Sb(builtin_func_index, -6, self.heap));

                self.generator
                    .add_instruction(Instruction::Sb(force_count, -5, self.heap));

                // Arguments still 0 here
                self.generator
                    .add_instruction(Instruction::Sw(Register::Zero, -4, self.heap));

                // Store new value in ret
                self.generator
                    .add_instruction(Instruction::Addi(ret, self.heap, -7));

                // If still have forces to apply then return
                self.generator.add_instruction(Instruction::Bne(
                    Register::Zero,
                    force_count,
                    "return".to_string(),
                ));

                let arguments_length = Register::T0;
                self.generator
                    .add_instruction(Instruction::Lw(arguments_length, 3, function));

                let arity_lookup = Register::T1;
                self.generator
                    .add_instruction(Instruction::La(arity_lookup, "arities".to_string()));

                self.generator.add_instruction(Instruction::Add(
                    arity_lookup,
                    arity_lookup,
                    builtin_func_index,
                ));

                let arity = Register::T3;
                self.generator
                    .add_instruction(Instruction::Lbu(arity, 0, arity_lookup));

                // If all arguments not applied then return.
                self.generator.add_instruction(Instruction::Bne(
                    arity,
                    arguments_length,
                    "return".to_string(),
                ));

                self.generator
                    .add_instruction(Instruction::J("eval_builtin_app".to_string()));
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

            let builtin_value_tag = Register::T1;
            self.generator
                .add_instruction(Instruction::Li(builtin_value_tag, 3));

            self.generator.add_instruction(Instruction::Bne(
                function_tag,
                builtin_value_tag,
                "apply_evaluate_error".to_string(),
            ));

            {
                let force_count = Register::T1;
                self.generator
                    .add_instruction(Instruction::Lbu(force_count, 2, function));

                self.generator.add_instruction(Instruction::Bne(
                    Register::Zero,
                    force_count,
                    "apply_evaluate_error".to_string(),
                ));

                let builtin_func_index = Register::T2;
                self.generator
                    .add_instruction(Instruction::Lbu(builtin_func_index, 1, function));

                let arguments_length = Register::T3;
                self.generator
                    .add_instruction(Instruction::Lw(arguments_length, 3, function));

                let arity_lookup = Register::T4;
                self.generator
                    .add_instruction(Instruction::La(arity_lookup, "arities".to_string()));

                self.generator.add_instruction(Instruction::Add(
                    arity_lookup,
                    arity_lookup,
                    builtin_func_index,
                ));

                let arity = Register::T4;
                self.generator
                    .add_instruction(Instruction::Lbu(arity, 0, arity_lookup));

                self.generator.add_instruction(Instruction::Beq(
                    arity,
                    arguments_length,
                    "apply_evaluate_error".to_string(),
                ));

                //Create clone of value with new arg included and assign to ret

                let new_args_length = Register::T1;
                self.generator.add_instruction(Instruction::Addi(
                    new_args_length,
                    arguments_length,
                    1,
                ));

                let heap_allocation = Register::T5;
                self.generator.add_instruction(Instruction::Slli(
                    heap_allocation,
                    new_args_length,
                    2,
                ));

                self.generator.add_instruction(Instruction::Addi(
                    heap_allocation,
                    heap_allocation,
                    7,
                ));

                let cloned_value = Register::T6;
                self.generator
                    .add_instruction(Instruction::Mv(cloned_value, self.heap));

                self.generator.add_instruction(Instruction::Add(
                    self.heap,
                    self.heap,
                    heap_allocation,
                ));

                self.generator
                    .add_instruction(Instruction::Sb(function_tag, 0, cloned_value));

                self.generator.add_instruction(Instruction::Sb(
                    builtin_func_index,
                    1,
                    cloned_value,
                ));

                // Forces was checked to be 0 above
                self.generator
                    .add_instruction(Instruction::Sb(Register::Zero, 2, cloned_value));

                // Arguments is now the new_args_length
                self.generator
                    .add_instruction(Instruction::Sw(new_args_length, 3, cloned_value));

                let store_new_args_length = Register::A7;
                self.generator
                    .add_instruction(Instruction::Mv(store_new_args_length, new_args_length));

                let store_new_arg = Register::A6;
                self.generator
                    .add_instruction(Instruction::Mv(store_new_arg, argument));

                let store_arity = Register::A5;
                self.generator
                    .add_instruction(Instruction::Mv(store_arity, arity));

                let new_value = Register::A4;
                self.generator
                    .add_instruction(Instruction::Mv(new_value, cloned_value));

                let size = Register::A2;
                self.generator
                    .add_instruction(Instruction::Mv(size, arguments_length));

                let dest_list = Register::A1;
                self.generator
                    .add_instruction(Instruction::Addi(dest_list, cloned_value, 7));

                let src_list = ret;
                self.generator
                    .add_instruction(Instruction::Addi(src_list, function, 7));

                let callback = Register::A3;
                self.generator
                    .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

                // We can store the new arg value one word before current heap since it's not modified anywhere yet
                self.generator
                    .add_instruction(Instruction::Sw(store_new_arg, -4, self.heap));

                self.generator
                    .add_instruction(Instruction::Mv(ret, new_value));

                // Check arity
                self.generator.add_instruction(Instruction::Bne(
                    store_arity,
                    store_new_args_length,
                    "return".to_string(),
                ));

                self.generator
                    .add_instruction(Instruction::J("apply_evaluate_builtin".to_string()));
            }

            {
                self.generator
                    .add_instruction(Instruction::Label("apply_evaluate_error".to_string()));

                self.generator
                    .add_instruction(Instruction::J("handle_error".to_string()));
            }
        }
    }

    pub fn lookup(&mut self) {
        let index = self.first_arg;
        let env = self.env;
        self.generator
            .add_instruction(Instruction::Label("lookup".to_string()));

        let current_index = self.first_temp;

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

            let current_env = self.second_temp;
            self.generator
                .add_instruction(Instruction::Lw(current_env, 4, env));

            self.generator
                .add_instruction(Instruction::Mv(env, current_env));

            let ret = self.return_reg;
            self.generator
                .add_instruction(Instruction::Mv(ret, current_index));

            self.generator
                .add_instruction(Instruction::J("lookup".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("lookup_return".to_string()));

            let ret = self.return_reg;
            self.generator.add_instruction(Instruction::Lw(ret, 0, env));

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

    pub fn initial_term(&mut self, bytes: Vec<u8>) {
        self.generator
            .add_instruction(Instruction::Section("data".to_string()));

        self.generator
            .add_instruction(Instruction::Label("initial_term".to_string()));

        self.generator.add_instruction(Instruction::Byte(bytes));

        self.generator
            .add_instruction(Instruction::Label("force_counts".to_string()));
        self.generator
            .add_instruction(Instruction::Byte(force_counts.to_vec()));

        self.generator
            .add_instruction(Instruction::Label("arities".to_string()));
        self.generator
            .add_instruction(Instruction::Byte(arities.to_vec()));
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use emulator::ExecutionResult;
    use risc_v_gen::emulator::verify_file;
    use uplc::ast::{DeBruijn, Program, Term};
    use uplc_serializer::serialize;

    use crate::cek::Cek;

    #[test]
    fn test_cek_machine() {
        // Initialize the CEK machine
        let mut cek = Cek::default();

        // Generate the core CEK implementation
        let ret = cek.init();
        cek.compute();
        cek.return_compute();
        cek.handle_var();
        cek.handle_delay();
        cek.handle_lambda();
        cek.handle_apply();
        cek.handle_constant();
        cek.handle_force();
        cek.handle_error();
        cek.handle_builtin();
        let (second_field, frames_arg, size, callback1) = cek.handle_constr(ret);
        let (list, list_dest, length, callback) = cek.handle_case(ret);

        assert!(
            second_field == list
                && list_dest == frames_arg
                && size == length
                && callback1 == callback
        );
        cek.handle_frame_await_fun_term(ret);
        let (function, argument) = cek.handle_frame_await_arg(ret);
        cek.handle_frame_await_fun_value(ret);
        cek.handle_frame_force();
        cek.handle_frame_constr(ret);
        cek.handle_frame_case(ret);
        cek.halt();
        cek.force_evaluate(ret);
        cek.apply_evaluate(ret, function, argument);
        cek.lookup();
        cek.clone_list(list, list_dest, length, callback);
        cek.reverse_clone_list(list, list_dest, length, callback);
        cek.initial_term(vec![
            /*apply*/ 3, /* arg pointer*/ 11, 0, 0, 144, /*lambda*/ 2,
            /*var*/ 0, 1, 0, 0, 0, /*constant*/ 4, 13, 0, 0, 0,
        ]);

        let code_gen = cek.generator;

        println!("{}", code_gen.generate());
    }

    #[test]
    #[ignore]
    fn test_compilation() {
        let thing = Cek::default();

        let gene = thing.cek_assembly(vec![
            /*apply*/ 3, /* arg pointer*/ 11, 0, 0, 144, /*lambda*/ 2,
            /*var*/ 0, 1, 0, 0, 0, /*constant*/ 4, 13, 0, 0, 0,
        ]);

        gene.save_to_file("../../test.s").unwrap();
    }

    #[test]
    fn test_apply_lambda_var_constant() {
        let thing = Cek::default();

        let gene = thing.cek_assembly(vec![
            /*apply*/ 3, /* arg pointer*/ 11, 0, 0, 144, /*lambda*/ 2,
            /*var*/ 0, 1, 0, 0, 0, /*constant*/ 4, 13, 0, 0, 0,
        ]);

        gene.save_to_file("test_apply.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_apply.o",
                "test_apply.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_apply.elf",
                "-T",
                "../../linker/link.ld",
                "test_apply.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_apply.elf").unwrap();

        match v.0 {
            ExecutionResult::Halt(result, _step) => assert_eq!(result, 13),
            _ => unreachable!("HOW?"),
        }
    }

    #[test]
    fn test_force_delay_error() {
        let thing = Cek::default();

        let gene = thing.cek_assembly(vec![5, 1, 6, 0]);

        gene.save_to_file("test_force.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_force.o",
                "test_force.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_force.elf",
                "-T",
                "../../linker/link.ld",
                "test_force.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_force.elf").unwrap();

        match v.0 {
            ExecutionResult::Halt(result, _step) => assert_eq!(result, u32::MAX),
            _ => unreachable!("HOW?"),
        }
    }

    #[test]
    fn test_force_delay_error_serialize() {
        let thing = Cek::default();

        let term = Term::Error.delay().force();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_serialize.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_serialize.o",
                "test_serialize.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_serialize.elf",
                "-T",
                "../../linker/link.ld",
                "test_serialize.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_serialize.elf").unwrap();

        match v.0 {
            ExecutionResult::Halt(result, _step) => assert_eq!(result, u32::MAX),
            _ => unreachable!("HOW?"),
        }
    }

    #[test]
    fn test_apply_lambda_force_var_delay_error_serialize() {
        let thing = Cek::default();

        let term = Term::var("x")
            .force()
            .lambda("x")
            .apply(Term::Error.delay());

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        dbg!(&riscv_program);

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_serialize_2.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_serialize_2.o",
                "test_serialize_2.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_serialize_2.elf",
                "-T",
                "../../linker/link.ld",
                "test_serialize_2.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_serialize_2.elf").unwrap();

        match v.0 {
            ExecutionResult::Halt(result, _step) => assert_eq!(result, u32::MAX),
            _ => unreachable!("HOW?"),
        }
    }
}
