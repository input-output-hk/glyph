use risc_v_gen::{CodeGenerator, Instruction, Register};
use uplc_serializer::constants::const_tag::{self, BOOL};
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
const FORCE_COUNTS: [u8; 88] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

const ARITIES: [u8; 88] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 1, 1, 3, 2,
    2, 1, 1, 3, 2, 1, 1, 1, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2,
    1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 2, 3, 3, 3, 1, 2, 3, 2, 2, 2, 1, 1, 1, 3,
];

#[derive(Debug)]
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
        self.init();
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
        self.handle_constr();
        self.handle_case();
        self.handle_frame_await_fun_term();
        self.handle_frame_await_arg();
        self.handle_frame_await_fun_value();
        self.handle_frame_force();
        self.handle_frame_constr();
        self.handle_frame_case();
        self.handle_no_frame();
        self.halt();
        self.force_evaluate();
        self.apply_evaluate();
        self.lookup();
        self.clone_list();
        self.reverse_clone_list();
        self.eval_builtin_app();
        self.unwrap_integer();
        self.unwrap_bytestring();
        self.add_integer();
        self.sub_integer();
        self.multiply_integer();
        self.divide_integer();
        self.quotient_integer();
        self.remainder_integer();
        self.mod_integer();
        self.equals_integer();
        self.less_than_integer();
        self.less_than_equals_integer();
        self.append_bytestring();
        self.add_signed_integers();
        self.compare_magnitude();
        self.sub_signed_integers();
        self.initial_term(bytes);
        // self.generator
        //     .add_instruction(Instruction::Section("heap".to_string()));

        // self.generator
        //     .add_instruction(Instruction::Label("heap".to_string()));

        // self.generator
        //     .add_instruction(Instruction::Byte(vec![0, 0, 0, 0]));

        self.generator
    }

    pub fn init(&mut self) {
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
    }

    // TODO: for both compute and return_compute, We can compute the jump via term offset
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

    pub fn handle_constr(&mut self) {
        let constr = self.first_arg;
        let heap = self.heap;
        let env = self.env;
        let frames = self.frames;

        self.generator
            .add_instruction(Instruction::Label("handle_constr".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Load the tag of the constr into T0".to_string(),
        ));

        let constr_tag = self.first_temp;

        self.generator
            .add_instruction(Instruction::Lw(constr_tag, 1, constr));

        self.generator.add_instruction(Instruction::Comment(
            "Load the length of the constr fields into T1".to_string(),
        ));

        let constr_len = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(constr_len, 5, constr));

        self.generator.add_instruction(Instruction::Beq(
            constr_len,
            Register::Zero,
            "handle_constr_empty".to_string(),
        ));

        {
            self.generator.add_instruction(Instruction::Comment(
                "-- Fields is not empty --".to_string(),
            ));

            // Overwriting constr_len
            let constr_len_popped = constr_len;

            self.generator
                .add_instruction(Instruction::Addi(constr_len_popped, constr_len, -1));

            self.generator.add_instruction(Instruction::Comment(
                "Minimum size for FrameConstr is 17 bytes".to_string(),
            ));

            let min_byte_size = self.third_temp;
            self.generator
                .add_instruction(Instruction::Li(min_byte_size, 17));

            let elements_byte_size = self.fourth_temp;

            self.generator.add_instruction(Instruction::Slli(
                elements_byte_size,
                constr_len_popped,
                2,
            ));
            // Overwriting min_byte_size
            let total_byte_size = min_byte_size;

            self.generator.add_instruction(Instruction::Add(
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
            self.generator
                .add_instruction(Instruction::Sub(frames, frames, total_byte_size));

            let frame_builder = self.fifth_temp;
            self.generator
                .add_instruction(Instruction::Mv(frame_builder, frames));

            // Overwriting elements_byte_size
            let constr_frame_tag = elements_byte_size;
            self.generator
                .add_instruction(Instruction::Li(constr_frame_tag, 4));

            self.generator
                .add_instruction(Instruction::Comment("store frame tag".to_string()));
            self.generator
                .add_instruction(Instruction::Sb(constr_frame_tag, 0, frame_builder));

            self.generator
                .add_instruction(Instruction::Comment("move up 1 byte".to_string()));
            self.generator
                .add_instruction(Instruction::Addi(frame_builder, frame_builder, 1));

            self.generator
                .add_instruction(Instruction::Comment(" store constr tag".to_string()));
            self.generator
                .add_instruction(Instruction::Sw(constr_tag, 0, frame_builder));
            self.generator
                .add_instruction(Instruction::Comment("move up 4 bytes".to_string()));
            self.generator
                .add_instruction(Instruction::Addi(frame_builder, frame_builder, 4));

            self.generator
                .add_instruction(Instruction::Comment("store environment".to_string()));
            self.generator
                .add_instruction(Instruction::Sw(env, 0, frame_builder));

            self.generator
                .add_instruction(Instruction::Comment("move up 4 bytes".to_string()));
            self.generator
                .add_instruction(Instruction::Addi(frame_builder, frame_builder, 4));

            self.generator
                .add_instruction(Instruction::Comment("store fields length -1".to_string()));

            self.generator
                .add_instruction(Instruction::Sw(constr_len_popped, 0, frame_builder));

            self.generator
                .add_instruction(Instruction::Comment("move up 4 bytes".to_string()));
            self.generator
                .add_instruction(Instruction::Addi(frame_builder, frame_builder, 4));

            self.generator
                .add_instruction(Instruction::Comment("Load first field to A4".to_string()));

            let first_field = self.fifth_arg;

            self.generator
                .add_instruction(Instruction::Lw(first_field, 9, constr));

            self.generator.add_instruction(Instruction::Comment(
                "move fields length - 1 to A2".to_string(),
            ));

            let size = self.third_arg;

            self.generator
                .add_instruction(Instruction::Mv(size, constr_len_popped));

            self.generator.add_instruction(Instruction::Comment(
                "move current stack pointer to A1".to_string(),
            ));

            let frames_arg = self.second_arg;

            self.generator
                .add_instruction(Instruction::Mv(frames_arg, frame_builder));

            self.generator.add_instruction(Instruction::Comment(
                "move A0 pointer to second element in fields (regardless if there or not)"
                    .to_string(),
            ));

            let second_field = self.first_arg;

            self.generator
                .add_instruction(Instruction::Addi(second_field, constr, 13));

            // Takes in A0 - elements pointer, A1 - destination pointer, A2 - length
            // A3 - return address

            let callback = self.fourth_arg;

            self.generator
                .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

            // frames_arg points to last 4 bytes in allocated frame
            self.generator
                .add_instruction(Instruction::Mv(frame_builder, frames_arg));

            self.generator.add_instruction(Instruction::Comment(
                "Store 0 for values length".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::Sw(Register::Zero, 0, frame_builder));

            self.generator.add_instruction(Instruction::Comment(
                "Mv A4 (pointer to first field term) to A0".to_string(),
            ));
            let ret = self.return_reg;
            self.generator
                .add_instruction(Instruction::Mv(ret, first_field));

            self.generator
                .add_instruction(Instruction::J("compute".to_string()));
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
                .add_instruction(Instruction::Addi(heap, heap, 9));

            let vconstr_tag = self.third_temp;

            self.generator
                .add_instruction(Instruction::Li(vconstr_tag, 4));

            self.generator
                .add_instruction(Instruction::Sb(vconstr_tag, -9, heap));

            self.generator
                .add_instruction(Instruction::Sw(constr_tag, -8, heap));

            self.generator
                .add_instruction(Instruction::Sw(constr_len, -4, heap));

            let ret = self.return_reg;
            self.generator
                .add_instruction(Instruction::Addi(ret, heap, -9));

            self.generator
                .add_instruction(Instruction::J("return".to_string()));
        }
    }

    pub fn handle_case(&mut self) {
        let case = self.first_arg;
        let frames = self.frames;
        let env = self.env;
        self.generator
            .add_instruction(Instruction::Label("handle_case".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Load the term pointer of the constr of case into A4".to_string(),
        ));

        // Store constr to compute on in A4
        let constr = self.fifth_arg;
        self.generator
            .add_instruction(Instruction::Lw(constr, 1, case));

        self.generator.add_instruction(Instruction::Comment(
            "Load the length of the constr fields into T1".to_string(),
        ));

        let size = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(size, 5, constr));

        self.generator.add_instruction(Instruction::Comment(
            "Minimum size for FrameCase is 9 bytes".to_string(),
        ));

        let min_bytes = self.second_temp;
        self.generator
            .add_instruction(Instruction::Li(min_bytes, 9));

        let elements_byte_size = self.third_temp;
        self.generator
            .add_instruction(Instruction::Slli(elements_byte_size, size, 2));

        // Overwrite elements_byte_size
        let total_byte_size = elements_byte_size;
        self.generator.add_instruction(Instruction::Add(
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
            .add_instruction(Instruction::Sub(frames, frames, total_byte_size));

        let frames_builder = self.fourth_temp;
        self.generator
            .add_instruction(Instruction::Mv(frames_builder, frames));

        // FrameCase tag
        let frame_case_tag = self.fifth_temp;
        self.generator
            .add_instruction(Instruction::Li(frame_case_tag, 5));

        self.generator
            .add_instruction(Instruction::Sb(frame_case_tag, 0, frames_builder));

        self.generator
            .add_instruction(Instruction::Addi(frames_builder, frames_builder, 1));

        self.generator
            .add_instruction(Instruction::Sw(env, 0, frames_builder));

        self.generator
            .add_instruction(Instruction::Addi(frames_builder, frames_builder, 4));

        self.generator
            .add_instruction(Instruction::Sw(size, 0, frames_builder));

        self.generator
            .add_instruction(Instruction::Addi(frames_builder, frames_builder, 4));
        // A0 pointer to terms array
        // A1 is new stack pointer
        // A2 is length of terms array
        // A3 holds return address

        let list_size = self.third_arg;
        self.generator
            .add_instruction(Instruction::Mv(list_size, size));

        let frames_arg = self.second_arg;

        self.generator
            .add_instruction(Instruction::Mv(frames_arg, frames_builder));

        let branches = self.first_arg;
        self.generator
            .add_instruction(Instruction::Addi(branches, case, 9));

        let callback = self.fourth_arg;

        self.generator
            .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

        let ret = self.return_reg;
        // Move term pointer into A0
        self.generator.add_instruction(Instruction::Mv(ret, constr));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));
    }

    pub fn handle_frame_await_arg(&mut self) {
        let arg = self.first_arg;
        let frames = self.frames;
        self.generator
            .add_instruction(Instruction::Label("handle_frame_await_arg".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "load function value pointer from stack".to_string(),
        ));

        let function = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(function, 1, frames));

        self.generator.add_instruction(Instruction::Comment(
            "reset stack Kontinuation pointer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Addi(frames, frames, 5));

        let second_eval_arg = self.second_arg;

        self.generator
            .add_instruction(Instruction::Mv(second_eval_arg, arg));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Mv(ret, function));

        self.generator
            .add_instruction(Instruction::J("apply_evaluate".to_string()));
    }

    // Takes in a0 and passes it to apply_evaluate
    pub fn handle_frame_await_fun_term(&mut self) {
        let function = self.first_arg;
        let frames = self.frames;
        let env = self.env;

        self.generator.add_instruction(Instruction::Label(
            "handle_frame_await_fun_term".to_string(),
        ));

        self.generator.add_instruction(Instruction::Comment(
            "load argument pointer from stack".to_string(),
        ));

        let argument = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(argument, 1, frames));

        self.generator.add_instruction(Instruction::Comment(
            "load environment from stack".to_string(),
        ));

        let environment = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(environment, 5, frames));

        self.generator.add_instruction(Instruction::Comment(
            "reset stack Kontinuation pointer".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(frames, frames, 9));

        self.generator.add_instruction(Instruction::Comment(
            "5 bytes for FrameAwaitArg allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(frames, frames, -5));

        self.generator.add_instruction(Instruction::Comment(
            "Tag is 0 for FrameAwaitArg".to_string(),
        ));

        let frame_tag = self.third_temp;
        self.generator
            .add_instruction(Instruction::Li(frame_tag, 0));

        self.generator
            .add_instruction(Instruction::Comment("Push tag onto stack".to_string()));

        self.generator
            .add_instruction(Instruction::Sb(frame_tag, 0, frames));

        self.generator.add_instruction(Instruction::Comment(
            "Push function value pointer onto stack".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Sw(function, 1, frames));

        self.generator.add_instruction(Instruction::Comment(
            "Set new environment pointer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Mv(env, environment));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Mv(ret, argument));

        self.generator
            .add_instruction(Instruction::J("compute".to_string()));
    }

    pub fn handle_frame_await_fun_value(&mut self) {
        let function = self.first_arg;
        let frames = self.frames;

        self.generator.add_instruction(Instruction::Label(
            "handle_frame_await_fun_value".to_string(),
        ));

        self.generator.add_instruction(Instruction::Comment(
            "load function value pointer from stack".to_string(),
        ));

        let arg = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(arg, 1, frames));

        self.generator.add_instruction(Instruction::Comment(
            "reset stack Kontinuation pointer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Addi(frames, frames, 5));

        let second_eval_arg = self.second_arg;

        self.generator
            .add_instruction(Instruction::Mv(second_eval_arg, arg));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Mv(ret, function));

        self.generator
            .add_instruction(Instruction::J("apply_evaluate".to_string()));
    }

    // Takes in a0 and passes it to force_evaluate
    pub fn handle_frame_force(&mut self) {
        let frames = self.frames;
        self.generator
            .add_instruction(Instruction::Label("handle_frame_force".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "reset stack Kontinuation pointer".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Addi(frames, frames, 1));

        self.generator
            .add_instruction(Instruction::J("force_evaluate".to_string()));
    }

    pub fn handle_frame_constr(&mut self) {
        let computed_value = self.first_arg;
        let frames = self.frames;
        let env = self.env;
        let heap = self.heap;
        self.generator
            .add_instruction(Instruction::Label("handle_frame_constr".to_string()));

        let constr_tag = self.first_temp;

        // Load the constructor tag from the frame
        self.generator
            .add_instruction(Instruction::Lw(constr_tag, 1, frames));

        let environment = self.second_temp;

        // Load the environment from the frame
        self.generator
            .add_instruction(Instruction::Lw(environment, 5, frames));

        let fields_len = self.third_temp;

        self.generator
            .add_instruction(Instruction::Lw(fields_len, 9, frames));

        // bytes offset from frame to values len based on fields length
        let bytes_offset = self.fourth_temp;

        self.generator
            .add_instruction(Instruction::Mv(bytes_offset, fields_len));

        self.generator
            .add_instruction(Instruction::Slli(bytes_offset, bytes_offset, 2));

        self.generator
            .add_instruction(Instruction::Addi(bytes_offset, bytes_offset, 13));

        self.generator
            .add_instruction(Instruction::Add(bytes_offset, bytes_offset, frames));

        let values_len = self.fifth_temp;

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
                .add_instruction(Instruction::Sw(current_field_len, 9, frames));

            // Field term to compute on in A5
            let first_field = self.sixth_arg;
            self.generator
                .add_instruction(Instruction::Lw(first_field, 13, frames));

            // Value to push onto the frame in A4
            let new_value = self.fifth_arg;
            self.generator
                .add_instruction(Instruction::Mv(new_value, computed_value));

            let length_arg = self.third_arg;
            self.generator
                .add_instruction(Instruction::Mv(length_arg, current_field_len));

            self.generator
                .add_instruction(Instruction::Add(length_arg, length_arg, values_len));

            let new_list = self.second_arg;
            self.generator
                .add_instruction(Instruction::Addi(new_list, frames, 13));

            let src_list = self.first_arg;
            self.generator
                .add_instruction(Instruction::Addi(src_list, frames, 17));

            let callback = self.fourth_arg;
            self.generator
                .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

            self.generator
                .add_instruction(Instruction::Sw(new_value, 0, new_list));

            self.generator
                .add_instruction(Instruction::Mv(env, environment));

            let ret = self.return_reg;
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
            // Overwrite fields_len
            let allocation_amount = fields_len;
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

            // Overwrite environment
            // since it is not used in return compute
            let allocator_space = environment;
            self.generator
                .add_instruction(Instruction::Mv(allocator_space, heap));

            // Allocate VConstr on the heap
            // 9 + 4 * value length
            self.generator
                .add_instruction(Instruction::Add(heap, heap, allocation_amount));

            let value_tag = self.sixth_temp;

            self.generator
                .add_instruction(Instruction::Li(value_tag, 4));

            self.generator
                .add_instruction(Instruction::Sb(value_tag, 0, allocator_space));

            self.generator
                .add_instruction(Instruction::Sw(constr_tag, 1, allocator_space));

            self.generator
                .add_instruction(Instruction::Sw(values_len, 5, allocator_space));

            // Value to return compute in A5
            let return_value = self.sixth_arg;
            self.generator
                .add_instruction(Instruction::Mv(return_value, allocator_space));

            self.generator
                .add_instruction(Instruction::Addi(allocator_space, allocator_space, 9));

            self.generator
                .add_instruction(Instruction::Addi(values_len, values_len, -1));

            // Overwrite constr_tag
            let list_byte_length = constr_tag;
            self.generator
                .add_instruction(Instruction::Mv(list_byte_length, values_len));

            self.generator.add_instruction(Instruction::Slli(
                list_byte_length,
                list_byte_length,
                2,
            ));

            // Overwrite list_byte_length
            let tail_list = list_byte_length;
            self.generator.add_instruction(Instruction::Add(
                tail_list,
                list_byte_length,
                bytes_offset,
            ));

            let next_frame = self.seventh_arg;
            self.generator
                .add_instruction(Instruction::Addi(next_frame, tail_list, 4));

            let new_value = self.fifth_arg;
            self.generator
                .add_instruction(Instruction::Mv(new_value, computed_value));

            let list_to_reverse = self.first_arg;

            self.generator
                .add_instruction(Instruction::Mv(list_to_reverse, tail_list));

            let dest = self.second_arg;
            self.generator
                .add_instruction(Instruction::Mv(dest, allocator_space));

            self.generator
                .add_instruction(Instruction::Sw(new_value, 0, dest));

            self.generator
                .add_instruction(Instruction::Addi(dest, dest, 4));

            let size = self.third_arg;
            self.generator
                .add_instruction(Instruction::Mv(size, values_len));

            let callback = self.fourth_arg;
            self.generator
                .add_instruction(Instruction::Jal(callback, "reverse_clone_list".to_string()));

            let ret = self.return_reg;
            self.generator
                .add_instruction(Instruction::Mv(ret, return_value));

            // Reset frame stack by moving to next frame
            self.generator
                .add_instruction(Instruction::Mv(frames, next_frame));

            self.generator
                .add_instruction(Instruction::J("return".to_string()));
        }
    }

    pub fn handle_frame_case(&mut self) {
        let first_arg = self.first_arg;
        let frames = self.frames;
        let env = self.env;

        self.generator
            .add_instruction(Instruction::Label("handle_frame_case".to_string()));

        let constr = self.first_temp;
        self.generator
            .add_instruction(Instruction::Mv(constr, first_arg));

        let constr_term_tag = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lbu(constr_term_tag, 0, constr));

        let expected_tag = self.third_temp;

        self.generator
            .add_instruction(Instruction::Li(expected_tag, 4));

        self.generator.add_instruction(Instruction::Bne(
            constr_term_tag,
            expected_tag,
            "handle_frame_case_error".to_string(),
        ));

        {
            let constr_tag = self.fourth_temp;
            self.generator
                .add_instruction(Instruction::Lw(constr_tag, 1, constr));

            // Overwrite expected_tag
            let branches_len = expected_tag;
            self.generator
                .add_instruction(Instruction::Lw(branches_len, 5, frames));

            self.generator.add_instruction(Instruction::Bge(
                constr_tag,
                branches_len,
                "handle_frame_case_error".to_string(),
            ));

            // Don't need scope here since handle error is already in another scope
            // {}

            // set env
            self.generator
                .add_instruction(Instruction::Lw(env, 1, frames));

            // Overwrite constr_tag
            let offset_to_branch = constr_tag;
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
                frames,
            ));

            // Put branch term in return register
            let ret = self.return_reg;
            self.generator
                .add_instruction(Instruction::Lw(ret, 0, offset_to_branch));

            // Overwrite branches_len
            let claim_stack_item = branches_len;
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

            // reset frame pointer
            self.generator
                .add_instruction(Instruction::Add(frames, frames, claim_stack_item));

            // Overwrite constr_term_tag
            let constr_fields_len = constr_term_tag;
            self.generator
                .add_instruction(Instruction::Lw(constr_fields_len, 5, constr));

            let current_index = self.fifth_temp;
            self.generator
                .add_instruction(Instruction::Mv(current_index, Register::Zero));

            let current_offset = self.sixth_temp;
            // 9 for constant offset
            // 1 for frame tag + 4 for constr tag + 4 for constr fields len
            self.generator
                .add_instruction(Instruction::Li(current_offset, 9));

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
                    .add_instruction(Instruction::Addi(frames, frames, -5));

                //Overwrite claim_stack_item
                let frame_tag = claim_stack_item;
                self.generator
                    .add_instruction(Instruction::Li(frame_tag, 2));

                self.generator
                    .add_instruction(Instruction::Sb(frame_tag, 0, frames));

                // Overwrite offset_to_branch
                let arg = offset_to_branch;

                self.generator
                    .add_instruction(Instruction::Lw(arg, 0, current_offset));

                self.generator
                    .add_instruction(Instruction::Sw(arg, 1, frames));

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

    pub fn handle_no_frame(&mut self) {
        let value = self.first_arg;
        self.generator
            .add_instruction(Instruction::Label("handle_no_frame".to_string()));

        let value_tag = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lbu(value_tag, 0, value));

        // We should only return constants. That greatly simplifies how we return a value to the user
        self.generator.add_instruction(Instruction::Bne(
            value_tag,
            Register::Zero,
            "handle_error".to_string(),
        ));

        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Lw(ret, 1, value));

        self.generator
            .add_instruction(Instruction::J("halt".to_string()));
    }

    pub fn halt(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("halt".to_string()));

        // exit code
        self.generator
            .add_instruction(Instruction::Li(self.eighth_arg, 93));

        self.generator.add_instruction(Instruction::Ecall);
    }

    pub fn force_evaluate(&mut self) {
        let function = self.first_arg;
        let env = self.env;
        let heap = self.heap;
        self.generator
            .add_instruction(Instruction::Label("force_evaluate".to_string()));

        self.generator.add_instruction(Instruction::Comment(
            "Value address should be in A0".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::Comment("Load value tag".to_string()));

        let tag = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lbu(tag, 0, function));

        self.generator
            .add_instruction(Instruction::Comment("Delay".to_string()));

        let delay_value_tag = self.second_temp;
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

            // Overwrite tag
            let body = tag;
            self.generator
                .add_instruction(Instruction::Lw(body, 1, function));

            self.generator.add_instruction(Instruction::Comment(
                "load environment from a0 which is Value".to_string(),
            ));

            // Overwrite delay_value_tag
            let environment = delay_value_tag;
            self.generator
                .add_instruction(Instruction::Lw(environment, 5, function));

            self.generator
                .add_instruction(Instruction::Mv(env, environment));

            let ret = self.return_reg;
            self.generator.add_instruction(Instruction::Mv(ret, body));

            self.generator
                .add_instruction(Instruction::J("compute".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("force_evaluate_builtin".to_string()));

            // Overwrite delay_value_tag
            let builtin_value_tag = delay_value_tag;
            self.generator
                .add_instruction(Instruction::Li(builtin_value_tag, 3));

            self.generator.add_instruction(Instruction::Bne(
                tag,
                builtin_value_tag,
                "force_evaluate_error".to_string(),
            ));

            {
                // Overwrite builtin_value_tag
                let force_count = builtin_value_tag;
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
                    .add_instruction(Instruction::Addi(heap, heap, 7));

                self.generator
                    .add_instruction(Instruction::Sb(tag, -7, heap));

                let builtin_func_index = self.third_temp;
                self.generator
                    .add_instruction(Instruction::Lbu(builtin_func_index, 1, function));

                self.generator
                    .add_instruction(Instruction::Sb(builtin_func_index, -6, heap));

                self.generator
                    .add_instruction(Instruction::Sb(force_count, -5, heap));

                // 0 Arguments applied so arg fields is just arg length (zero)
                self.generator
                    .add_instruction(Instruction::Sw(Register::Zero, -4, heap));

                // Store new value in ret
                let ret = self.return_reg;
                self.generator
                    .add_instruction(Instruction::Addi(ret, heap, -7));

                // If still have forces to apply then return
                self.generator.add_instruction(Instruction::Bne(
                    Register::Zero,
                    force_count,
                    "return".to_string(),
                ));

                // Overwrite tag
                let arguments_length = tag;
                self.generator
                    .add_instruction(Instruction::Lw(arguments_length, 3, function));

                // Overwrite force count
                let arity_lookup = force_count;
                self.generator
                    .add_instruction(Instruction::La(arity_lookup, "arities".to_string()));

                self.generator.add_instruction(Instruction::Add(
                    arity_lookup,
                    arity_lookup,
                    builtin_func_index,
                ));

                let arity = self.fourth_temp;
                self.generator
                    .add_instruction(Instruction::Lbu(arity, 0, arity_lookup));

                // If all arguments not applied then return.
                self.generator.add_instruction(Instruction::Bne(
                    arity,
                    arguments_length,
                    "return".to_string(),
                ));

                let function_index = self.second_arg;
                self.generator
                    .add_instruction(Instruction::Mv(function_index, builtin_func_index));

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

    pub fn apply_evaluate(&mut self) {
        let function = self.first_arg;
        let argument = self.second_arg;
        let env = self.env;
        let heap = self.heap;

        self.generator
            .add_instruction(Instruction::Label("apply_evaluate".to_string()));

        //Value address should be in A0
        self.generator
            .add_instruction(Instruction::Comment("Load function value tag".to_string()));

        let function_tag = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lbu(function_tag, 0, function));

        self.generator
            .add_instruction(Instruction::Comment("Lambda".to_string()));

        let lambda_value_tag = self.second_temp;

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

            // Overwrite function_tag
            let body = function_tag;

            self.generator
                .add_instruction(Instruction::Lw(body, 1, function));

            self.generator.add_instruction(Instruction::Comment(
                "load environment from a0 which is function Value".to_string(),
            ));

            // Overwrite lambda_value_tag
            let environment = lambda_value_tag;

            self.generator
                .add_instruction(Instruction::Lw(environment, 5, function));

            self.generator
                .add_instruction(Instruction::Mv(env, environment));

            self.generator.add_instruction(Instruction::Comment(
                "Important this is the only place we modify environment".to_string(),
            ));

            self.generator.add_instruction(Instruction::Comment(
                "Allocate 8 bytes on the heap".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::Addi(heap, heap, 8));

            self.generator.add_instruction(Instruction::Comment(
                "pointer to argument value".to_string(),
            ));
            self.generator
                .add_instruction(Instruction::Sw(argument, -8, heap));

            self.generator.add_instruction(Instruction::Comment(
                "pointer to previous environment".to_string(),
            ));
            self.generator
                .add_instruction(Instruction::Sw(env, -4, heap));

            self.generator.add_instruction(Instruction::Comment(
                "Save allocated heap location in environment pointer".to_string(),
            ));
            self.generator
                .add_instruction(Instruction::Addi(env, heap, -8));

            let ret = self.return_reg;
            self.generator.add_instruction(Instruction::Mv(ret, body));

            self.generator
                .add_instruction(Instruction::J("compute".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("apply_evaluate_builtin".to_string()));

            // Overwrite lambda_value_tag
            let builtin_value_tag = lambda_value_tag;
            self.generator
                .add_instruction(Instruction::Li(builtin_value_tag, 3));

            self.generator.add_instruction(Instruction::Bne(
                function_tag,
                builtin_value_tag,
                "apply_evaluate_error".to_string(),
            ));

            {
                // Overwrite builtin_value_tag
                let force_count = builtin_value_tag;
                self.generator
                    .add_instruction(Instruction::Lbu(force_count, 2, function));

                self.generator.add_instruction(Instruction::Bne(
                    Register::Zero,
                    force_count,
                    "apply_evaluate_error".to_string(),
                ));

                let builtin_func_index = self.third_temp;
                self.generator
                    .add_instruction(Instruction::Lbu(builtin_func_index, 1, function));

                let arguments_length = self.fourth_temp;
                self.generator
                    .add_instruction(Instruction::Lw(arguments_length, 3, function));

                let arity_lookup = self.fifth_temp;
                self.generator
                    .add_instruction(Instruction::La(arity_lookup, "arities".to_string()));

                self.generator.add_instruction(Instruction::Add(
                    arity_lookup,
                    arity_lookup,
                    builtin_func_index,
                ));

                // Overwrite arity_lookup
                let arity = arity_lookup;
                self.generator
                    .add_instruction(Instruction::Lbu(arity, 0, arity_lookup));

                self.generator.add_instruction(Instruction::Beq(
                    arity,
                    arguments_length,
                    "apply_evaluate_error".to_string(),
                ));

                //
                let new_args_length = force_count;
                self.generator.add_instruction(Instruction::Addi(
                    new_args_length,
                    arguments_length,
                    1,
                ));

                let heap_allocation = self.sixth_temp;
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

                let cloned_value = self.seventh_temp;
                self.generator
                    .add_instruction(Instruction::Mv(cloned_value, heap));

                self.generator
                    .add_instruction(Instruction::Add(heap, heap, heap_allocation));

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

                let store_new_args_length = self.eighth_arg;
                self.generator
                    .add_instruction(Instruction::Mv(store_new_args_length, new_args_length));

                let store_new_arg = self.seventh_arg;
                self.generator
                    .add_instruction(Instruction::Mv(store_new_arg, argument));

                let store_arity = self.sixth_arg;
                self.generator
                    .add_instruction(Instruction::Mv(store_arity, arity));

                let new_value = self.fifth_arg;
                self.generator
                    .add_instruction(Instruction::Mv(new_value, cloned_value));

                let size = self.third_arg;
                self.generator
                    .add_instruction(Instruction::Mv(size, arguments_length));

                let dest_list = self.second_arg;
                self.generator
                    .add_instruction(Instruction::Addi(dest_list, cloned_value, 7));

                let src_list = self.first_arg;
                self.generator
                    .add_instruction(Instruction::Addi(src_list, function, 7));

                let callback = self.fourth_arg;
                self.generator
                    .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

                // We can store the new arg value one word before current heap since it's not modified anywhere yet
                self.generator
                    .add_instruction(Instruction::Sw(store_new_arg, -4, heap));

                let ret = self.return_reg;
                self.generator
                    .add_instruction(Instruction::Mv(ret, new_value));

                // Check arity
                self.generator.add_instruction(Instruction::Bne(
                    store_arity,
                    store_new_args_length,
                    "return".to_string(),
                ));

                let function_index = self.second_arg;
                self.generator
                    .add_instruction(Instruction::Mv(function_index, builtin_func_index));

                self.generator
                    .add_instruction(Instruction::J("eval_builtin_app".to_string()));
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
    pub fn clone_list(&mut self) {
        let src_list = self.first_arg;
        let dest_list = self.second_arg;
        let length = self.third_arg;
        let callback = self.fourth_arg;
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
            let list_item = self.first_temp;
            self.generator
                .add_instruction(Instruction::Lw(list_item, 0, src_list));

            // Store term in new storage
            self.generator
                .add_instruction(Instruction::Sw(list_item, 0, dest_list));

            // move fields up by 4 bytes
            self.generator
                .add_instruction(Instruction::Addi(src_list, src_list, 4));

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
    pub fn reverse_clone_list(&mut self) {
        let src_list = self.first_arg;
        let dest_list = self.second_arg;
        let length = self.third_arg;
        let callback = self.fourth_arg;
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
            let list_item = self.first_temp;
            self.generator
                .add_instruction(Instruction::Lw(list_item, 0, src_list));

            // Store term in new storage
            self.generator
                .add_instruction(Instruction::Sw(list_item, 0, dest_list));

            // move backwards by 4 bytes
            self.generator
                .add_instruction(Instruction::Addi(src_list, src_list, -4));

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
            .add_instruction(Instruction::Byte(FORCE_COUNTS.to_vec()));

        self.generator
            .add_instruction(Instruction::Label("arities".to_string()));
        self.generator
            .add_instruction(Instruction::Byte(ARITIES.to_vec()));
    }

    pub fn eval_builtin_app(&mut self) {
        let builtin_value = self.first_arg;
        let builtin_func_index = self.second_arg;
        self.generator
            .add_instruction(Instruction::Label("eval_builtin_app".to_string()));

        // offset to actual args for the builtin call
        let builtin_args = self.first_arg;
        self.generator
            .add_instruction(Instruction::Addi(builtin_args, builtin_value, 7));

        let builtin_call_jump = self.first_temp;
        self.generator.add_instruction(Instruction::La(
            builtin_call_jump,
            "eval_builtin_call".to_string(),
        ));

        let builtin_index_offset = self.second_temp;
        self.generator.add_instruction(Instruction::Slli(
            builtin_index_offset,
            builtin_func_index,
            2,
        ));

        self.generator.add_instruction(Instruction::Add(
            builtin_call_jump,
            builtin_call_jump,
            builtin_index_offset,
        ));

        self.generator
            .add_instruction(Instruction::Jalr(self.discard, builtin_call_jump, 0));

        self.generator
            .add_instruction(Instruction::Label("eval_builtin_call".to_string()));

        // 0 - add_integer
        self.generator
            .add_instruction(Instruction::J("add_integer".to_string()));

        // 1 - sub_integer
        self.generator
            .add_instruction(Instruction::J("sub_integer".to_string()));

        // 2 - multiply_integer
        self.generator
            .add_instruction(Instruction::J("multiply_integer".to_string()));

        // 3 - divide_integer
        self.generator
            .add_instruction(Instruction::J("divide_integer".to_string()));

        // 4 - quotient_integer
        self.generator
            .add_instruction(Instruction::J("quotient_integer".to_string()));

        // 5 - remainder_integer
        self.generator
            .add_instruction(Instruction::J("remainder_integer".to_string()));

        // 6 - mod_integer
        self.generator
            .add_instruction(Instruction::J("mod_integer".to_string()));

        // 7 - equals_integer
        self.generator
            .add_instruction(Instruction::J("equals_integer".to_string()));

        // 8 - less_than_equals_integer
        self.generator
            .add_instruction(Instruction::J("less_than_equals_integer".to_string()));

        // 9 - less_than_integer
        self.generator
            .add_instruction(Instruction::J("less_than_integer".to_string()));

        // 10 - append_bytestring
        self.generator
            .add_instruction(Instruction::J("append_bytestring".to_string()));
    }

    pub fn unwrap_integer(&mut self) {
        let arg = self.first_arg;
        let return_address = self.second_arg;
        self.generator
            .add_instruction(Instruction::Label("unwrap_integer".to_string()));

        let expected_tag = self.first_temp;
        self.generator
            .add_instruction(Instruction::Li(expected_tag, 0));

        let value_tag = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lbu(value_tag, 0, arg));

        self.generator.add_instruction(Instruction::Bne(
            expected_tag,
            value_tag,
            "handle_error".to_string(),
        ));

        let constant_value = self.third_temp;
        self.generator
            .add_instruction(Instruction::Lw(constant_value, 1, arg));

        // Overwrite expected_tag
        let expected_type = expected_tag;
        self.generator.add_instruction(Instruction::Li(
            expected_type,
            1 + 256 * (const_tag::INTEGER as i32),
        ));

        // The type ends up being [0x01, 0x00] which in little endian is 1
        // Overwrite value_tag
        let arg_type = value_tag;
        self.generator
            .add_instruction(Instruction::Lhu(arg_type, 0, constant_value));

        self.generator.add_instruction(Instruction::Bne(
            expected_type,
            arg_type,
            "handle_error".to_string(),
        ));

        // Return pointer to integer value
        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Addi(ret, constant_value, 2));

        self.generator
            .add_instruction(Instruction::Jalr(self.discard, return_address, 0));
    }

    pub fn unwrap_bytestring(&mut self) {
        let arg = self.first_arg;
        let return_address = self.second_arg;
        self.generator
            .add_instruction(Instruction::Label("unwrap_bytestring".to_string()));

        let expected_tag = self.first_temp;
        self.generator
            .add_instruction(Instruction::Li(expected_tag, 0));

        let value_tag = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lbu(value_tag, 0, arg));

        self.generator.add_instruction(Instruction::Bne(
            expected_tag,
            value_tag,
            "handle_error".to_string(),
        ));

        let constant_value = self.third_temp;
        self.generator
            .add_instruction(Instruction::Lw(constant_value, 1, arg));

        // Overwrite expected_tag
        let expected_type = expected_tag;
        self.generator.add_instruction(Instruction::Li(
            expected_type,
            1 + 256 * (const_tag::BYTESTRING as i32),
        ));

        // The type ends up being [0x01, 0x00] which in little endian is 1
        // Overwrite value_tag
        let arg_type = value_tag;
        self.generator
            .add_instruction(Instruction::Lhu(arg_type, 0, constant_value));

        self.generator.add_instruction(Instruction::Bne(
            expected_type,
            arg_type,
            "handle_error".to_string(),
        ));

        // Return pointer to bytestring value
        let ret = self.return_reg;
        self.generator
            .add_instruction(Instruction::Addi(ret, constant_value, 2));

        self.generator
            .add_instruction(Instruction::Jalr(self.discard, return_address, 0));
    }

    pub fn add_integer(&mut self) {
        let args = self.first_arg;
        self.generator
            .add_instruction(Instruction::Label("add_integer".to_string()));

        let x_value = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(x_value, 0, args));

        let y_value = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(y_value, 4, args));

        // Overwrite args
        let first_arg = args;
        self.generator
            .add_instruction(Instruction::Mv(first_arg, x_value));

        let store_y = self.third_arg;
        self.generator
            .add_instruction(Instruction::Mv(store_y, y_value));

        let callback = self.second_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Overwrite x_value
        let x_integer = x_value;
        self.generator
            .add_instruction(Instruction::Mv(x_integer, first_arg));

        self.generator
            .add_instruction(Instruction::Mv(first_arg, store_y));

        // Overwrite store_y
        let store_x = store_y;
        self.generator
            .add_instruction(Instruction::Mv(store_x, x_integer));

        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Now move things back
        self.generator
            .add_instruction(Instruction::Mv(x_integer, store_x));

        // Overwrite y_value
        let y_integer = y_value;
        self.generator
            .add_instruction(Instruction::Mv(y_integer, first_arg));

        // Overwrite first_arg
        let x_sign = self.return_reg;
        self.generator
            .add_instruction(Instruction::Lbu(x_sign, 0, x_integer));

        // Overwrite callback
        let y_sign = callback;
        self.generator
            .add_instruction(Instruction::Lbu(y_sign, 0, y_integer));

        let x_magnitude = store_x;
        self.generator
            .add_instruction(Instruction::Addi(x_magnitude, x_integer, 1));

        let y_magnitude = self.fourth_arg;
        self.generator
            .add_instruction(Instruction::Addi(y_magnitude, y_integer, 1));

        self.generator
            .add_instruction(Instruction::J("add_signed_integers".to_string()));
    }

    pub fn sub_integer(&mut self) {
        let args = self.first_arg;
        self.generator
            .add_instruction(Instruction::Label("sub_integer".to_string()));

        let x_value = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(x_value, 0, args));

        let y_value = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(y_value, 4, args));

        // Overwrite args
        let first_arg = args;
        self.generator
            .add_instruction(Instruction::Mv(first_arg, x_value));

        let store_y = self.third_arg;
        self.generator
            .add_instruction(Instruction::Mv(store_y, y_value));

        let callback = self.second_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Overwrite x_value
        let x_integer = x_value;
        self.generator
            .add_instruction(Instruction::Mv(x_integer, first_arg));

        self.generator
            .add_instruction(Instruction::Mv(first_arg, store_y));

        // Overwrite store_y
        let store_x = store_y;
        self.generator
            .add_instruction(Instruction::Mv(store_x, x_integer));

        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Now move things back
        self.generator
            .add_instruction(Instruction::Mv(x_integer, store_x));

        // Overwrite y_value
        let y_integer = y_value;
        self.generator
            .add_instruction(Instruction::Mv(y_integer, first_arg));

        // Overwrite first_arg
        let x_sign = self.return_reg;
        self.generator
            .add_instruction(Instruction::Lbu(x_sign, 0, x_integer));

        // Overwrite callback
        let y_sign = callback;
        self.generator
            .add_instruction(Instruction::Lbu(y_sign, 0, y_integer));

        // flip y_sign
        self.generator
            .add_instruction(Instruction::Xori(y_sign, y_sign, 1));

        let x_magnitude = store_x;
        self.generator
            .add_instruction(Instruction::Addi(x_magnitude, x_integer, 1));

        let y_magnitude = self.fourth_arg;
        self.generator
            .add_instruction(Instruction::Addi(y_magnitude, y_integer, 1));

        self.generator
            .add_instruction(Instruction::J("add_signed_integers".to_string()));
    }

    pub fn multiply_integer(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("multiply_integer".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn divide_integer(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("divide_integer".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn quotient_integer(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("quotient_integer".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn remainder_integer(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("remainder_integer".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn mod_integer(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("mod_integer".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn equals_integer(&mut self) {
        let args = self.first_arg;
        let heap = self.heap;
        self.generator
            .add_instruction(Instruction::Label("equals_integer".to_string()));

        let x_value = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(x_value, 0, args));

        let y_value = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(y_value, 4, args));

        // Overwrite args
        let first_arg = args;
        self.generator
            .add_instruction(Instruction::Mv(first_arg, x_value));

        let store_y = self.third_arg;
        self.generator
            .add_instruction(Instruction::Mv(store_y, y_value));

        let callback = self.second_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Overwrite x_value
        let x_integer = x_value;
        self.generator
            .add_instruction(Instruction::Mv(x_integer, first_arg));

        self.generator
            .add_instruction(Instruction::Mv(first_arg, store_y));

        // Overwrite store_y
        let store_x = store_y;
        self.generator
            .add_instruction(Instruction::Mv(store_x, x_integer));

        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Now move things back
        self.generator
            .add_instruction(Instruction::Mv(x_integer, store_x));

        // Overwrite y_value
        let y_integer = y_value;
        self.generator
            .add_instruction(Instruction::Mv(y_integer, first_arg));

        // Overwrite first_arg
        let x_sign = self.return_reg;
        self.generator
            .add_instruction(Instruction::Lbu(x_sign, 0, x_integer));

        // Overwrite callback
        let y_sign = callback;
        self.generator
            .add_instruction(Instruction::Lbu(y_sign, 0, y_integer));

        let x_magnitude = store_x;
        self.generator
            .add_instruction(Instruction::Addi(x_magnitude, x_integer, 1));

        let y_magnitude = self.fourth_arg;
        self.generator
            .add_instruction(Instruction::Addi(y_magnitude, y_integer, 1));

        let callback = self.fifth_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "compare_magnitude".to_string()));

        // Overwrite first_arg
        let equality = x_sign;
        let bool_value = x_integer;
        self.generator
            .add_instruction(Instruction::Mv(bool_value, equality));

        // heap allocation = 1 byte for value tag + 4 bytes for pointer + 1 byte for type length
        // + 1 byte for type + 1 byte for bool value
        // Overwrite equality
        let ret = self.return_reg;
        self.generator.add_instruction(Instruction::Mv(ret, heap));

        self.generator
            .add_instruction(Instruction::Addi(heap, heap, 8));

        self.generator
            .add_instruction(Instruction::Sb(Register::Zero, -8, heap));

        // Overwrite y_integer
        let constant_pointer = y_integer;
        self.generator
            .add_instruction(Instruction::Addi(constant_pointer, heap, -3));

        self.generator
            .add_instruction(Instruction::Sw(constant_pointer, -7, heap));

        // Overwrite constant_pointer
        let bool_type = constant_pointer;
        self.generator
            .add_instruction(Instruction::Li(bool_type, 1 + 256 * (BOOL as i32)));

        self.generator
            .add_instruction(Instruction::Sh(bool_type, -3, heap));

        self.generator
            .add_instruction(Instruction::Sb(bool_value, -1, heap));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn less_than_equals_integer(&mut self) {
        let args = self.first_arg;
        let heap = self.heap;
        self.generator
            .add_instruction(Instruction::Label("less_than_equals_integer".to_string()));

        let x_value = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(x_value, 0, args));

        let y_value = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(y_value, 4, args));

        // Overwrite args
        let first_arg = args;
        self.generator
            .add_instruction(Instruction::Mv(first_arg, x_value));

        let store_y = self.third_arg;
        self.generator
            .add_instruction(Instruction::Mv(store_y, y_value));

        let callback = self.second_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Overwrite x_value
        let x_integer = x_value;
        self.generator
            .add_instruction(Instruction::Mv(x_integer, first_arg));

        self.generator
            .add_instruction(Instruction::Mv(first_arg, store_y));

        // Overwrite store_y
        let store_x = store_y;
        self.generator
            .add_instruction(Instruction::Mv(store_x, x_integer));

        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Now move things back
        self.generator
            .add_instruction(Instruction::Mv(x_integer, store_x));

        // Overwrite y_value
        let y_integer = y_value;
        self.generator
            .add_instruction(Instruction::Mv(y_integer, first_arg));

        // Overwrite first_arg
        let x_sign = self.return_reg;
        self.generator
            .add_instruction(Instruction::Lbu(x_sign, 0, x_integer));

        // Overwrite callback
        let y_sign = callback;
        self.generator
            .add_instruction(Instruction::Lbu(y_sign, 0, y_integer));

        let x_magnitude = store_x;
        self.generator
            .add_instruction(Instruction::Addi(x_magnitude, x_integer, 1));

        let first_magnitude = self.seventh_arg;
        self.generator
            .add_instruction(Instruction::Mv(first_magnitude, x_magnitude));

        let y_magnitude = self.fourth_arg;
        self.generator
            .add_instruction(Instruction::Addi(y_magnitude, y_integer, 1));

        let callback = self.fifth_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "compare_magnitude".to_string()));

        // Overwrite first_arg
        let equality = x_sign;
        let greater_magnitude_value = y_sign;
        let bool_value = x_integer;
        self.generator
            .add_instruction(Instruction::Mv(bool_value, equality));

        // heap allocation = 1 byte for value tag + 4 bytes for pointer + 1 byte for type length
        // + 1 byte for type + 1 byte for bool value
        // Overwrite equality
        let ret = self.return_reg;
        self.generator.add_instruction(Instruction::Mv(ret, heap));

        self.generator
            .add_instruction(Instruction::Addi(heap, heap, 8));

        self.generator
            .add_instruction(Instruction::Sb(Register::Zero, -8, heap));

        // Overwrite y_integer
        let constant_pointer = y_integer;
        self.generator
            .add_instruction(Instruction::Addi(constant_pointer, heap, -3));

        self.generator
            .add_instruction(Instruction::Sw(constant_pointer, -7, heap));

        // Overwrite constant_pointer
        let bool_type = constant_pointer;
        self.generator
            .add_instruction(Instruction::Li(bool_type, 1 + 256 * (BOOL as i32)));

        self.generator
            .add_instruction(Instruction::Sh(bool_type, -3, heap));

        self.generator.add_instruction(Instruction::Bne(
            greater_magnitude_value,
            first_magnitude,
            "less_than".to_string(),
        ));

        {
            self.generator.add_instruction(Instruction::J(
                "finish_less_than_equals_integer".to_string(),
            ));
        }
        {
            self.generator
                .add_instruction(Instruction::Label("less_than".to_string()));

            self.generator
                .add_instruction(Instruction::Li(bool_value, 1));
        }

        self.generator.add_instruction(Instruction::Label(
            "finish_less_than_equals_integer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::Sb(bool_value, -1, heap));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn less_than_integer(&mut self) {
        let args = self.first_arg;
        let heap = self.heap;
        self.generator
            .add_instruction(Instruction::Label("less_than_integer".to_string()));

        let x_value = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(x_value, 0, args));

        let y_value = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(y_value, 4, args));

        // Overwrite args
        let first_arg = args;
        self.generator
            .add_instruction(Instruction::Mv(first_arg, x_value));

        let store_y = self.third_arg;
        self.generator
            .add_instruction(Instruction::Mv(store_y, y_value));

        let callback = self.second_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Overwrite x_value
        let x_integer = x_value;
        self.generator
            .add_instruction(Instruction::Mv(x_integer, first_arg));

        self.generator
            .add_instruction(Instruction::Mv(first_arg, store_y));

        // Overwrite store_y
        let store_x = store_y;
        self.generator
            .add_instruction(Instruction::Mv(store_x, x_integer));

        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_integer".to_string()));

        // Now move things back
        self.generator
            .add_instruction(Instruction::Mv(x_integer, store_x));

        // Overwrite y_value
        let y_integer = y_value;
        self.generator
            .add_instruction(Instruction::Mv(y_integer, first_arg));

        // Overwrite first_arg
        let x_sign = self.return_reg;
        self.generator
            .add_instruction(Instruction::Lbu(x_sign, 0, x_integer));

        // Overwrite callback
        let y_sign = callback;
        self.generator
            .add_instruction(Instruction::Lbu(y_sign, 0, y_integer));

        let x_magnitude = store_x;
        self.generator
            .add_instruction(Instruction::Addi(x_magnitude, x_integer, 1));

        let first_magnitude = self.seventh_arg;
        self.generator
            .add_instruction(Instruction::Mv(first_magnitude, x_magnitude));

        let y_magnitude = self.fourth_arg;
        self.generator
            .add_instruction(Instruction::Addi(y_magnitude, y_integer, 1));

        let callback = self.fifth_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "compare_magnitude".to_string()));

        // Overwrite first_arg
        let equality = x_sign;
        let greater_magnitude_value = y_sign;
        let bool_value = x_integer;
        // Flip equality
        self.generator
            .add_instruction(Instruction::Xori(bool_value, equality, 1));

        // heap allocation = 1 byte for value tag + 4 bytes for pointer + 1 byte for type length
        // + 1 byte for type + 1 byte for bool value
        // Overwrite equality
        let ret = self.return_reg;
        self.generator.add_instruction(Instruction::Mv(ret, heap));

        self.generator
            .add_instruction(Instruction::Addi(heap, heap, 8));

        self.generator
            .add_instruction(Instruction::Sb(Register::Zero, -8, heap));

        // Overwrite y_integer
        let constant_pointer = y_integer;
        self.generator
            .add_instruction(Instruction::Addi(constant_pointer, heap, -3));

        self.generator
            .add_instruction(Instruction::Sw(constant_pointer, -7, heap));

        // Overwrite constant_pointer
        let bool_type = constant_pointer;
        self.generator
            .add_instruction(Instruction::Li(bool_type, 1 + 256 * (BOOL as i32)));

        self.generator
            .add_instruction(Instruction::Sh(bool_type, -3, heap));

        self.generator.add_instruction(Instruction::Bne(
            greater_magnitude_value,
            first_magnitude,
            "less_than_int".to_string(),
        ));

        {
            self.generator
                .add_instruction(Instruction::J("finish_less_than_integer".to_string()));
        }
        {
            self.generator
                .add_instruction(Instruction::Label("less_than_int".to_string()));

            self.generator
                .add_instruction(Instruction::Li(bool_value, 1));
        }

        self.generator
            .add_instruction(Instruction::Label("finish_less_than_integer".to_string()));

        self.generator
            .add_instruction(Instruction::Sb(bool_value, -1, heap));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn append_bytestring(&mut self) {
        let args = self.first_arg;
        let heap = self.heap;
        self.generator
            .add_instruction(Instruction::Label("append_bytestring".to_string()));

        let x_value = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(x_value, 0, args));

        let y_value = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(y_value, 4, args));

        // Overwrite args
        let first_arg = args;
        self.generator
            .add_instruction(Instruction::Mv(first_arg, x_value));

        let store_y = self.third_arg;
        self.generator
            .add_instruction(Instruction::Mv(store_y, y_value));

        let callback = self.second_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_bytestring".to_string()));

        // Overwrite x_value
        let x_bytestring = x_value;
        self.generator
            .add_instruction(Instruction::Mv(x_bytestring, first_arg));

        self.generator
            .add_instruction(Instruction::Mv(first_arg, store_y));

        // Overwrite store_y
        let store_x = store_y;
        self.generator
            .add_instruction(Instruction::Mv(store_x, x_bytestring));

        self.generator
            .add_instruction(Instruction::Jal(callback, "unwrap_bytestring".to_string()));

        // Now move things back
        self.generator
            .add_instruction(Instruction::Mv(x_bytestring, store_x));

        // Overwrite y_value
        let y_bytestring = y_value;
        self.generator
            .add_instruction(Instruction::Mv(y_bytestring, first_arg));

        let x_length = self.third_temp;
        self.generator
            .add_instruction(Instruction::Lw(x_value, 0, x_bytestring));

        let y_length = self.fourth_temp;
        self.generator
            .add_instruction(Instruction::Lw(y_value, 0, y_bytestring));

        let total_length = self.fifth_temp;
        self.generator
            .add_instruction(Instruction::Add(total_length, x_length, y_length));

        let ret = self.return_reg;

        self.generator.add_instruction(Instruction::Mv(ret, heap));

        let value_builder = self.sixth_temp;
        self.generator
            .add_instruction(Instruction::Mv(value_builder, heap));

        // Overwrite total_length
        let total_allocation = self.seventh_temp;
        self.generator
            .add_instruction(Instruction::Slli(total_allocation, total_length, 2));

        // heap allocation = 1 byte for value tag + 4 bytes for pointer + 1 byte for type length
        // + 1 byte for type + 4 bytes for bytestring length + 4 * bytestring length bytes
        self.generator
            .add_instruction(Instruction::Addi(total_allocation, total_allocation, 11));

        self.generator
            .add_instruction(Instruction::Add(heap, heap, total_allocation));

        self.generator
            .add_instruction(Instruction::Sb(Register::Zero, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 1));

        let constant_pointer = total_allocation;
        self.generator
            .add_instruction(Instruction::Addi(constant_pointer, value_builder, 4));

        self.generator
            .add_instruction(Instruction::Sw(constant_pointer, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 4));

        let bytestring_type = constant_pointer;
        self.generator
            .add_instruction(Instruction::Li(bytestring_type, 257));

        self.generator
            .add_instruction(Instruction::Sh(bytestring_type, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 2));

        self.generator
            .add_instruction(Instruction::Sw(total_length, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 4));

        let return_temp = self.seventh_arg;
        self.generator
            .add_instruction(Instruction::Mv(return_temp, ret));

        let second_list = self.sixth_arg;
        self.generator
            .add_instruction(Instruction::Mv(second_list, y_bytestring));

        let second_list_len = self.fifth_arg;
        self.generator
            .add_instruction(Instruction::Mv(second_list_len, y_length));

        let size = self.third_arg;
        self.generator
            .add_instruction(Instruction::Mv(size, x_length));

        let dst_list = self.second_arg;

        self.generator
            .add_instruction(Instruction::Mv(dst_list, value_builder));

        let src_list = ret;
        self.generator
            .add_instruction(Instruction::Addi(src_list, x_bytestring, 4));

        // clone x into heap
        let callback = self.fourth_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

        self.generator
            .add_instruction(Instruction::Mv(size, second_list_len));

        self.generator
            .add_instruction(Instruction::Addi(src_list, second_list, 4));

        // clone y into heap
        self.generator
            .add_instruction(Instruction::Jal(callback, "clone_list".to_string()));

        self.generator
            .add_instruction(Instruction::Mv(self.return_reg, return_temp));

        self.generator
            .add_instruction(Instruction::J("return".to_string()));
    }

    pub fn add_signed_integers(&mut self) {
        let first_sign = self.first_arg;
        let second_sign = self.second_arg;
        let first_magnitude = self.third_arg;
        let second_magnitude = self.fourth_arg;
        let heap = self.heap;

        self.generator
            .add_instruction(Instruction::Label("add_signed_integers".to_string()));

        // If the signs are not equal we subtract the larger magnitude from the smaller magnitude
        // Then we use the largers sign. Except if equal magnitudes then we set the value to 0 and the sign to positive
        self.generator.add_instruction(Instruction::Bne(
            first_sign,
            second_sign,
            "sub_signed_integers".to_string(),
        ));

        let first_magnitude_length = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(first_magnitude_length, 0, first_magnitude));

        let second_magnitude_length = self.second_temp;
        self.generator.add_instruction(Instruction::Lw(
            second_magnitude_length,
            0,
            second_magnitude,
        ));

        let max_magnitude_length = self.third_temp;
        let bigger_magnitude = self.fifth_arg;
        let smaller_magnitude = self.sixth_arg;
        // Overwrite first_magnitude
        let smaller_length = first_magnitude;
        self.generator.add_instruction(Instruction::Bltu(
            first_magnitude_length,
            second_magnitude_length,
            "first_smaller".to_string(),
        ));

        {
            self.generator.add_instruction(Instruction::Mv(
                max_magnitude_length,
                first_magnitude_length,
            ));

            self.generator
                .add_instruction(Instruction::Mv(bigger_magnitude, first_magnitude));

            self.generator
                .add_instruction(Instruction::Mv(smaller_magnitude, second_magnitude));

            self.generator
                .add_instruction(Instruction::Mv(smaller_length, second_magnitude_length));

            self.generator
                .add_instruction(Instruction::J("allocate_heap".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("first_smaller".to_string()));

            self.generator.add_instruction(Instruction::Mv(
                max_magnitude_length,
                second_magnitude_length,
            ));

            self.generator
                .add_instruction(Instruction::Mv(bigger_magnitude, second_magnitude));

            self.generator
                .add_instruction(Instruction::Mv(smaller_magnitude, first_magnitude));

            self.generator
                .add_instruction(Instruction::Mv(smaller_length, first_magnitude_length));

            // No need for jump since next instruction is allocate_heap
        }

        self.generator
            .add_instruction(Instruction::Label("allocate_heap".to_string()));

        let max_heap_allocation = self.fourth_temp;
        self.generator.add_instruction(Instruction::Addi(
            max_heap_allocation,
            max_magnitude_length,
            1,
        ));

        self.generator.add_instruction(Instruction::Slli(
            max_heap_allocation,
            max_heap_allocation,
            2,
        ));

        // Add fixed constant for creating a constant integer value on the heap
        // 1 for value tag + 4 for constant pointer + 1 for type length + 1 for type integer
        // + 1 for sign + 4 for magnitude length + (4 * (largest magnitude length + 1))
        self.generator.add_instruction(Instruction::Addi(
            max_heap_allocation,
            max_heap_allocation,
            12,
        ));

        let value_builder = self.fifth_temp;
        self.generator
            .add_instruction(Instruction::Mv(value_builder, heap));

        // Overwrite first_sign
        let ret = self.return_reg;
        self.generator.add_instruction(Instruction::Mv(ret, heap));

        // Add maximum heap value needed possibly and reclaim later after addition
        self.generator
            .add_instruction(Instruction::Add(heap, heap, max_heap_allocation));

        // Overwrite max_heap_allocation
        let value_tag = max_heap_allocation;
        self.generator
            .add_instruction(Instruction::Li(value_tag, 0));

        self.generator
            .add_instruction(Instruction::Sb(value_tag, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 1));

        // Overwrite value_tag
        let integer_pointer = value_tag;
        // We store integer immediately after the pointer so we simply add 4 to point to the location
        self.generator
            .add_instruction(Instruction::Addi(integer_pointer, value_builder, 4));

        self.generator
            .add_instruction(Instruction::Sw(integer_pointer, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 4));

        // Overwrite integer_pointer
        let integer_type = integer_pointer;
        self.generator
            .add_instruction(Instruction::Li(integer_type, 1));

        self.generator
            .add_instruction(Instruction::Sh(integer_type, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 2));

        // Store second sign. In this case the signs are the same
        self.generator
            .add_instruction(Instruction::Sb(second_sign, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 1));

        // Overwrite integer_type
        let length_pointer = integer_type;
        self.generator
            .add_instruction(Instruction::Mv(length_pointer, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 4));

        // Overwrite first_magnitude_length
        let current_word_index = first_magnitude_length;
        self.generator
            .add_instruction(Instruction::Li(current_word_index, 0));

        // Overwrite second_magnitude_length
        // First carry is always 0
        let carry = second_magnitude_length;
        self.generator.add_instruction(Instruction::Li(carry, 0));

        // Overwrite bigger_magnitude
        let bigger_arg_word_location = bigger_magnitude;
        self.generator.add_instruction(Instruction::Addi(
            bigger_arg_word_location,
            bigger_magnitude,
            4,
        ));

        // Overwrite smaller_magnitude
        let smaller_arg_word_location = smaller_magnitude;
        self.generator.add_instruction(Instruction::Addi(
            smaller_arg_word_location,
            smaller_magnitude,
            4,
        ));

        {
            self.generator
                .add_instruction(Instruction::Label("add_words".to_string()));

            self.generator.add_instruction(Instruction::Beq(
                current_word_index,
                max_magnitude_length,
                "finalize_int_value".to_string(),
            ));

            let bigger = self.sixth_temp;
            self.generator
                .add_instruction(Instruction::Lw(bigger, 0, bigger_arg_word_location));

            let smaller = self.seventh_temp;
            self.generator.add_instruction(Instruction::Bge(
                current_word_index,
                smaller_length,
                "smaller_length".to_string(),
            ));

            {
                self.generator.add_instruction(Instruction::Lw(
                    smaller,
                    0,
                    smaller_arg_word_location,
                ));

                self.generator
                    .add_instruction(Instruction::J("result".to_string()));
            }
            {
                self.generator
                    .add_instruction(Instruction::Label("smaller_length".to_string()));

                self.generator.add_instruction(Instruction::Li(smaller, 0));
            }

            self.generator
                .add_instruction(Instruction::Label("result".to_string()));

            // Overwrite smaller
            let result = smaller;
            self.generator
                .add_instruction(Instruction::Add(result, bigger, result));

            // Add previous carry
            self.generator
                .add_instruction(Instruction::Add(result, result, carry));

            // Set carry if we overflowed
            self.generator
                .add_instruction(Instruction::Sltu(carry, result, bigger));

            self.generator
                .add_instruction(Instruction::Sw(result, 0, value_builder));

            self.generator
                .add_instruction(Instruction::Addi(value_builder, value_builder, 4));

            self.generator.add_instruction(Instruction::Addi(
                bigger_arg_word_location,
                bigger_arg_word_location,
                4,
            ));

            self.generator.add_instruction(Instruction::Addi(
                smaller_arg_word_location,
                smaller_arg_word_location,
                4,
            ));

            self.generator.add_instruction(Instruction::Addi(
                current_word_index,
                current_word_index,
                1,
            ));

            self.generator
                .add_instruction(Instruction::J("add_words".to_string()));
        }

        self.generator
            .add_instruction(Instruction::Label("finalize_int_value".to_string()));

        self.generator.add_instruction(Instruction::Bne(
            carry,
            Register::Zero,
            "handle_final_carry".to_string(),
        ));
        {
            self.generator.add_instruction(Instruction::Sw(
                max_magnitude_length,
                0,
                length_pointer,
            ));

            // Reclaim 4 bytes since no carry is used
            self.generator
                .add_instruction(Instruction::Addi(heap, heap, -4));

            // ret is set earlier and never overwritten
            self.generator
                .add_instruction(Instruction::J("return".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::Label("handle_final_carry".to_string()));

            // handle carry increasing word length
            self.generator.add_instruction(Instruction::Addi(
                max_magnitude_length,
                max_magnitude_length,
                1,
            ));

            self.generator.add_instruction(Instruction::Sw(
                max_magnitude_length,
                0,
                length_pointer,
            ));

            self.generator
                .add_instruction(Instruction::Sw(carry, 0, value_builder));

            // ret is set earlier and never overwritten
            self.generator
                .add_instruction(Instruction::J("return".to_string()));
        }
    }

    // returns
    // equality flag(1:0)
    // greater magnitude
    // lesser magnitude
    // greater magnitude sign(0 if equal)
    pub fn compare_magnitude(&mut self) {
        let first_sign = self.first_arg;
        let second_sign = self.second_arg;
        let first_value = self.third_arg;
        let second_value = self.fourth_arg;
        let callback = self.fifth_arg;

        self.generator
            .add_instruction(Instruction::Label("compare_magnitude".to_string()));

        let first_magnitude_len = self.first_temp;
        self.generator
            .add_instruction(Instruction::Lw(first_magnitude_len, 0, first_value));

        let second_magnitude_len = self.second_temp;
        self.generator
            .add_instruction(Instruction::Lw(second_magnitude_len, 0, second_value));

        // Check absolute lengths of the magnitudes
        let comparison_check = self.third_temp;
        self.generator.add_instruction(Instruction::Sltu(
            comparison_check,
            first_magnitude_len,
            second_magnitude_len,
        ));

        // If lengths are unequal we don't need to compare the
        // individual words in the magnitudes
        self.generator.add_instruction(Instruction::Bne(
            first_magnitude_len,
            second_magnitude_len,
            "unequal_values".to_string(),
        ));

        {
            // Magnitudes are the same length
            let magnitude_len = first_magnitude_len;
            self.generator
                .add_instruction(Instruction::Mv(magnitude_len, first_magnitude_len));

            let word_offset = self.fourth_temp;
            self.generator
                .add_instruction(Instruction::Slli(word_offset, magnitude_len, 2));

            {
                // Loop to compare values
                // either they are equal when we have checked each
                // word in the length or we find a word where there is
                // a difference. We do a comparison check and then branch
                // to unequal_values in that case
                self.generator
                    .add_instruction(Instruction::Label("compare_words".to_string()));

                self.generator.add_instruction(Instruction::Beq(
                    magnitude_len,
                    Register::Zero,
                    "equal_values".to_string(),
                ));

                let first_arg_offset = self.fifth_temp;
                self.generator.add_instruction(Instruction::Add(
                    first_arg_offset,
                    word_offset,
                    first_value,
                ));

                // Overwrite word_offset
                let second_arg_offset = second_magnitude_len;
                self.generator.add_instruction(Instruction::Add(
                    second_arg_offset,
                    word_offset,
                    second_value,
                ));

                let first_arg_values = self.sixth_temp;
                self.generator.add_instruction(Instruction::Lw(
                    first_arg_values,
                    0,
                    first_arg_offset,
                ));

                let second_arg_values = self.seventh_temp;
                self.generator.add_instruction(Instruction::Lw(
                    second_arg_values,
                    0,
                    second_arg_offset,
                ));

                self.generator.add_instruction(Instruction::Sltu(
                    comparison_check,
                    first_arg_values,
                    second_arg_values,
                ));

                self.generator.add_instruction(Instruction::Bne(
                    first_arg_values,
                    second_arg_values,
                    "unequal_values".to_string(),
                ));

                self.generator
                    .add_instruction(Instruction::Addi(magnitude_len, magnitude_len, -1));

                self.generator
                    .add_instruction(Instruction::Addi(word_offset, word_offset, -4));

                self.generator
                    .add_instruction(Instruction::J("compare_words".to_string()));
            }

            self.generator
                .add_instruction(Instruction::Label("equal_values".to_string()));

            // Overwrite first_sign
            let equality = self.return_reg;
            self.generator.add_instruction(Instruction::Li(equality, 1));

            // Overwrite second_sign
            let greater_magnitude = second_sign;
            self.generator
                .add_instruction(Instruction::Mv(greater_magnitude, first_value));

            // Overwrite first_value
            let lesser_magnitude = first_value;
            self.generator
                .add_instruction(Instruction::Mv(lesser_magnitude, second_value));

            // Overwrite second_value
            let sign = second_value;
            self.generator.add_instruction(Instruction::Li(sign, 0));

            self.generator
                .add_instruction(Instruction::Jalr(self.discard, callback, 0));
        }
        {
            self.generator
                .add_instruction(Instruction::Label("unequal_values".to_string()));

            self.generator.add_instruction(Instruction::Bne(
                comparison_check,
                Register::Zero,
                "first_value_smaller".to_string(),
            ));

            {
                let first_sign_temp = self.fourth_temp;
                self.generator
                    .add_instruction(Instruction::Mv(first_sign_temp, first_sign));

                // Overwrite first_sign
                let equality = self.return_reg;
                // Values are not equal so return 0
                self.generator
                    .add_instruction(Instruction::Mv(equality, Register::Zero));

                // Overwrite second_sign
                let greater_value = second_sign;
                self.generator
                    .add_instruction(Instruction::Mv(greater_value, first_value));

                // Overwrite first_value
                let lesser_value = first_value;
                self.generator
                    .add_instruction(Instruction::Mv(lesser_value, second_value));

                // Overwrite second_value
                let greater_sign = second_value;
                self.generator
                    .add_instruction(Instruction::Mv(greater_sign, first_sign_temp));

                self.generator
                    .add_instruction(Instruction::Jalr(self.discard, callback, 0));
            }
            {
                self.generator
                    .add_instruction(Instruction::Label("first_value_smaller".to_string()));

                let second_sign_temp = self.fourth_temp;
                self.generator
                    .add_instruction(Instruction::Mv(second_sign_temp, second_sign));

                // Overwrite first_sign
                let equality = self.return_reg;
                // Values are not equal so return 0
                self.generator
                    .add_instruction(Instruction::Mv(equality, Register::Zero));

                // Overwrite second_sign
                let greater_value = second_sign;
                self.generator
                    .add_instruction(Instruction::Mv(greater_value, second_value));

                // Overwrite first_value
                let lesser_value = first_value;
                self.generator
                    .add_instruction(Instruction::Mv(lesser_value, first_value));

                let greater_sign = second_value;
                self.generator
                    .add_instruction(Instruction::Mv(greater_sign, second_sign_temp));

                self.generator
                    .add_instruction(Instruction::Jalr(self.discard, callback, 0));
            }
        }
    }

    pub fn sub_signed_integers(&mut self) {
        let first_sign = self.first_arg;
        let second_sign = self.second_arg;
        let first_value = self.third_arg;
        let second_value = self.fourth_arg;
        let heap = self.heap;
        self.generator
            .add_instruction(Instruction::Label("sub_signed_integers".to_string()));

        let callback = self.fifth_arg;
        self.generator
            .add_instruction(Instruction::Jal(callback, "compare_magnitude".to_string()));
        // Overwrite first_sign
        let equality = first_sign;
        // Overwrite second_sign
        let greater_value = second_sign;
        // Overwrite first_value
        let lesser_value = first_value;
        // Overwrite second_value
        let greater_sign = second_value;

        let equality_temp = self.first_temp;
        self.generator
            .add_instruction(Instruction::Mv(equality_temp, equality));

        let value_builder = self.second_temp;
        self.generator
            .add_instruction(Instruction::Mv(value_builder, heap));

        // Overwrite equality
        let ret = self.return_reg;
        self.generator.add_instruction(Instruction::Mv(ret, heap));

        let greater_magnitude_len = self.third_temp;
        self.generator
            .add_instruction(Instruction::Lw(greater_magnitude_len, 0, greater_value));

        let lesser_magnitude_len = self.seventh_arg;
        self.generator
            .add_instruction(Instruction::Lw(lesser_magnitude_len, 0, lesser_value));

        let max_heap_allocation = self.fourth_temp;
        self.generator.add_instruction(Instruction::Slli(
            max_heap_allocation,
            greater_magnitude_len,
            2,
        ));

        self.generator.add_instruction(Instruction::Addi(
            max_heap_allocation,
            max_heap_allocation,
            12,
        ));
        // Add maximum heap value needed possibly and reclaim later after addition
        self.generator
            .add_instruction(Instruction::Add(heap, heap, max_heap_allocation));

        // Overwrite max_heap_allocation
        let value_tag = max_heap_allocation;
        self.generator
            .add_instruction(Instruction::Li(value_tag, 0));

        self.generator
            .add_instruction(Instruction::Sb(value_tag, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 1));

        // Overwrite value_tag
        let integer_pointer = value_tag;
        // We store integer immediately after the pointer so we simply add 4 to point to the location
        self.generator
            .add_instruction(Instruction::Addi(integer_pointer, value_builder, 4));

        self.generator
            .add_instruction(Instruction::Sw(integer_pointer, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 4));

        // Overwrite integer_pointer
        let integer_type = integer_pointer;
        self.generator
            .add_instruction(Instruction::Li(integer_type, 1));

        self.generator
            .add_instruction(Instruction::Sh(integer_type, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 2));

        // Store greater sign
        self.generator
            .add_instruction(Instruction::Sb(greater_sign, 0, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 1));

        // Overwrite integer_type
        let length_pointer = integer_type;
        self.generator
            .add_instruction(Instruction::Mv(length_pointer, value_builder));

        self.generator
            .add_instruction(Instruction::Addi(value_builder, value_builder, 4));

        self.generator.add_instruction(Instruction::Bne(
            equality_temp,
            Register::Zero,
            "equal_value_subtraction".to_string(),
        ));
        {
            // Overwrite equality_temp
            let current_word_index = equality_temp;
            self.generator
                .add_instruction(Instruction::Li(current_word_index, 0));

            // First carry is always 0
            let carry = self.fifth_temp;
            self.generator.add_instruction(Instruction::Li(carry, 0));

            // Overwrite first_magnitude
            let greater_arg_word_location = greater_value;
            self.generator.add_instruction(Instruction::Addi(
                greater_arg_word_location,
                greater_value,
                4,
            ));

            // Overwrite second_magnitude
            let lesser_arg_word_location = lesser_value;
            self.generator.add_instruction(Instruction::Addi(
                lesser_arg_word_location,
                lesser_value,
                4,
            ));

            let reclaim_heap_amount = callback;
            self.generator
                .add_instruction(Instruction::Li(reclaim_heap_amount, 0));

            let final_length = self.sixth_arg;
            self.generator
                .add_instruction(Instruction::Li(final_length, 0));

            {
                self.generator
                    .add_instruction(Instruction::Label("sub_words".to_string()));

                self.generator.add_instruction(Instruction::Beq(
                    current_word_index,
                    greater_magnitude_len,
                    "finalize_sub_int_value".to_string(),
                ));

                let arg_word_greater = self.sixth_temp;
                self.generator.add_instruction(Instruction::Lw(
                    arg_word_greater,
                    0,
                    greater_arg_word_location,
                ));

                let arg_word_smaller = self.seventh_temp;
                self.generator.add_instruction(Instruction::Bge(
                    current_word_index,
                    lesser_magnitude_len,
                    "lesser".to_string(),
                ));

                {
                    self.generator.add_instruction(Instruction::Lw(
                        arg_word_smaller,
                        0,
                        lesser_arg_word_location,
                    ));

                    self.generator
                        .add_instruction(Instruction::J("sub_result".to_string()));
                }
                {
                    self.generator
                        .add_instruction(Instruction::Label("lesser".to_string()));

                    self.generator
                        .add_instruction(Instruction::Li(arg_word_smaller, 0));
                }

                self.generator
                    .add_instruction(Instruction::Label("sub_result".to_string()));

                // Overwrite arg_word_smaller
                let result = arg_word_smaller;
                self.generator.add_instruction(Instruction::Sub(
                    result,
                    arg_word_greater,
                    arg_word_smaller,
                ));

                // Overwrite greater_sign
                let first_carry_check = greater_sign;
                // Check result is more than first arg thus needing a carry
                self.generator.add_instruction(Instruction::Sltu(
                    first_carry_check,
                    arg_word_greater,
                    result,
                ));

                // Sub previous carry
                self.generator
                    .add_instruction(Instruction::Sub(result, result, carry));

                // Set carry if we overflowed
                self.generator
                    .add_instruction(Instruction::Sltu(carry, arg_word_greater, result));

                // OR the carry checks
                self.generator
                    .add_instruction(Instruction::Or(carry, carry, first_carry_check));

                self.generator
                    .add_instruction(Instruction::Sw(result, 0, value_builder));

                self.generator
                    .add_instruction(Instruction::Addi(value_builder, value_builder, 4));

                self.generator.add_instruction(Instruction::Addi(
                    greater_arg_word_location,
                    greater_arg_word_location,
                    4,
                ));

                self.generator.add_instruction(Instruction::Addi(
                    lesser_arg_word_location,
                    lesser_arg_word_location,
                    4,
                ));

                self.generator.add_instruction(Instruction::Addi(
                    current_word_index,
                    current_word_index,
                    1,
                ));

                self.generator.add_instruction(Instruction::Beq(
                    Register::Zero,
                    result,
                    "can_reclaim_heap_word".to_string(),
                ));

                {
                    self.generator.add_instruction(Instruction::Addi(
                        final_length,
                        final_length,
                        1,
                    ));

                    self.generator.add_instruction(Instruction::Add(
                        final_length,
                        final_length,
                        reclaim_heap_amount,
                    ));

                    self.generator
                        .add_instruction(Instruction::Li(reclaim_heap_amount, 0));

                    self.generator
                        .add_instruction(Instruction::J("sub_words".to_string()));
                }
                {
                    self.generator
                        .add_instruction(Instruction::Label("can_reclaim_heap_word".to_string()));

                    self.generator.add_instruction(Instruction::Addi(
                        reclaim_heap_amount,
                        reclaim_heap_amount,
                        1,
                    ));

                    self.generator
                        .add_instruction(Instruction::J("sub_words".to_string()));
                }
            }

            self.generator
                .add_instruction(Instruction::Label("finalize_sub_int_value".to_string()));

            self.generator
                .add_instruction(Instruction::Sw(final_length, 0, length_pointer));

            self.generator.add_instruction(Instruction::Slli(
                reclaim_heap_amount,
                reclaim_heap_amount,
                2,
            ));

            self.generator
                .add_instruction(Instruction::Sub(heap, heap, reclaim_heap_amount));

            self.generator
                .add_instruction(Instruction::J("return".to_string()));
        }
        {
            self.generator
                .add_instruction(Instruction::Label("equal_value_subtraction".to_string()));

            // In this case equality_temp stores 1 anyway
            // but to make sure we still set to 1
            self.generator
                .add_instruction(Instruction::Li(equality_temp, 1));

            self.generator
                .add_instruction(Instruction::Sw(equality_temp, 0, length_pointer));

            self.generator
                .add_instruction(Instruction::Sw(Register::Zero, 0, value_builder));

            // reclaim heap since value is 0 and length is 1
            self.generator
                .add_instruction(Instruction::Addi(heap, value_builder, 4));

            self.generator
                .add_instruction(Instruction::J("return".to_string()));
        }
    }
}

impl Default for Cek {
    fn default() -> Self {
        Self::new()
    }
}

pub fn u32_vec_to_u8_vec(input: Vec<u32>) -> Vec<u8> {
    input
        .into_iter()
        .flat_map(|num| num.to_be_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use emulator::ExecutionResult;
    use risc_v_gen::emulator::verify_file;
    use uplc::ast::{DeBruijn, Name, Program, Term};
    use uplc_serializer::{constants::const_tag, serialize};

    use crate::cek::{u32_vec_to_u8_vec, Cek};

    #[test]
    fn test_cek_machine() {
        // Initialize the CEK machine
        let mut cek = Cek::default();

        // Generate the core CEK implementation
        cek.init();
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
        cek.handle_constr();
        cek.handle_case();
        cek.handle_frame_await_fun_term();
        cek.handle_frame_await_arg();
        cek.handle_frame_await_fun_value();
        cek.handle_frame_force();
        cek.handle_frame_constr();
        cek.handle_frame_case();
        cek.handle_no_frame();
        cek.halt();
        cek.force_evaluate();
        cek.apply_evaluate();
        cek.lookup();
        cek.clone_list();
        cek.reverse_clone_list();
        cek.eval_builtin_app();
        cek.unwrap_integer();
        cek.unwrap_bytestring();
        cek.add_integer();
        cek.sub_integer();
        cek.multiply_integer();
        cek.divide_integer();
        cek.quotient_integer();
        cek.remainder_integer();
        cek.mod_integer();
        cek.equals_integer();
        cek.less_than_integer();
        cek.less_than_equals_integer();
        cek.append_bytestring();
        cek.add_signed_integers();
        cek.compare_magnitude();
        cek.sub_signed_integers();
        cek.initial_term(vec![
            /*apply*/ 3, /* arg pointer*/ 11, 0, 0, 144, /*lambda*/ 2,
            /*var*/ 0, 1, 0, 0, 0, /*constant*/ 4, 13, 0, 0, 0,
        ]);

        let code_gen = cek.generator;

        println!("{}", code_gen.generate());
    }

    #[test]
    fn test_apply_lambda_var_constant() {
        let thing = Cek::default();

        let gene = thing.cek_assembly(vec![
            /*apply*/ 3, /* arg pointer*/ 11, 0, 0, 144, /*lambda*/ 2,
            /*var*/ 0, 1, 0, 0, 0, /*constant*/ 4, /* type length in bytes */ 1,
            /* integer */ 0, /* sign */ 0, /* length */ 1, 0, 0, 0,
            /*value (little-endian) */ 13, 0, 0, 0,
        ]);

        // println!("{}", gene.generate());

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

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            _ => unreachable!("HOW?"),
        };

        assert_ne!(result_pointer, u32::MAX);

        let section = v.2.find_section(result_pointer).unwrap();

        let section_data = u32_vec_to_u8_vec(section.data.clone());

        let offset_index = (result_pointer - section.start) as usize;

        let type_length = section_data[offset_index];

        assert_eq!(type_length, 1);

        let constant_type = section_data[offset_index + 1];

        assert_eq!(constant_type, 0);

        let sign = section_data[offset_index + 2];

        assert_eq!(sign, 0);

        let word_length = *section_data[(offset_index + 3)..(offset_index + 7)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(word_length, 1);

        let value = *section_data[(offset_index + 7)..(offset_index + 11)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(value, 13);
    }

    #[test]
    fn test_add_integer_double() {
        let thing = Cek::default();

        let gene = thing.cek_assembly(vec![
            /*apply*/ 3, /* arg pointer*/ 28, 0, 0, 144, /*lambda*/ 2,
            /*apply */ 3, /* arg pointer*/ 18, 0, 0, 144, /*apply */ 3,
            /* arg pointer*/ 23, 0, 0, 144, /*add_integer */ 7, 0, /*var*/ 0, 1, 0,
            0, 0, /*var*/ 0, 1, 0, 0, 0, /*constant*/ 4,
            /* type length in bytes */ 1, /* integer */ 0, /* sign */ 0,
            /* length */ 1, 0, 0, 0, /*value (little-endian) */ 13, 0, 0, 0,
        ]);

        // println!("{}", gene.generate());

        gene.save_to_file("test_add.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_add.o",
                "test_add.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_add.elf",
                "-T",
                "../../linker/link.ld",
                "test_add.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_add.elf").unwrap();

        // let mut file = File::create("bbbb.txt").unwrap();
        // write!(
        //     &mut file,
        //     "{}",
        //     v.1.iter()
        //         .map(|(item, _)| {
        //             format!(
        //                 "Step number: {}, Opcode: {:#?}, hex: {:#x}\nFull: {:#?}",
        //                 item.step_number,
        //                 riscv_decode::decode(item.read_pc.opcode),
        //                 item.read_pc.opcode,
        //                 item,
        //             )
        //         })
        //         .collect::<Vec<String>>()
        //         .join("\n")
        // )
        // .unwrap();
        // file.flush().unwrap();

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            _ => unreachable!("HOW?"),
        };

        assert_ne!(result_pointer, u32::MAX);

        let section = v.2.find_section(result_pointer).unwrap();

        let section_data = u32_vec_to_u8_vec(section.data.clone());

        let offset_index = (result_pointer - section.start) as usize;

        let type_length = section_data[offset_index];

        assert_eq!(type_length, 1);

        let constant_type = section_data[offset_index + 1];

        assert_eq!(constant_type, 0);

        let sign = section_data[offset_index + 2];

        assert_eq!(sign, 0);

        let word_length = *section_data[(offset_index + 3)..(offset_index + 7)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(word_length, 1);

        let value = *section_data[(offset_index + 7)..(offset_index + 11)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(value, 26);
    }

    #[test]
    fn test_force_delay_error() {
        let thing = Cek::default();

        let gene = thing.cek_assembly(vec![
            /*force */ 5, /*delay */ 1, /*error */ 6, 0,
        ]);

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
    fn test_case_constr_lambda_lambda_var_constant() {
        let thing = Cek::default();

        let gene = thing.cek_assembly(vec![
            /*case */ 9, /*const pointer */ 17, 0, 0, 144, /*branches length */ 2,
            0, 0, 0, /*first branch pointer */ 58, 0, 0, 144,
            /*second branch pointer */ 59, 0, 0, 144, /*constr*/ 8, /* tag*/ 1, 0,
            0, 0, /*fields length */ 2, 0, 0, 0, /*first field pointer */ 34, 0, 0, 144,
            /*second field pointer */ 46, 0, 0, 144, /* first field constant */ 4,
            /* type length in bytes */ 1, /* integer */ 0, /* sign */ 0,
            /* length */ 1, 0, 0, 0, /*integer 99 */ 99, 0, 0, 0,
            /*second field constant */ 4, /* type length in bytes */ 1,
            /* integer */ 0, /* sign */ 0, /* length */ 1, 0, 0, 0,
            /*integer 13 */ 13, 0, 0, 0, /*first branch error*/ 6,
            /*second branch lambda */ 2, /*lambda */ 2, /*var */ 0,
            /*second debruijn index */ 2, 0, 0, 0, 0, 0,
        ]);

        gene.save_to_file("test_case_constr.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_case_constr.o",
                "test_case_constr.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_case_constr.elf",
                "-T",
                "../../linker/link.ld",
                "test_case_constr.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_case_constr.elf").unwrap();

        // let mut file = File::create("bbbb.txt").unwrap();
        // write!(
        //     &mut file,
        //     "{}",
        //     v.1.iter()
        //         .map(|(item, _)| {
        //             format!(
        //                 "Step number: {}, Opcode: {:#?}, hex: {:#x}\nFull: {:#?}",
        //                 item.step_number,
        //                 riscv_decode::decode(item.read_pc.opcode),
        //                 item.read_pc.opcode,
        //                 item,
        //             )
        //         })
        //         .collect::<Vec<String>>()
        //         .join("\n")
        // )
        // .unwrap();
        // file.flush().unwrap();
        //

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            _ => unreachable!("HOW?"),
        };

        assert_ne!(result_pointer, u32::MAX);

        let section = v.2.find_section(result_pointer).unwrap();

        let section_data = u32_vec_to_u8_vec(section.data.clone());

        let offset_index = (result_pointer - section.start) as usize;

        let type_length = section_data[offset_index];

        assert_eq!(type_length, 1);

        let constant_type = section_data[offset_index + 1];

        assert_eq!(constant_type, 0);

        let sign = section_data[offset_index + 2];

        assert_eq!(sign, 0);

        let word_length = *section_data[(offset_index + 3)..(offset_index + 7)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(word_length, 1);

        let value = *section_data[(offset_index + 7)..(offset_index + 11)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(value, 99);
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

        // (apply (lambda x (force x)) (delay (error)))
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
    #[test]
    fn test_apply_apply_builtin_serialize() {
        let thing = Cek::default();

        // (apply (lambda x (force x)) (delay (error)))
        let term: Term<Name> = Term::add_integer()
            .apply(Term::Error.delay())
            .apply(Term::Error.delay());

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_serialize_3.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_serialize_3.o",
                "test_serialize_3.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_serialize_3.elf",
                "-T",
                "../../linker/link.ld",
                "test_serialize_3.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_serialize_3.elf").unwrap();

        // let mut file = File::create("bbbb.txt").unwrap();
        // write!(
        //     &mut file,
        //     "{}",
        //     v.1.iter()
        //         .map(|(item, _)| {
        //             format!(
        //                 "Step number: {}, Opcode: {:#?}, hex: {:#x}\nFull: {:#?}",
        //                 item.step_number,
        //                 riscv_decode::decode(item.read_pc.opcode),
        //                 item.read_pc.opcode,
        //                 item,
        //             )
        //         })
        //         .collect::<Vec<String>>()
        //         .join("\n")
        // )
        // .unwrap();
        // file.flush().unwrap();

        match v.0 {
            ExecutionResult::Halt(result, _step) => assert_eq!(result, u32::MAX),
            g => unreachable!("HOW? {:#?}", g),
        }
    }

    #[test]
    fn test_add_2_different_size_numbers() {
        let thing = Cek::default();

        // (apply (lambda x (force x)) (delay (error)))
        let term: Term<Name> = Term::add_integer()
            .apply(Term::integer((-5_000_000_000_i128).into()))
            .apply(Term::integer((-5).into()));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        // println!("{:#?}", riscv_program);

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_add_big_int.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_add_big_int.o",
                "test_add_big_int.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_add_big_int.elf",
                "-T",
                "../../linker/link.ld",
                "test_add_big_int.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_add_big_int.elf").unwrap();

        // let mut file = File::create("bbbb.txt").unwrap();
        // write!(
        //     &mut file,
        //     "{}",
        //     v.1.iter()
        //         .map(|(item, _)| {
        //             format!(
        //                 "Step number: {}, Opcode: {:#?}, hex: {:#x}\nFull: {:#?}",
        //                 item.step_number,
        //                 riscv_decode::decode(item.read_pc.opcode),
        //                 item.read_pc.opcode,
        //                 item,
        //             )
        //         })
        //         .collect::<Vec<String>>()
        //         .join("\n")
        // )
        // .unwrap();
        // file.flush().unwrap();

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            _ => unreachable!("HOW?"),
        };

        assert_ne!(result_pointer, u32::MAX);

        let section = v.2.find_section(result_pointer).unwrap();

        let section_data = u32_vec_to_u8_vec(section.data.clone());

        let offset_index = (result_pointer - section.start) as usize;

        // println!("{:#?}", &section_data[offset_index..(offset_index + 100)]);

        let type_length = section_data[offset_index];

        assert_eq!(type_length, 1);

        let constant_type = section_data[offset_index + 1];

        assert_eq!(constant_type, 0);

        let sign = section_data[offset_index + 2];

        assert_eq!(sign, 1);

        let word_length = *section_data[(offset_index + 3)..(offset_index + 7)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(word_length, 2);

        let value = section_data[(offset_index + 7)..(offset_index + 15)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>();

        let result = value[0] as u64 + value[1] as u64 * 256_u64.pow(4);

        assert_eq!(result, 5000000005);
    }

    #[test]
    fn test_add_2_different_sign_numbers() {
        let thing = Cek::default();

        // (apply (lambda x (force x)) (delay (error)))
        let term: Term<Name> = Term::add_integer()
            .apply(Term::integer((-5_000_000_000_i128).into()))
            .apply(Term::integer((5).into()));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_add_big_int_sign.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_add_big_int_sign.o",
                "test_add_big_int_sign.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_add_big_int_sign.elf",
                "-T",
                "../../linker/link.ld",
                "test_add_big_int_sign.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_add_big_int_sign.elf").unwrap();

        // let mut file = File::create("bbbb.txt").unwrap();
        // write!(
        //     &mut file,
        //     "{}",
        //     v.1.iter()
        //         .map(|(item, _)| {
        //             format!(
        //                 "Step number: {}, Opcode: {:#?}, hex: {:#x}\nFull: {:#?}",
        //                 item.step_number,
        //                 riscv_decode::decode(item.read_pc.opcode),
        //                 item.read_pc.opcode,
        //                 item,
        //             )
        //         })
        //         .collect::<Vec<String>>()
        //         .join("\n")
        // )
        // .unwrap();
        // file.flush().unwrap();

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            a => unreachable!("HOW? {:#?}", a),
        };

        assert_ne!(result_pointer, u32::MAX);

        let section = v.2.find_section(result_pointer).unwrap();

        let section_data = u32_vec_to_u8_vec(section.data.clone());

        let offset_index = (result_pointer - section.start) as usize;

        let type_length = section_data[offset_index];

        assert_eq!(type_length, 1);

        let constant_type = section_data[offset_index + 1];

        assert_eq!(constant_type, 0);

        let sign = section_data[offset_index + 2];

        assert_eq!(sign, 1);

        let word_length = *section_data[(offset_index + 3)..(offset_index + 7)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(word_length, 2);

        let value = section_data[(offset_index + 7)..(offset_index + 15)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>();

        let result = value[0] as u64 + value[1] as u64 * 256_u64.pow(4);

        assert_eq!(result, 4999999995);
    }

    #[test]
    fn test_sub_2_numbers() {
        let thing = Cek::default();

        // (apply (lambda x (force x)) (delay (error)))
        let term: Term<Name> = Term::subtract_integer()
            .apply(Term::integer((5_000_000_000_i128).into()))
            .apply(Term::integer((5).into()));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_sub_big_int.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_sub_big_int.o",
                "test_sub_big_int.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_sub_big_int.elf",
                "-T",
                "../../linker/link.ld",
                "test_sub_big_int.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_sub_big_int.elf").unwrap();

        // let mut file = File::create("bbbb.txt").unwrap();
        // write!(
        //     &mut file,
        //     "{}",
        //     v.1.iter()
        //         .map(|(item, _)| {
        //             format!(
        //                 "Step number: {}, Opcode: {:#?}, hex: {:#x}\nFull: {:#?}",
        //                 item.step_number,
        //                 riscv_decode::decode(item.read_pc.opcode),
        //                 item.read_pc.opcode,
        //                 item,
        //             )
        //         })
        //         .collect::<Vec<String>>()
        //         .join("\n")
        // )
        // .unwrap();
        // file.flush().unwrap();

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            a => unreachable!("HOW? {:#?}", a),
        };

        assert_ne!(result_pointer, u32::MAX);

        let section = v.2.find_section(result_pointer).unwrap();

        let section_data = u32_vec_to_u8_vec(section.data.clone());

        let offset_index = (result_pointer - section.start) as usize;

        let type_length = section_data[offset_index];

        assert_eq!(type_length, 1);

        let constant_type = section_data[offset_index + 1];

        assert_eq!(constant_type, 0);

        let sign = section_data[offset_index + 2];

        assert_eq!(sign, 0);

        let word_length = *section_data[(offset_index + 3)..(offset_index + 7)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(word_length, 2);

        let value = section_data[(offset_index + 7)..(offset_index + 15)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>();

        let result = value[0] as u64 + value[1] as u64 * 256_u64.pow(4);

        assert_eq!(result, 4999999995);
    }

    #[test]
    fn test_equals_numbers() {
        let thing = Cek::default();

        // (apply (lambda x (force x)) (delay (error)))
        let term: Term<Name> = Term::equals_integer()
            .apply(
                Term::subtract_integer()
                    .apply(Term::integer((5_000_000_000_i128).into()))
                    .apply(Term::integer((5).into())),
            )
            .apply(Term::integer((4_999_999_995_i128).into()));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_equal_big_int.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_equal_big_int.o",
                "test_equal_big_int.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_equal_big_int.elf",
                "-T",
                "../../linker/link.ld",
                "test_equal_big_int.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_equal_big_int.elf").unwrap();

        // let mut file = File::create("bbbb.txt").unwrap();
        // write!(
        //     &mut file,
        //     "{}",
        //     v.1.iter()
        //         .map(|(item, _)| {
        //             format!(
        //                 "Step number: {}, Opcode: {:#?}, hex: {:#x}\nFull: {:#?}",
        //                 item.step_number,
        //                 riscv_decode::decode(item.read_pc.opcode),
        //                 item.read_pc.opcode,
        //                 item,
        //             )
        //         })
        //         .collect::<Vec<String>>()
        //         .join("\n")
        // )
        // .unwrap();
        // file.flush().unwrap();

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            a => unreachable!("HOW? {:#?}", a),
        };

        assert_ne!(result_pointer, u32::MAX);

        let section = v.2.find_section(result_pointer).unwrap();

        let section_data = u32_vec_to_u8_vec(section.data.clone());

        let offset_index = (result_pointer - section.start) as usize;

        let type_length = section_data[offset_index];

        assert_eq!(type_length, 1);

        let constant_type = section_data[offset_index + 1];

        assert_eq!(constant_type, const_tag::BOOL);

        let boolean = section_data[offset_index + 2];

        assert_eq!(boolean, 1)
    }

    #[test]
    fn test_less_numbers() {
        let thing = Cek::default();

        // (apply (lambda x (force x)) (delay (error)))
        let term: Term<Name> = Term::less_than_integer()
            .apply(
                Term::subtract_integer()
                    .apply(Term::integer((5_000_000_000_i128).into()))
                    .apply(Term::integer((5).into())),
            )
            .apply(Term::integer((4_999_999_995_i128).into()));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_less_big_int.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_less_big_int.o",
                "test_less_big_int.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_less_big_int.elf",
                "-T",
                "../../linker/link.ld",
                "test_less_big_int.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_less_big_int.elf").unwrap();

        // let mut file = File::create("bbbb.txt").unwrap();
        // write!(
        //     &mut file,
        //     "{}",
        //     v.1.iter()
        //         .map(|(item, _)| {
        //             format!(
        //                 "Step number: {}, Opcode: {:#?}, hex: {:#x}\nFull: {:#?}",
        //                 item.step_number,
        //                 riscv_decode::decode(item.read_pc.opcode),
        //                 item.read_pc.opcode,
        //                 item,
        //             )
        //         })
        //         .collect::<Vec<String>>()
        //         .join("\n")
        // )
        // .unwrap();
        // file.flush().unwrap();

        let result_pointer = match v.0 {
            ExecutionResult::Halt(result, _step) => result,
            a => unreachable!("HOW? {:#?}", a),
        };

        assert_ne!(result_pointer, u32::MAX);

        let section = v.2.find_section(result_pointer).unwrap();

        let section_data = u32_vec_to_u8_vec(section.data.clone());

        let offset_index = (result_pointer - section.start) as usize;

        let type_length = section_data[offset_index];

        assert_eq!(type_length, 1);

        let constant_type = section_data[offset_index + 1];

        assert_eq!(constant_type, const_tag::BOOL);

        let boolean = section_data[offset_index + 2];

        assert_eq!(boolean, 1)
    }
}
