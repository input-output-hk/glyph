use risc_v_gen::{Assign, CodeGenerator, Instruction, Register};
use strum::IntoEnumIterator;
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

macro_rules! var {
    ($var:ident = $cek:ident . $register:ident) => {
        let mut $var = $cek.register_map.var(stringify!($var), $cek.$register);
    };
}

macro_rules! constnt {
    ($var:ident = $cek:ident . $register:ident) => {
        let mut $var = $cek.register_map.constnt(stringify!($var), $cek.$register);
    };
}

macro_rules! argument {
    ($var:ident = $cek:ident . $register:ident) => {
        let mut $var = $cek.register_map.constnt(stringify!($var), $cek.$register);
        $var.assigned = true;
    };
}

macro_rules! var_argument {
    ($var:ident = $cek:ident . $register:ident) => {
        let mut $var = $cek.register_map.var(stringify!($var), $cek.$register);
        $var.assigned = true;
    };
}

macro_rules! var_overwrite {
    ($var:ident = $value:expr) => {
        let mut $var = $value.var_overwrite(stringify!($var));
    };
}

macro_rules! constnt_overwrite {
    ($var:ident = $value:expr) => {
        let mut $var = $value.constnt_overwrite(stringify!($var));
    };
}

// macro_rules! var_arg_overwrite {
//     ($var:ident = $value:expr) => {
//         let mut $var = $value.var_overwrite(stringify!($var));
//         $var.assigned = true;
//     };
// }

macro_rules! constnt_arg_overwrite {
    ($var:ident = $value:expr) => {
        let mut $var = $value.constnt_overwrite(stringify!($var));
        $var.assigned = true;
    };
}

// This is used to automatically free up registers that
// are only used with vars in blocks
macro_rules! create_block {
    // Match a block containing zero or more statements
    ($cek:ident, { $($stmt:tt)* }) => {
        {
            let mut var_assigns = Vec::new();
            create_block!(var_assigns, @process $($stmt)*);
            $cek.register_map.free_assigns(var_assigns);

        }
    };
    ($vec:ident, @process) => {};

    ($vec:ident, @process var ! $inner:tt; $($rest:tt)*) => {
        create_block!($vec, @parse_var $inner);
        create_block!($vec, @process $($rest)*);
    };

    ($vec:ident, @process constnt ! $inner:tt; $($rest:tt)*) => {
        create_block!($vec, @parse_constnt $inner);
        create_block!($vec, @process $($rest)*);
    };

    ($vec:ident, @parse_var ($name:ident = $cek:ident . $register:ident)) => {
        var!($name = $cek.$register);
        $vec.push($name.clone());
    };

    ($vec:ident, @parse_constnt ($name:ident = $cek:ident . $register:ident)) => {
        constnt!($name = $cek.$register);
        $vec.push($name.clone());
    };

    ($vec:ident, @process $thing:stmt; $($rest:tt)*) => {
        $thing
        create_block!($vec, @process $($rest)*);
    };
    ($vec:ident, @process $thing:tt $($rest:tt)*) => {
        $thing
        create_block!($vec, @process $($rest)*);
    };
}

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
    register_map: RegisterMap,
    frames: Register,
    env: Register,
    heap: Register,
    saved: Register,
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
    zero: Register,
}

impl Cek {
    pub fn new() -> Cek {
        Cek {
            generator: CodeGenerator::default(),
            register_map: RegisterMap::default(),
            frames: Register::Sp,
            env: Register::S1,
            heap: Register::S2,
            saved: Register::S3,
            discard: Register::Zero,
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
            zero: Register::Zero,
        }
    }

    pub fn cek_assembly(mut self, bytes: Vec<u8>) -> CodeGenerator {
        // Generate the core CEK implementation
        self.generator
            .add_instruction(Instruction::global("_start".to_string()));
        self.generator
            .add_instruction(Instruction::label("_start".to_string()));
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
        self.allocate_integer_type();
        self.allocate_bytestring_type();
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
        self.cons_bytestring();
        self.slice_bytestring();
        self.length_bytestring();
        self.index_bytestring();
        self.equals_bytestring();
        self.less_than_bytestring();
        self.less_than_equals_bytestring();
        self.sha2_256();
        self.sha3_256();
        self.blake2b_256();
        self.verify_ed25519_signature();
        self.append_string();
        self.equals_string();
        self.encode_utf8();
        self.decode_utf8();
        self.if_then_else();
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

    pub fn init(&mut self) -> Freed {
        constnt!(heap = self.heap);
        var_argument!(frames = self.frames);
        constnt!(env = self.env);
        argument!(zero = self.zero);

        self.generator
            .add_instruction(Instruction::section("text".to_string()));

        self.generator
            .add_instruction(Instruction::label("init".to_string()));

        self.generator
            .add_instruction(Instruction::lui(&mut heap, 0xc0000));

        // self.generator
        //     .add_instruction(Instruction::Lui(self.frames, 0xe0000));

        self.generator.add_instruction(Instruction::comment(
            "1 byte for NoFrame allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::addi(&mut frames.clone(), &frames, -1));

        self.generator
            .add_instruction(Instruction::comment("Tag is 6 for NoFrame".to_string()));

        constnt!(frame_tag = self.first_temp);

        self.generator
            .add_instruction(Instruction::li(&mut frame_tag, 6));

        self.generator.add_instruction(Instruction::comment(
            "Push NoFrame tag onto stack".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::sb(&frame_tag, 0, &frames));

        self.generator.add_instruction(Instruction::comment(
            "Environment stack pointer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::mv(&mut env, &zero));

        self.generator
            .add_instruction(Instruction::comment("A0 is return register".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "Load address of initial_term".to_string(),
        ));

        constnt!(ret = self.first_arg);
        self.generator
            .add_instruction(Instruction::la(&mut ret, "initial_term".to_string()));

        self.generator
            .add_instruction(Instruction::j("compute".to_string()));

        self.register_map.free_all()
    }

    // TODO: for both compute and return_compute, We can compute the jump via term offset
    pub fn compute(&mut self) -> Freed {
        argument!(term = self.first_arg);
        argument!(zero = self.zero);
        self.generator.add_instruction(Instruction::comment(
            "  s0: KP - Continuation stack pointer (points to top of K stack)".to_string(),
        ));
        self.generator.add_instruction(Instruction::comment(
            "  s1: E  - Environment pointer (points to current environment linked list)"
                .to_string(),
        ));
        self.generator.add_instruction(Instruction::comment(
            "  s2: HP - Heap pointer (points to next free heap address)".to_string(),
        ));
        self.generator.add_instruction(Instruction::comment(
            "  a0: Storage for C (control) pointer".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::label("compute".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "Term address should be in A0".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Load term tag".to_string()));

        constnt!(term_tag = self.first_temp);

        self.generator
            .add_instruction(Instruction::lbu(&mut term_tag, 0, &term));

        self.generator
            .add_instruction(Instruction::comment("Var".to_string()));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &zero,
            "handle_var".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Delay".to_string()));

        var!(match_tag = self.second_temp);

        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 1));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &match_tag,
            "handle_delay".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Lambda".to_string()));
        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 2));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &match_tag,
            "handle_lambda".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Apply".to_string()));
        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 3));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &match_tag,
            "handle_apply".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Constant".to_string()));
        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 4));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &match_tag,
            "handle_constant".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Force".to_string()));
        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 5));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &match_tag,
            "handle_force".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Error".to_string()));
        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 6));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &match_tag,
            "handle_error".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Builtin".to_string()));
        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 7));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &match_tag,
            "handle_builtin".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Constr".to_string()));

        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 8));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &match_tag,
            "handle_constr".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("Case".to_string()));

        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 9));

        self.generator.add_instruction(Instruction::beq(
            &term_tag,
            &match_tag,
            "handle_case".to_string(),
        ));

        self.register_map.free_all()
    }

    pub fn return_compute(&mut self) -> Freed {
        argument!(zero = self.zero);
        argument!(frames = self.frames);
        self.generator
            .add_instruction(Instruction::label("return".to_string()));

        self.generator
            .add_instruction(Instruction::comment("Load Frame from S0".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "Frame tag is first byte of frame".to_string(),
        ));

        constnt!(frame_tag = self.first_temp);

        self.generator
            .add_instruction(Instruction::lbu(&mut frame_tag, 0, &frames));

        self.generator
            .add_instruction(Instruction::comment("FrameAwaitArg".to_string()));

        self.generator.add_instruction(Instruction::beq(
            &frame_tag,
            &zero,
            "handle_frame_await_arg".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("FrameAwaitFunTerm".to_string()));

        var!(match_tag = self.second_temp);

        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 1));

        self.generator.add_instruction(Instruction::beq(
            &frame_tag,
            &match_tag,
            "handle_frame_await_fun_term".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("FrameAwaitFunValue".to_string()));

        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 2));

        self.generator.add_instruction(Instruction::beq(
            &frame_tag,
            &match_tag,
            "handle_frame_await_fun_value".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("FrameForce".to_string()));

        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 3));

        self.generator.add_instruction(Instruction::beq(
            &frame_tag,
            &match_tag,
            "handle_frame_force".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 4));

        self.generator.add_instruction(Instruction::beq(
            &frame_tag,
            &match_tag,
            "handle_frame_constr".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 5));

        self.generator.add_instruction(Instruction::beq(
            &frame_tag,
            &match_tag,
            "handle_frame_case".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::comment("NoFrame".to_string()));

        self.generator
            .add_instruction(Instruction::li(&mut match_tag, 6));

        self.generator.add_instruction(Instruction::beq(
            &frame_tag,
            &match_tag,
            "handle_no_frame".to_string(),
        ));

        self.register_map.free_all()
    }

    pub fn handle_var(&mut self) -> Freed {
        argument!(var = self.first_arg);
        self.generator
            .add_instruction(Instruction::label("handle_var".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "load debruijn index into temp".to_string(),
        ));

        constnt!(var_index = self.first_temp);

        self.generator
            .add_instruction(Instruction::lw(&mut var_index, 1, &var));

        self.generator.add_instruction(Instruction::comment(
            "Put debruijn index into A0".to_string(),
        ));

        constnt_overwrite!(ret = var);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &var_index));

        self.generator
            .add_instruction(Instruction::j("lookup".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_delay(&mut self) -> Freed {
        argument!(delay_term = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(env = self.env);
        self.generator
            .add_instruction(Instruction::label("handle_delay".to_string()));

        self.generator
            .add_instruction(Instruction::comment("Term body is next byte".to_string()));

        constnt!(body = self.first_temp);

        self.generator
            .add_instruction(Instruction::addi(&mut body, &delay_term, 1));

        self.generator.add_instruction(Instruction::comment(
            "9 bytes for DelayValue allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 9));

        self.generator
            .add_instruction(Instruction::comment("tag is 1 in rust".to_string()));

        constnt!(vdelay_tag = self.second_temp);

        self.generator
            .add_instruction(Instruction::li(&mut vdelay_tag, 1));

        self.generator
            .add_instruction(Instruction::comment("first byte is tag".to_string()));

        self.generator
            .add_instruction(Instruction::sb(&vdelay_tag, -9, &heap));

        self.generator
            .add_instruction(Instruction::comment("Store body pointer".to_string()));

        self.generator
            .add_instruction(Instruction::sw(&body, -8, &heap));

        self.generator
            .add_instruction(Instruction::comment("Store environment".to_string()));

        self.generator
            .add_instruction(Instruction::sw(&env, -4, &heap));

        self.generator
            .add_instruction(Instruction::comment("Put return value into A0".to_string()));

        constnt_overwrite!(ret = delay_term);
        self.generator
            .add_instruction(Instruction::addi(&mut ret, &heap, -9));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_lambda(&mut self) -> Freed {
        argument!(lambda_term = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(env = self.env);

        self.generator
            .add_instruction(Instruction::label("handle_lambda".to_string()));

        self.generator
            .add_instruction(Instruction::comment("Term body is next byte".to_string()));

        constnt!(body = self.first_temp);

        self.generator
            .add_instruction(Instruction::addi(&mut body, &lambda_term, 1));

        self.generator.add_instruction(Instruction::comment(
            "9 bytes for LambdaValue allocation".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 9));

        self.generator
            .add_instruction(Instruction::comment("tag is 2 in rust".to_string()));

        constnt!(vlambda_tag = self.second_temp);

        self.generator
            .add_instruction(Instruction::li(&mut vlambda_tag, 2));

        self.generator
            .add_instruction(Instruction::comment("first byte is tag".to_string()));

        self.generator
            .add_instruction(Instruction::sb(&vlambda_tag, -9, &heap));

        self.generator
            .add_instruction(Instruction::comment("Store body".to_string()));

        self.generator
            .add_instruction(Instruction::sw(&body, -8, &heap));

        self.generator
            .add_instruction(Instruction::comment("Store environment".to_string()));

        self.generator
            .add_instruction(Instruction::sw(&env, -4, &heap));

        self.generator
            .add_instruction(Instruction::comment("Put return value into A0".to_string()));

        constnt_overwrite!(ret = lambda_term);
        self.generator
            .add_instruction(Instruction::addi(&mut ret, &heap, -9));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_apply(&mut self) -> Freed {
        argument!(apply_term = self.first_arg);
        var_argument!(frames = self.frames);
        argument!(env = self.env);

        self.generator
            .add_instruction(Instruction::label("handle_apply".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "Apply is tag |argument address| function".to_string(),
        ));
        self.generator.add_instruction(Instruction::comment(
            "Function is 5 bytes after tag location".to_string(),
        ));

        constnt!(function = self.first_temp);

        self.generator
            .add_instruction(Instruction::addi(&mut function, &apply_term, 5));

        self.generator
            .add_instruction(Instruction::comment("Load argument into temp".to_string()));

        constnt!(argument = self.second_temp);

        self.generator
            .add_instruction(Instruction::lw(&mut argument, 1, &apply_term));

        self.generator.add_instruction(Instruction::comment(
            "9 bytes for FrameAwaitFunTerm allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::addi(&mut frames.clone(), &frames, -9));

        self.generator.add_instruction(Instruction::comment(
            "Tag is 1 for FrameAwaitFunTerm".to_string(),
        ));

        constnt!(frame_tag = self.third_temp);

        self.generator
            .add_instruction(Instruction::li(&mut frame_tag, 1));

        self.generator
            .add_instruction(Instruction::comment("Push tag onto stack".to_string()));

        self.generator
            .add_instruction(Instruction::sb(&frame_tag, 0, &frames));

        self.generator
            .add_instruction(Instruction::comment("Push argument onto stack".to_string()));

        self.generator
            .add_instruction(Instruction::sw(&argument, 1, &frames));

        self.generator.add_instruction(Instruction::comment(
            "Push environment onto stack".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::sw(&env, 5, &frames));

        self.generator.add_instruction(Instruction::comment(
            "Put function address into A0".to_string(),
        ));

        constnt_overwrite!(ret = apply_term);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &function));

        self.generator
            .add_instruction(Instruction::j("compute".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_constant(&mut self) -> Freed {
        argument!(constant_term = self.first_arg);
        var_argument!(heap = self.heap);

        self.generator
            .add_instruction(Instruction::label("handle_constant".to_string()));

        // store pointer to constant in T0
        constnt!(constant = self.first_temp);

        self.generator
            .add_instruction(Instruction::addi(&mut constant, &constant_term, 1));

        // allocate 5 bytes on the heap
        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 5));

        constnt!(constant_tag = self.second_temp);

        self.generator
            .add_instruction(Instruction::li(&mut constant_tag, 0));

        self.generator
            .add_instruction(Instruction::sb(&constant_tag, -5, &heap));

        self.generator
            .add_instruction(Instruction::sw(&constant, -4, &heap));

        constnt_overwrite!(ret = constant_term);
        self.generator
            .add_instruction(Instruction::addi(&mut ret, &heap, -5));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_force(&mut self) -> Freed {
        argument!(force_term = self.first_arg);
        var_argument!(frames = self.frames);

        self.generator
            .add_instruction(Instruction::label("handle_force".to_string()));

        self.generator
            .add_instruction(Instruction::comment("Load term body".to_string()));

        constnt!(body = self.first_temp);
        self.generator
            .add_instruction(Instruction::addi(&mut body, &force_term, 1));

        self.generator.add_instruction(Instruction::comment(
            "1 byte for FrameForce allocation".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::addi(&mut frames.clone(), &frames, -1));

        self.generator
            .add_instruction(Instruction::comment("Tag is 3 for FrameForce".to_string()));

        constnt!(tag = self.second_temp);

        self.generator.add_instruction(Instruction::li(&mut tag, 3));

        self.generator.add_instruction(Instruction::comment(
            "Push FrameForce tag onto stack".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::sb(&tag, 0, &frames));

        self.generator.add_instruction(Instruction::comment(
            "Put term body address into A0".to_string(),
        ));

        constnt_overwrite!(ret = force_term);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &body));

        self.generator
            .add_instruction(Instruction::j("compute".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_error(&mut self) -> Freed {
        self.generator
            .add_instruction(Instruction::label("handle_error".to_string()));

        self.generator
            .add_instruction(Instruction::comment("Load -1 into A0".to_string()));

        constnt!(ret = self.return_reg);
        self.generator
            .add_instruction(Instruction::li(&mut ret, -1));

        self.generator
            .add_instruction(Instruction::j("halt".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_builtin(&mut self) -> Freed {
        argument!(builtin = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);

        self.generator
            .add_instruction(Instruction::label("handle_builtin".to_string()));

        constnt!(builtin_func_index = self.first_temp);

        self.generator
            .add_instruction(Instruction::lbu(&mut builtin_func_index, 1, &builtin));

        // 1 byte for value tag, 1 byte for func index, 1 byte for forces, 4 bytes for args length 0
        self.generator.add_instruction(Instruction::comment(
            "7 bytes for VBuiltin allocation".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 7));

        constnt!(value_tag = self.second_temp);
        self.generator
            .add_instruction(Instruction::li(&mut value_tag, 3));

        self.generator
            .add_instruction(Instruction::sb(&value_tag, -7, &heap));

        self.generator
            .add_instruction(Instruction::sb(&builtin_func_index, -6, &heap));

        var!(force_lookup = self.third_temp);
        self.generator.add_instruction(Instruction::la(
            &mut force_lookup,
            "force_counts".to_string(),
        ));

        self.generator.add_instruction(Instruction::add(
            &mut force_lookup.clone(),
            &force_lookup,
            &builtin_func_index,
        ));

        constnt!(forces = self.fourth_temp);
        self.generator
            .add_instruction(Instruction::lbu(&mut forces, 0, &force_lookup));

        self.generator
            .add_instruction(Instruction::sb(&forces, -5, &heap));

        self.generator
            .add_instruction(Instruction::sw(&zero, -4, &heap));

        constnt_overwrite!(ret = builtin);
        self.generator
            .add_instruction(Instruction::addi(&mut ret, &heap, -7));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_constr(&mut self) -> Freed {
        argument!(constr = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(env = self.env);
        var_argument!(frames = self.frames);
        argument!(zero = self.zero);

        self.generator
            .add_instruction(Instruction::label("handle_constr".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "Load the tag of the constr into T0".to_string(),
        ));

        constnt!(constr_tag = self.first_temp);

        self.generator
            .add_instruction(Instruction::lw(&mut constr_tag, 1, &constr));

        self.generator.add_instruction(Instruction::comment(
            "Load the length of the constr fields into T1".to_string(),
        ));

        constnt!(constr_len = self.second_temp);

        self.generator
            .add_instruction(Instruction::lw(&mut constr_len, 5, &constr));

        self.generator.add_instruction(Instruction::beq(
            &constr_len,
            &zero,
            "handle_constr_empty".to_string(),
        ));

        create_block!(self, {
            // In branching terminating scopes we clone the terms that are overwritten
            let mut constr_len = constr_len.clone();
            let mut constr = constr.clone();
            self.generator.add_instruction(Instruction::comment(
                "-- Fields is not empty --".to_string(),
            ));

            constnt!(constr_len_popped = self.third_temp);
            self.generator.add_instruction(Instruction::addi(
                &mut constr_len_popped,
                &constr_len,
                -1,
            ));

            self.generator.add_instruction(Instruction::comment(
                "Minimum size for FrameConstr is 17 bytes".to_string(),
            ));

            constnt!(elements_byte_size = self.fourth_temp);
            self.generator.add_instruction(Instruction::slli(
                &mut elements_byte_size,
                &constr_len_popped,
                2,
            ));

            constnt_overwrite!(total_byte_size = constr_len);
            self.generator.add_instruction(Instruction::addi(
                &mut total_byte_size,
                &elements_byte_size,
                17,
            ));

            self.generator.add_instruction(Instruction::comment(
                "Allocate 17 + 4 * constr fields length + 4 * values length".to_string(),
            ));
            self.generator.add_instruction(Instruction::comment(
                "where values length is 0 and fields length is fields in constr - 1".to_string(),
            ));
            self.generator.add_instruction(Instruction::comment(
                "frame tag 1 byte + constr tag 4 bytes + environment 4 bytes".to_string(),
            ));
            self.generator.add_instruction(Instruction::comment(
                "+ fields length 4 bytes + 4 * fields length in bytes".to_string(),
            ));
            self.generator.add_instruction(Instruction::comment(
                "+ values length 4 bytes + 4 * values length in bytes".to_string(),
            ));
            self.generator.add_instruction(Instruction::comment(
                "Remember this is subtracting the above value".to_string(),
            ));

            self.generator.add_instruction(Instruction::sub(
                &mut frames.clone(),
                &frames,
                &total_byte_size,
            ));

            var!(frame_builder = self.fifth_temp);
            self.generator
                .add_instruction(Instruction::mv(&mut frame_builder, &frames));

            // Overwriting elements_byte_size
            constnt_overwrite!(constr_frame_tag = elements_byte_size);
            self.generator
                .add_instruction(Instruction::li(&mut constr_frame_tag, 4));

            self.generator
                .add_instruction(Instruction::comment("store frame tag".to_string()));

            self.generator
                .add_instruction(Instruction::sb(&constr_frame_tag, 0, &frame_builder));

            self.generator
                .add_instruction(Instruction::comment("move up 1 byte".to_string()));

            self.generator.add_instruction(Instruction::addi(
                &mut frame_builder.clone(),
                &frame_builder,
                1,
            ));

            self.generator
                .add_instruction(Instruction::comment(" store constr tag".to_string()));

            self.generator
                .add_instruction(Instruction::sw(&constr_tag, 0, &frame_builder));

            self.generator
                .add_instruction(Instruction::comment("move up 4 bytes".to_string()));

            self.generator.add_instruction(Instruction::addi(
                &mut frame_builder.clone(),
                &frame_builder,
                4,
            ));

            self.generator
                .add_instruction(Instruction::comment("store environment".to_string()));

            self.generator
                .add_instruction(Instruction::sw(&env, 0, &frame_builder));

            self.generator
                .add_instruction(Instruction::comment("move up 4 bytes".to_string()));

            self.generator.add_instruction(Instruction::addi(
                &mut frame_builder.clone(),
                &frame_builder,
                4,
            ));

            self.generator
                .add_instruction(Instruction::comment("store fields length -1".to_string()));

            self.generator
                .add_instruction(Instruction::sw(&constr_len_popped, 0, &frame_builder));

            self.generator
                .add_instruction(Instruction::comment("move up 4 bytes".to_string()));

            self.generator.add_instruction(Instruction::addi(
                &mut frame_builder.clone(),
                &frame_builder,
                4,
            ));

            self.generator
                .add_instruction(Instruction::comment("Load first field to A4".to_string()));

            constnt!(first_field = self.fifth_arg);

            self.generator
                .add_instruction(Instruction::lw(&mut first_field, 9, &constr));

            self.generator.add_instruction(Instruction::comment(
                "move fields length - 1 to A2".to_string(),
            ));

            constnt!(size = self.third_arg);

            self.generator
                .add_instruction(Instruction::mv(&mut size, &constr_len_popped));

            self.generator.add_instruction(Instruction::comment(
                "move current stack pointer to A1".to_string(),
            ));

            constnt!(frames_arg = self.second_arg);

            self.generator
                .add_instruction(Instruction::mv(&mut frames_arg, &frame_builder));

            self.generator.add_instruction(Instruction::comment(
                "move A0 pointer to second element in fields (regardless if there or not)"
                    .to_string(),
            ));

            var!(constr_temp = self.sixth_temp);

            self.generator
                .add_instruction(Instruction::mv(&mut constr_temp, &constr));

            constnt_overwrite!(second_field = constr);

            self.generator
                .add_instruction(Instruction::addi(&mut second_field, &constr_temp, 13));

            // Takes in A0 - elements pointer, A1 - destination pointer, A2 - length
            // A3 - return address

            constnt!(callback = self.fourth_arg);

            self.generator
                .add_instruction(Instruction::jal(&mut callback, "clone_list".to_string()));

            // frames_arg points to last 4 bytes in allocated frame
            self.generator
                .add_instruction(Instruction::mv(&mut frame_builder, &frames_arg));

            self.generator.add_instruction(Instruction::comment(
                "Store 0 for values length".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::sw(&zero, 0, &frame_builder));

            self.generator.add_instruction(Instruction::comment(
                "Mv A4 (pointer to first field term) to A0".to_string(),
            ));
            constnt_overwrite!(ret = second_field);
            self.generator
                .add_instruction(Instruction::mv(&mut ret, &first_field));

            self.generator
                .add_instruction(Instruction::j("compute".to_string()));
        });

        create_block!(self, {
            self.generator
                .add_instruction(Instruction::comment("-- Empty fields --".to_string()));

            self.generator
                .add_instruction(Instruction::label("handle_constr_empty".to_string()));

            self.generator.add_instruction(Instruction::comment(
                "9 bytes allocated on heap".to_string(),
            ));
            self.generator.add_instruction(Instruction::comment(
                "1 byte value tag + 4 bytes constr tag + 4 bytes constr fields length which is 0"
                    .to_string(),
            ));
            self.generator
                .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 9));

            constnt!(vconstr_tag = self.third_temp);

            self.generator
                .add_instruction(Instruction::li(&mut vconstr_tag, 4));

            self.generator
                .add_instruction(Instruction::sb(&vconstr_tag, -9, &heap));

            self.generator
                .add_instruction(Instruction::sw(&constr_tag, -8, &heap));

            self.generator
                .add_instruction(Instruction::sw(&constr_len, -4, &heap));

            constnt_overwrite!(ret = constr);
            self.generator
                .add_instruction(Instruction::addi(&mut ret, &heap, -9));

            self.generator
                .add_instruction(Instruction::j("return".to_string()));
        });

        self.register_map.free_all()
    }

    pub fn handle_case(&mut self) -> Freed {
        argument!(case = self.first_arg);
        var_argument!(frames = self.frames);
        argument!(env = self.env);
        self.generator
            .add_instruction(Instruction::label("handle_case".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "Load the term pointer of the constr of case into A4".to_string(),
        ));

        // Store constr to compute on in A4
        constnt!(constr = self.fifth_arg);
        self.generator
            .add_instruction(Instruction::lw(&mut constr, 1, &case));

        self.generator.add_instruction(Instruction::comment(
            "Load the length of the constr fields into T1".to_string(),
        ));

        constnt!(size = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut size, 5, &constr));

        self.generator.add_instruction(Instruction::comment(
            "Minimum size for FrameCase is 9 bytes".to_string(),
        ));

        constnt!(elements_byte_size = self.second_temp);
        self.generator
            .add_instruction(Instruction::slli(&mut elements_byte_size, &size, 2));

        constnt!(total_byte_size = self.third_temp);
        self.generator.add_instruction(Instruction::addi(
            &mut total_byte_size,
            &elements_byte_size,
            9,
        ));

        self.generator.add_instruction(Instruction::comment(
            "Allocate 9 + 4 * cases length".to_string(),
        ));

        self.generator.add_instruction(Instruction::comment(
            "frame tag 1 byte + environment 4 bytes".to_string(),
        ));
        self.generator.add_instruction(Instruction::comment(
            "+ cases length 4 bytes + 4 * cases length in bytes".to_string(),
        ));

        self.generator.add_instruction(Instruction::comment(
            "Remember this is subtracting the above value".to_string(),
        ));

        self.generator.add_instruction(Instruction::sub(
            &mut frames.clone(),
            &frames,
            &total_byte_size,
        ));

        var!(frames_builder = self.fourth_temp);
        self.generator
            .add_instruction(Instruction::mv(&mut frames_builder, &frames));

        // FrameCase tag
        constnt!(frame_case_tag = self.fifth_temp);
        self.generator
            .add_instruction(Instruction::li(&mut frame_case_tag, 5));

        self.generator
            .add_instruction(Instruction::sb(&frame_case_tag, 0, &frames_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut frames_builder.clone(),
            &frames_builder,
            1,
        ));

        self.generator
            .add_instruction(Instruction::sw(&env, 0, &frames_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut frames_builder.clone(),
            &frames_builder,
            4,
        ));

        self.generator
            .add_instruction(Instruction::sw(&size, 0, &frames_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut frames_builder.clone(),
            &frames_builder,
            4,
        ));
        // A0 pointer to terms array
        // A1 is new stack pointer
        // A2 is length of terms array
        // A3 holds return address

        constnt!(list_size = self.third_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut list_size, &size));

        constnt!(frames_arg = self.second_arg);

        self.generator
            .add_instruction(Instruction::mv(&mut frames_arg, &frames_builder));

        constnt!(case_temp = self.sixth_temp);

        self.generator
            .add_instruction(Instruction::mv(&mut case_temp, &case));

        constnt_overwrite!(branches = case);
        self.generator
            .add_instruction(Instruction::addi(&mut branches, &case_temp, 9));

        constnt!(callback = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::jal(&mut callback, "clone_list".to_string()));

        constnt_overwrite!(ret = branches);
        // Move term pointer into A0
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &constr));

        self.generator
            .add_instruction(Instruction::j("compute".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_frame_await_arg(&mut self) -> Freed {
        argument!(arg = self.first_arg);
        var_argument!(frames = self.frames);
        self.generator
            .add_instruction(Instruction::label("handle_frame_await_arg".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "load function value pointer from stack".to_string(),
        ));

        constnt!(function = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut function, 1, &frames));

        self.generator.add_instruction(Instruction::comment(
            "reset stack Kontinuation pointer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::addi(&mut frames.clone(), &frames, 5));

        constnt!(second_eval_arg = self.second_arg);

        self.generator
            .add_instruction(Instruction::mv(&mut second_eval_arg, &arg));

        constnt_overwrite!(ret = arg);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &function));

        self.generator
            .add_instruction(Instruction::j("apply_evaluate".to_string()));

        self.register_map.free_all()
    }

    // Takes in a0 and passes it to apply_evaluate
    pub fn handle_frame_await_fun_term(&mut self) -> Freed {
        argument!(function = self.first_arg);
        var_argument!(frames = self.frames);

        self.generator.add_instruction(Instruction::label(
            "handle_frame_await_fun_term".to_string(),
        ));

        self.generator.add_instruction(Instruction::comment(
            "load argument pointer from stack".to_string(),
        ));

        constnt!(argument = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut argument, 1, &frames));

        self.generator.add_instruction(Instruction::comment(
            "load environment from stack".to_string(),
        ));

        constnt!(environment = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut environment, 5, &frames));

        self.generator.add_instruction(Instruction::comment(
            "reset stack Kontinuation pointer".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::addi(&mut frames.clone(), &frames, 9));

        self.generator.add_instruction(Instruction::comment(
            "5 bytes for FrameAwaitArg allocation".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::addi(&mut frames.clone(), &frames, -5));

        self.generator.add_instruction(Instruction::comment(
            "Tag is 0 for FrameAwaitArg".to_string(),
        ));

        constnt!(frame_tag = self.third_temp);
        self.generator
            .add_instruction(Instruction::li(&mut frame_tag, 0));

        self.generator
            .add_instruction(Instruction::comment("Push tag onto stack".to_string()));

        self.generator
            .add_instruction(Instruction::sb(&frame_tag, 0, &frames));

        self.generator.add_instruction(Instruction::comment(
            "Push function value pointer onto stack".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::sw(&function, 1, &frames));

        self.generator.add_instruction(Instruction::comment(
            "Set new environment pointer".to_string(),
        ));
        constnt!(env = self.env);
        self.generator
            .add_instruction(Instruction::mv(&mut env, &environment));

        constnt_overwrite!(ret = function);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &argument));

        self.generator
            .add_instruction(Instruction::j("compute".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_frame_await_fun_value(&mut self) -> Freed {
        argument!(function = self.first_arg);
        var_argument!(frames = self.frames);

        self.generator.add_instruction(Instruction::label(
            "handle_frame_await_fun_value".to_string(),
        ));

        self.generator.add_instruction(Instruction::comment(
            "load function value pointer from stack".to_string(),
        ));

        constnt!(arg = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut arg, 1, &frames));

        self.generator.add_instruction(Instruction::comment(
            "reset stack Kontinuation pointer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::addi(&mut frames.clone(), &frames, 5));

        constnt!(second_eval_arg = self.second_arg);

        self.generator
            .add_instruction(Instruction::mv(&mut second_eval_arg, &arg));

        self.generator
            .add_instruction(Instruction::j("apply_evaluate".to_string()));

        self.register_map.free_all()
    }

    // Takes in a0 and passes it to force_evaluate
    pub fn handle_frame_force(&mut self) -> Freed {
        var_argument!(frames = self.frames);
        self.generator
            .add_instruction(Instruction::label("handle_frame_force".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "reset stack Kontinuation pointer".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::addi(&mut frames.clone(), &frames, 1));

        self.generator
            .add_instruction(Instruction::j("force_evaluate".to_string()));

        self.register_map.free_all()
    }

    pub fn handle_frame_constr(&mut self) -> Freed {
        argument!(computed_value = self.first_arg);
        var_argument!(frames = self.frames);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::label("handle_frame_constr".to_string()));

        constnt!(constr_tag = self.first_temp);
        // Load the constructor tag from the frame
        self.generator
            .add_instruction(Instruction::lw(&mut constr_tag, 1, &frames));

        constnt!(environment = self.second_temp);
        // Load the environment from the frame
        self.generator
            .add_instruction(Instruction::lw(&mut environment, 5, &frames));

        constnt!(fields_len = self.third_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut fields_len, 9, &frames));

        // bytes offset from frame to values len based on fields length
        var!(bytes_offset = self.fourth_temp);
        self.generator
            .add_instruction(Instruction::mv(&mut bytes_offset, &fields_len));

        self.generator.add_instruction(Instruction::slli(
            &mut bytes_offset.clone(),
            &bytes_offset,
            2,
        ));

        self.generator.add_instruction(Instruction::addi(
            &mut bytes_offset.clone(),
            &bytes_offset,
            13,
        ));

        self.generator.add_instruction(Instruction::add(
            &mut bytes_offset.clone(),
            &bytes_offset,
            &frames,
        ));

        var!(values_len = self.fifth_temp);

        self.generator
            .add_instruction(Instruction::lw(&mut values_len, 0, &bytes_offset));

        self.generator
            .add_instruction(Instruction::addi(&mut values_len.clone(), &values_len, 1));

        self.generator
            .add_instruction(Instruction::sw(&values_len, 0, &bytes_offset));

        self.generator.add_instruction(Instruction::beq(
            &fields_len,
            &zero,
            "handle_frame_constr_empty".to_string(),
        ));

        create_block!(self, {
            let mut computed_value = computed_value.clone();
            // Mutate fields length to be 1 less since we are popping from the front
            constnt!(current_field_len = self.sixth_temp);

            self.generator.add_instruction(Instruction::addi(
                &mut current_field_len,
                &fields_len,
                -1,
            ));

            self.generator
                .add_instruction(Instruction::sw(&current_field_len, 9, &frames));

            // Field term to compute on in A5
            constnt!(first_field = self.sixth_arg);
            self.generator
                .add_instruction(Instruction::lw(&mut first_field, 13, &frames));

            // Value to push onto the frame in A4
            constnt!(new_value = self.fifth_arg);
            self.generator
                .add_instruction(Instruction::mv(&mut new_value, &computed_value));

            var!(length_arg = self.third_arg);
            self.generator
                .add_instruction(Instruction::mv(&mut length_arg, &current_field_len));

            self.generator.add_instruction(Instruction::add(
                &mut length_arg.clone(),
                &length_arg,
                &values_len,
            ));

            constnt!(new_list = self.second_arg);
            self.generator
                .add_instruction(Instruction::addi(&mut new_list, &frames, 13));

            constnt_overwrite!(src_list = computed_value);
            self.generator
                .add_instruction(Instruction::addi(&mut src_list, &frames, 17));

            constnt!(callback = self.fourth_arg);
            self.generator
                .add_instruction(Instruction::jal(&mut callback, "clone_list".to_string()));

            self.generator
                .add_instruction(Instruction::sw(&new_value, 0, &new_list));

            constnt!(env = self.env);
            self.generator
                .add_instruction(Instruction::mv(&mut env, &environment));

            constnt_overwrite!(ret = src_list);
            self.generator
                .add_instruction(Instruction::mv(&mut ret, &first_field));

            self.generator
                .add_instruction(Instruction::j("compute".to_string()));
        });

        create_block!(self, {
            self.generator
                .add_instruction(Instruction::label("handle_frame_constr_empty".to_string()));

            // fields length is 0 and not needed
            // allocation amount in bytes is 4 * value length + 9
            // 1 for frame tag, 4 for constr tag, and 4 for value length
            // Overwrite fields_len
            var_overwrite!(allocation_amount = fields_len);
            self.generator
                .add_instruction(Instruction::mv(&mut allocation_amount, &values_len));

            self.generator.add_instruction(Instruction::slli(
                &mut allocation_amount.clone(),
                &allocation_amount,
                2,
            ));

            self.generator.add_instruction(Instruction::addi(
                &mut allocation_amount.clone(),
                &allocation_amount,
                9,
            ));

            // Overwrite environment
            // since it is not used in return compute
            var_overwrite!(allocator_space = environment);
            self.generator
                .add_instruction(Instruction::mv(&mut allocator_space, &heap));

            // Allocate VConstr on the heap
            // 9 + 4 * value length
            self.generator.add_instruction(Instruction::add(
                &mut heap.clone(),
                &heap,
                &allocation_amount,
            ));

            constnt!(value_tag = self.sixth_temp);

            self.generator
                .add_instruction(Instruction::li(&mut value_tag, 4));

            self.generator
                .add_instruction(Instruction::sb(&value_tag, 0, &allocator_space));

            self.generator
                .add_instruction(Instruction::sw(&constr_tag, 1, &allocator_space));

            self.generator
                .add_instruction(Instruction::sw(&values_len, 5, &allocator_space));

            // Value to return compute in A5
            constnt!(return_value = self.sixth_arg);
            self.generator
                .add_instruction(Instruction::mv(&mut return_value, &allocator_space));

            self.generator.add_instruction(Instruction::addi(
                &mut allocator_space.clone(),
                &allocator_space,
                9,
            ));

            self.generator.add_instruction(Instruction::addi(
                &mut values_len.clone(),
                &values_len,
                -1,
            ));

            var_overwrite!(list_byte_length = constr_tag);
            self.generator
                .add_instruction(Instruction::mv(&mut list_byte_length, &values_len));

            self.generator.add_instruction(Instruction::slli(
                &mut list_byte_length.clone(),
                &list_byte_length,
                2,
            ));

            constnt_overwrite!(tail_list = value_tag);
            self.generator.add_instruction(Instruction::add(
                &mut tail_list,
                &list_byte_length,
                &bytes_offset,
            ));

            constnt!(next_frame = self.seventh_arg);
            self.generator
                .add_instruction(Instruction::addi(&mut next_frame, &tail_list, 4));

            constnt!(new_value = self.fifth_arg);
            self.generator
                .add_instruction(Instruction::mv(&mut new_value, &computed_value));

            constnt_overwrite!(list_to_reverse = computed_value);

            self.generator
                .add_instruction(Instruction::mv(&mut list_to_reverse, &tail_list));

            var!(dest = self.second_arg);
            self.generator
                .add_instruction(Instruction::mv(&mut dest, &allocator_space));

            self.generator
                .add_instruction(Instruction::sw(&new_value, 0, &dest));

            self.generator
                .add_instruction(Instruction::addi(&mut dest.clone(), &dest, 4));

            constnt!(size = self.third_arg);
            self.generator
                .add_instruction(Instruction::mv(&mut size, &values_len));

            constnt!(callback = self.fourth_arg);
            self.generator.add_instruction(Instruction::jal(
                &mut callback,
                "reverse_clone_list".to_string(),
            ));

            constnt_overwrite!(ret = list_to_reverse);
            self.generator
                .add_instruction(Instruction::mv(&mut ret, &return_value));

            // Reset frame stack by moving to next frame
            self.generator
                .add_instruction(Instruction::mv(&mut frames, &next_frame));

            self.generator
                .add_instruction(Instruction::j("return".to_string()));
        });

        self.register_map.free_all()
    }

    pub fn handle_frame_case(&mut self) -> Freed {
        argument!(first_arg = self.first_arg);
        var_argument!(frames = self.frames);
        argument!(zero = self.zero);

        self.generator
            .add_instruction(Instruction::label("handle_frame_case".to_string()));

        constnt!(constr = self.first_temp);
        self.generator
            .add_instruction(Instruction::mv(&mut constr, &first_arg));

        constnt!(constr_term_tag = self.second_temp);
        self.generator
            .add_instruction(Instruction::lbu(&mut constr_term_tag, 0, &constr));

        constnt!(expected_tag = self.third_temp);

        self.generator
            .add_instruction(Instruction::li(&mut expected_tag, 4));

        self.generator.add_instruction(Instruction::bne(
            &constr_term_tag,
            &expected_tag,
            "handle_frame_case_error".to_string(),
        ));

        {
            var!(constr_tag = self.fourth_temp);
            self.generator
                .add_instruction(Instruction::lw(&mut constr_tag, 1, &constr));

            // Overwrite expected_tag
            var_overwrite!(branches_len = expected_tag);
            self.generator
                .add_instruction(Instruction::lw(&mut branches_len, 5, &frames));

            self.generator.add_instruction(Instruction::bge(
                &constr_tag,
                &branches_len,
                "handle_frame_case_error".to_string(),
            ));

            // set env
            constnt!(env = self.env);
            self.generator
                .add_instruction(Instruction::lw(&mut env, 1, &frames));

            let mut offset_to_branch = constr_tag;
            self.generator.add_instruction(Instruction::slli(
                &mut offset_to_branch.clone(),
                &offset_to_branch,
                2,
            ));

            self.generator.add_instruction(Instruction::addi(
                &mut offset_to_branch.clone(),
                &offset_to_branch,
                9,
            ));

            self.generator.add_instruction(Instruction::add(
                &mut offset_to_branch.clone(),
                &offset_to_branch,
                &frames,
            ));

            // Put branch term in return register
            constnt_overwrite!(ret = first_arg);
            self.generator
                .add_instruction(Instruction::lw(&mut ret, 0, &offset_to_branch));

            let mut claim_stack_item = branches_len;

            self.generator.add_instruction(Instruction::slli(
                &mut claim_stack_item.clone(),
                &claim_stack_item,
                2,
            ));

            self.generator.add_instruction(Instruction::addi(
                &mut claim_stack_item.clone(),
                &claim_stack_item,
                9,
            ));

            // reset frame pointer
            self.generator.add_instruction(Instruction::add(
                &mut frames.clone(),
                &frames,
                &claim_stack_item,
            ));

            // Overwrite constr_term_tag
            constnt_overwrite!(constr_fields_len = constr_term_tag);
            self.generator
                .add_instruction(Instruction::lw(&mut constr_fields_len, 5, &constr));

            var!(current_index = self.fifth_temp);
            self.generator
                .add_instruction(Instruction::mv(&mut current_index, &zero));

            var!(current_offset = self.sixth_temp);
            // 9 for constant offset
            // 1 for frame tag + 4 for constr tag + 4 for constr fields len
            self.generator
                .add_instruction(Instruction::li(&mut current_offset, 9));

            self.generator.add_instruction(Instruction::add(
                &mut current_offset.clone(),
                &current_offset,
                &constr,
            ));

            {
                self.generator
                    .add_instruction(Instruction::label("transfer_fields_as_args".to_string()));

                self.generator.add_instruction(Instruction::beq(
                    &current_index,
                    &constr_fields_len,
                    "compute".to_string(),
                ));

                // Allocate for FrameAwaitFunValue to stack
                // 5 bytes, 1 for frame tag, 4 for argument value pointer
                self.generator
                    .add_instruction(Instruction::addi(&mut frames.clone(), &frames, -5));

                constnt_overwrite!(frame_tag = claim_stack_item);
                self.generator
                    .add_instruction(Instruction::li(&mut frame_tag, 2));

                self.generator
                    .add_instruction(Instruction::sb(&frame_tag, 0, &frames));

                constnt_overwrite!(arg = offset_to_branch);
                self.generator
                    .add_instruction(Instruction::lw(&mut arg, 0, &current_offset));

                self.generator
                    .add_instruction(Instruction::sw(&arg, 1, &frames));

                self.generator.add_instruction(Instruction::addi(
                    &mut current_offset.clone(),
                    &current_offset,
                    4,
                ));

                self.generator.add_instruction(Instruction::addi(
                    &mut current_index.clone(),
                    &current_index,
                    1,
                ));

                self.generator
                    .add_instruction(Instruction::j("transfer_fields_as_args".to_string()));
            }
        };

        {
            self.generator
                .add_instruction(Instruction::label("handle_frame_case_error".to_string()));

            self.generator
                .add_instruction(Instruction::j("handle_error".to_string()));
        };

        self.register_map.free_all()
    }

    pub fn handle_no_frame(&mut self) -> Freed {
        argument!(value = self.first_arg);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::label("handle_no_frame".to_string()));

        constnt!(value_tag = self.first_temp);
        self.generator
            .add_instruction(Instruction::lbu(&mut value_tag, 0, &value));

        // We should only return constants. That greatly simplifies how we return a value to the user
        self.generator.add_instruction(Instruction::bne(
            &value_tag,
            &zero,
            "handle_error".to_string(),
        ));

        constnt!(ret_temp = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut ret_temp, 1, &value));

        constnt_overwrite!(ret = value);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &ret_temp));

        self.generator
            .add_instruction(Instruction::j("halt".to_string()));

        self.register_map.free_all()
    }

    pub fn halt(&mut self) -> Freed {
        self.generator
            .add_instruction(Instruction::label("halt".to_string()));

        // exit code
        constnt!(exit_code = self.eighth_arg);
        self.generator
            .add_instruction(Instruction::li(&mut exit_code, 93));

        self.generator.add_instruction(Instruction::ecall());

        self.register_map.free_all()
    }

    pub fn force_evaluate(&mut self) -> Freed {
        argument!(function = self.first_arg);
        argument!(zero = self.zero);
        var_argument!(heap = self.heap);

        self.generator
            .add_instruction(Instruction::label("force_evaluate".to_string()));

        self.generator.add_instruction(Instruction::comment(
            "Value address should be in A0".to_string(),
        ));
        self.generator
            .add_instruction(Instruction::comment("Load value tag".to_string()));

        constnt!(tag = self.first_temp);
        self.generator
            .add_instruction(Instruction::lbu(&mut tag, 0, &function));

        self.generator
            .add_instruction(Instruction::comment("Delay".to_string()));

        constnt!(delay_value_tag = self.second_temp);
        self.generator
            .add_instruction(Instruction::li(&mut delay_value_tag, 1));

        self.generator.add_instruction(Instruction::bne(
            &tag,
            &delay_value_tag,
            "force_evaluate_builtin".to_string(),
        ));

        {
            let mut delay_value_tag = delay_value_tag.clone();
            let mut tag = tag.clone();
            let mut function = function.clone();

            self.generator.add_instruction(Instruction::comment(
                "load body pointer from a0 which is Value".to_string(),
            ));

            constnt_overwrite!(body = tag);
            self.generator
                .add_instruction(Instruction::lw(&mut body, 1, &function));

            self.generator.add_instruction(Instruction::comment(
                "load environment from a0 which is Value".to_string(),
            ));

            constnt_overwrite!(environment = delay_value_tag);
            self.generator
                .add_instruction(Instruction::lw(&mut environment, 5, &function));

            constnt!(env = self.env);
            self.generator
                .add_instruction(Instruction::mv(&mut env, &environment));

            constnt_overwrite!(ret = function);
            self.generator
                .add_instruction(Instruction::mv(&mut ret, &body));

            self.generator
                .add_instruction(Instruction::j("compute".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::label("force_evaluate_builtin".to_string()));

            // Overwrite delay_value_tag
            constnt_overwrite!(builtin_value_tag = delay_value_tag);
            self.generator
                .add_instruction(Instruction::li(&mut builtin_value_tag, 3));

            self.generator.add_instruction(Instruction::bne(
                &tag,
                &builtin_value_tag,
                "force_evaluate_error".to_string(),
            ));

            {
                // Overwrite builtin_value_tag
                var_overwrite!(force_count = builtin_value_tag);
                self.generator
                    .add_instruction(Instruction::lbu(&mut force_count, 2, &function));

                self.generator.add_instruction(Instruction::beq(
                    &zero,
                    &force_count,
                    "force_evaluate_error".to_string(),
                ));

                self.generator.add_instruction(Instruction::addi(
                    &mut force_count.clone(),
                    &force_count,
                    -1,
                ));

                // Create clone of current value with number of forces changed
                self.generator
                    .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 7));

                self.generator
                    .add_instruction(Instruction::sb(&tag, -7, &heap));

                constnt!(builtin_func_index = self.third_temp);
                self.generator.add_instruction(Instruction::lbu(
                    &mut builtin_func_index,
                    1,
                    &function,
                ));

                self.generator
                    .add_instruction(Instruction::sb(&builtin_func_index, -6, &heap));

                self.generator
                    .add_instruction(Instruction::sb(&force_count, -5, &heap));

                // 0 Arguments applied so arg fields is just arg length (zero)
                self.generator
                    .add_instruction(Instruction::sw(&zero, -4, &heap));

                // Store new value in ret
                constnt_overwrite!(ret = function);
                self.generator
                    .add_instruction(Instruction::addi(&mut ret, &heap, -7));

                // If still have forces to apply then return
                self.generator.add_instruction(Instruction::bne(
                    &zero,
                    &force_count,
                    "return".to_string(),
                ));

                constnt_overwrite!(arguments_length = tag);
                self.generator
                    .add_instruction(Instruction::lw(&mut arguments_length, 3, &ret));

                var_overwrite!(arity_lookup = force_count);
                self.generator
                    .add_instruction(Instruction::la(&mut arity_lookup, "arities".to_string()));

                self.generator.add_instruction(Instruction::add(
                    &mut arity_lookup.clone(),
                    &arity_lookup,
                    &builtin_func_index,
                ));

                constnt!(arity = self.fourth_temp);
                self.generator
                    .add_instruction(Instruction::lbu(&mut arity, 0, &arity_lookup));

                // If all arguments not applied then return.
                self.generator.add_instruction(Instruction::bne(
                    &arity,
                    &arguments_length,
                    "return".to_string(),
                ));

                constnt!(function_index = self.second_arg);
                self.generator
                    .add_instruction(Instruction::mv(&mut function_index, &builtin_func_index));

                self.generator
                    .add_instruction(Instruction::j("eval_builtin_app".to_string()));
            }

            {
                self.generator
                    .add_instruction(Instruction::label("force_evaluate_error".to_string()));

                self.generator
                    .add_instruction(Instruction::j("handle_error".to_string()));
            }
        }

        self.register_map.free_all()
    }

    pub fn apply_evaluate(&mut self) -> Freed {
        argument!(function = self.first_arg);
        argument!(argument = self.second_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);

        self.generator
            .add_instruction(Instruction::label("apply_evaluate".to_string()));

        //Value address should be in A0
        self.generator
            .add_instruction(Instruction::comment("Load function value tag".to_string()));

        constnt!(function_tag = self.first_temp);
        self.generator
            .add_instruction(Instruction::lbu(&mut function_tag, 0, &function));

        self.generator
            .add_instruction(Instruction::comment("Lambda".to_string()));

        constnt!(lambda_value_tag = self.second_temp);

        self.generator
            .add_instruction(Instruction::li(&mut lambda_value_tag, 2));

        self.generator.add_instruction(Instruction::bne(
            &function_tag,
            &lambda_value_tag,
            "apply_evaluate_builtin".to_string(),
        ));

        {
            let mut lambda_value_tag = lambda_value_tag.clone();
            let mut function_tag = function_tag.clone();
            let mut function = function.clone();

            self.generator.add_instruction(Instruction::comment(
                "load body pointer from a0 which is function Value".to_string(),
            ));

            constnt_overwrite!(body = function_tag);
            self.generator
                .add_instruction(Instruction::lw(&mut body, 1, &function));

            self.generator.add_instruction(Instruction::comment(
                "load environment from a0 which is function Value".to_string(),
            ));

            constnt_overwrite!(environment = lambda_value_tag);

            self.generator
                .add_instruction(Instruction::lw(&mut environment, 5, &function));

            var!(env = self.env);
            self.generator
                .add_instruction(Instruction::mv(&mut env, &environment));

            self.generator.add_instruction(Instruction::comment(
                "Important this is the only place we modify environment".to_string(),
            ));

            self.generator.add_instruction(Instruction::comment(
                "Allocate 8 bytes on the heap".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 8));

            self.generator.add_instruction(Instruction::comment(
                "pointer to argument value".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::sw(&argument, -8, &heap));

            self.generator.add_instruction(Instruction::comment(
                "pointer to previous environment".to_string(),
            ));

            self.generator
                .add_instruction(Instruction::sw(&env, -4, &heap));

            self.generator.add_instruction(Instruction::comment(
                "Save allocated heap location in environment pointer".to_string(),
            ));
            self.generator
                .add_instruction(Instruction::addi(&mut env, &heap, -8));

            constnt_overwrite!(ret = function);
            self.generator
                .add_instruction(Instruction::mv(&mut ret, &body));

            self.generator
                .add_instruction(Instruction::j("compute".to_string()));
        };

        {
            self.generator
                .add_instruction(Instruction::label("apply_evaluate_builtin".to_string()));

            // Overwrite lambda_value_tag
            constnt_overwrite!(builtin_value_tag = lambda_value_tag);
            self.generator
                .add_instruction(Instruction::li(&mut builtin_value_tag, 3));

            self.generator.add_instruction(Instruction::bne(
                &function_tag,
                &builtin_value_tag,
                "apply_evaluate_error".to_string(),
            ));

            {
                constnt_overwrite!(force_count = builtin_value_tag);
                self.generator
                    .add_instruction(Instruction::lbu(&mut force_count, 2, &function));

                self.generator.add_instruction(Instruction::bne(
                    &zero,
                    &force_count,
                    "apply_evaluate_error".to_string(),
                ));

                constnt!(builtin_func_index = self.third_temp);
                self.generator.add_instruction(Instruction::lbu(
                    &mut builtin_func_index,
                    1,
                    &function,
                ));

                constnt!(arguments_length = self.fourth_temp);
                self.generator.add_instruction(Instruction::lw(
                    &mut arguments_length,
                    3,
                    &function,
                ));

                var!(arity_lookup = self.fifth_temp);
                self.generator
                    .add_instruction(Instruction::la(&mut arity_lookup, "arities".to_string()));

                self.generator.add_instruction(Instruction::add(
                    &mut arity_lookup.clone(),
                    &arity_lookup,
                    &builtin_func_index,
                ));

                constnt!(arity = self.sixth_temp);
                self.generator
                    .add_instruction(Instruction::lbu(&mut arity, 0, &arity_lookup));

                self.generator.add_instruction(Instruction::beq(
                    &arity,
                    &arguments_length,
                    "apply_evaluate_error".to_string(),
                ));

                constnt_overwrite!(new_args_length = force_count);
                self.generator.add_instruction(Instruction::addi(
                    &mut new_args_length,
                    &arguments_length,
                    1,
                ));

                var_overwrite!(heap_allocation = arity_lookup);
                self.generator.add_instruction(Instruction::slli(
                    &mut heap_allocation,
                    &new_args_length,
                    2,
                ));

                self.generator.add_instruction(Instruction::addi(
                    &mut heap_allocation.clone(),
                    &heap_allocation,
                    7,
                ));

                constnt!(cloned_value = self.seventh_temp);
                self.generator
                    .add_instruction(Instruction::mv(&mut cloned_value, &heap));

                self.generator.add_instruction(Instruction::add(
                    &mut heap.clone(),
                    &heap,
                    &heap_allocation,
                ));

                self.generator
                    .add_instruction(Instruction::sb(&function_tag, 0, &cloned_value));

                self.generator.add_instruction(Instruction::sb(
                    &builtin_func_index,
                    1,
                    &cloned_value,
                ));

                // Forces was checked to be 0 above
                self.generator
                    .add_instruction(Instruction::sb(&zero, 2, &cloned_value));

                // Arguments is now the new_args_length
                self.generator
                    .add_instruction(Instruction::sw(&new_args_length, 3, &cloned_value));

                constnt!(store_new_args_length = self.eighth_arg);
                self.generator.add_instruction(Instruction::mv(
                    &mut store_new_args_length,
                    &new_args_length,
                ));

                constnt!(store_new_arg = self.seventh_arg);
                self.generator
                    .add_instruction(Instruction::mv(&mut store_new_arg, &argument));

                constnt!(store_arity = self.sixth_arg);
                self.generator
                    .add_instruction(Instruction::mv(&mut store_arity, &arity));

                constnt!(new_value = self.fifth_arg);
                self.generator
                    .add_instruction(Instruction::mv(&mut new_value, &cloned_value));

                constnt!(size = self.third_arg);
                self.generator
                    .add_instruction(Instruction::mv(&mut size, &arguments_length));

                constnt_overwrite!(dest_list = argument);
                self.generator
                    .add_instruction(Instruction::addi(&mut dest_list, &cloned_value, 7));

                constnt_overwrite!(src_list_temp = arguments_length);
                self.generator
                    .add_instruction(Instruction::addi(&mut src_list_temp, &function, 7));

                constnt_overwrite!(src_list = function);
                self.generator
                    .add_instruction(Instruction::mv(&mut src_list, &src_list_temp));

                constnt!(callback = self.fourth_arg);
                self.generator
                    .add_instruction(Instruction::jal(&mut callback, "clone_list".to_string()));

                // We can store the new arg value one word before current heap since it's not modified anywhere yet
                self.generator
                    .add_instruction(Instruction::sw(&store_new_arg, -4, &heap));

                constnt_overwrite!(ret = src_list);
                self.generator
                    .add_instruction(Instruction::mv(&mut ret, &new_value));

                // Check arity
                self.generator.add_instruction(Instruction::bne(
                    &store_arity,
                    &store_new_args_length,
                    "return".to_string(),
                ));

                constnt_overwrite!(function_index = dest_list);
                self.generator
                    .add_instruction(Instruction::mv(&mut function_index, &builtin_func_index));

                self.generator
                    .add_instruction(Instruction::j("eval_builtin_app".to_string()));
            };

            {
                self.generator
                    .add_instruction(Instruction::label("apply_evaluate_error".to_string()));

                self.generator
                    .add_instruction(Instruction::j("handle_error".to_string()));
            };
        };

        self.register_map.free_all()
    }

    pub fn lookup(&mut self) -> Freed {
        argument!(index = self.first_arg);
        var_argument!(env = self.env);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::label("lookup".to_string()));

        var!(current_index = self.first_temp);

        self.generator
            .add_instruction(Instruction::mv(&mut current_index, &index));

        self.generator.add_instruction(Instruction::addi(
            &mut current_index.clone(),
            &current_index,
            -1,
        ));

        self.generator.add_instruction(Instruction::beq(
            &current_index,
            &zero,
            "lookup_return".to_string(),
        ));

        {
            let mut index = index.clone();
            self.generator.add_instruction(Instruction::comment(
                "pointer to next environment node".to_string(),
            ));

            constnt!(current_env = self.second_temp);
            self.generator
                .add_instruction(Instruction::lw(&mut current_env, 4, &env));

            self.generator
                .add_instruction(Instruction::mv(&mut env, &current_env));

            constnt_overwrite!(ret = index);
            self.generator
                .add_instruction(Instruction::mv(&mut ret, &current_index));

            self.generator
                .add_instruction(Instruction::j("lookup".to_string()));
        };

        {
            self.generator
                .add_instruction(Instruction::label("lookup_return".to_string()));

            constnt_overwrite!(ret = index);
            self.generator
                .add_instruction(Instruction::lw(&mut ret, 0, &env));

            self.generator
                .add_instruction(Instruction::j("return".to_string()));
        };

        self.register_map.free_all()
    }

    // A0 pointer to terms array
    // A1 is new stack pointer
    // A2 is length of terms array
    // A3 is the return address
    pub fn clone_list(&mut self) -> Freed {
        var_argument!(src_list = self.first_arg);
        var_argument!(dest_list = self.second_arg);
        var_argument!(length = self.third_arg);
        argument!(callback = self.fourth_arg);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::label("clone_list".to_string()));

        self.generator
            .add_instruction(Instruction::comment("A2 contains terms length".to_string()));

        self.generator.add_instruction(Instruction::beq(
            &length,
            &zero,
            "clone_list_return".to_string(),
        ));

        {
            constnt!(list_item = self.first_temp);
            self.generator
                .add_instruction(Instruction::lw(&mut list_item, 0, &src_list));

            // Store term in new storage
            self.generator
                .add_instruction(Instruction::sw(&list_item, 0, &dest_list));

            // move fields up by 4 bytes
            self.generator
                .add_instruction(Instruction::addi(&mut src_list.clone(), &src_list, 4));

            // move pointer up by 4 bytes
            self.generator.add_instruction(Instruction::addi(
                &mut dest_list.clone(),
                &dest_list,
                4,
            ));

            // decrement terms length
            self.generator
                .add_instruction(Instruction::addi(&mut length.clone(), &length, -1));

            self.generator
                .add_instruction(Instruction::j("clone_list".to_string()));
        };

        {
            self.generator
                .add_instruction(Instruction::label("clone_list_return".to_string()));

            self.generator.add_instruction(Instruction::comment(
                "A3 contains return address".to_string(),
            ));

            var_overwrite!(discard = zero);
            self.generator
                .add_instruction(Instruction::jalr(&mut discard, &callback, 0));
        };

        self.register_map.free_all()
    }

    // A0 pointer to terms array to decrement from
    // A1 is new stack pointer
    // A2 is length of terms array
    // A3 is the return address
    pub fn reverse_clone_list(&mut self) -> Freed {
        var_argument!(src_list = self.first_arg);
        var_argument!(dest_list = self.second_arg);
        var_argument!(length = self.third_arg);
        argument!(callback = self.fourth_arg);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::label("reverse_clone_list".to_string()));

        self.generator
            .add_instruction(Instruction::comment("A2 contains terms length".to_string()));

        self.generator.add_instruction(Instruction::beq(
            &length,
            &zero,
            "reverse_clone_list_return".to_string(),
        ));

        {
            constnt!(list_item = self.first_temp);
            self.generator
                .add_instruction(Instruction::lw(&mut list_item, 0, &src_list));

            // Store term in new storage
            self.generator
                .add_instruction(Instruction::sw(&list_item, 0, &dest_list));

            // move backwards by 4 bytes
            self.generator
                .add_instruction(Instruction::addi(&mut src_list.clone(), &src_list, -4));

            // move pointer up by 4 bytes
            self.generator.add_instruction(Instruction::addi(
                &mut dest_list.clone(),
                &dest_list,
                4,
            ));

            // decrement terms length
            self.generator
                .add_instruction(Instruction::addi(&mut length.clone(), &length, -1));

            self.generator
                .add_instruction(Instruction::j("reverse_clone_list".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::label("reverse_clone_list_return".to_string()));

            self.generator.add_instruction(Instruction::comment(
                "A3 contains return address".to_string(),
            ));

            constnt_overwrite!(discard = zero);
            self.generator
                .add_instruction(Instruction::jalr(&mut discard, &callback, 0));
        };

        self.register_map.free_all()
    }

    pub fn initial_term(&mut self, bytes: Vec<u8>) {
        self.generator
            .add_instruction(Instruction::section("data".to_string()));

        self.generator
            .add_instruction(Instruction::label("initial_term".to_string()));

        self.generator.add_instruction(Instruction::byte(bytes));

        self.generator
            .add_instruction(Instruction::label("force_counts".to_string()));
        self.generator
            .add_instruction(Instruction::byte(FORCE_COUNTS.to_vec()));

        self.generator
            .add_instruction(Instruction::label("arities".to_string()));
        self.generator
            .add_instruction(Instruction::byte(ARITIES.to_vec()));
    }

    pub fn eval_builtin_app(&mut self) -> Freed {
        argument!(builtin_value = self.first_arg);
        argument!(builtin_func_index = self.second_arg);
        self.generator
            .add_instruction(Instruction::label("eval_builtin_app".to_string()));

        // offset to actual args for the builtin call
        constnt!(builtin_args_temp = self.first_temp);
        self.generator.add_instruction(Instruction::addi(
            &mut builtin_args_temp,
            &builtin_value,
            7,
        ));

        constnt_overwrite!(builtin_args = builtin_value);
        self.generator
            .add_instruction(Instruction::mv(&mut builtin_args, &builtin_args_temp));

        var_overwrite!(builtin_call_jump = builtin_args_temp);
        self.generator.add_instruction(Instruction::la(
            &mut builtin_call_jump,
            "eval_builtin_call".to_string(),
        ));

        constnt!(builtin_index_offset = self.second_temp);
        self.generator.add_instruction(Instruction::slli(
            &mut builtin_index_offset,
            &builtin_func_index,
            2,
        ));

        self.generator.add_instruction(Instruction::add(
            &mut builtin_call_jump.clone(),
            &builtin_call_jump,
            &builtin_index_offset,
        ));

        constnt!(discard = self.discard);
        self.generator
            .add_instruction(Instruction::jalr(&mut discard, &builtin_call_jump, 0));

        self.generator
            .add_instruction(Instruction::label("eval_builtin_call".to_string()));

        // 0 - add_integer
        self.generator
            .add_instruction(Instruction::j("add_integer".to_string()));

        // 1 - sub_integer
        self.generator
            .add_instruction(Instruction::j("sub_integer".to_string()));

        // 2 - multiply_integer
        self.generator
            .add_instruction(Instruction::j("multiply_integer".to_string()));

        // 3 - divide_integer
        self.generator
            .add_instruction(Instruction::j("divide_integer".to_string()));

        // 4 - quotient_integer
        self.generator
            .add_instruction(Instruction::j("quotient_integer".to_string()));

        // 5 - remainder_integer
        self.generator
            .add_instruction(Instruction::j("remainder_integer".to_string()));

        // 6 - mod_integer
        self.generator
            .add_instruction(Instruction::j("mod_integer".to_string()));

        // 7 - equals_integer
        self.generator
            .add_instruction(Instruction::j("equals_integer".to_string()));

        // 8 - less_than_integer
        self.generator
            .add_instruction(Instruction::j("less_than_integer".to_string()));

        // 9 - less_than_equals_integer
        self.generator
            .add_instruction(Instruction::j("less_than_equals_integer".to_string()));

        // 10 - append_bytestring
        self.generator
            .add_instruction(Instruction::j("append_bytestring".to_string()));

        // 11 - cons_bytestring
        self.generator
            .add_instruction(Instruction::j("cons_bytestring".to_string()));

        // 12 - slice_bytestring
        self.generator
            .add_instruction(Instruction::j("slice_bytestring".to_string()));

        // 13 - length_bytestring
        self.generator
            .add_instruction(Instruction::j("length_bytestring".to_string()));

        // 14 - index_bytestring
        self.generator
            .add_instruction(Instruction::j("index_bytestring".to_string()));

        // 15 - equals_bytestring
        self.generator
            .add_instruction(Instruction::j("equals_bytestring".to_string()));

        // 16 - less_than_bytestring
        self.generator
            .add_instruction(Instruction::j("less_than_bytestring".to_string()));

        // 17 - less_than_equals_bytestring
        self.generator
            .add_instruction(Instruction::j("less_than_equals_bytestring".to_string()));

        // 18 - sha2_256
        self.generator
            .add_instruction(Instruction::j("sha2_256".to_string()));

        // 19 - sha3_256
        self.generator
            .add_instruction(Instruction::j("sha3_256".to_string()));

        // 20 - blake2b_256
        self.generator
            .add_instruction(Instruction::j("blake2b_256".to_string()));

        // 21 - verify_ed25519_signature
        self.generator
            .add_instruction(Instruction::j("verify_ed25519_signature".to_string()));

        // 22 - append_string
        self.generator
            .add_instruction(Instruction::j("append_string".to_string()));

        // 23 - equals_string
        self.generator
            .add_instruction(Instruction::j("equals_string".to_string()));

        // 24 - encode_utf8
        self.generator
            .add_instruction(Instruction::j("encode_utf8".to_string()));

        // 25 - decode_utf8
        self.generator
            .add_instruction(Instruction::j("decode_utf8".to_string()));

        // 26 - if_then_else
        self.generator
            .add_instruction(Instruction::j("if_then_else".to_string()));

        self.register_map.free_all()
    }

    pub fn unwrap_integer(&mut self) -> Freed {
        argument!(arg = self.first_arg);
        argument!(callback = self.second_arg);

        self.generator
            .add_instruction(Instruction::label("unwrap_integer".to_string()));

        constnt!(expected_tag = self.first_temp);
        self.generator
            .add_instruction(Instruction::li(&mut expected_tag, 0));

        constnt!(value_tag = self.second_temp);
        self.generator
            .add_instruction(Instruction::lbu(&mut value_tag, 0, &arg));

        self.generator.add_instruction(Instruction::bne(
            &expected_tag,
            &value_tag,
            "handle_error".to_string(),
        ));

        constnt!(constant_value = self.third_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut constant_value, 1, &arg));

        // Overwrite expected_tag
        constnt_overwrite!(expected_type = expected_tag);
        self.generator.add_instruction(Instruction::li(
            &mut expected_type,
            1 + 256 * (const_tag::INTEGER as i32),
        ));

        // The type ends up being [0x01, 0x00] which in little endian is 1
        // Overwrite value_tag
        constnt_overwrite!(arg_type = value_tag);
        self.generator
            .add_instruction(Instruction::lhu(&mut arg_type, 0, &constant_value));

        self.generator.add_instruction(Instruction::bne(
            &expected_type,
            &arg_type,
            "handle_error".to_string(),
        ));

        // Return pointer to integer value
        constnt_overwrite!(ret = arg);
        self.generator
            .add_instruction(Instruction::addi(&mut ret, &constant_value, 2));

        constnt!(discard = self.discard);
        self.generator
            .add_instruction(Instruction::jalr(&mut discard, &callback, 0));

        self.register_map.free_all()
    }

    // purely allocate integer type + constant value on heap
    pub fn allocate_integer_type(&mut self) -> Freed {
        var_argument!(heap = self.heap);
        argument!(allocate_temp = self.saved);
        argument!(callback = self.eighth_arg);

        self.generator
            .add_instruction(Instruction::label("allocate_integer_type".to_string()));

        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 7));

        // Overwrite allocate_temp
        constnt_overwrite!(value_tag = allocate_temp);
        self.generator
            .add_instruction(Instruction::li(&mut value_tag, 0));

        self.generator
            .add_instruction(Instruction::sb(&value_tag, -7, &heap));

        // Overwrite value_tag
        constnt_overwrite!(integer_pointer = value_tag);
        // We store integer immediately after the pointer so we simply add 4 to point to the location
        self.generator
            .add_instruction(Instruction::addi(&mut integer_pointer, &heap, -2));

        self.generator
            .add_instruction(Instruction::sw(&integer_pointer, -6, &heap));

        // Overwrite integer_pointer
        constnt_overwrite!(integer_type = integer_pointer);
        self.generator.add_instruction(Instruction::li(
            &mut integer_type,
            1 + (const_tag::INTEGER as i32) * 256,
        ));

        self.generator
            .add_instruction(Instruction::sh(&integer_type, -2, &heap));

        constnt!(discard = self.discard);
        self.generator
            .add_instruction(Instruction::jalr(&mut discard, &callback, 0));

        self.register_map.free_all()
    }

    // purely allocate integer type + constant value on heap
    pub fn allocate_bytestring_type(&mut self) -> Freed {
        var_argument!(heap = self.heap);
        argument!(allocate_temp = self.saved);
        argument!(callback = self.eighth_arg);

        self.generator
            .add_instruction(Instruction::label("allocate_bytestring_type".to_string()));

        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 7));

        // Overwrite allocate_temp
        constnt_overwrite!(value_tag = allocate_temp);
        self.generator
            .add_instruction(Instruction::li(&mut value_tag, 0));

        self.generator
            .add_instruction(Instruction::sb(&value_tag, -7, &heap));

        // Overwrite value_tag
        constnt_overwrite!(bytestring_pointer = value_tag);
        // We store integer immediately after the pointer so we simply add 4 to point to the location
        self.generator
            .add_instruction(Instruction::addi(&mut bytestring_pointer, &heap, -2));

        self.generator
            .add_instruction(Instruction::sw(&bytestring_pointer, -6, &heap));

        // Overwrite bytestring_pointer
        constnt_overwrite!(integer_type = bytestring_pointer);
        self.generator.add_instruction(Instruction::li(
            &mut integer_type,
            1 + (const_tag::BYTESTRING as i32) * 256,
        ));

        self.generator
            .add_instruction(Instruction::sh(&integer_type, -2, &heap));

        constnt!(discard = self.discard);
        self.generator
            .add_instruction(Instruction::jalr(&mut discard, &callback, 0));

        self.register_map.free_all()
    }

    pub fn unwrap_bytestring(&mut self) -> Freed {
        argument!(arg = self.first_arg);
        argument!(callback = self.second_arg);
        self.generator
            .add_instruction(Instruction::label("unwrap_bytestring".to_string()));

        constnt!(expected_tag = self.first_temp);
        self.generator
            .add_instruction(Instruction::li(&mut expected_tag, 0));

        constnt!(value_tag = self.second_temp);
        self.generator
            .add_instruction(Instruction::lbu(&mut value_tag, 0, &arg));

        self.generator.add_instruction(Instruction::bne(
            &expected_tag,
            &value_tag,
            "handle_error".to_string(),
        ));

        constnt!(constant_value = self.third_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut constant_value, 1, &arg));

        // Overwrite expected_tag
        constnt_overwrite!(expected_type = expected_tag);
        self.generator.add_instruction(Instruction::li(
            &mut expected_type,
            1 + 256 * (const_tag::BYTESTRING as i32),
        ));

        // The type ends up being [0x01, 0x00] which in little endian is 1
        // Overwrite value_tag
        constnt_overwrite!(arg_type = value_tag);
        self.generator
            .add_instruction(Instruction::lhu(&mut arg_type, 0, &constant_value));

        self.generator.add_instruction(Instruction::bne(
            &expected_type,
            &arg_type,
            "handle_error".to_string(),
        ));

        // Return pointer to bytestring value
        constnt_overwrite!(ret = arg);
        self.generator
            .add_instruction(Instruction::addi(&mut ret, &constant_value, 2));

        constnt!(discard = self.discard);
        self.generator
            .add_instruction(Instruction::jalr(&mut discard, &callback, 0));

        self.register_map.free_all()
    }

    pub fn add_integer(&mut self) -> Freed {
        argument!(args = self.first_arg);
        self.generator
            .add_instruction(Instruction::label("add_integer".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt!(y_value = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut y_value, 4, &args));

        // Overwrite args
        constnt_overwrite!(first_arg = args);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &x_value));

        constnt!(store_y = self.third_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut store_y, &y_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        // Overwrite x_value
        constnt_overwrite!(x_integer = x_value);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &first_arg));

        constnt_overwrite!(first_arg = first_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &store_y));

        // Overwrite store_y
        constnt_overwrite!(store_x = store_y);
        self.generator
            .add_instruction(Instruction::mv(&mut store_x, &x_integer));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        // Now move things back
        constnt_overwrite!(x_integer = x_integer);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &store_x));

        // Overwrite y_value
        constnt_overwrite!(y_integer = y_value);
        self.generator
            .add_instruction(Instruction::mv(&mut y_integer, &first_arg));

        // Overwrite first_arg
        constnt_overwrite!(x_sign = first_arg);
        self.generator
            .add_instruction(Instruction::lbu(&mut x_sign, 0, &x_integer));

        // Overwrite callback
        constnt_overwrite!(y_sign = callback);
        self.generator
            .add_instruction(Instruction::lbu(&mut y_sign, 0, &y_integer));

        constnt_overwrite!(x_magnitude = store_x);
        self.generator
            .add_instruction(Instruction::addi(&mut x_magnitude, &x_integer, 1));

        constnt!(y_magnitude = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::addi(&mut y_magnitude, &y_integer, 1));

        self.generator
            .add_instruction(Instruction::j("add_signed_integers".to_string()));

        self.register_map.free_all()
    }

    pub fn sub_integer(&mut self) -> Freed {
        argument!(args = self.first_arg);
        self.generator
            .add_instruction(Instruction::label("sub_integer".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt!(y_value = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut y_value, 4, &args));

        constnt_overwrite!(first_arg = args);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &x_value));

        constnt!(store_y = self.third_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut store_y, &y_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        constnt_overwrite!(x_integer = x_value);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &first_arg));

        constnt_overwrite!(first_arg = first_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &store_y));

        constnt_overwrite!(store_x = store_y);
        self.generator
            .add_instruction(Instruction::mv(&mut store_x, &x_integer));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        constnt_overwrite!(x_integer = x_integer);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &store_x));

        // Overwrite y_value
        constnt_overwrite!(y_integer = y_value);
        self.generator
            .add_instruction(Instruction::mv(&mut y_integer, &first_arg));

        // Overwrite first_arg
        constnt_overwrite!(x_sign = first_arg);
        self.generator
            .add_instruction(Instruction::lbu(&mut x_sign, 0, &x_integer));

        var_overwrite!(y_sign = callback);
        self.generator
            .add_instruction(Instruction::lbu(&mut y_sign, 0, &y_integer));

        // flip y_sign
        self.generator
            .add_instruction(Instruction::xori(&mut y_sign.clone(), &y_sign, 1));

        constnt_overwrite!(x_magnitude = store_x);
        self.generator
            .add_instruction(Instruction::addi(&mut x_magnitude, &x_integer, 1));

        constnt!(y_magnitude = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::addi(&mut y_magnitude, &y_integer, 1));

        self.generator
            .add_instruction(Instruction::j("add_signed_integers".to_string()));

        self.register_map.free_all()
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

    pub fn equals_integer(&mut self) -> Freed {
        argument!(args = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::label("equals_integer".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt!(y_value = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut y_value, 4, &args));

        constnt_overwrite!(first_arg = args);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &x_value));

        constnt!(store_y = self.third_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut store_y, &y_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        // Overwrite x_value
        constnt_overwrite!(x_integer = x_value);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &first_arg));

        constnt_overwrite!(first_arg = first_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &store_y));

        constnt_overwrite!(store_x = store_y);
        self.generator
            .add_instruction(Instruction::mv(&mut store_x, &x_integer));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        constnt_overwrite!(x_integer = x_integer);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &store_x));

        // Overwrite y_value
        constnt_overwrite!(y_integer = y_value);
        self.generator
            .add_instruction(Instruction::mv(&mut y_integer, &first_arg));

        constnt_overwrite!(x_sign = first_arg);
        self.generator
            .add_instruction(Instruction::lbu(&mut x_sign, 0, &x_integer));

        constnt_overwrite!(y_sign = callback);
        self.generator
            .add_instruction(Instruction::lbu(&mut y_sign, 0, &y_integer));

        constnt_overwrite!(x_magnitude = store_x);
        self.generator
            .add_instruction(Instruction::addi(&mut x_magnitude, &x_integer, 1));

        constnt!(y_magnitude = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::addi(&mut y_magnitude, &y_integer, 1));

        constnt!(callback = self.fifth_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "compare_magnitude".to_string(),
        ));

        // Overwrite first_arg
        constnt_arg_overwrite!(equality = x_sign);
        constnt_overwrite!(bool_value = x_integer);
        self.generator
            .add_instruction(Instruction::mv(&mut bool_value, &equality));

        // heap allocation = 1 byte for value tag + 4 bytes for pointer + 1 byte for type length
        // + 1 byte for type + 1 byte for bool value
        // Overwrite equality
        constnt_overwrite!(ret = equality);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 8));

        self.generator
            .add_instruction(Instruction::sb(&zero, -8, &heap));

        constnt_overwrite!(constant_pointer = y_integer);
        self.generator
            .add_instruction(Instruction::addi(&mut constant_pointer, &heap, -3));

        self.generator
            .add_instruction(Instruction::sw(&constant_pointer, -7, &heap));

        constnt_overwrite!(bool_type = constant_pointer);
        self.generator
            .add_instruction(Instruction::li(&mut bool_type, 1 + 256 * (BOOL as i32)));

        self.generator
            .add_instruction(Instruction::sh(&bool_type, -3, &heap));

        self.generator
            .add_instruction(Instruction::sb(&bool_value, -1, &heap));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn less_than_equals_integer(&mut self) -> Freed {
        argument!(args = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::label("less_than_equals_integer".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt!(y_value = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut y_value, 4, &args));

        // Overwrite args
        constnt_overwrite!(first_arg = args);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &x_value));

        constnt!(store_y = self.third_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut store_y, &y_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        // Overwrite x_value
        constnt_overwrite!(x_integer = x_value);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &first_arg));

        constnt_overwrite!(first_arg = first_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &store_y));

        constnt_overwrite!(store_x = store_y);
        self.generator
            .add_instruction(Instruction::mv(&mut store_x, &x_integer));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        // Now move things back
        constnt_overwrite!(x_integer = x_integer);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &store_x));

        // Overwrite y_value
        constnt_overwrite!(y_integer = y_value);
        self.generator
            .add_instruction(Instruction::mv(&mut y_integer, &first_arg));

        // Overwrite first_arg
        constnt_overwrite!(x_sign = first_arg);
        self.generator
            .add_instruction(Instruction::lbu(&mut x_sign, 0, &x_integer));

        // Overwrite callback
        constnt_overwrite!(y_sign = callback);
        self.generator
            .add_instruction(Instruction::lbu(&mut y_sign, 0, &y_integer));

        constnt_overwrite!(x_magnitude = store_x);
        self.generator
            .add_instruction(Instruction::addi(&mut x_magnitude, &x_integer, 1));

        constnt!(first_magnitude = self.seventh_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut first_magnitude, &x_magnitude));

        constnt!(y_magnitude = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::addi(&mut y_magnitude, &y_integer, 1));

        constnt!(callback = self.fifth_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "compare_magnitude".to_string(),
        ));

        // Overwrite first_arg
        constnt_arg_overwrite!(equality = x_sign);
        constnt_arg_overwrite!(greater_magnitude_value = y_sign);
        var_overwrite!(bool_value = x_integer);
        self.generator
            .add_instruction(Instruction::mv(&mut bool_value, &equality));

        // heap allocation = 1 byte for value tag + 4 bytes for pointer + 1 byte for type length
        // + 1 byte for type + 1 byte for bool value
        // Overwrite equality
        constnt_overwrite!(ret = equality);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 8));

        self.generator
            .add_instruction(Instruction::sb(&zero, -8, &heap));

        constnt_overwrite!(constant_pointer = y_integer);
        self.generator
            .add_instruction(Instruction::addi(&mut constant_pointer, &heap, -3));

        self.generator
            .add_instruction(Instruction::sw(&constant_pointer, -7, &heap));

        // Overwrite constant_pointer
        constnt_overwrite!(bool_type = constant_pointer);
        self.generator
            .add_instruction(Instruction::li(&mut bool_type, 1 + 256 * (BOOL as i32)));

        self.generator
            .add_instruction(Instruction::sh(&bool_type, -3, &heap));

        self.generator.add_instruction(Instruction::bne(
            &greater_magnitude_value,
            &first_magnitude,
            "less_than".to_string(),
        ));

        {
            self.generator.add_instruction(Instruction::j(
                "finish_less_than_equals_integer".to_string(),
            ));
        }
        {
            self.generator
                .add_instruction(Instruction::label("less_than".to_string()));

            self.generator
                .add_instruction(Instruction::li(&mut bool_value, 1));
        }

        self.generator.add_instruction(Instruction::label(
            "finish_less_than_equals_integer".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::sb(&bool_value, -1, &heap));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn less_than_integer(&mut self) -> Freed {
        argument!(args = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::label("less_than_integer".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt!(y_value = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut y_value, 4, &args));

        // Overwrite args
        constnt_overwrite!(first_arg = args);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &x_value));

        constnt!(store_y = self.third_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut store_y, &y_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        // Overwrite x_value
        constnt_overwrite!(x_integer = x_value);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &first_arg));

        constnt_overwrite!(first_arg = first_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &store_y));

        constnt_overwrite!(store_x = store_y);
        self.generator
            .add_instruction(Instruction::mv(&mut store_x, &x_integer));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        // Now move things back
        constnt_overwrite!(x_integer = x_integer);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &store_x));

        // Overwrite y_value
        constnt_overwrite!(y_integer = y_value);
        self.generator
            .add_instruction(Instruction::mv(&mut y_integer, &first_arg));

        // Overwrite first_arg
        constnt_overwrite!(x_sign = first_arg);
        self.generator
            .add_instruction(Instruction::lbu(&mut x_sign, 0, &x_integer));

        // Overwrite callback
        constnt_overwrite!(y_sign = callback);
        self.generator
            .add_instruction(Instruction::lbu(&mut y_sign, 0, &y_integer));

        constnt_overwrite!(x_magnitude = store_x);
        self.generator
            .add_instruction(Instruction::addi(&mut x_magnitude, &x_integer, 1));

        constnt!(first_magnitude = self.seventh_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut first_magnitude, &x_magnitude));

        constnt!(y_magnitude = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::addi(&mut y_magnitude, &y_integer, 1));

        constnt!(callback = self.fifth_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "compare_magnitude".to_string(),
        ));

        // Overwrite first_arg
        constnt_arg_overwrite!(equality = x_sign);
        constnt_arg_overwrite!(greater_magnitude_value = y_sign);
        var_overwrite!(bool_value = x_integer);
        // Flip equality
        self.generator
            .add_instruction(Instruction::xori(&mut bool_value, &equality, 1));

        // heap allocation = 1 byte for value tag + 4 bytes for pointer + 1 byte for type length
        // + 1 byte for type + 1 byte for bool value
        // Overwrite equality
        constnt_overwrite!(ret = equality);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 8));

        self.generator
            .add_instruction(Instruction::sb(&zero, -8, &heap));

        constnt_overwrite!(constant_pointer = y_integer);
        self.generator
            .add_instruction(Instruction::addi(&mut constant_pointer, &heap, -3));

        self.generator
            .add_instruction(Instruction::sw(&constant_pointer, -7, &heap));

        // Overwrite constant_pointer
        constnt_overwrite!(bool_type = constant_pointer);
        self.generator
            .add_instruction(Instruction::li(&mut bool_type, 1 + 256 * (BOOL as i32)));

        self.generator
            .add_instruction(Instruction::sh(&bool_type, -3, &heap));

        self.generator.add_instruction(Instruction::bne(
            &greater_magnitude_value,
            &first_magnitude,
            "less_than_int".to_string(),
        ));

        {
            self.generator
                .add_instruction(Instruction::j("finish_less_than_integer".to_string()));

            self.generator
                .add_instruction(Instruction::li(&mut bool_value, 0));
        }
        {
            self.generator
                .add_instruction(Instruction::label("less_than_int".to_string()));
        }

        self.generator
            .add_instruction(Instruction::label("finish_less_than_integer".to_string()));

        self.generator
            .add_instruction(Instruction::sb(&bool_value, -1, &heap));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn append_bytestring(&mut self) -> Freed {
        argument!(args = self.first_arg);
        var_argument!(heap = self.heap);
        self.generator
            .add_instruction(Instruction::label("append_bytestring".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt!(y_value = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut y_value, 4, &args));

        // Overwrite args
        constnt_overwrite!(first_arg = args);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &x_value));

        constnt!(store_y = self.third_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut store_y, &y_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_bytestring".to_string(),
        ));

        // Overwrite x_value
        constnt_overwrite!(x_bytestring = x_value);
        self.generator
            .add_instruction(Instruction::mv(&mut x_bytestring, &first_arg));

        constnt_overwrite!(first_arg = first_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &store_y));

        constnt_overwrite!(store_x = store_y);
        self.generator
            .add_instruction(Instruction::mv(&mut store_x, &x_bytestring));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_bytestring".to_string(),
        ));

        constnt_overwrite!(x_bytestring = x_bytestring);
        self.generator
            .add_instruction(Instruction::mv(&mut x_bytestring, &store_x));

        constnt_overwrite!(y_bytestring = y_value);
        self.generator
            .add_instruction(Instruction::mv(&mut y_bytestring, &first_arg));

        constnt!(x_length = self.third_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_length, 0, &x_bytestring));

        constnt!(y_length = self.fourth_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut y_length, 0, &y_bytestring));

        constnt!(total_length = self.fifth_temp);
        self.generator
            .add_instruction(Instruction::add(&mut total_length, &x_length, &y_length));

        constnt_overwrite!(ret = first_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        constnt!(callback_alloc = self.eighth_arg);
        // This function does not modify temps and only changes heap and s3
        // and eighth arg
        self.generator.add_instruction(Instruction::jal(
            &mut callback_alloc,
            "allocate_bytestring_type".to_string(),
        ));

        var!(value_builder = self.sixth_temp);
        self.generator
            .add_instruction(Instruction::mv(&mut value_builder, &heap));

        // Overwrite total_length
        var!(total_allocation = self.seventh_temp);
        self.generator
            .add_instruction(Instruction::slli(&mut total_allocation, &total_length, 2));

        // heap allocation = 1 byte for value tag + 4 bytes for pointer + 1 byte for type length
        // + 1 byte for type + 4 bytes for bytestring length + 4 * bytestring length bytes
        self.generator.add_instruction(Instruction::addi(
            &mut total_allocation.clone(),
            &total_allocation,
            4,
        ));

        self.generator.add_instruction(Instruction::add(
            &mut heap.clone(),
            &heap,
            &total_allocation,
        ));

        self.generator
            .add_instruction(Instruction::sw(&total_length, 0, &value_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut value_builder.clone(),
            &value_builder,
            4,
        ));

        constnt!(return_temp = self.seventh_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut return_temp, &ret));

        constnt!(second_list = self.sixth_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut second_list, &y_bytestring));

        constnt!(second_list_len = self.fifth_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut second_list_len, &y_length));

        constnt_overwrite!(size = store_x);
        self.generator
            .add_instruction(Instruction::mv(&mut size, &x_length));

        constnt_overwrite!(dst_list = callback);
        self.generator
            .add_instruction(Instruction::mv(&mut dst_list, &value_builder));

        constnt_overwrite!(src_list = ret);
        self.generator
            .add_instruction(Instruction::addi(&mut src_list, &x_bytestring, 4));

        // clone x into heap
        constnt!(callback = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::jal(&mut callback, "clone_list".to_string()));

        constnt_overwrite!(size = size);
        self.generator
            .add_instruction(Instruction::mv(&mut size, &second_list_len));

        constnt_overwrite!(src_list = src_list);
        self.generator
            .add_instruction(Instruction::addi(&mut src_list, &second_list, 4));

        constnt_overwrite!(callback = callback);
        self.generator
            .add_instruction(Instruction::jal(&mut callback, "clone_list".to_string()));

        constnt_overwrite!(ret = src_list);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &return_temp));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn cons_bytestring(&mut self) -> Freed {
        argument!(args = self.first_arg);
        var_argument!(heap = self.heap);
        self.generator
            .add_instruction(Instruction::label("cons_bytestring".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt!(y_value = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut y_value, 4, &args));

        // Overwrite args
        constnt_overwrite!(first_arg = args);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &x_value));

        constnt!(store_y = self.third_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut store_y, &y_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        // Overwrite x_value
        constnt_overwrite!(x_integer = x_value);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &first_arg));

        constnt_overwrite!(first_arg = first_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut first_arg, &store_y));

        // Overwrite store_y
        constnt_overwrite!(store_x = store_y);
        self.generator
            .add_instruction(Instruction::mv(&mut store_x, &x_integer));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_bytestring".to_string(),
        ));

        constnt_overwrite!(x_integer = x_integer);
        self.generator
            .add_instruction(Instruction::mv(&mut x_integer, &store_x));

        constnt_overwrite!(y_bytestring = y_value);
        self.generator
            .add_instruction(Instruction::mv(&mut y_bytestring, &first_arg));

        constnt!(integer_size = self.third_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut integer_size, 1, &x_integer));

        constnt!(expected_size = self.fourth_temp);
        self.generator
            .add_instruction(Instruction::li(&mut expected_size, 1));

        self.generator.add_instruction(Instruction::bne(
            &expected_size,
            &integer_size,
            "handle_error".to_string(),
        ));

        // Overwrite integer_size
        constnt_overwrite!(integer_word_value = integer_size);
        self.generator
            .add_instruction(Instruction::lw(&mut integer_word_value, 5, &x_integer));

        // Overwrite expected_size
        constnt_overwrite!(max_value = expected_size);
        self.generator
            .add_instruction(Instruction::li(&mut max_value, 255));

        self.generator.add_instruction(Instruction::bltu(
            &max_value,
            &integer_word_value,
            "handle_error".to_string(),
        ));

        constnt_overwrite!(ret = first_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        constnt!(callback_alloc = self.eighth_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback_alloc,
            "allocate_bytestring_type".to_string(),
        ));

        // Overwrite max_value
        var_overwrite!(value_builder = max_value);
        self.generator
            .add_instruction(Instruction::mv(&mut value_builder, &heap));

        var!(new_bytes_length = self.fifth_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut new_bytes_length, 0, &y_bytestring));

        self.generator.add_instruction(Instruction::addi(
            &mut new_bytes_length.clone(),
            &new_bytes_length,
            1,
        ));

        var!(heap_allocation = self.sixth_temp);
        self.generator.add_instruction(Instruction::slli(
            &mut heap_allocation,
            &new_bytes_length,
            2,
        ));

        // Add 4 for bytes length
        self.generator.add_instruction(Instruction::addi(
            &mut heap_allocation.clone(),
            &heap_allocation,
            4,
        ));

        self.generator.add_instruction(Instruction::add(
            &mut heap.clone(),
            &heap,
            &heap_allocation,
        ));

        self.generator
            .add_instruction(Instruction::sw(&new_bytes_length, 0, &value_builder));

        self.generator
            .add_instruction(Instruction::sw(&integer_word_value, 4, &value_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut value_builder.clone(),
            &value_builder,
            8,
        ));

        constnt!(store_ret = self.fifth_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut store_ret, &ret));

        constnt_overwrite!(bytes_to_copy = store_x);
        self.generator.add_instruction(Instruction::addi(
            &mut bytes_to_copy,
            &new_bytes_length,
            -1,
        ));

        constnt_overwrite!(dst_list = callback);
        self.generator
            .add_instruction(Instruction::mv(&mut dst_list, &value_builder));

        constnt_overwrite!(src_list = ret);
        self.generator
            .add_instruction(Instruction::addi(&mut src_list, &y_bytestring, 4));

        constnt!(callback = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::jal(&mut callback, "clone_list".to_string()));

        constnt_overwrite!(ret = src_list);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &store_ret));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn slice_bytestring(&mut self) -> Freed {
        argument!(args = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::label("slice_bytestring".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt!(y_value = self.third_arg);
        self.generator
            .add_instruction(Instruction::lw(&mut y_value, 4, &args));

        constnt!(z_value = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::lw(&mut z_value, 8, &args));

        constnt_overwrite!(unwrap_val = args);
        self.generator
            .add_instruction(Instruction::mv(&mut unwrap_val, &x_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        constnt!(x_store = self.fifth_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut x_store, &unwrap_val));

        constnt_overwrite!(unwrap_val = unwrap_val);
        self.generator
            .add_instruction(Instruction::mv(&mut unwrap_val, &y_value));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        constnt!(y_store = self.sixth_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut y_store, &unwrap_val));

        constnt_overwrite!(unwrap_val = unwrap_val);
        self.generator
            .add_instruction(Instruction::mv(&mut unwrap_val, &z_value));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_bytestring".to_string(),
        ));
        constnt_overwrite!(bytestring_val = y_value);
        self.generator
            .add_instruction(Instruction::mv(&mut bytestring_val, &unwrap_val));

        constnt!(max_int_length = self.second_temp);
        self.generator
            .add_instruction(Instruction::li(&mut max_int_length, 1));

        constnt_overwrite!(x_length = x_value);
        self.generator
            .add_instruction(Instruction::lw(&mut x_length, 1, &x_store));

        self.generator.add_instruction(Instruction::bne(
            &x_length,
            &max_int_length,
            "handle_error".to_string(),
        ));
        constnt!(y_length = self.third_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut y_length, 1, &y_store));

        self.generator.add_instruction(Instruction::bne(
            &y_length,
            &max_int_length,
            "handle_error".to_string(),
        ));

        constnt!(bytestring_len = self.fourth_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut bytestring_len, 0, &bytestring_val));

        var!(starting_index = self.fifth_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut starting_index, 5, &x_store));

        var!(taking_index = self.sixth_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut taking_index, 5, &y_store));

        // Use bytes register to offset by 4 * starting_index + 4 for length word
        var!(bytes = self.seventh_temp);
        self.generator
            .add_instruction(Instruction::slli(&mut bytes, &starting_index, 2));

        self.generator
            .add_instruction(Instruction::addi(&mut bytes.clone(), &bytes, 4));

        self.generator.add_instruction(Instruction::add(
            &mut bytes.clone(),
            &bytes,
            &bytestring_val,
        ));

        constnt_overwrite!(ret = unwrap_val);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        constnt!(callback_alloc = self.eighth_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback_alloc,
            "allocate_bytestring_type".to_string(),
        ));

        var_overwrite!(value_builder = z_value);
        self.generator
            .add_instruction(Instruction::mv(&mut value_builder, &heap));

        var_overwrite!(new_length = bytestring_val);
        self.generator
            .add_instruction(Instruction::li(&mut new_length, 0));

        constnt!(length_pointer = self.seventh_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut length_pointer, &value_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut value_builder.clone(),
            &value_builder,
            4,
        ));

        {
            self.generator
                .add_instruction(Instruction::label("slice_loop".to_string()));

            self.generator.add_instruction(Instruction::bgeu(
                &starting_index,
                &bytestring_len,
                "finalize_slice".to_string(),
            ));

            self.generator.add_instruction(Instruction::beq(
                &taking_index,
                &zero,
                "finalize_slice".to_string(),
            ));

            constnt_overwrite!(byte = x_length);
            self.generator
                .add_instruction(Instruction::lw(&mut byte, 0, &bytes));

            self.generator
                .add_instruction(Instruction::sw(&byte, 0, &value_builder));

            self.generator.add_instruction(Instruction::addi(
                &mut value_builder.clone(),
                &value_builder,
                4,
            ));

            self.generator
                .add_instruction(Instruction::addi(&mut bytes.clone(), &bytes, 4));

            self.generator.add_instruction(Instruction::addi(
                &mut taking_index.clone(),
                &taking_index,
                -1,
            ));

            self.generator.add_instruction(Instruction::addi(
                &mut starting_index.clone(),
                &starting_index,
                1,
            ));

            self.generator.add_instruction(Instruction::addi(
                &mut new_length.clone(),
                &new_length,
                1,
            ));

            self.generator
                .add_instruction(Instruction::j("slice_loop".to_string()));
        }

        self.generator
            .add_instruction(Instruction::label("finalize_slice".to_string()));

        self.generator
            .add_instruction(Instruction::sw(&new_length, 0, &length_pointer));

        self.generator
            .add_instruction(Instruction::mv(&mut heap, &value_builder));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn length_bytestring(&mut self) -> Freed {
        argument!(args = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::Label("length_bytestring".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt_overwrite!(unwrap_val = args);
        self.generator
            .add_instruction(Instruction::mv(&mut unwrap_val, &x_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_bytestring".to_string(),
        ));

        constnt_overwrite!(bytestring = x_value);
        self.generator
            .add_instruction(Instruction::mv(&mut bytestring, &unwrap_val));

        constnt!(bytestring_len = self.second_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut bytestring_len, 0, &bytestring));

        constnt_overwrite!(ret = unwrap_val);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        constnt!(callback_alloc = self.eighth_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback_alloc,
            "allocate_integer_type".to_string(),
        ));

        // 1 byte for sign + 4 bytes for length of 1 + 4 bytes to represent bytestring len
        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 9));

        self.generator
            .add_instruction(Instruction::sb(&zero, -9, &heap));

        constnt_overwrite!(integer_size = bytestring);
        self.generator
            .add_instruction(Instruction::li(&mut integer_size, 1));

        self.generator
            .add_instruction(Instruction::sw(&integer_size, -8, &heap));

        self.generator
            .add_instruction(Instruction::sw(&bytestring_len, -4, &heap));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn index_bytestring(&mut self) -> Freed {
        argument!(args = self.first_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);
        self.generator
            .add_instruction(Instruction::Label("index_bytestring".to_string()));

        constnt!(x_value = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut x_value, 0, &args));

        constnt!(y_value = self.third_arg);
        self.generator
            .add_instruction(Instruction::lw(&mut y_value, 4, &args));

        constnt_overwrite!(unwrap_val = args);
        self.generator
            .add_instruction(Instruction::mv(&mut unwrap_val, &x_value));

        constnt!(callback = self.second_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_bytestring".to_string(),
        ));

        constnt!(bytestring = self.fourth_arg);
        self.generator
            .add_instruction(Instruction::mv(&mut bytestring, &unwrap_val));

        constnt_overwrite!(unwrap_val = unwrap_val);
        self.generator
            .add_instruction(Instruction::mv(&mut unwrap_val, &y_value));

        constnt_overwrite!(callback = callback);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "unwrap_integer".to_string(),
        ));

        constnt_overwrite!(index = x_value);
        self.generator
            .add_instruction(Instruction::mv(&mut index, &unwrap_val));

        constnt!(sign = self.second_temp);
        self.generator
            .add_instruction(Instruction::lbu(&mut sign, 0, &index));

        self.generator
            .add_instruction(Instruction::bne(&sign, &zero, "handle_error".to_string()));

        constnt!(max_index_size = self.third_temp);
        self.generator
            .add_instruction(Instruction::li(&mut max_index_size, 1));

        constnt!(index_size = self.fourth_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut index_size, 1, &index));

        self.generator.add_instruction(Instruction::bne(
            &index_size,
            &max_index_size,
            "handle_error".to_string(),
        ));

        var_overwrite!(index_val = index_size);
        self.generator
            .add_instruction(Instruction::lw(&mut index_val, 5, &index));

        constnt!(byte_len = self.fifth_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut byte_len, 0, &bytestring));

        self.generator.add_instruction(Instruction::bgeu(
            &index_val,
            &byte_len,
            "handle_error".to_string(),
        ));

        // offset index past length word
        self.generator
            .add_instruction(Instruction::addi(&mut index_val.clone(), &index_val, 1));

        self.generator
            .add_instruction(Instruction::slli(&mut index_val.clone(), &index_val, 2));

        self.generator.add_instruction(Instruction::add(
            &mut index_val.clone(),
            &index_val,
            &bytestring,
        ));

        constnt!(byte = self.fifth_arg);
        self.generator
            .add_instruction(Instruction::lw(&mut byte, 0, &index_val));

        constnt_overwrite!(ret = unwrap_val);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        constnt!(callback_alloc = self.eighth_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback_alloc,
            "allocate_integer_type".to_string(),
        ));

        self.generator
            .add_instruction(Instruction::addi(&mut heap.clone(), &heap, 9));

        self.generator
            .add_instruction(Instruction::sb(&zero, -9, &heap));

        self.generator
            .add_instruction(Instruction::sw(&max_index_size, -8, &heap));

        self.generator
            .add_instruction(Instruction::sw(&byte, -4, &heap));

        self.generator
            .add_instruction(Instruction::j("return".to_string()));

        self.register_map.free_all()
    }

    pub fn equals_bytestring(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("equals_bytestring".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn less_than_bytestring(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("less_than_bytestring".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn less_than_equals_bytestring(&mut self) {
        self.generator.add_instruction(Instruction::Label(
            "less_than_equals_bytestring".to_string(),
        ));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn sha2_256(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("sha2_256".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn sha3_256(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("sha3_256".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn blake2b_256(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("blake2b_256".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn verify_ed25519_signature(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("verify_ed25519_signature".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn append_string(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("append_string".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn equals_string(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("equals_string".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn encode_utf8(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("encode_utf8".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn decode_utf8(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("decode_utf8".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    pub fn if_then_else(&mut self) {
        self.generator
            .add_instruction(Instruction::Label("if_then_else".to_string()));

        self.generator.add_instruction(Instruction::Nop);
    }

    // This uses all argument and temp registers
    pub fn add_signed_integers(&mut self) -> Freed {
        argument!(first_sign = self.first_arg);
        argument!(second_sign = self.second_arg);
        argument!(first_magnitude = self.third_arg);
        argument!(second_magnitude = self.fourth_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);

        self.generator
            .add_instruction(Instruction::label("add_signed_integers".to_string()));

        // If the signs are not equal we subtract the larger magnitude from the smaller magnitude
        // Then we use the largers sign. Except if equal magnitudes then we set the value to 0 and the sign to positive
        self.generator.add_instruction(Instruction::bne(
            &first_sign,
            &second_sign,
            "sub_signed_integers".to_string(),
        ));

        constnt!(first_magnitude_length = self.first_temp);
        self.generator.add_instruction(Instruction::lw(
            &mut first_magnitude_length,
            0,
            &first_magnitude,
        ));

        constnt!(second_magnitude_length = self.second_temp);
        self.generator.add_instruction(Instruction::lw(
            &mut second_magnitude_length,
            0,
            &second_magnitude,
        ));

        var!(max_magnitude_length = self.third_temp);
        constnt!(bigger_magnitude = self.fifth_arg);
        constnt!(smaller_magnitude = self.sixth_arg);
        constnt!(smaller_length = self.seventh_arg);
        self.generator.add_instruction(Instruction::bltu(
            &first_magnitude_length,
            &second_magnitude_length,
            "first_smaller".to_string(),
        ));

        {
            let mut max_magnitude_length = max_magnitude_length.clone();
            let mut bigger_magnitude = bigger_magnitude.clone();
            let mut smaller_magnitude = smaller_magnitude.clone();
            let mut smaller_length = smaller_length.clone();
            self.generator.add_instruction(Instruction::mv(
                &mut max_magnitude_length,
                &first_magnitude_length,
            ));

            self.generator
                .add_instruction(Instruction::mv(&mut bigger_magnitude, &first_magnitude));

            self.generator
                .add_instruction(Instruction::mv(&mut smaller_magnitude, &second_magnitude));

            self.generator.add_instruction(Instruction::mv(
                &mut smaller_length,
                &second_magnitude_length,
            ));

            self.generator
                .add_instruction(Instruction::j("allocate_heap".to_string()));
        }

        {
            self.generator
                .add_instruction(Instruction::label("first_smaller".to_string()));

            self.generator.add_instruction(Instruction::mv(
                &mut max_magnitude_length,
                &second_magnitude_length,
            ));

            self.generator
                .add_instruction(Instruction::mv(&mut bigger_magnitude, &second_magnitude));

            self.generator
                .add_instruction(Instruction::mv(&mut smaller_magnitude, &first_magnitude));

            self.generator.add_instruction(Instruction::mv(
                &mut smaller_length,
                &first_magnitude_length,
            ));

            // No need for jump since next instruction is allocate_heap
        }

        self.generator
            .add_instruction(Instruction::label("allocate_heap".to_string()));

        var!(max_heap_allocation = self.fourth_temp);
        self.generator.add_instruction(Instruction::addi(
            &mut max_heap_allocation,
            &max_magnitude_length,
            1,
        ));

        self.generator.add_instruction(Instruction::slli(
            &mut max_heap_allocation.clone(),
            &max_heap_allocation,
            2,
        ));

        // Add fixed constant for creating a constant integer value on the heap
        // + 1 for sign + 4 for magnitude length + (4 * (largest magnitude length + 1))
        self.generator.add_instruction(Instruction::addi(
            &mut max_heap_allocation.clone(),
            &max_heap_allocation,
            5,
        ));

        constnt_overwrite!(ret = first_sign);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        constnt!(callback = self.eighth_arg);
        // This function does not modify temps and only changes heap and s3
        // and eighth arg
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "allocate_integer_type".to_string(),
        ));

        var!(value_builder = self.fifth_temp);
        self.generator
            .add_instruction(Instruction::mv(&mut value_builder, &heap));

        // Add maximum heap value needed possibly and reclaim later after addition
        self.generator.add_instruction(Instruction::add(
            &mut heap.clone(),
            &heap,
            &max_heap_allocation,
        ));

        // Store second sign. In this case the signs are the same
        self.generator
            .add_instruction(Instruction::sb(&second_sign, 0, &value_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut value_builder.clone(),
            &value_builder,
            1,
        ));

        constnt_overwrite!(length_pointer = max_heap_allocation);
        self.generator
            .add_instruction(Instruction::mv(&mut length_pointer, &value_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut value_builder.clone(),
            &value_builder,
            4,
        ));

        var_overwrite!(current_word_index = first_magnitude_length);
        self.generator
            .add_instruction(Instruction::li(&mut current_word_index, 0));

        // Overwrite second_magnitude_length
        // First carry is always 0
        var_overwrite!(carry = second_magnitude_length);
        self.generator
            .add_instruction(Instruction::li(&mut carry, 0));

        var!(bigger_arg_word_location = self.sixth_temp);
        self.generator.add_instruction(Instruction::addi(
            &mut bigger_arg_word_location,
            &bigger_magnitude,
            4,
        ));

        var!(smaller_arg_word_location = self.seventh_temp);
        self.generator.add_instruction(Instruction::addi(
            &mut smaller_arg_word_location,
            &smaller_magnitude,
            4,
        ));

        {
            self.generator
                .add_instruction(Instruction::label("add_words".to_string()));

            self.generator.add_instruction(Instruction::beq(
                &current_word_index,
                &max_magnitude_length,
                "finalize_int_value".to_string(),
            ));

            constnt_overwrite!(bigger = bigger_magnitude);
            self.generator.add_instruction(Instruction::lw(
                &mut bigger,
                0,
                &bigger_arg_word_location,
            ));

            constnt_overwrite!(smaller = smaller_magnitude);
            self.generator.add_instruction(Instruction::bge(
                &current_word_index,
                &smaller_length,
                "smaller_length".to_string(),
            ));

            {
                let mut smaller = smaller.clone();
                self.generator.add_instruction(Instruction::lw(
                    &mut smaller,
                    0,
                    &smaller_arg_word_location,
                ));

                self.generator
                    .add_instruction(Instruction::j("result".to_string()));
            }
            {
                self.generator
                    .add_instruction(Instruction::label("smaller_length".to_string()));

                self.generator
                    .add_instruction(Instruction::li(&mut smaller, 0));
            }

            self.generator
                .add_instruction(Instruction::label("result".to_string()));

            var_overwrite!(result = first_magnitude);
            self.generator
                .add_instruction(Instruction::add(&mut result, &bigger, &smaller));

            // Add previous carry
            self.generator
                .add_instruction(Instruction::add(&mut result.clone(), &result, &carry));

            // Set carry if we overflowed
            self.generator
                .add_instruction(Instruction::sltu(&mut carry, &result, &bigger));

            self.generator
                .add_instruction(Instruction::sw(&result, 0, &value_builder));

            self.generator.add_instruction(Instruction::addi(
                &mut value_builder.clone(),
                &value_builder,
                4,
            ));

            self.generator.add_instruction(Instruction::addi(
                &mut bigger_arg_word_location.clone(),
                &bigger_arg_word_location,
                4,
            ));

            self.generator.add_instruction(Instruction::addi(
                &mut smaller_arg_word_location.clone(),
                &smaller_arg_word_location,
                4,
            ));

            self.generator.add_instruction(Instruction::addi(
                &mut current_word_index.clone(),
                &current_word_index,
                1,
            ));

            self.generator
                .add_instruction(Instruction::j("add_words".to_string()));
        }

        self.generator
            .add_instruction(Instruction::label("finalize_int_value".to_string()));

        self.generator.add_instruction(Instruction::bne(
            &carry,
            &zero,
            "handle_final_carry".to_string(),
        ));
        {
            self.generator.add_instruction(Instruction::sw(
                &max_magnitude_length,
                0,
                &length_pointer,
            ));

            // Reclaim 4 bytes since no carry is used
            self.generator
                .add_instruction(Instruction::addi(&mut heap.clone(), &heap, -4));

            // ret is set earlier and never overwritten
            assert!(ret.register.is_some());
            self.generator
                .add_instruction(Instruction::j("return".to_string()));
        };

        {
            self.generator
                .add_instruction(Instruction::label("handle_final_carry".to_string()));

            // handle carry increasing word length
            self.generator.add_instruction(Instruction::addi(
                &mut max_magnitude_length.clone(),
                &max_magnitude_length,
                1,
            ));

            self.generator.add_instruction(Instruction::sw(
                &max_magnitude_length,
                0,
                &length_pointer,
            ));

            self.generator
                .add_instruction(Instruction::sw(&carry, 0, &value_builder));

            // ret is set earlier and never overwritten
            self.generator
                .add_instruction(Instruction::j("return".to_string()));
        };

        self.register_map.free_all()
    }

    // returns
    // equality flag(1:0)
    // greater magnitude
    // lesser magnitude
    // greater magnitude sign(0 if equal)
    pub fn compare_magnitude(&mut self) -> Freed {
        argument!(first_sign = self.first_arg);
        argument!(second_sign = self.second_arg);
        argument!(first_value = self.third_arg);
        argument!(second_value = self.fourth_arg);
        argument!(callback = self.fifth_arg);
        argument!(zero = self.zero);

        self.generator
            .add_instruction(Instruction::label("compare_magnitude".to_string()));

        constnt!(first_magnitude_len = self.first_temp);
        self.generator
            .add_instruction(Instruction::lw(&mut first_magnitude_len, 0, &first_value));

        constnt!(second_magnitude_len = self.second_temp);
        self.generator.add_instruction(Instruction::lw(
            &mut second_magnitude_len,
            0,
            &second_value,
        ));

        // Check absolute lengths of the magnitudes
        var!(comparison_check = self.third_temp);
        self.generator.add_instruction(Instruction::sltu(
            &mut comparison_check,
            &first_magnitude_len,
            &second_magnitude_len,
        ));

        // If lengths are unequal we don't need to compare the
        // individual words in the magnitudes
        self.generator.add_instruction(Instruction::bne(
            &first_magnitude_len,
            &second_magnitude_len,
            "unequal_values".to_string(),
        ));

        create_block!(self, {
            // Magnitudes are the same length
            var!(magnitude_len = self.fourth_temp);
            self.generator
                .add_instruction(Instruction::mv(&mut magnitude_len, &first_magnitude_len));

            var_overwrite!(word_offset = first_magnitude_len);
            self.generator
                .add_instruction(Instruction::slli(&mut word_offset, &magnitude_len, 2));

            {
                // Loop to compare values
                // either they are equal when we have checked each
                // word in the length or we find a word where there is
                // a difference. We do a comparison check and then branch
                // to unequal_values in that case
                self.generator
                    .add_instruction(Instruction::label("compare_words".to_string()));

                self.generator.add_instruction(Instruction::beq(
                    &magnitude_len,
                    &zero,
                    "equal_values".to_string(),
                ));

                constnt!(first_arg_offset = self.fifth_temp);
                self.generator.add_instruction(Instruction::add(
                    &mut first_arg_offset,
                    &word_offset,
                    &first_value,
                ));

                // Overwrite word_offset
                constnt_overwrite!(second_arg_offset = second_magnitude_len);
                self.generator.add_instruction(Instruction::add(
                    &mut second_arg_offset,
                    &word_offset,
                    &second_value,
                ));

                constnt!(first_arg_values = self.sixth_temp);
                self.generator.add_instruction(Instruction::lw(
                    &mut first_arg_values,
                    0,
                    &first_arg_offset,
                ));

                constnt!(second_arg_values = self.seventh_temp);
                self.generator.add_instruction(Instruction::lw(
                    &mut second_arg_values,
                    0,
                    &second_arg_offset,
                ));

                self.generator.add_instruction(Instruction::sltu(
                    &mut comparison_check,
                    &first_arg_values,
                    &second_arg_values,
                ));

                self.generator.add_instruction(Instruction::bne(
                    &first_arg_values,
                    &second_arg_values,
                    "unequal_values".to_string(),
                ));

                self.generator.add_instruction(Instruction::addi(
                    &mut magnitude_len.clone(),
                    &magnitude_len,
                    -1,
                ));

                self.generator.add_instruction(Instruction::addi(
                    &mut word_offset.clone(),
                    &word_offset,
                    -4,
                ));

                self.generator
                    .add_instruction(Instruction::j("compare_words".to_string()));
            }

            // Terminating case that ends in JALR
            self.generator
                .add_instruction(Instruction::label("equal_values".to_string()));

            let mut zero = zero.clone();
            let mut first_sign = first_sign.clone();
            let mut second_sign = second_sign.clone();
            let mut first_value = first_value.clone();
            let mut second_value = second_value.clone();

            constnt_overwrite!(equality = first_sign);
            self.generator
                .add_instruction(Instruction::li(&mut equality, 1));

            constnt_overwrite!(greater_magnitude = second_sign);
            self.generator
                .add_instruction(Instruction::mv(&mut greater_magnitude, &first_value));

            constnt_overwrite!(lesser_magnitude = first_value);
            self.generator
                .add_instruction(Instruction::mv(&mut lesser_magnitude, &second_value));

            constnt_overwrite!(sign = second_value);
            self.generator
                .add_instruction(Instruction::li(&mut sign, 0));

            constnt_overwrite!(discard = zero);
            self.generator
                .add_instruction(Instruction::jalr(&mut discard, &callback, 0));
        });
        {
            self.generator
                .add_instruction(Instruction::label("unequal_values".to_string()));

            self.generator.add_instruction(Instruction::bne(
                &comparison_check,
                &zero,
                "first_value_smaller".to_string(),
            ));

            create_block!(self, {
                constnt!(first_sign_temp = self.fourth_temp);
                self.generator
                    .add_instruction(Instruction::mv(&mut first_sign_temp, &first_sign));

                let mut zero = zero.clone();
                let mut first_sign = first_sign.clone();
                let mut second_sign = second_sign.clone();
                let mut first_value = first_value.clone();
                let mut second_value = second_value.clone();

                // Overwrite first_sign
                constnt_overwrite!(equality = first_sign);
                // Values are not equal so return 0
                self.generator
                    .add_instruction(Instruction::mv(&mut equality, &zero));

                // Overwrite second_sign
                constnt_overwrite!(greater_value = second_sign);
                self.generator
                    .add_instruction(Instruction::mv(&mut greater_value, &first_value));

                // Overwrite first_value
                constnt_overwrite!(lesser_value = first_value);
                self.generator
                    .add_instruction(Instruction::mv(&mut lesser_value, &second_value));

                // Overwrite second_value
                constnt_overwrite!(greater_sign = second_value);
                self.generator
                    .add_instruction(Instruction::mv(&mut greater_sign, &first_sign_temp));

                constnt_overwrite!(discard = zero);
                self.generator
                    .add_instruction(Instruction::jalr(&mut discard, &callback, 0));
            });
            {
                self.generator
                    .add_instruction(Instruction::label("first_value_smaller".to_string()));

                constnt!(second_sign_temp = self.fourth_temp);
                self.generator
                    .add_instruction(Instruction::mv(&mut second_sign_temp, &second_sign));

                constnt_overwrite!(equality = first_sign);
                // Values are not equal so return 0
                self.generator
                    .add_instruction(Instruction::mv(&mut equality, &zero));

                constnt_overwrite!(greater_value = second_sign);
                self.generator
                    .add_instruction(Instruction::mv(&mut greater_value, &second_value));

                constnt_arg_overwrite!(lesser_value = first_value);

                constnt_overwrite!(greater_sign = second_value);
                self.generator
                    .add_instruction(Instruction::mv(&mut greater_sign, &second_sign_temp));

                constnt_overwrite!(discard = zero);
                self.generator
                    .add_instruction(Instruction::jalr(&mut discard, &callback, 0));
            }
        }

        self.register_map.free_all()
    }

    pub fn sub_signed_integers(&mut self) -> Freed {
        argument!(first_sign = self.first_arg);
        argument!(second_sign = self.second_arg);
        argument!(first_value = self.third_arg);
        argument!(second_value = self.fourth_arg);
        var_argument!(heap = self.heap);
        argument!(zero = self.zero);

        self.generator
            .add_instruction(Instruction::label("sub_signed_integers".to_string()));

        constnt!(callback = self.fifth_arg);
        self.generator.add_instruction(Instruction::jal(
            &mut callback,
            "compare_magnitude".to_string(),
        ));

        // Compare magnitude overwrites these arguments
        constnt_arg_overwrite!(equality = first_sign);
        constnt_arg_overwrite!(greater_value = second_sign);
        constnt_arg_overwrite!(lesser_value = first_value);
        constnt_arg_overwrite!(greater_sign = second_value);

        constnt!(equality_temp = self.first_temp);
        self.generator
            .add_instruction(Instruction::mv(&mut equality_temp, &equality));

        // Overwrite equality
        constnt_overwrite!(ret = equality);
        self.generator
            .add_instruction(Instruction::mv(&mut ret, &heap));

        constnt!(greater_magnitude_len = self.second_temp);
        self.generator.add_instruction(Instruction::lw(
            &mut greater_magnitude_len,
            0,
            &greater_value,
        ));

        constnt!(lesser_magnitude_len = self.third_temp);
        self.generator.add_instruction(Instruction::lw(
            &mut lesser_magnitude_len,
            0,
            &lesser_value,
        ));

        var!(max_heap_allocation = self.fourth_temp);
        self.generator.add_instruction(Instruction::slli(
            &mut max_heap_allocation,
            &greater_magnitude_len,
            2,
        ));

        self.generator.add_instruction(Instruction::addi(
            &mut max_heap_allocation.clone(),
            &max_heap_allocation,
            5,
        ));

        constnt!(callback_alloc = self.eighth_arg);
        // This function does not modify temps and only changes heap and s3
        // and eighth arg
        self.generator.add_instruction(Instruction::jal(
            &mut callback_alloc,
            "allocate_integer_type".to_string(),
        ));

        var!(value_builder = self.fifth_temp);
        self.generator
            .add_instruction(Instruction::mv(&mut value_builder, &heap));

        // Add maximum heap value needed possibly and reclaim later after addition
        self.generator.add_instruction(Instruction::add(
            &mut heap.clone(),
            &heap,
            &max_heap_allocation,
        ));

        // Store greater sign
        self.generator
            .add_instruction(Instruction::sb(&greater_sign, 0, &value_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut value_builder.clone(),
            &value_builder,
            1,
        ));

        constnt_overwrite!(length_pointer = max_heap_allocation);
        self.generator
            .add_instruction(Instruction::mv(&mut length_pointer, &value_builder));

        self.generator.add_instruction(Instruction::addi(
            &mut value_builder.clone(),
            &value_builder,
            4,
        ));

        self.generator.add_instruction(Instruction::bne(
            &equality_temp,
            &zero,
            "equal_value_subtraction".to_string(),
        ));
        {
            let mut equality_temp = equality_temp.clone();

            var_overwrite!(current_word_index = equality_temp);
            self.generator
                .add_instruction(Instruction::li(&mut current_word_index, 0));

            // First carry is always 0
            var!(carry = self.sixth_temp);
            self.generator
                .add_instruction(Instruction::li(&mut carry, 0));

            // Overwrite first_magnitude
            var!(greater_arg_word_location = self.seventh_temp);
            self.generator.add_instruction(Instruction::addi(
                &mut greater_arg_word_location,
                &greater_value,
                4,
            ));

            // Overwrite second_magnitude
            var_overwrite!(lesser_arg_word_location = callback);
            self.generator.add_instruction(Instruction::addi(
                &mut lesser_arg_word_location,
                &lesser_value,
                4,
            ));

            var_overwrite!(reclaim_heap_amount = greater_value);
            self.generator
                .add_instruction(Instruction::li(&mut reclaim_heap_amount, 0));

            var!(final_length = self.sixth_arg);
            self.generator
                .add_instruction(Instruction::li(&mut final_length, 0));

            {
                self.generator
                    .add_instruction(Instruction::label("sub_words".to_string()));

                self.generator.add_instruction(Instruction::beq(
                    &current_word_index,
                    &greater_magnitude_len,
                    "finalize_sub_int_value".to_string(),
                ));

                constnt_overwrite!(arg_word_greater = callback_alloc);
                self.generator.add_instruction(Instruction::lw(
                    &mut arg_word_greater,
                    0,
                    &greater_arg_word_location,
                ));

                constnt_overwrite!(arg_word_smaller = lesser_value);
                self.generator.add_instruction(Instruction::bge(
                    &current_word_index,
                    &lesser_magnitude_len,
                    "lesser".to_string(),
                ));

                {
                    let mut arg_word_smaller = arg_word_smaller.clone();
                    self.generator.add_instruction(Instruction::lw(
                        &mut arg_word_smaller,
                        0,
                        &lesser_arg_word_location,
                    ));

                    self.generator
                        .add_instruction(Instruction::j("sub_result".to_string()));
                }
                {
                    self.generator
                        .add_instruction(Instruction::label("lesser".to_string()));

                    self.generator
                        .add_instruction(Instruction::li(&mut arg_word_smaller, 0));
                }

                self.generator
                    .add_instruction(Instruction::label("sub_result".to_string()));

                // Overwrite arg_word_smaller
                var!(result = self.seventh_arg);
                self.generator.add_instruction(Instruction::sub(
                    &mut result,
                    &arg_word_greater,
                    &arg_word_smaller,
                ));

                // Overwrite greater_sign
                constnt_overwrite!(first_carry_check = greater_sign);
                // Check result is more than first arg thus needing a carry
                self.generator.add_instruction(Instruction::sltu(
                    &mut first_carry_check,
                    &arg_word_greater,
                    &result,
                ));

                // Sub previous carry
                self.generator.add_instruction(Instruction::sub(
                    &mut result.clone(),
                    &result,
                    &carry,
                ));

                // Set carry if we overflowed
                self.generator.add_instruction(Instruction::sltu(
                    &mut carry.clone(),
                    &arg_word_greater,
                    &result,
                ));

                // OR the carry checks
                self.generator.add_instruction(Instruction::or(
                    &mut carry.clone(),
                    &carry,
                    &first_carry_check,
                ));

                self.generator
                    .add_instruction(Instruction::sw(&result, 0, &value_builder));

                self.generator.add_instruction(Instruction::addi(
                    &mut value_builder.clone(),
                    &value_builder,
                    4,
                ));

                self.generator.add_instruction(Instruction::addi(
                    &mut greater_arg_word_location.clone(),
                    &greater_arg_word_location,
                    4,
                ));

                self.generator.add_instruction(Instruction::addi(
                    &mut lesser_arg_word_location.clone(),
                    &lesser_arg_word_location,
                    4,
                ));

                self.generator.add_instruction(Instruction::addi(
                    &mut current_word_index.clone(),
                    &current_word_index,
                    1,
                ));

                self.generator.add_instruction(Instruction::beq(
                    &zero,
                    &result,
                    "can_reclaim_heap_word".to_string(),
                ));

                {
                    self.generator.add_instruction(Instruction::addi(
                        &mut final_length.clone(),
                        &final_length,
                        1,
                    ));

                    self.generator.add_instruction(Instruction::add(
                        &mut final_length.clone(),
                        &final_length,
                        &reclaim_heap_amount,
                    ));

                    self.generator
                        .add_instruction(Instruction::li(&mut reclaim_heap_amount, 0));

                    self.generator
                        .add_instruction(Instruction::j("sub_words".to_string()));
                }
                {
                    self.generator
                        .add_instruction(Instruction::label("can_reclaim_heap_word".to_string()));

                    self.generator.add_instruction(Instruction::addi(
                        &mut reclaim_heap_amount.clone(),
                        &reclaim_heap_amount,
                        1,
                    ));

                    self.generator
                        .add_instruction(Instruction::j("sub_words".to_string()));
                }
            }

            self.generator
                .add_instruction(Instruction::label("finalize_sub_int_value".to_string()));

            self.generator
                .add_instruction(Instruction::sw(&final_length, 0, &length_pointer));

            self.generator.add_instruction(Instruction::slli(
                &mut reclaim_heap_amount.clone(),
                &reclaim_heap_amount,
                2,
            ));

            self.generator.add_instruction(Instruction::sub(
                &mut heap.clone(),
                &heap,
                &reclaim_heap_amount,
            ));

            self.generator
                .add_instruction(Instruction::j("return".to_string()));
        }
        {
            self.generator
                .add_instruction(Instruction::label("equal_value_subtraction".to_string()));

            constnt_overwrite!(size = equality_temp);
            self.generator
                .add_instruction(Instruction::li(&mut size, 1));

            self.generator
                .add_instruction(Instruction::sw(&size, 0, &length_pointer));

            self.generator
                .add_instruction(Instruction::sw(&zero, 0, &value_builder));

            // reclaim heap since value is 0 and length is 1
            self.generator
                .add_instruction(Instruction::addi(&mut heap, &value_builder, 4));

            self.generator
                .add_instruction(Instruction::j("return".to_string()));
        }

        self.register_map.free_all()
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
    use risc_v_gen::{emulator::verify_file, Instruction};
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
        cek.allocate_integer_type();
        cek.allocate_bytestring_type();
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
        cek.cons_bytestring();
        cek.slice_bytestring();
        cek.length_bytestring();
        cek.index_bytestring();
        cek.equals_bytestring();
        cek.less_than_bytestring();
        cek.less_than_equals_bytestring();
        cek.sha2_256();
        cek.sha3_256();
        cek.blake2b_256();
        cek.verify_ed25519_signature();
        cek.append_string();
        cek.equals_string();
        cek.encode_utf8();
        cek.decode_utf8();
        cek.if_then_else();
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

        assert_eq!(constant_type, const_tag::INTEGER);

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

        assert_eq!(constant_type, const_tag::INTEGER);

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

        let type_length = section_data[offset_index];

        assert_eq!(type_length, 1);

        let constant_type = section_data[offset_index + 1];

        assert_eq!(constant_type, const_tag::INTEGER);

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

        Command::new("ls").args(["../../linker"]).status().unwrap();

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

        Command::new("ls").args(["."]).status().unwrap();

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

        assert_eq!(constant_type, const_tag::INTEGER);

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

        assert_eq!(constant_type, const_tag::INTEGER);

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

        assert_eq!(boolean, 0)
    }

    #[test]
    fn test_less_equals_numbers() {
        let thing = Cek::default();

        // (apply (lambda x (force x)) (delay (error)))
        let term: Term<Name> = Term::less_than_equals_integer()
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

        gene.save_to_file("test_less_equals_big_int.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_less_equals_big_int.o",
                "test_less_equals_big_int.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_less_equals_big_int.elf",
                "-T",
                "../../linker/link.ld",
                "test_less_equals_big_int.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_less_equals_big_int.elf").unwrap();

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
    fn test_append_bytestring() {
        let thing = Cek::default();

        let term: Term<Name> = Term::append_bytearray()
            .apply(Term::byte_string(vec![255, 255]))
            .apply(Term::byte_string(vec![254, 245]));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_append_bytes.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_append_bytes.o",
                "test_append_bytes.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_append_bytes.elf",
                "-T",
                "../../linker/link.ld",
                "test_append_bytes.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_append_bytes.elf").unwrap();

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

        assert_eq!(constant_type, const_tag::BYTESTRING);

        let byte_length = *section_data[(offset_index + 2)..(offset_index + 6)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(byte_length, 4);

        let value = section_data[(offset_index + 6)..(offset_index + 22)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>();

        assert_eq!(value, vec![255, 255, 254, 245])
    }

    #[test]
    fn test_con_bytestring() {
        let thing = Cek::default();

        let term: Term<Name> = Term::cons_bytearray()
            .apply(Term::integer(20.into()))
            .apply(Term::byte_string(vec![254, 245]));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_cons_bytes.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_cons_bytes.o",
                "test_cons_bytes.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_cons_bytes.elf",
                "-T",
                "../../linker/link.ld",
                "test_cons_bytes.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_cons_bytes.elf").unwrap();

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

        assert_eq!(constant_type, const_tag::BYTESTRING);

        let byte_length = *section_data[(offset_index + 2)..(offset_index + 6)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(byte_length, 3);

        let value = section_data[(offset_index + 6)..(offset_index + 18)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>();

        assert_eq!(value, vec![20, 254, 245])
    }

    #[test]
    fn test_slice_bytestring() {
        let thing = Cek::default();

        let term: Term<Name> = Term::slice_bytearray()
            .apply(Term::integer(1.into()))
            .apply(Term::integer(2.into()))
            .apply(Term::byte_string(vec![251, 251, 254, 254]));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_slice_bytes.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_slice_bytes.o",
                "test_slice_bytes.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_slice_bytes.elf",
                "-T",
                "../../linker/link.ld",
                "test_slice_bytes.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_slice_bytes.elf").unwrap();

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

        assert_eq!(constant_type, const_tag::BYTESTRING);

        let byte_length = *section_data[(offset_index + 2)..(offset_index + 6)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(byte_length, 2);

        let value = section_data[(offset_index + 6)..(offset_index + 14)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>();

        assert_eq!(value, vec![251, 254])
    }

    #[test]
    fn test_bytestring_length() {
        let thing = Cek::default();

        let term: Term<Name> =
            Term::length_of_bytearray().apply(Term::byte_string(vec![251, 251, 254, 254]));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_len_bytes.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_len_bytes.o",
                "test_len_bytes.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_len_bytes.elf",
                "-T",
                "../../linker/link.ld",
                "test_len_bytes.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_len_bytes.elf").unwrap();

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

        assert_eq!(constant_type, const_tag::INTEGER);

        let sign = section_data[offset_index + 2];

        assert_eq!(sign, 0);

        let integer_length = *section_data[(offset_index + 3)..(offset_index + 7)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(integer_length, 1);

        let value = *section_data[(offset_index + 7)..(offset_index + 11)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(value, 4);
    }

    #[test]
    fn test_index_bytestring() {
        let thing = Cek::default();

        let term: Term<Name> = Term::index_bytearray()
            .apply(Term::byte_string(vec![251, 251, 254, 254]))
            .apply(Term::integer(2.into()));

        let term_debruijn: Term<DeBruijn> = term.try_into().unwrap();

        let program: Program<DeBruijn> = Program {
            version: (1, 1, 0),
            term: term_debruijn,
        };

        let riscv_program = serialize(&program, 0x90000000).unwrap();

        let gene = thing.cek_assembly(riscv_program);

        gene.save_to_file("test_index_bytes.s").unwrap();

        Command::new("riscv64-elf-as")
            .args([
                "-march=rv32i",
                "-mabi=ilp32",
                "-o",
                "test_index_bytes.o",
                "test_index_bytes.s",
            ])
            .status()
            .unwrap();

        Command::new("riscv64-elf-ld")
            .args([
                "-m",
                "elf32lriscv",
                "-o",
                "test_index_bytes.elf",
                "-T",
                "../../linker/link.ld",
                "test_index_bytes.o",
            ])
            .status()
            .unwrap();

        let v = verify_file("test_index_bytes.elf").unwrap();

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

        assert_eq!(constant_type, const_tag::INTEGER);

        let sign = section_data[offset_index + 2];

        assert_eq!(sign, 0);

        let integer_length = *section_data[(offset_index + 3)..(offset_index + 7)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(integer_length, 1);

        let value = *section_data[(offset_index + 7)..(offset_index + 11)]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>()
            .first()
            .unwrap();

        assert_eq!(value, 254);
    }

    #[test]
    fn blocks() {
        let mut thing = Cek::default();

        argument!(yes = thing.first_arg);

        create_block!(thing, {
            var!(no = thing.second_arg);
            constnt!(sure = thing.third_arg);

            thing
                .generator
                .add_instruction(Instruction::li(&mut no, 42));

            thing
                .generator
                .add_instruction(Instruction::li(&mut sure, 421));

            thing
                .generator
                .add_instruction(Instruction::mv(&mut no, &yes));
        });
        println!("THING {:#?}", thing.generator);
    }
}
