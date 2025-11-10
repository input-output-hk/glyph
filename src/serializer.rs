use std::convert::TryFrom;
use std::io::Write;
use std::ops::Deref;
use std::rc::Rc;

use byteorder::{LittleEndian, WriteBytesExt};
use constants::{bool_val, const_tag, data_tag, term_tag};
use num_bigint::{BigInt as NumBigInt, BigUint, Sign};
use uplc::BigInt;
use uplc::PlutusData;
use uplc::ast::{Constant, DeBruijn, Program, Term, Type};
use uplc::builtins::DefaultFunction;
use uplc::machine::runtime::Compressable;

use std::io;
use thiserror::Error;

pub mod constants;

const CONST_TYPE_INTEGER: u32 = 0;
const CONST_TYPE_BYTESTRING: u32 = 1;
const CONST_TYPE_STRING: u32 = 2;
const CONST_TYPE_UNIT: u32 = 3;
const CONST_TYPE_BOOL: u32 = 4;
const CONST_TYPE_LIST: u32 = 5;
const CONST_TYPE_PAIR: u32 = 6;
const CONST_TYPE_DATA: u32 = 7;
const CONST_TYPE_BLS12_381_G1_ELEMENT: u32 = 8;
const CONST_TYPE_BLS12_381_G2_ELEMENT: u32 = 9;
const CONST_TYPE_BLS12_381_MLRESULT: u32 = 10;
const LARGE_CONSTR_TAG_FLAG: u32 = 0x8000_0000;

/// Error type for serialization failures
#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("IO error: {0}")]
    IoError(#[from] io::Error),

    #[error("Invalid term type: {0}")]
    InvalidTermType(String),

    #[error("Integer too large to serialize: {0}")]
    IntegerTooLarge(String),

    #[error("String too large to serialize: {0}")]
    StringTooLarge(String),

    #[error("ByteString too large to serialize: {0}")]
    ByteStringTooLarge(String),

    #[error("Data structure too complex: {0}")]
    DataTooComplex(String),

    #[error("Memory layout error: {0}")]
    MemoryLayoutError(String),
}

/// Result type for serialization operations
pub type Result<T> = std::result::Result<T, SerializationError>;

/// Serialize a UPLC program to a binary format suitable for the RISC-V CEK machine.
///
/// Each term is serialized according to its type (see `constants.rs` for tag values):
///
/// - Variable (0x00):
///   - DeBruijn index (4 bytes)
///
/// - Lambda (0x01):
///   - Followed by body term (0 bytes)
///
/// - Apply (0x02):
///   - Argument reference (4 bytes)
///   - Followed by function term (0 bytes)
///
/// - Force (0x03):
///   - Followed by body term (0 bytes)
///
/// - Delay (0x04):
///  - Followed by body term (0 bytes)
///
/// - Constant (0x05):
///   - Followed by constant encoding (0 bytes)
///
/// - Builtin (0x06):
///   - Builtin function ID (1 byte)
///
/// - Error (0x07):
///   - No additional data
///
/// - Constructor (0x08):
///   - Tag (2 bytes)
///   - Field count (4 bytes)
///   - x Field references (4*x bytes)
///
/// - Case (0x09):
///   - Match term reference (4 bytes)
///   - Branch count (4 bytes)
///   - x Branch references (4*x bytes)
///
/// # Arguments
///
/// * `program` - The UPLC program to serialize
///
/// # Returns
///
/// A `Result` containing the serialized program as a `Vec<u8>` or a `SerializationError`
pub fn serialize_program(program: &Program<DeBruijn>) -> Result<Vec<u8>> {
    serialize(program, 0, true)
}

/// Parse a UPLC file and serialize it to binary format
///
/// This is a convenience function that parses a UPLC program from text format
/// and then serializes it to the binary format for the RISC-V CEK machine.
///
/// # Arguments
///
/// * `uplc_text` - The UPLC program text
///
/// # Returns
///
/// A `Result` containing the serialized program as a `Vec<u8>` or a `SerializationError`
pub fn parse_and_serialize(uplc_text: &str) -> Result<Vec<u8>> {
    let program = uplc::parser::program(uplc_text)
        .map_err(|e| SerializationError::InvalidTermType(format!("Parse error: {e}")))?
        .to_debruijn()
        .map_err(|e| {
            SerializationError::InvalidTermType(format!("DeBruijn conversion error: {e}"))
        })?;

    serialize_program(&program)
}

/// Deserialize binary format back to a UPLC program (placeholder for future implementation)
pub fn deserialize(_binary: &[u8]) -> Result<Program<DeBruijn>> {
    unimplemented!("Deserialization is not yet implemented")
}

/// Serialize the program to a binary format
pub fn serialize(
    program: &Program<DeBruijn>,
    mut preceeding_byte_size: u32,
    space_for_input: bool,
) -> Result<Vec<u8>> {
    // Now serialize the root term
    let mut x: Vec<u8> = Vec::new();

    if space_for_input {
        x.write_u32::<LittleEndian>(term_tag::APPLY)?;

        // the input argument will be inserted by the emulator into .bss section aka 0xA0000000
        x.write_u32::<LittleEndian>(0xA0000000)?;

        preceeding_byte_size += 8;
    }

    let serialized_bytes = serialize_term(preceeding_byte_size, &program.term)?;

    x.write_all(&serialized_bytes)?;

    // Ensure the final bytestring is divisible by 4 in length
    let padding_size = (4 - (x.len() % 4)) % 4;
    if padding_size > 0 {
        x.write_all(&vec![0; padding_size])?;
    }

    // Return the serialized program
    Ok(x)
}

/// Serialize a term and return its address
fn serialize_term(preceeding_byte_size: u32, term: &Term<DeBruijn>) -> Result<Vec<u8>> {
    // Serialize the term based on its type
    match term {
        Term::Var(index) => serialize_var(index.inner()),
        Term::Lambda {
            parameter_name: _,
            body,
        } => serialize_lambda(preceeding_byte_size, body),
        Term::Apply { function, argument } => {
            serialize_apply(preceeding_byte_size, function, argument)
        }
        Term::Force(term) => serialize_force(preceeding_byte_size, term),
        Term::Delay(term) => serialize_delay(preceeding_byte_size, term),
        Term::Constant(constant) => serialize_constant(preceeding_byte_size, constant),
        Term::Builtin(builtin) => serialize_builtin(*builtin),
        Term::Error => serialize_error(),
        Term::Constr { tag, fields } => serialize_constructor(preceeding_byte_size, *tag, fields),
        Term::Case { constr, branches } => serialize_case(preceeding_byte_size, constr, branches),
    }
}

/// Serialize a variable term
fn serialize_var(index: usize) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte
    x.write_u32::<LittleEndian>(term_tag::VARIABLE)?;

    // DeBruijn index (4 bytes, little-endian)
    x.write_u32::<LittleEndian>(index as u32)?;

    Ok(x)
}

/// Serialize a lambda term
fn serialize_lambda(preceeding_byte_size: u32, body: &Rc<Term<DeBruijn>>) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte
    x.write_u32::<LittleEndian>(term_tag::LAMBDA)?;

    // Serialize the body (recursively)
    let body_ser = serialize_term(preceeding_byte_size + 4, body)?;

    // Write body address
    x.write_all(&body_ser)?;

    Ok(x)
}

/// Serialize an apply term
fn serialize_apply(
    preceeding_byte_size: u32,
    function: &Rc<Term<DeBruijn>>,
    argument: &Rc<Term<DeBruijn>>,
) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte
    x.write_u32::<LittleEndian>(term_tag::APPLY)?;

    // Serialize the function and argument (recursively)
    let function_pointer = preceeding_byte_size + 8;
    let function_ser = serialize_term(function_pointer, function).unwrap();
    let argument_pointer = function_pointer + function_ser.len() as u32;
    let argument_ser = serialize_term(argument_pointer, argument).unwrap();

    // We need to provide the size of each term before writing the terms
    x.write_u32::<LittleEndian>(argument_pointer)?;

    // Write function and argument addresses
    let _ = x.write_all(&function_ser);
    let _ = x.write_all(&argument_ser);

    Ok(x)
}

/// Serialize a force term
fn serialize_force(preceeding_byte_size: u32, term: &Rc<Term<DeBruijn>>) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte
    x.write_u32::<LittleEndian>(term_tag::FORCE)?;

    // Serialize the term being forced (recursively)
    // preceeding_byte_size + 1 for the force tag
    let term_ser = serialize_term(preceeding_byte_size + 4, term)?;

    // Write term address
    x.write_all(&term_ser)?;

    Ok(x)
}

/// Serialize a delay term
fn serialize_delay(preceeding_byte_size: u32, term: &Rc<Term<DeBruijn>>) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte
    x.write_u32::<LittleEndian>(term_tag::DELAY)?;

    // Serialize the term being delayed (recursively)
    // preceeding_byte_size + 1 for the delay tag
    let term_ser = serialize_term(preceeding_byte_size + 4, term)?;

    // Write term address
    x.write_all(&term_ser)?;

    Ok(x)
}

/// Serialize a constant term
fn serialize_constant(preceeding_byte_size: u32, constant: &Rc<Constant>) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte for constant
    x.write_u32::<LittleEndian>(term_tag::CONSTANT)?;

    // Serialize the constant based on its type
    let serialized_data = serialize_constant_body(preceeding_byte_size + 4, constant.deref())?;

    // Write serialized data to our buffer
    x.write_all(&serialized_data)?;

    Ok(x)
}

fn serialize_constant_body(preceeding_byte_size: u32, constant: &Constant) -> Result<Vec<u8>> {
    match constant {
        Constant::Integer(int) => serialize_integer_constant(preceeding_byte_size, int),
        Constant::ByteString(bytes) => serialize_bytestring_constant(preceeding_byte_size, bytes),
        Constant::String(s) => serialize_string_constant(preceeding_byte_size, s.as_str()),
        Constant::Unit => serialize_unit_constant(preceeding_byte_size),
        Constant::Bool(b) => serialize_bool_constant(preceeding_byte_size, *b),
        Constant::Data(data) => serialize_data_constant(preceeding_byte_size, data),
        Constant::ProtoList(ty, elements) => {
            serialize_protolist_constant(preceeding_byte_size, ty, elements)
        }
        Constant::ProtoPair(fst_ty, snd_ty, fst_value, snd_value) => {
            serialize_protopair_constant(preceeding_byte_size, fst_ty, snd_ty, fst_value, snd_value)
        }
        Constant::Bls12_381G1Element(point) => {
            serialize_bls12_381_g1_constant(preceeding_byte_size, point.as_ref())
        }
        Constant::Bls12_381G2Element(point) => {
            serialize_bls12_381_g2_constant(preceeding_byte_size, point.as_ref())
        }
        Constant::Bls12_381MlResult(fp12) => {
            serialize_bls12_381_mlresult_constant(preceeding_byte_size, fp12.as_ref())
        }
        other => Err(SerializationError::InvalidTermType(format!(
            "Unsupported constant type: {other:?}",
        ))),
    }
}

/// Serialize an integer constant
fn serialize_integer_constant(
    preceeding_byte_size: u32,
    int: &num_bigint::BigInt,
) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    let mut int_type: Vec<u8> = Vec::new();

    // BigInt (variable size)
    let (sign, bytes) = int.to_u32_digits();

    // Write sign byte
    let sign_byte = if sign == num_bigint::Sign::Minus {
        1
    } else {
        0
    };

    x.write_u32::<LittleEndian>(sign_byte)?;

    x.write_u32::<LittleEndian>(bytes.len().try_into().unwrap())?;

    // Write the magnitude bytes
    x.write_all(
        &bytes
            .into_iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;

    int_type.write_u32::<LittleEndian>(1)?;

    int_type.write_u32::<LittleEndian>(preceeding_byte_size + x.len() as u32 + 8)?;

    x.write_u32::<LittleEndian>(const_tag::INTEGER)?;

    int_type.extend(x);

    Ok(int_type)
}

/// Serialize a bytestring constant
///
/// ByteStrings are stored in unpacked format: one byte value per u32 word.
/// This matches the runtime representation used by bytestring operations.
fn serialize_bytestring_constant(preceeding_byte_size: u32, bytes: &[u8]) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    let mut bytes_type: Vec<u8> = Vec::new();
    // Check size
    if bytes.len() > (u32::MAX as usize) {
        return Err(SerializationError::ByteStringTooLarge(format!(
            "ByteString too large: {} bytes",
            bytes.len()
        )));
    }

    // Store the byte count
    let byte_count: u32 = bytes.len() as u32;
    x.write_u32::<LittleEndian>(byte_count)?;

    // Write each byte as a separate u32 word (unpacked format)
    for b in bytes.iter() {
        x.write_u32::<LittleEndian>(*b as u32)?;
    }

    bytes_type.write_u32::<LittleEndian>(1)?;

    bytes_type.write_u32::<LittleEndian>(preceeding_byte_size + x.len() as u32 + 8)?;

    x.write_u32::<LittleEndian>(const_tag::BYTESTRING)?;

    bytes_type.extend(x);

    Ok(bytes_type)
}

fn serialize_bls_bytes(preceeding_byte_size: u32, payload: &[u8], tag: u32) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    let mut header: Vec<u8> = Vec::new();

    for chunk in payload.chunks(4) {
        let mut word: u32 = 0;
        for (i, b) in chunk.iter().enumerate() {
            word |= (*b as u32) << (8 * i as u32);
        }
        x.write_u32::<LittleEndian>(word)?;
    }

    header.write_u32::<LittleEndian>(1)?;
    header.write_u32::<LittleEndian>(preceeding_byte_size + x.len() as u32 + 8)?;

    x.write_u32::<LittleEndian>(tag)?;
    header.extend(x);

    Ok(header)
}

fn serialize_bls12_381_g1_constant(
    preceeding_byte_size: u32,
    point: &blst::blst_p1,
) -> Result<Vec<u8>> {
    let bytes = point.compress();
    serialize_bls_bytes(
        preceeding_byte_size,
        &bytes,
        const_tag::BLS12_381_G1_ELEMENT,
    )
}

fn serialize_bls12_381_g2_constant(
    preceeding_byte_size: u32,
    point: &blst::blst_p2,
) -> Result<Vec<u8>> {
    let bytes = point.compress();
    serialize_bls_bytes(
        preceeding_byte_size,
        &bytes,
        const_tag::BLS12_381_G2_ELEMENT,
    )
}

fn serialize_bls12_381_mlresult_constant(
    preceeding_byte_size: u32,
    fp12: &blst::blst_fp12,
) -> Result<Vec<u8>> {
    let mut bytes = [0u8; 576];
    unsafe {
        blst::blst_bendian_from_fp12(bytes.as_mut_ptr(), fp12);
    }
    serialize_bls_bytes(preceeding_byte_size, &bytes, const_tag::BLS12_381_MLRESULT)
}

/// Serialize a string constant (UTF-8), using the same packed-word layout as ByteString
fn serialize_string_constant(preceeding_byte_size: u32, s: &str) -> Result<Vec<u8>> {
    let bytes = s.as_bytes();
    let mut x: Vec<u8> = Vec::new();
    let mut string_type: Vec<u8> = Vec::new();

    if bytes.len() > (u32::MAX as usize) * 4 {
        return Err(SerializationError::StringTooLarge(format!(
            "String too large: {} bytes",
            bytes.len()
        )));
    }

    let word_count: u32 = if bytes.is_empty() {
        0
    } else {
        bytes.len().div_ceil(4) as u32
    };

    x.write_u32::<LittleEndian>(word_count)?;

    for chunk in bytes.chunks(4) {
        let mut word: u32 = 0;
        for (i, b) in chunk.iter().enumerate() {
            word |= (*b as u32) << (8 * i as u32);
        }
        x.write_u32::<LittleEndian>(word)?;
    }

    string_type.write_u32::<LittleEndian>(1)?;

    string_type.write_u32::<LittleEndian>(preceeding_byte_size + x.len() as u32 + 8)?;

    x.write_u32::<LittleEndian>(const_tag::STRING)?;

    string_type.extend(x);

    Ok(string_type)
}

/// Serialize a unit constant
fn serialize_unit_constant(preceeding_byte_size: u32) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();

    x.write_u32::<LittleEndian>(1)?;

    // type pointer
    x.write_u32::<LittleEndian>(preceeding_byte_size + 8)?;

    // Write constant type tag
    x.write_u32::<LittleEndian>(const_tag::UNIT)?;

    // No additional data for unit
    Ok(x)
}

/// Serialize a boolean constant
fn serialize_bool_constant(preceeding_byte_size: u32, value: bool) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();

    x.write_u32::<LittleEndian>(1)?;

    // type pointer
    x.write_u32::<LittleEndian>(preceeding_byte_size + 12)?;

    // Write boolean value (0x00 for false, 0x01 for true)
    x.write_u32::<LittleEndian>(if value {
        bool_val::TRUE
    } else {
        bool_val::FALSE
    })?;

    // Write constant type tag
    x.write_u32::<LittleEndian>(const_tag::BOOL)?;

    Ok(x)
}

fn serialize_protolist_constant(
    preceeding_byte_size: u32,
    element_type: &Type,
    elements: &[Constant],
) -> Result<Vec<u8>> {
    let mut payload: Vec<u8> = Vec::new();
    payload.write_u32::<LittleEndian>(usize_to_u32(elements.len())?)?;

    let head_ptr_offset = payload.len();
    payload.write_u32::<LittleEndian>(0)?;

    let mut current_payload_len = payload.len();
    let mut element_value_ptrs = Vec::with_capacity(elements.len());

    for element in elements {
        let element_offset = current_payload_len + 8;
        let element_base = add_bytes(preceeding_byte_size, element_offset)?;
        let element_bytes = serialize_constant_body(element_base, element)?;
        let element_value_ptr = add_bytes(element_base, 8)?;

        element_value_ptrs.push(element_value_ptr);
        payload.extend_from_slice(&element_bytes);
        current_payload_len += element_bytes.len();
    }

    let node_section_offset = current_payload_len;
    let mut head_ptr_value = 0;

    for (idx, value_ptr) in element_value_ptrs.iter().enumerate() {
        let node_offset = node_section_offset + idx * 8;
        let node_addr = add_bytes(preceeding_byte_size, 8 + node_offset)?;
        if idx == 0 {
            head_ptr_value = node_addr;
        }

        payload.write_u32::<LittleEndian>(*value_ptr)?;

        let next_ptr = if idx + 1 < element_value_ptrs.len() {
            add_bytes(
                preceeding_byte_size,
                8 + node_section_offset + (idx + 1) * 8,
            )?
        } else {
            0
        };

        payload.write_u32::<LittleEndian>(next_ptr)?;
        current_payload_len += 8;
    }

    if !element_value_ptrs.is_empty() {
        payload[head_ptr_offset..head_ptr_offset + 4]
            .copy_from_slice(&head_ptr_value.to_le_bytes());
    }

    let payload_len = payload.len();
    let mut type_entries = Vec::new();
    type_entries.push(CONST_TYPE_LIST);
    encode_type_descriptor(element_type, &mut type_entries)?;

    let mut body: Vec<u8> = Vec::new();
    body.write_u32::<LittleEndian>(usize_to_u32(type_entries.len())?)?;
    let type_ptr = add_bytes(preceeding_byte_size, 8 + payload_len)?;
    body.write_u32::<LittleEndian>(type_ptr)?;
    body.extend_from_slice(&payload);
    for entry in type_entries {
        body.write_u32::<LittleEndian>(entry)?;
    }

    Ok(body)
}

fn serialize_protopair_constant(
    preceeding_byte_size: u32,
    first_type: &Type,
    second_type: &Type,
    first_value: &Constant,
    second_value: &Constant,
) -> Result<Vec<u8>> {
    // Reserve space for the two field pointers; we'll patch them once the
    // component payload addresses are known.
    let mut payload: Vec<u8> = vec![0; 8];

    let first_base = add_bytes(preceeding_byte_size, 8 + payload.len())?;
    let first_bytes = serialize_constant_body(first_base, first_value)?;
    let first_value_ptr = add_bytes(first_base, 8)?;
    payload.extend_from_slice(&first_bytes);

    let second_base = add_bytes(preceeding_byte_size, 8 + payload.len())?;
    let second_bytes = serialize_constant_body(second_base, second_value)?;
    let second_value_ptr = add_bytes(second_base, 8)?;
    payload.extend_from_slice(&second_bytes);

    payload[0..4].copy_from_slice(&first_value_ptr.to_le_bytes());
    payload[4..8].copy_from_slice(&second_value_ptr.to_le_bytes());

    let mut type_entries = Vec::new();
    type_entries.push(CONST_TYPE_PAIR);
    encode_type_descriptor(first_type, &mut type_entries)?;
    encode_type_descriptor(second_type, &mut type_entries)?;

    let mut body: Vec<u8> = Vec::new();
    body.write_u32::<LittleEndian>(usize_to_u32(type_entries.len())?)?;
    let type_ptr = add_bytes(preceeding_byte_size, 8 + payload.len())?;
    body.write_u32::<LittleEndian>(type_ptr)?;
    body.extend_from_slice(&payload);
    for entry in type_entries {
        body.write_u32::<LittleEndian>(entry)?;
    }

    Ok(body)
}

fn encode_type_descriptor(ty: &Type, out: &mut Vec<u32>) -> Result<()> {
    match ty {
        Type::Bool => out.push(CONST_TYPE_BOOL),
        Type::Integer => out.push(CONST_TYPE_INTEGER),
        Type::String => out.push(CONST_TYPE_STRING),
        Type::ByteString => out.push(CONST_TYPE_BYTESTRING),
        Type::Unit => out.push(CONST_TYPE_UNIT),
        Type::Data => out.push(CONST_TYPE_DATA),
        Type::Bls12_381G1Element => out.push(CONST_TYPE_BLS12_381_G1_ELEMENT),
        Type::Bls12_381G2Element => out.push(CONST_TYPE_BLS12_381_G2_ELEMENT),
        Type::Bls12_381MlResult => out.push(CONST_TYPE_BLS12_381_MLRESULT),
        Type::List(inner) => {
            out.push(CONST_TYPE_LIST);
            encode_type_descriptor(inner.as_ref(), out)?;
        }
        Type::Pair(fst, snd) => {
            out.push(CONST_TYPE_PAIR);
            encode_type_descriptor(fst.as_ref(), out)?;
            encode_type_descriptor(snd.as_ref(), out)?;
        }
    }

    Ok(())
}

fn add_bytes(base: u32, extra: usize) -> Result<u32> {
    let extra = usize_to_u32(extra)?;
    base.checked_add(extra).ok_or_else(|| {
        SerializationError::MemoryLayoutError(
            "Address space overflow while serializing list constant".to_string(),
        )
    })
}

fn usize_to_u32(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        SerializationError::MemoryLayoutError(format!(
            "Value {value} does not fit in 32-bit address space"
        ))
    })
}

/// Serialize a Plutus Data constant
fn serialize_data_constant(_: u32, data: &PlutusData) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    x.write_u32::<LittleEndian>(const_tag::DATA)?;
    let mut payload = encode_plutus_data(data)?;
    let content_size = bytes_to_words(payload.len());
    x.write_u32::<LittleEndian>(content_size)?;
    x.write_all(&payload)?;

    let padding_size = (4 - (payload.len() % 4)) % 4;
    if padding_size > 0 {
        x.write_all(&vec![0; padding_size])?;
    }

    Ok(x)
}

fn encode_plutus_data(data: &PlutusData) -> Result<Vec<u8>> {
    let mut buf = Vec::new();

    match data {
        PlutusData::Constr(constr_data) => {
            buf.write_u32::<LittleEndian>(data_tag::CONSTR)?;
            let encoded_tag = match constr_data.any_constructor {
                Some(ix) => encode_large_constr_tag(ix)?,
                None => u32::try_from(constr_data.tag).map_err(|_| {
                    SerializationError::DataTooComplex(
                        "constructor tag does not fit in 32 bits".to_string(),
                    )
                })?,
            };
            buf.write_u32::<LittleEndian>(encoded_tag)?;

            let fields: Vec<PlutusData> = constr_data.fields.clone().into();
            buf.write_u32::<LittleEndian>(u32::try_from(fields.len()).map_err(|_| {
                SerializationError::DataTooComplex("too many constructor fields".to_string())
            })?)?;

            for field in &fields {
                write_nested_data(&mut buf, field)?;
            }
        }
        PlutusData::Map(map_data) => {
            buf.write_u32::<LittleEndian>(data_tag::MAP)?;
            let pairs: Vec<(PlutusData, PlutusData)> = map_data.clone().into();
            buf.write_u32::<LittleEndian>(u32::try_from(pairs.len()).map_err(|_| {
                SerializationError::DataTooComplex("map contains too many entries".to_string())
            })?)?;

            for (key, value) in &pairs {
                write_nested_data(&mut buf, key)?;
                write_nested_data(&mut buf, value)?;
            }
        }
        PlutusData::Array(array_data) => {
            buf.write_u32::<LittleEndian>(data_tag::LIST)?;
            let elements: Vec<PlutusData> = array_data.clone().into();
            buf.write_u32::<LittleEndian>(u32::try_from(elements.len()).map_err(|_| {
                SerializationError::DataTooComplex("list contains too many elements".to_string())
            })?)?;

            for element in &elements {
                write_nested_data(&mut buf, element)?;
            }
        }
        PlutusData::BigInt(int_data) => {
            buf.write_u32::<LittleEndian>(data_tag::INTEGER)?;
            let (sign, words) = plutus_bigint_to_words(int_data)?;
            buf.write_u8(sign)?;
            buf.write_u32::<LittleEndian>(u32::try_from(words.len()).map_err(|_| {
                SerializationError::DataTooComplex(
                    "integer representation is too large".to_string(),
                )
            })?)?;
            for word in words {
                buf.write_u32::<LittleEndian>(word)?;
            }
        }
        PlutusData::BoundedBytes(bytes) => {
            buf.write_u32::<LittleEndian>(data_tag::BYTESTRING)?;
            let payload: Vec<u8> = bytes.clone().into();
            buf.write_u32::<LittleEndian>(u32::try_from(payload.len()).map_err(|_| {
                SerializationError::DataTooComplex("bytestring too large".to_string())
            })?)?;
            buf.write_all(&payload)?;
        }
    }

    pad_to_word_boundary(&mut buf);
    Ok(buf)
}

fn write_nested_data(buf: &mut Vec<u8>, value: &PlutusData) -> Result<()> {
    let nested = encode_plutus_data(value)?;
    let word_count = bytes_to_words(nested.len());
    buf.write_u32::<LittleEndian>(word_count)?;
    buf.write_all(&nested)?;
    Ok(())
}

fn encode_large_constr_tag(ix: u64) -> Result<u32> {
    let tag = u32::try_from(ix).map_err(|_| {
        SerializationError::DataTooComplex(format!(
            "constructor tag {ix} does not fit in 32 bits"
        ))
    })?;
    Ok(LARGE_CONSTR_TAG_FLAG | tag)
}

fn pad_to_word_boundary(buf: &mut Vec<u8>) {
    let padding = (4 - (buf.len() % 4)) % 4;
    for _ in 0..padding {
        buf.push(0);
    }
}

fn bytes_to_words(len: usize) -> u32 {
    if len == 0 { 0 } else { ((len + 3) / 4) as u32 }
}

fn plutus_bigint_to_words(int_data: &BigInt) -> Result<(u8, Vec<u32>)> {
    let numeric = match int_data {
        BigInt::Int(value) => {
            let repr: i128 = value.clone().into();
            NumBigInt::from(repr)
        }
        BigInt::BigUInt(bytes) => {
            let raw: Vec<u8> = bytes.clone().into();
            NumBigInt::from_biguint(Sign::Plus, BigUint::from_bytes_be(&raw))
        }
        BigInt::BigNInt(bytes) => {
            let raw: Vec<u8> = bytes.clone().into();
            let mut magnitude = BigUint::from_bytes_be(&raw);
            magnitude += BigUint::from(1u8);
            NumBigInt::from_biguint(Sign::Minus, magnitude)
        }
    };

    let (sign, words) = numeric.to_u32_digits();
    let sign_byte = if sign == Sign::Minus { 1 } else { 0 };
    Ok((sign_byte, words))
}

/// Serialize a builtin term
fn serialize_builtin(builtin: DefaultFunction) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte
    x.write_u32::<LittleEndian>(term_tag::BUILTIN)?;

    // Builtin function identifier
    x.write_u32::<LittleEndian>(builtin as u32)?;

    Ok(x)
}

/// Serialize an error term
fn serialize_error() -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte
    x.write_u32::<LittleEndian>(term_tag::ERROR)?;

    // No additional data for error

    Ok(x)
}

/// Serialize a constructor term
fn serialize_constructor(
    preceeding_byte_size: u32,
    tag: usize,
    fields: &[Term<DeBruijn>],
) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte
    x.write_u32::<LittleEndian>(term_tag::CONSTRUCTOR)?;

    // Constructor tag (32-bit)
    x.write_u32::<LittleEndian>(tag as u32)?;

    // Field count (32-bit)
    x.write_u32::<LittleEndian>(fields.len() as u32)?;

    // Calculate the base offset for the first field
    // Initial offset includes:
    // - 4 byte for constructor term tag
    // - 4 bytes for constructor tag value
    // - 4 bytes for field count
    // - 4 bytes per field for pointers
    let mut current_offset = preceeding_byte_size + 4 + 4 + 4 + (fields.len() as u32 * 4);

    // Serialize each field with its appropriate offset and collect results
    let mut field_bodies = Vec::with_capacity(fields.len());

    for field in fields {
        x.write_u32::<LittleEndian>(current_offset)?;
        let field_body = serialize_term(current_offset, field)?;
        current_offset += field_body.len() as u32;
        field_bodies.push(field_body);
    }

    // Write field bodies
    for field_body in field_bodies {
        x.write_all(&field_body)?;
    }

    Ok(x)
}

/// Serialize a case term
fn serialize_case(
    preceeding_byte_size: u32,
    constr: &Rc<Term<DeBruijn>>,
    branches: &[Term<DeBruijn>],
) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte
    x.write_u32::<LittleEndian>(term_tag::CASE)?;

    // Calculate base offset for the constructor expression
    // Initial offset includes:
    // - 4 bytes for case term tag
    // - 4 bytes for the constructor pointer
    // - 4 bytes for the branch count
    // - 4 bytes per branch for branch pointers
    let mut current_offset = preceeding_byte_size + 4 + 4 + 4 + (branches.len() as u32 * 4);

    // Constructor pointer
    x.write_u32::<LittleEndian>(current_offset)?;

    // Case count (32-bit)
    x.write_u32::<LittleEndian>(branches.len() as u32)?;

    // Serialize the constructor expression
    let constr_ser = serialize_term(current_offset, constr)?;
    current_offset += constr_ser.len() as u32;

    // Serialize each branch and collect pointers
    let mut branch_bodies = Vec::with_capacity(branches.len());

    for branch in branches {
        x.write_u32::<LittleEndian>(current_offset)?;
        let branch_ser = serialize_term(current_offset, branch)?;
        current_offset += branch_ser.len() as u32;
        branch_bodies.push(branch_ser);
    }

    // Write constructor body
    x.write_all(&constr_ser)?;

    // Write branch bodies
    for branch_body in branch_bodies {
        x.write_all(&branch_body)?;
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_serialization() {
        // A simple program: (program 1.0.0 (con integer 42))
        let uplc_text = "(program 1.0.0 (con integer 42))";
        let binary = parse_and_serialize(uplc_text).unwrap();

        // The program should be successfully serialized
        assert!(
            binary.len() > 1,
            "Binary should be larger than just the header"
        );
    }

    #[test]
    fn test_term_types() {
        // A program with various term types
        let uplc_text = r#"
        (program 1.0.0
          (lam x
            [
              (force
                (delay x)
              )
              (con integer 1)
            ]
          )
        )
        "#;

        let binary = parse_and_serialize(uplc_text).unwrap();

        // We can't check exact binary contents as the memory layout might change,
        // but we can verify the program was serialized
        assert!(
            binary.len() > 20,
            "Binary should contain data beyond the header"
        );
    }

    #[test]
    fn test_constants() {
        // Test integer constants
        let int_test = "(program 1.0.0 (con integer 42))";
        let int_binary = parse_and_serialize(int_test).unwrap();
        assert!(int_binary.len() > 1);

        // Test boolean constant
        let bool_test = "(program 1.0.0 (con bool True))";
        let bool_binary = parse_and_serialize(bool_test).unwrap();
        assert!(bool_binary.len() > 1);

        // Test unit constant
        let unit_test = "(program 1.0.0 (con unit ()))";
        let unit_binary = parse_and_serialize(unit_test).unwrap();
        assert!(unit_binary.len() > 1);

        // Test bytestring constant
        let bytestring_test = "(program 1.0.0 (con bytestring #01020304))";
        let bytestring_binary = parse_and_serialize(bytestring_test).unwrap();
        assert!(bytestring_binary.len() > 1);
    }

    #[test]
    fn test_complex_program() {
        // A more complex program with nested terms
        let complex_text = r#"
        (program 1.0.0
          (lam f
            (lam x
              (force
                [
                  (builtin addInteger)
                  x
                  (con integer 2)
                ]
              )
            )
          )
        )
        "#;

        let binary = parse_and_serialize(complex_text).unwrap();

        // Again, we can't check exact contents, but we want to verify
        // that the complex program was serialized without errors
        assert!(
            !binary.is_empty(),
            "Binary should contain substantial data for a complex program"
        );
    }

    #[test]
    fn test_error_handling() {
        // Test handling of invalid UPLC
        let invalid_uplc = "(program 1.0.0 (invalid node))";
        let result = parse_and_serialize(invalid_uplc);
        assert!(
            result.is_err(),
            "Serializing invalid UPLC should return an error"
        );
    }
}
