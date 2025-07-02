use std::io::{Cursor, Write};
use std::ops::Deref;
use std::rc::Rc;

use byteorder::{LittleEndian, WriteBytesExt};
use uplc::BigInt;
use uplc::PlutusData;
use uplc::ast::{Constant, DeBruijn, Program, Term};
use uplc::builtins::DefaultFunction;

use crate::constants::{bool_val, const_tag, data_tag, term_tag};
use crate::{Result, SerializationError};

/// Serialize the program to a binary format
pub fn serialize(program: &Program<DeBruijn>, preceeding_byte_size: u32) -> Result<Vec<u8>> {
    // Now serialize the root term
    let mut x: Vec<u8> = Vec::new();
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
        },
        Term::Force(term) => serialize_force(preceeding_byte_size, term),
        Term::Delay(term) => serialize_delay(preceeding_byte_size, term),
        Term::Constant(constant) => serialize_constant(constant),
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
    let body_ser = serialize_term(preceeding_byte_size + 1, body)?;

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
    let term_ser = serialize_term(preceeding_byte_size + 1, term)?;

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
    let term_ser = serialize_term(preceeding_byte_size + 1, term)?;

    // Write term address
    x.write_all(&term_ser)?;

    Ok(x)
}

/// Serialize a constant term
fn serialize_constant(constant: &Rc<Constant>) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Tag byte for constant
    x.write_u32::<LittleEndian>(term_tag::CONSTANT)?;

    // Determine the type length and store it
    // (This is a placeholder - you may need to calculate the actual type length)
    let type_length: u32 = 1; // For simple types
    x.write_u32::<LittleEndian>(type_length)?;

    // Serialize the constant based on its type
    let serialized_data = match &**constant {
        Constant::Integer(int) => serialize_integer_constant(int)?,
        Constant::ByteString(bytes) => serialize_bytestring_constant(bytes)?,
        // Constant::String(s) => serialize_string_constant(s)?, wait wut todo
        Constant::Unit => serialize_unit_constant()?,
        Constant::Bool(b) => serialize_bool_constant(*b)?,
        Constant::Data(data) => serialize_data_constant(data)?,
        _ => {
            return Err(SerializationError::InvalidTermType(format!(
                "Unsupported constant type: {constant:?}",
            )));
        },
    };

    // Write serialized data to our buffer
    x.write_all(&serialized_data)?;

    Ok(x)
}

/// Serialize an integer constant
fn serialize_integer_constant(int: &num_bigint::BigInt) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Write constant type tag
    x.write_u32::<LittleEndian>(const_tag::INTEGER)?;

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

    Ok(x)
}

/// Serialize a bytestring constant
fn serialize_bytestring_constant(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Check if the bytestring is too large
    if bytes.len() > u32::MAX as usize {
        return Err(SerializationError::ByteStringTooLarge(format!(
            "ByteString too large: {} bytes",
            bytes.len()
        )));
    }

    // Write constant type tag
    x.write_u32::<LittleEndian>(const_tag::BYTESTRING)?;

    // Write actual length in bytes
    x.write_u32::<LittleEndian>(bytes.len() as u32)?;

    // Write each byte as a word
    x.write_all(
        &bytes
            .iter()
            .flat_map(|byte| (*byte as u32).to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;

    Ok(x)
}

/// Serialize a unit constant
fn serialize_unit_constant() -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Write constant type tag
    x.write_u32::<LittleEndian>(const_tag::UNIT)?;

    // No additional data for unit
    Ok(x)
}

/// Serialize a boolean constant
fn serialize_bool_constant(value: bool) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Write constant type tag
    x.write_u32::<LittleEndian>(const_tag::BOOL)?;

    // Write boolean value (0x00 for false, 0x01 for true)
    x.write_u32::<LittleEndian>(if value {
        bool_val::TRUE
    } else {
        bool_val::FALSE
    })?;

    Ok(x)
}

/// Serialize a Plutus Data constant
fn serialize_data_constant(data: &PlutusData) -> Result<Vec<u8>> {
    let mut x: Vec<u8> = Vec::new();
    // Constant type tag
    x.write_u32::<LittleEndian>(const_tag::DATA)?;

    // For this implementation, we'll treat all Data as "black-box" with a simple representation
    // A full implementation would serialize the structure recursively

    // Create a temporary buffer for the serialized data
    let mut data_buffer = Cursor::new(Vec::new());

    // Serialize the data based on its variant
    match data {
        PlutusData::Constr(constr_data) => {
            // Write the tag
            data_buffer.write_u32::<LittleEndian>(data_tag::CONSTR)?;

            // Serialize constructor tag - in PlutusData, the tag is a usize
            data_buffer.write_u32::<LittleEndian>(constr_data.tag as u32)?;

            // Write a simple placeholder for fields
            // In a real implementation, you'd recursively serialize each field
            data_buffer.write_u32::<LittleEndian>(constr_data.fields.len() as u32)?;
        },
        PlutusData::Map(map_data) => {
            // Write the map tag
            data_buffer.write_u32::<LittleEndian>(data_tag::MAP)?;

            // Write map size
            data_buffer.write_u32::<LittleEndian>(map_data.len() as u32)?;

            // A simplified representation - in reality you'd serialize each key-value pair
            // This is just a placeholder
        },
        PlutusData::Array(array_data) => {
            // Write the list tag
            data_buffer.write_u32::<LittleEndian>(data_tag::LIST)?;

            // Write list size
            data_buffer.write_u32::<LittleEndian>(array_data.len() as u32)?;

            // A simplified representation - in reality you'd serialize each list element
            // This is just a placeholder
        },
        PlutusData::BigInt(int_data) => {
            // Write the integer tag
            data_buffer.write_u32::<LittleEndian>(data_tag::INTEGER)?;

            match int_data {
                BigInt::Int(int_val) => {
                    // Since we don't have access to details about the Int type's internals,
                    // we'll convert it to a string and use the first character to check sign
                    let int_str = format!("{int_val:?}");
                    let is_negative = int_str.starts_with('-');

                    // Write sign (0 for positive, 1 for negative)
                    data_buffer.write_u8(if is_negative { 1u8 } else { 0u8 })?;

                    // For simplicity, we'll use a basic byte representation
                    // In a production system, you'd want to extract proper bytes from Int
                    let bytes = [0, 0, 0, 1]; // Simple placeholder

                    // Write the length of bytes
                    data_buffer.write_u32::<LittleEndian>(bytes.len() as u32)?;

                    // Write the actual bytes
                    data_buffer.write_all(&bytes)?;
                },
                BigInt::BigUInt(bytes_val) => {
                    // Positive big integer
                    data_buffer.write_u8(0)?; // Sign byte (0 for positive)

                    // Get the bytes from BoundedBytes (which is a wrapper around Vec<u8>)
                    let bytes = bytes_val.deref();

                    // Write the length of bytes
                    data_buffer.write_u32::<LittleEndian>(bytes.len() as u32)?;

                    // Write the actual bytes
                    data_buffer.write_all(bytes)?;
                },
                BigInt::BigNInt(bytes_val) => {
                    // Negative big integer
                    data_buffer.write_u8(1)?; // Sign byte (1 for negative)

                    // Get the bytes from BoundedBytes
                    let bytes = bytes_val.deref();

                    // Write the length of bytes
                    data_buffer.write_u32::<LittleEndian>(bytes.len() as u32)?;

                    // Write the actual bytes
                    data_buffer.write_all(bytes)?;
                },
            }
        },
        PlutusData::BoundedBytes(bytes) => {
            // Write the bytestring tag
            data_buffer.write_u32::<LittleEndian>(data_tag::BYTESTRING)?;

            // Write length and bytes
            data_buffer.write_u32::<LittleEndian>(bytes.len() as u32)?;
            data_buffer.write_all(bytes)?;
        },
    }

    // Get the serialized data
    let data_bytes = data_buffer.into_inner();

    // Calculate content size in words (4 bytes each)
    let content_size = data_bytes.len().div_ceil(4) as u32; // Round up to nearest word
    x.write_u32::<LittleEndian>(content_size)?;

    // Write the data
    x.write_all(&data_bytes)?;

    // Pad to complete the last word if necessary
    let padding_size = (4 - (data_bytes.len() % 4)) % 4;
    if padding_size > 0 {
        x.write_all(&vec![0; padding_size])?;
    }

    Ok(x)
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
    let mut field_pointers = Vec::with_capacity(fields.len());

    for field in fields {
        field_pointers.push(current_offset);
        let field_body = serialize_term(current_offset, field)?;
        current_offset += field_body.len() as u32;
        field_bodies.push(field_body);
    }

    // Write field pointers (not sizes)
    for pointer in &field_pointers {
        x.write_u32::<LittleEndian>(*pointer)?;
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

    // Serialize the constructor expression
    let constr_pointer = current_offset;
    let constr_ser = serialize_term(constr_pointer, constr)?;
    current_offset += constr_ser.len() as u32;

    // Case count (32-bit)
    x.write_u32::<LittleEndian>(branches.len() as u32)?;

    // Write constructor pointer (not size)
    x.write_u32::<LittleEndian>(constr_pointer)?;

    // Serialize each branch and collect pointers
    let mut branch_pointers = Vec::with_capacity(branches.len());
    let mut branch_bodies = Vec::with_capacity(branches.len());

    for branch in branches {
        branch_pointers.push(current_offset);
        let branch_ser = serialize_term(current_offset, branch)?;
        current_offset += branch_ser.len() as u32;
        branch_bodies.push(branch_ser);
    }

    // Write branch pointers (not sizes)
    for branch_pointer in &branch_pointers {
        x.write_u32::<LittleEndian>(*branch_pointer)?;
    }

    // Write constructor body
    x.write_all(&constr_ser)?;

    // Write branch bodies
    for branch_body in branch_bodies {
        x.write_all(&branch_body)?;
    }

    Ok(x)
}
