use std::io::{Cursor, Write};
use std::ops::Deref;
use std::rc::Rc;

use byteorder::{LittleEndian, WriteBytesExt};
use num_traits::ToPrimitive;
use uplc::ast::{Constant, DeBruijn, Program, Term};
use uplc::builtins::DefaultFunction;
use uplc::BigInt;
use uplc::PlutusData;

use crate::constants::{bool_val, const_tag, data_tag, term_tag};
use crate::{Result, SerializationError};

/// A serializer for UPLC terms
pub struct UPLCSerializer<'a> {
    /// The program being serialized
    program: &'a Program<DeBruijn>,

    /// Output buffer
    output: Cursor<Vec<u8>>,
}

impl<'a> UPLCSerializer<'a> {
    /// Create a new serializer for a UPLC program
    pub fn new(program: &'a Program<DeBruijn>) -> Self {
        Self {
            program,
            output: Cursor::new(Vec::new()),
        }
    }

    /// Serialize the program to a binary format
    pub fn serialize(mut self) -> Result<Vec<u8>> {
        // Now serialize the root term
        let serialized_bytes = self.serialize_term(&self.program.term)?;

        self.output.write_all(&serialized_bytes)?;

        // Return the serialized program
        Ok(self.output.into_inner())
    }

    /// Serialize a term and return its address
    fn serialize_term(&mut self, term: &Term<DeBruijn>) -> Result<Vec<u8>> {
        // Serialize the term based on its type
        match term {
            Term::Var(index) => self.serialize_var(index.inner()),
            Term::Lambda {
                parameter_name: _,
                body,
            } => self.serialize_lambda(body),
            Term::Apply { function, argument } => self.serialize_apply(function, argument),
            Term::Force(term) => self.serialize_force(term),
            Term::Delay(term) => self.serialize_delay(term),
            Term::Constant(constant) => self.serialize_constant(constant),
            Term::Builtin(builtin) => self.serialize_builtin(*builtin),
            Term::Error => self.serialize_error(),
            Term::Constr { tag, fields } => self.serialize_constructor(*tag, fields),
            Term::Case { constr, branches } => self.serialize_case(constr, branches),
        }
    }

    /// Serialize a variable term
    fn serialize_var(&self, index: usize) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte
        x.write_u8(term_tag::VARIABLE)?;

        // DeBruijn index (4 bytes, little-endian)
        x.write_u32::<LittleEndian>(index as u32)?;

        Ok(x)
    }

    /// Serialize a lambda term
    fn serialize_lambda(&mut self, body: &Rc<Term<DeBruijn>>) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte
        x.write_u8(term_tag::LAMBDA)?;

        // Serialize the body (recursively)
        let body_ser = self.serialize_term(body)?;

        // Write body address
        x.write_all(&body_ser)?;

        Ok(x)
    }

    /// Serialize an apply term
    fn serialize_apply(
        &mut self,
        function: &Rc<Term<DeBruijn>>,
        argument: &Rc<Term<DeBruijn>>,
    ) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte
        x.write_u8(term_tag::APPLY)?;

        // Serialize the function and argument (recursively)
        let function_ser = self.serialize_term(function).unwrap();
        let argument_ser = self.serialize_term(argument).unwrap();

        // We need to provide the size of each term before writing the terms
        x.write_u32::<LittleEndian>(function_ser.len() as u32)?;
        x.write_u32::<LittleEndian>(argument_ser.len() as u32)?;

        // Write function and argument addresses
        let _ = x.write_all(&function_ser);
        let _ = x.write_all(&argument_ser);

        Ok(x)
    }

    /// Serialize a force term
    fn serialize_force(&mut self, term: &Rc<Term<DeBruijn>>) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte
        x.write_u8(term_tag::FORCE)?;

        // Serialize the term being forced (recursively)
        let term_ser = self.serialize_term(term).unwrap();

        // Write term address
        let _ = x.write_all(&term_ser);

        Ok(x)
    }

    /// Serialize a delay term
    fn serialize_delay(&mut self, term: &Rc<Term<DeBruijn>>) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte
        x.write_u8(term_tag::DELAY)?;

        // Serialize the term being delayed (recursively)
        let term_ser = self.serialize_term(term).unwrap();

        // Write term address
        let _ = x.write_all(&term_ser);

        Ok(x)
    }

    /// Serialize a constant term
    fn serialize_constant(&mut self, constant: &Rc<Constant>) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte for constant
        x.write_u8(term_tag::CONSTANT)?;

        // Determine the type length and store it
        // (This is a placeholder - you may need to calculate the actual type length)
        let type_length: u8 = 1; // For simple types
        x.write_u8(type_length)?;

        // Serialize the constant based on its type
        let serialized_data = match &**constant {
            Constant::Integer(int) => self.serialize_integer_constant(int)?,
            Constant::ByteString(bytes) => self.serialize_bytestring_constant(bytes)?,
            // Constant::String(s) => self.serialize_string_constant(s)?, wait wut todo
            Constant::Unit => self.serialize_unit_constant()?,
            Constant::Bool(b) => self.serialize_bool_constant(*b)?,
            Constant::Data(data) => self.serialize_data_constant(data)?,
            _ => {
                return Err(SerializationError::InvalidTermType(format!(
                    "Unsupported constant type: {:?}",
                    constant
                )))
            },
        };

        // Write serialized data to our buffer
        x.write_all(&serialized_data)?;

        Ok(x)
    }

    /// Serialize an integer constant
    fn serialize_integer_constant(&mut self, int: &num_bigint::BigInt) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Write constant type tag
        x.write_u8(const_tag::INTEGER)?;

        // Determine content size and write it (1 word = 4 bytes)
        let size_in_bytes = (int.bits() + 7) / 8; // Round up to nearest byte
        let content_size = ((size_in_bytes + 3) / 4) as u32; // Round up to nearest word (4 bytes)
        x.write_u32::<LittleEndian>(content_size)?;

        // Determine the format based on size
        if int.bits() <= 8
            && int >= &num_bigint::BigInt::from(-128)
            && int <= &num_bigint::BigInt::from(127)
        {
            // Small integer (1 byte)
            if let Some(i) = int.to_i8() {
                x.write_i8(i)?;
                // Pad to complete the word
                x.write_all(&[0, 0, 0])?;
            } else {
                return Err(SerializationError::IntegerTooLarge(format!(
                    "Integer doesn't fit in i8: {}",
                    int
                )));
            }
        } else if int.bits() <= 16
            && int >= &num_bigint::BigInt::from(-32768)
            && int <= &num_bigint::BigInt::from(32767)
        {
            // Medium integer (2 bytes)
            if let Some(i) = int.to_i16() {
                x.write_i16::<LittleEndian>(i)?;
                // Pad to complete the word
                x.write_all(&[0, 0])?;
            } else {
                return Err(SerializationError::IntegerTooLarge(format!(
                    "Integer doesn't fit in i16: {}",
                    int
                )));
            }
        } else if int.bits() <= 32
            && int >= &num_bigint::BigInt::from(i32::MIN)
            && int <= &num_bigint::BigInt::from(i32::MAX)
        {
            // Large integer (4 bytes)
            if let Some(i) = int.to_i32() {
                x.write_i32::<LittleEndian>(i)?;
            } else {
                return Err(SerializationError::IntegerTooLarge(format!(
                    "Integer doesn't fit in i32: {}",
                    int
                )));
            }
        } else if int.bits() <= 64
            && int >= &num_bigint::BigInt::from(i64::MIN)
            && int <= &num_bigint::BigInt::from(i64::MAX)
        {
            // Extra large integer (8 bytes = 2 words)
            if let Some(i) = int.to_i64() {
                x.write_i64::<LittleEndian>(i)?;
            } else {
                return Err(SerializationError::IntegerTooLarge(format!(
                    "Integer doesn't fit in i64: {}",
                    int
                )));
            }
        } else {
            // BigInt (variable size)
            let (sign, bytes) = int.to_bytes_le();

            // Write sign byte
            let sign_byte = if sign == num_bigint::Sign::Minus {
                1
            } else {
                0
            };
            x.write_u8(sign_byte)?;

            // Write the magnitude bytes
            x.write_all(&bytes)?;

            // Pad to complete the last word if necessary
            let padding_size = (4 - (bytes.len() + 1) % 4) % 4;
            if padding_size > 0 {
                x.write_all(&vec![0; padding_size])?;
            }
        }

        Ok(x)
    }

    /// Serialize a bytestring constant
    fn serialize_bytestring_constant(&mut self, bytes: &[u8]) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Check if the bytestring is too large
        if bytes.len() > u32::MAX as usize {
            return Err(SerializationError::ByteStringTooLarge(format!(
                "ByteString too large: {} bytes",
                bytes.len()
            )));
        }

        // Write constant type tag
        x.write_u8(const_tag::BYTESTRING)?;

        // Calculate content size in words (4 bytes each)
        let content_size = ((bytes.len() + 3) / 4) as u32; // Round up to nearest word
        x.write_u32::<LittleEndian>(content_size)?;

        // Write actual length in bytes
        x.write_u32::<LittleEndian>(bytes.len() as u32)?;

        // Write bytes
        x.write_all(bytes)?;

        // Pad to complete the last word if necessary
        let padding_size = (4 - (bytes.len() % 4)) % 4;
        if padding_size > 0 {
            x.write_all(&vec![0; padding_size])?;
        }

        Ok(x)
    }

    /// Serialize a unit constant
    fn serialize_unit_constant(&mut self) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Write constant type tag
        x.write_u8(const_tag::UNIT)?;

        // No additional data for unit

        Ok(x)
    }

    /// Serialize a boolean constant
    fn serialize_bool_constant(&mut self, value: bool) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Write constant type tag
        x.write_u8(const_tag::BOOL)?;

        // Write boolean value (0x00 for false, 0x01 for true)
        x.write_u8(if value {
            bool_val::TRUE
        } else {
            bool_val::FALSE
        })?;

        Ok(x)
    }

    /// Serialize a Plutus Data constant
    fn serialize_data_constant(&mut self, data: &PlutusData) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Constant type tag
        x.write_u8(const_tag::DATA)?;

        // For this implementation, we'll treat all Data as "black-box" with a simple representation
        // A full implementation would serialize the structure recursively

        // Create a temporary buffer for the serialized data
        let mut data_buffer = Cursor::new(Vec::new());

        // Serialize the data based on its variant
        match data {
            PlutusData::Constr(constr_data) => {
                // Write the tag
                data_buffer.write_u8(data_tag::CONSTR)?;

                // Serialize constructor tag - in PlutusData, the tag is a usize
                data_buffer.write_u32::<LittleEndian>(constr_data.tag as u32)?;

                // Write a simple placeholder for fields
                // In a real implementation, you'd recursively serialize each field
                data_buffer.write_u32::<LittleEndian>(constr_data.fields.len() as u32)?;
            },
            PlutusData::Map(map_data) => {
                // Write the map tag
                data_buffer.write_u8(data_tag::MAP)?;

                // Write map size
                data_buffer.write_u32::<LittleEndian>(map_data.len() as u32)?;

                // A simplified representation - in reality you'd serialize each key-value pair
                // This is just a placeholder
            },
            PlutusData::Array(array_data) => {
                // Write the list tag
                data_buffer.write_u8(data_tag::LIST)?;

                // Write list size
                data_buffer.write_u32::<LittleEndian>(array_data.len() as u32)?;

                // A simplified representation - in reality you'd serialize each list element
                // This is just a placeholder
            },
            PlutusData::BigInt(int_data) => {
                // Write the integer tag
                data_buffer.write_u8(data_tag::INTEGER)?;

                match int_data {
                    BigInt::Int(int_val) => {
                        // Since we don't have access to details about the Int type's internals,
                        // we'll convert it to a string and use the first character to check sign
                        let int_str = format!("{:?}", int_val);
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
                data_buffer.write_u8(data_tag::BYTESTRING)?;

                // Write length and bytes
                data_buffer.write_u32::<LittleEndian>(bytes.len() as u32)?;
                data_buffer.write_all(bytes)?;
            },
        }

        // Get the serialized data
        let data_bytes = data_buffer.into_inner();

        // Calculate content size in words (4 bytes each)
        let content_size = ((data_bytes.len() + 3) / 4) as u32; // Round up to nearest word
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
    fn serialize_builtin(&mut self, builtin: DefaultFunction) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte
        x.write_u8(term_tag::BUILTIN)?;

        // Builtin function identifier
        x.write_u8(builtin as u8)?;

        Ok(x)
    }

    /// Serialize an error term
    fn serialize_error(&mut self) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte
        x.write_u8(term_tag::ERROR)?;

        // No additional data for error

        Ok(x)
    }

    /// Serialize a constructor term
    fn serialize_constructor(&mut self, tag: usize, fields: &[Term<DeBruijn>]) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte
        x.write_u8(term_tag::CONSTRUCTOR)?;

        // Constructor tag (16-bit)
        x.write_u16::<LittleEndian>(tag as u16)?;

        // Field count (32-bit)
        x.write_u32::<LittleEndian>(fields.len() as u32)?;

        let field_bodies = fields
            .iter()
            .map(|field| self.serialize_term(field))
            .collect::<Result<Vec<_>>>()?;

        // Write field sizes
        for field_body in &field_bodies {
            x.write_u32::<LittleEndian>(field_body.len().try_into().unwrap())?;
        }

        // Write field bodies
        for field_body in field_bodies {
            x.write_all(&field_body)?;
        }

        Ok(x)
    }

    /// Serialize a case term
    fn serialize_case(
        &mut self,
        constr: &Rc<Term<DeBruijn>>,
        branches: &[Term<DeBruijn>],
    ) -> Result<Vec<u8>> {
        let mut x: Vec<u8> = Vec::new();
        // Tag byte
        x.write_u8(term_tag::CASE)?;

        // Serialize the constructor expression
        let constr_ser = self.serialize_term(constr)?;

        // Case count (32-bit)
        x.write_u32::<LittleEndian>(branches.len() as u32)?;

        // Write constructor size
        x.write_u32::<LittleEndian>(constr_ser.len().try_into().unwrap())?;

        // Write constructor body
        x.write_all(&constr_ser)?;

        // Serialize each branch and collect sizes
        let mut branch_sizes = Vec::with_capacity(branches.len());
        let mut branch_bodies = Vec::with_capacity(branches.len());
        for branch in branches {
            let branch_ser = self.serialize_term(branch)?;
            branch_sizes.push(branch_ser.len().try_into().unwrap());
            branch_bodies.push(branch_ser);
        }

        // Write branch sizes
        for branch_size in &branch_sizes {
            x.write_u32::<LittleEndian>(*branch_size)?;
        }

        // Write branch bodies
        for branch_body in branch_bodies {
            x.write_all(&branch_body)?;
        }

        Ok(x)
    }
}
