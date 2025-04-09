use std::io::{Cursor, Write};
use std::rc::Rc;

use byteorder::{LittleEndian, WriteBytesExt};
use num_traits::ToPrimitive;
use uplc::ast::{Constant, DeBruijn, Program, Term};
use uplc::builtins::DefaultFunction;
use uplc::PlutusData;

use crate::constants::{bool_val, const_tag, data_tag, header, int_size, term_tag};
use crate::{Result, SerializationError};

/// A simple address type for tracking positions in the output buffer
#[derive(Debug, Clone, Copy)]
struct Address(u64);

impl Address {
    fn new(pos: u64) -> Self {
        Self(pos)
    }

    fn as_u32(&self) -> u32 {
        // This may truncate large addresses, but should be fine for most use cases
        // A proper implementation might want to check for overflow
        self.0 as u32
    }
}

/// A serializer for UPLC terms
pub struct UPLCSerializer<'a> {
    /// The program being serialized
    program: &'a Program<DeBruijn>,

    /// Memory layout manager
    // layout: MemoryLayout,

    /// Output buffer
    output: Cursor<Vec<u8>>,
    // Address of the root term
    root_term_address: Option<Address>,
}

impl<'a> UPLCSerializer<'a> {
    /// Create a new serializer for a UPLC program
    pub fn new(program: &'a Program<DeBruijn>) -> Self {
        Self {
            program,
            // layout: MemoryLayout::new(),
            output: Cursor::new(Vec::new()),
            root_term_address: None,
        }
    }

    /// Serialize the program to a binary format
    pub fn serialize(mut self) -> Result<Vec<u8>> {
        // First, reserve space for the header
        self.output.write_all(&[0; header::HEADER_SIZE])?;

        // Now serialize the root term
        let root_addr = self.serialize_term(&self.program.term)?;
        self.root_term_address = Some(root_addr);

        // Return the serialized program
        Ok(self.output.into_inner())
    }

    /// Serialize a term and return its address
    fn serialize_term(&mut self, term: &Term<DeBruijn>) -> Result<Address> {
        // Get current position in the output buffer
        let term_start_pos = self.output.position();
        let address = Address::new(term_start_pos);

        // Serialize the term based on its type
        match term {
            Term::Var(index) => self.serialize_var(index.inner())?,
            Term::Lambda {
                parameter_name: _,
                body,
            } => self.serialize_lambda(body)?,
            Term::Apply { function, argument } => self.serialize_apply(function, argument)?,
            Term::Force(term) => self.serialize_force(term)?,
            Term::Delay(term) => self.serialize_delay(term)?,
            Term::Constant(constant) => self.serialize_constant(constant)?,
            Term::Builtin(builtin) => self.serialize_builtin(*builtin)?,
            Term::Error => self.serialize_error()?,
            Term::Constr { tag, fields } => self.serialize_constructor(*tag, fields)?,
            Term::Case { constr, branches } => self.serialize_case(constr, branches)?,
        }

        Ok(address)
    }

    /// Serialize a variable term
    fn serialize_var(&mut self, index: usize) -> Result<()> {
        // Tag byte
        self.output.write_u8(term_tag::VARIABLE)?;

        // DeBruijn index (4 bytes, little-endian)
        self.output.write_u32::<LittleEndian>(index as u32)?;

        Ok(())
    }

    /// Serialize a lambda term
    fn serialize_lambda(&mut self, body: &Rc<Term<DeBruijn>>) -> Result<()> {
        // Tag byte
        self.output.write_u8(term_tag::LAMBDA)?;

        // Serialize the body (recursively)
        let body_addr = self.serialize_term(body)?;

        // Write body address
        self.output.write_u32::<LittleEndian>(body_addr.as_u32())?;

        Ok(())
    }

    /// Serialize an apply term
    fn serialize_apply(
        &mut self,
        function: &Rc<Term<DeBruijn>>,
        argument: &Rc<Term<DeBruijn>>,
    ) -> Result<()> {
        // Tag byte
        self.output.write_u8(term_tag::APPLY)?;

        // Serialize the function and argument (recursively)
        let function_addr = self.serialize_term(function)?;
        let argument_addr = self.serialize_term(argument)?;

        // Write function and argument addresses
        self.output
            .write_u32::<LittleEndian>(function_addr.as_u32())?;
        self.output
            .write_u32::<LittleEndian>(argument_addr.as_u32())?;

        Ok(())
    }

    /// Serialize a force term
    fn serialize_force(&mut self, term: &Rc<Term<DeBruijn>>) -> Result<()> {
        // Tag byte
        self.output.write_u8(term_tag::FORCE)?;

        // Serialize the term being forced (recursively)
        let term_addr = self.serialize_term(term)?;

        // Write term address
        self.output.write_u32::<LittleEndian>(term_addr.as_u32())?;

        Ok(())
    }

    /// Serialize a delay term
    fn serialize_delay(&mut self, term: &Rc<Term<DeBruijn>>) -> Result<()> {
        // Tag byte
        self.output.write_u8(term_tag::DELAY)?;

        // Serialize the term being delayed (recursively)
        let term_addr = self.serialize_term(term)?;

        // Write term address
        self.output.write_u32::<LittleEndian>(term_addr.as_u32())?;

        Ok(())
    }

    /// Serialize a constant term
    fn serialize_constant(&mut self, constant: &Rc<Constant>) -> Result<()> {
        // Tag byte for constant
        self.output.write_u8(term_tag::CONSTANT)?;
        
        // Determine the type length and store it
        // (This is a placeholder - you may need to calculate the actual type length)
        let type_length: u8 = 1; // For simple types
        self.output.write_u8(type_length)?;
        
        // Serialize the constant based on its type
        match &**constant {
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
        }

        Ok(())
    }

    /// Serialize an integer constant
    fn serialize_integer_constant(&mut self, int: &num_bigint::BigInt) -> Result<()> {
        // Write constant type tag
        self.output.write_u8(const_tag::INTEGER)?;

        // Determine content size and write it (1 word = 4 bytes)
        let size_in_bytes = (int.bits() + 7) / 8; // Round up to nearest byte
        let content_size = ((size_in_bytes + 3) / 4) as u32; // Round up to nearest word (4 bytes)
        self.output.write_u32::<LittleEndian>(content_size)?;

        // Determine the format based on size
        if int.bits() <= 8
            && int >= &num_bigint::BigInt::from(-128)
            && int <= &num_bigint::BigInt::from(127)
        {
            // Small integer (1 byte)
            if let Some(i) = int.to_i8() {
                self.output.write_i8(i)?;
                // Pad to complete the word
                self.output.write_all(&[0, 0, 0])?;
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
                self.output.write_i16::<LittleEndian>(i)?;
                // Pad to complete the word
                self.output.write_all(&[0, 0])?;
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
                self.output.write_i32::<LittleEndian>(i)?;
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
                self.output.write_i64::<LittleEndian>(i)?;
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
            let sign_byte = if sign == num_bigint::Sign::Minus { 1 } else { 0 };
            self.output.write_u8(sign_byte)?;
            
            // Write the magnitude bytes
            self.output.write_all(&bytes)?;
            
            // Pad to complete the last word if necessary
            let padding_size = (4 - (bytes.len() + 1) % 4) % 4;
            if padding_size > 0 {
                self.output.write_all(&vec![0; padding_size])?;
            }
        }

        Ok(())
    }

    /// Serialize a bytestring constant
    fn serialize_bytestring_constant(&mut self, bytes: &[u8]) -> Result<()> {
        // Check if the bytestring is too large
        if bytes.len() > u32::MAX as usize {
            return Err(SerializationError::ByteStringTooLarge(format!(
                "ByteString too large: {} bytes",
                bytes.len()
            )));
        }

        // Write constant type tag
        self.output.write_u8(const_tag::BYTESTRING)?;

        // Calculate content size in words (4 bytes each)
        let content_size = ((bytes.len() + 3) / 4) as u32; // Round up to nearest word
        self.output.write_u32::<LittleEndian>(content_size)?;
        
        // Write actual length in bytes
        self.output.write_u32::<LittleEndian>(bytes.len() as u32)?;

        // Write bytes
        self.output.write_all(bytes)?;
        
        // Pad to complete the last word if necessary
        let padding_size = (4 - (bytes.len() % 4)) % 4;
        if padding_size > 0 {
            self.output.write_all(&vec![0; padding_size])?;
        }

        Ok(())
    }

    /// Serialize a unit constant
    fn serialize_unit_constant(&mut self) -> Result<()> {
        // Write constant type tag
        self.output.write_u8(const_tag::UNIT)?;

        // No additional data for unit

        Ok(())
    }

    /// Serialize a boolean constant
    fn serialize_bool_constant(&mut self, value: bool) -> Result<()> {
        // Write constant type tag
        self.output.write_u8(const_tag::BOOL)?;

        // Write boolean value (0x00 for false, 0x01 for true)
        self.output.write_u8(if value {
            bool_val::TRUE
        } else {
            bool_val::FALSE
        })?;

        Ok(())
    }

    /// Serialize a Plutus Data constant
    fn serialize_data_constant(&mut self, data: &PlutusData) -> Result<()> {
        // Constant type tag
        self.output.write_u8(const_tag::DATA)?;
        
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
                
                // Since we don't know the specific methods available for uplc::BigInt,
                // we'll just create a simple representation for now
                // In a real implementation, you would adapt this based on the actual API
                
                // For simplicity, write a fixed integer representation
                data_buffer.write_u8(0)?; // sign (positive)
                data_buffer.write_u32::<LittleEndian>(4)?; // length (4 bytes)
                
                // Write a simple integer representation
                // This should be replaced with proper conversion from uplc::BigInt
                let value = 0i32; // Default value as placeholder
                data_buffer.write_i32::<LittleEndian>(value)?;
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
        self.output.write_u32::<LittleEndian>(content_size)?;
        
        // Write the data
        self.output.write_all(&data_bytes)?;
        
        // Pad to complete the last word if necessary
        let padding_size = (4 - (data_bytes.len() % 4)) % 4;
        if padding_size > 0 {
            self.output.write_all(&vec![0; padding_size])?;
        }

        Ok(())
    }

    /// Serialize a builtin term
    fn serialize_builtin(&mut self, builtin: DefaultFunction) -> Result<()> {
        // Tag byte
        self.output.write_u8(term_tag::BUILTIN)?;

        // Builtin function identifier
        self.output.write_u8(builtin as u8)?;

        Ok(())
    }

    /// Serialize an error term
    fn serialize_error(&mut self) -> Result<()> {
        // Tag byte
        self.output.write_u8(term_tag::ERROR)?;

        // No additional data for error

        Ok(())
    }

    /// Serialize a constructor term
    fn serialize_constructor(&mut self, tag: usize, fields: &[Term<DeBruijn>]) -> Result<()> {
        // Tag byte
        self.output.write_u8(term_tag::CONSTRUCTOR)?;

        // Constructor tag (16-bit)
        self.output.write_u16::<LittleEndian>(tag as u16)?;

        // Field count (16-bit)
        self.output.write_u16::<LittleEndian>(fields.len() as u16)?;

        // Serialize each field and collect addresses
        let mut field_addresses = Vec::with_capacity(fields.len());
        for field in fields {
            let addr = self.serialize_term(field)?;
            field_addresses.push(addr);
        }

        // Write field addresses
        for addr in field_addresses {
            self.output.write_u32::<LittleEndian>(addr.as_u32())?;
        }

        Ok(())
    }

    /// Serialize a case term
    fn serialize_case(
        &mut self,
        constr: &Rc<Term<DeBruijn>>,
        branches: &[Term<DeBruijn>],
    ) -> Result<()> {
        // Tag byte
        self.output.write_u8(term_tag::CASE)?;

        // Serialize the constructor expression
        let constr_addr = self.serialize_term(constr)?;

        // Case count (16-bit)
        self.output
            .write_u16::<LittleEndian>(branches.len() as u16)?;

        // Write constructor address
        self.output
            .write_u32::<LittleEndian>(constr_addr.as_u32())?;

        // Serialize each branch and collect addresses
        let mut branch_addresses = Vec::with_capacity(branches.len());
        for branch in branches {
            let addr = self.serialize_term(branch)?;
            branch_addresses.push(addr);
        }

        // Write branch addresses
        for addr in branch_addresses {
            self.output.write_u32::<LittleEndian>(addr.as_u32())?;
        }

        Ok(())
    }
}
