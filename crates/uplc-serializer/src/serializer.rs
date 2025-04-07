use std::io::{Cursor, Write};
use std::rc::Rc;

use byteorder::{LittleEndian, WriteBytesExt};
use num_traits::ToPrimitive;
use uplc::ast::{Constant, DeBruijn, Program, Term};
use uplc::builtins::DefaultFunction;
use uplc::PlutusData;

use crate::constants::{bool_val, const_tag, data_tag, header, int_size, term_tag};
use crate::memory_layout::{Address, MemoryLayout};
use crate::{Result, SerializationError};

/// A serializer for UPLC terms
pub struct UPLCSerializer<'a> {
    /// The program being serialized
    program: &'a Program<DeBruijn>,

    /// Memory layout manager
    layout: MemoryLayout,

    /// Output buffer
    output: Cursor<Vec<u8>>,

    /// Address of the root term
    root_term_address: Option<Address>,
}

impl<'a> UPLCSerializer<'a> {
    /// Create a new serializer for a UPLC program
    pub fn new(program: &'a Program<DeBruijn>) -> Self {
        Self {
            program,
            layout: MemoryLayout::new(),
            output: Cursor::new(Vec::new()),
            root_term_address: None,
        }
    }

    /// Serialize the program to a binary format
    pub fn serialize(mut self) -> Result<Vec<u8>> {
        // First, reserve space for the header
        self.output.write_all(&[0; header::HEADER_SIZE])?;

        // Calculate the size of the entire program and prepare memory layout
        self.calculate_layout()?;

        // Now serialize the root term
        let root_addr = self.serialize_term(&self.program.term)?;
        self.root_term_address = Some(root_addr);

        // Write the header (magic bytes, version, and root address)
        self.write_header()?;

        // Return the serialized program
        Ok(self.output.into_inner())
    }

    /// Write the program header
    fn write_header(&mut self) -> Result<()> {
        // Save current position
        let current_pos = self.output.position();

        // Move to the start of the output
        self.output.set_position(0);

        // Write magic bytes
        self.output.write_all(header::MAGIC)?;

        // Write version (as 3 individual bytes)
        self.output.write_u8(self.program.version.0 as u8)?;
        self.output.write_u8(self.program.version.1 as u8)?;
        self.output.write_u8(self.program.version.2 as u8)?;

        // Write a reserved byte (for alignment)
        self.output.write_u8(0)?;

        // Write root term address
        let root_addr = self.root_term_address.ok_or_else(|| {
            SerializationError::MemoryLayoutError("Root term address not set".to_string())
        })?;
        self.output.write_u32::<LittleEndian>(root_addr.as_u32())?;

        // Restore position
        self.output.set_position(current_pos);

        Ok(())
    }

    /// Calculate the memory layout for the program
    fn calculate_layout(&mut self) -> Result<()> {
        // This function would traverse the term tree to calculate the total size
        // and assign memory addresses to each term and constant.
        // For now, we'll leave this as a placeholder since our approach doesn't
        // require a separate layout calculation pass.
        Ok(())
    }

    /// Serialize a term and return its address
    fn serialize_term(&mut self, term: &Term<DeBruijn>) -> Result<Address> {
        // Check if this term has already been serialized (for deduplication)
        if let Some(addr) = self.layout.lookup_term(term) {
            return Ok(addr);
        }

        // Get current position in the output buffer
        let _term_start_pos = self.output.position();

        // The address will be in the term region, adjusted by the current position
        let address = self.layout.allocate_term(0)?; // Size will be calculated later

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

        // Register the serialized term with its address for deduplication
        self.layout.register_term(term, address);

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
        // Check if this constant has already been serialized (for deduplication)
        if let Some(addr) = self.layout.lookup_constant(constant) {
            // Write tag and constant address
            self.output.write_u8(term_tag::CONSTANT)?;
            self.output.write_u32::<LittleEndian>(addr.as_u32())?;
            return Ok(());
        }

        // Get current position
        let const_start_pos = self.output.position();

        // The address will be in the appropriate constant pool, allocated later
        let address = match &**constant {
            Constant::Integer(_) => self.layout.allocate_integer(0)?,
            Constant::ByteString(_) => self.layout.allocate_bytestring(0)?,
            Constant::String(_) => self.layout.allocate_string(0)?,
            _ => self.layout.allocate_complex_data(0)?,
        };

        // Write tag and constant address
        self.output.write_u8(term_tag::CONSTANT)?;
        self.output.write_u32::<LittleEndian>(address.as_u32())?;

        // Remember current position
        let return_pos = self.output.position();

        // Move to the constant address in the output stream
        self.output.set_position(address.as_u32() as u64);

        // Serialize the constant based on its type
        match &**constant {
            Constant::Integer(int) => self.serialize_integer_constant(int)?,
            Constant::ByteString(bytes) => self.serialize_bytestring_constant(bytes)?,
            Constant::String(s) => self.serialize_string_constant(s)?,
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

        // Register the serialized constant with its address for deduplication
        self.layout.register_constant(constant, address);

        // Restore position
        self.output.set_position(return_pos);

        Ok(())
    }

    /// Serialize an integer constant
    fn serialize_integer_constant(&mut self, int: &num_bigint::BigInt) -> Result<()> {
        // Write constant type tag
        self.output.write_u8(const_tag::INTEGER)?;

        // Determine the size needed for this integer
        if int.bits() <= 8
            && int >= &num_bigint::BigInt::from(-128)
            && int <= &num_bigint::BigInt::from(127)
        {
            // Small integer (1 byte)
            self.output.write_u8(int_size::SMALL)?;
            if let Some(i) = int.to_i8() {
                self.output.write_i8(i)?;
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
            self.output.write_u8(int_size::MEDIUM)?;
            if let Some(i) = int.to_i16() {
                self.output.write_i16::<LittleEndian>(i)?;
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
            self.output.write_u8(int_size::LARGE)?;
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
            // Extra large integer (8 bytes)
            self.output.write_u8(int_size::XLARGE)?;
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
            self.output.write_u8(int_size::BIGINT)?;

            // Get the sign and magnitude
            let (sign, bytes) = int.to_bytes_le();
            let sign_byte = if sign == num_bigint::Sign::Minus {
                0x01
            } else {
                0x00
            };

            // Write sign
            self.output.write_u8(sign_byte)?;

            // Write length (4 bytes)
            if bytes.len() > u32::MAX as usize {
                return Err(SerializationError::IntegerTooLarge(format!(
                    "BigInt too large: {} bytes",
                    bytes.len()
                )));
            }
            self.output.write_u32::<LittleEndian>(bytes.len() as u32)?;

            // Write magnitude bytes
            self.output.write_all(&bytes)?;
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

        // Write length (4 bytes)
        self.output.write_u32::<LittleEndian>(bytes.len() as u32)?;

        // Write bytes
        self.output.write_all(bytes)?;

        Ok(())
    }

    /// Serialize a string constant
    fn serialize_string_constant(&mut self, s: &str) -> Result<()> {
        // Get the UTF-8 encoded bytes
        let bytes = s.as_bytes();

        // Check if the string is too large
        if bytes.len() > u32::MAX as usize {
            return Err(SerializationError::StringTooLarge(format!(
                "String too large: {} bytes",
                bytes.len()
            )));
        }

        // Write constant type tag
        self.output.write_u8(const_tag::STRING)?;

        // Write length (4 bytes)
        self.output.write_u32::<LittleEndian>(bytes.len() as u32)?;

        // Write UTF-8 bytes
        self.output.write_all(bytes)?;

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

    /// Serialize a Plutus Data constant (simple implementation for now)
    fn serialize_data_constant(&mut self, data: &PlutusData) -> Result<()> {
        // Constant type tag
        self.output.write_u8(const_tag::DATA)?;

        // This is a placeholder - a real implementation would serialize
        // the data structure in more detail based on its type

        // For now, we'll just identify the data variant type
        match data {
            PlutusData::Constr(_) => {
                self.output.write_u8(data_tag::CONSTR)?;
            },
            PlutusData::Map(_) => {
                self.output.write_u8(data_tag::MAP)?;
            },
            // The PlutusData enum may have different variant names in the uplc crate
            // Let's use the correct ones based on the actual enum definition
            PlutusData::Array(_) => {
                self.output.write_u8(data_tag::LIST)?;
            },
            PlutusData::BigInt(_) => {
                self.output.write_u8(data_tag::INTEGER)?;
            },
            PlutusData::BoundedBytes(_) => {
                self.output.write_u8(data_tag::BYTESTRING)?;
            },
        }

        // A real implementation would serialize the full structure here
        // For now, we'll just write a placeholder size
        self.output.write_u32::<LittleEndian>(0)?;

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
