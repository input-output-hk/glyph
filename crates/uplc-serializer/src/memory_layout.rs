use std::collections::HashMap;
use std::rc::Rc;

use crate::constants::memory_region;
use crate::SerializationError;
use num_traits::ToPrimitive;
use uplc::ast::{Constant, DeBruijn, Term};

/// Represents a memory address in the serialized format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Address(pub u32);

impl Address {
    /// Create a new address from a u32
    pub fn new(addr: u32) -> Self {
        Self(addr)
    }

    /// Get the raw u32 value
    pub fn as_u32(&self) -> u32 {
        self.0
    }

    /// Convert to a byte array (little-endian)
    pub fn to_le_bytes(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }
}

/// A simplified memory management system for tracking addresses
/// during the serialization process
pub struct MemoryLayout {
    /// Current position in the term region
    term_position: u32,

    /// Current position in the integer pool
    integer_pool_position: u32,

    /// Current position in the bytestring pool
    bytestring_pool_position: u32,

    /// Current position in the string pool
    string_pool_position: u32,

    /// Current position in the complex data pool
    complex_data_pool_position: u32,

    /// Maps of terms and constants to their addresses (for deduplication)
    term_addresses: HashMap<TermKey, Address>,
    constant_addresses: HashMap<ConstantKey, Address>,
}

/// Key type for term deduplication
#[derive(Debug, PartialEq, Eq, Hash)]
enum TermKey {
    Variable(usize),
    Lambda,
    Apply,
    Delay,
    Force,
    Error,
    Constant(ConstantKey),
    Builtin(u8),
    Constructor(usize),
    Case,
}

/// Key type for constant deduplication
#[derive(Debug, PartialEq, Eq, Hash)]
enum ConstantKey {
    Integer(Vec<u8>), // Serialized bytes
    ByteString(Vec<u8>),
    String(String),
    Unit,
    Bool(bool),
    Data(Vec<u8>), // Placeholder for now
}

impl Default for MemoryLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryLayout {
    /// Create a new memory layout
    pub fn new() -> Self {
        Self {
            term_position: memory_region::TERM_REGION,
            integer_pool_position: memory_region::INTEGER_POOL,
            bytestring_pool_position: memory_region::BYTESTRING_POOL,
            string_pool_position: memory_region::STRING_POOL,
            complex_data_pool_position: memory_region::COMPLEX_DATA_POOL,
            term_addresses: HashMap::new(),
            constant_addresses: HashMap::new(),
        }
    }

    /// Allocate space in the term region and return its address
    pub fn allocate_term(&mut self, size: u32) -> Result<Address, SerializationError> {
        let addr = Address::new(self.term_position);
        self.term_position += size;

        if self.term_position > memory_region::TERM_REGION_END {
            return Err(SerializationError::MemoryLayoutError(
                "Term region overflow".to_string(),
            ));
        }

        Ok(addr)
    }

    /// Allocate space in the integer pool
    pub fn allocate_integer(&mut self, size: u32) -> Result<Address, SerializationError> {
        let addr = Address::new(self.integer_pool_position);
        self.integer_pool_position += size;

        if self.integer_pool_position > memory_region::BYTESTRING_POOL {
            return Err(SerializationError::MemoryLayoutError(
                "Integer pool overflow".to_string(),
            ));
        }

        Ok(addr)
    }

    /// Allocate space in the bytestring pool
    pub fn allocate_bytestring(&mut self, size: u32) -> Result<Address, SerializationError> {
        let addr = Address::new(self.bytestring_pool_position);
        self.bytestring_pool_position += size;

        if self.bytestring_pool_position > memory_region::STRING_POOL {
            return Err(SerializationError::MemoryLayoutError(
                "ByteString pool overflow".to_string(),
            ));
        }

        Ok(addr)
    }

    /// Allocate space in the string pool
    pub fn allocate_string(&mut self, size: u32) -> Result<Address, SerializationError> {
        let addr = Address::new(self.string_pool_position);
        self.string_pool_position += size;

        if self.string_pool_position > memory_region::COMPLEX_DATA_POOL {
            return Err(SerializationError::MemoryLayoutError(
                "String pool overflow".to_string(),
            ));
        }

        Ok(addr)
    }

    /// Allocate space in the complex data pool
    pub fn allocate_complex_data(&mut self, size: u32) -> Result<Address, SerializationError> {
        let addr = Address::new(self.complex_data_pool_position);
        self.complex_data_pool_position += size;

        if self.complex_data_pool_position > memory_region::ENVIRONMENT {
            return Err(SerializationError::MemoryLayoutError(
                "Complex data pool overflow".to_string(),
            ));
        }

        Ok(addr)
    }

    /// Register a term with its address for deduplication
    pub fn register_term(&mut self, term: &Term<DeBruijn>, address: Address) {
        let key = self.term_to_key(term);
        self.term_addresses.insert(key, address);
    }

    /// Look up if a term has already been allocated
    pub fn lookup_term(&self, term: &Term<DeBruijn>) -> Option<Address> {
        let key = self.term_to_key(term);
        self.term_addresses.get(&key).copied()
    }

    /// Register a constant with its address for deduplication
    pub fn register_constant(&mut self, constant: &Rc<Constant>, address: Address) {
        let key = self.constant_to_key(constant);
        self.constant_addresses.insert(key, address);
    }

    /// Look up if a constant has already been allocated
    pub fn lookup_constant(&self, constant: &Rc<Constant>) -> Option<Address> {
        let key = self.constant_to_key(constant);
        self.constant_addresses.get(&key).copied()
    }

    /// Convert a term to a key for deduplication
    fn term_to_key(&self, term: &Term<DeBruijn>) -> TermKey {
        match term {
            Term::Var(index) => TermKey::Variable(index.inner()),
            Term::Lambda { .. } => TermKey::Lambda,
            Term::Apply { .. } => TermKey::Apply,
            Term::Delay(_) => TermKey::Delay,
            Term::Force(_) => TermKey::Force,
            Term::Constant(constant) => TermKey::Constant(self.constant_to_key(constant)),
            Term::Error => TermKey::Error,
            Term::Builtin(builtin) => TermKey::Builtin(*builtin as u8),
            Term::Constr { tag, .. } => TermKey::Constructor(*tag),
            Term::Case { .. } => TermKey::Case,
        }
    }

    /// Convert a constant to a key for deduplication
    fn constant_to_key(&self, constant: &Rc<Constant>) -> ConstantKey {
        match &**constant {
            Constant::Integer(int) => {
                // We use the serialized bytes as the key
                let bytes = if int.bits() <= 64 {
                    if let Some(i) = int.to_i64() {
                        i.to_le_bytes().to_vec()
                    } else {
                        int.to_bytes_le().1
                    }
                } else {
                    int.to_bytes_le().1
                };
                ConstantKey::Integer(bytes)
            },
            Constant::ByteString(bytes) => ConstantKey::ByteString(bytes.clone()),
            Constant::String(s) => ConstantKey::String(s.clone()),
            Constant::Unit => ConstantKey::Unit,
            Constant::Bool(b) => ConstantKey::Bool(*b),
            Constant::Data(_) => {
                // This is a placeholder that should be improved
                ConstantKey::Data(vec![])
            },
            _ => ConstantKey::Unit, // Default fallback
        }
    }
}
