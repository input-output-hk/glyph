/// Term type tags (1 byte)
pub mod term_tag {
    pub const VARIABLE: u8 = 0x00;
    pub const LAMBDA: u8 = 0x01;
    pub const APPLY: u8 = 0x02;
    pub const FORCE: u8 = 0x03;
    pub const DELAY: u8 = 0x04;
    pub const CONSTANT: u8 = 0x05;
    pub const BUILTIN: u8 = 0x06;
    pub const ERROR: u8 = 0x07;
    pub const CONSTRUCTOR: u8 = 0x08;
    pub const CASE: u8 = 0x09;
}

/// Constant type tags (1 byte)
pub mod const_tag {
    pub const INTEGER: u8 = 0x00;
    pub const BYTESTRING: u8 = 0x01;
    pub const STRING: u8 = 0x02;
    pub const UNIT: u8 = 0x03;
    pub const BOOL: u8 = 0x04;
    pub const DATA: u8 = 0x05;
}

/// Data variant tags (1 byte)
pub mod data_tag {
    pub const CONSTR: u8 = 0x00;
    pub const MAP: u8 = 0x01;
    pub const LIST: u8 = 0x02;
    pub const INTEGER: u8 = 0x03;
    pub const BYTESTRING: u8 = 0x04;
}

/// Integer size indicators
pub mod int_size {
    pub const SMALL: u8 = 0x01;   // 1 byte (-128 to 127)
    pub const MEDIUM: u8 = 0x02;  // 2 bytes (-32768 to 32767)
    pub const LARGE: u8 = 0x04;   // 4 bytes (-2^31 to 2^31-1)
    pub const XLARGE: u8 = 0x08;  // 8 bytes (64-bit integers)
    pub const BIGINT: u8 = 0xFF;  // Variable length BigInt
}

/// Boolean values
pub mod bool_val {
    pub const FALSE: u8 = 0x00;
    pub const TRUE: u8 = 0x01;
}

/// Memory regions (base addresses)
pub mod memory_region {
    // First 64KB is for static data
    pub const PROGRAM_HEADER: u32 = 0x00000000; // Size: 16 bytes
    pub const STATIC_RESERVED: u32 = 0x00000010; // Rest of static region
    
    // Term region (next 64KB)
    pub const TERM_REGION: u32 = 0x00010000;    // Size: 64KB
    pub const TERM_REGION_END: u32 = 0x0001FFFF;
    
    // Constant pool (next 256KB)
    pub const INTEGER_POOL: u32 = 0x00020000;   // Size: 64KB
    pub const BYTESTRING_POOL: u32 = 0x00030000; // Size: 64KB
    pub const STRING_POOL: u32 = 0x00040000;    // Size: 64KB
    pub const COMPLEX_DATA_POOL: u32 = 0x00050000; // Size: 64KB
    
    // Dynamic memory (next 640KB)
    pub const ENVIRONMENT: u32 = 0x00060000;    // Size: 128KB
    pub const CONTINUATION_STACK: u32 = 0x00080000; // Size: 128KB
    pub const HEAP: u32 = 0x000A0000;           // Size: 384KB
    pub const MEMORY_END: u32 = 0x000FFFFF;
}

/// Program header 
pub mod header {
    pub const MAGIC: &[u8; 4] = b"UPLC";  // Magic bytes
    
    // Version encoding:
    // - Version is stored as 3 bytes (major, minor, patch)
    // - Each version component is a single byte (0-255)
    pub const VERSION_MAJOR_OFFSET: usize = 4;
    pub const VERSION_MINOR_OFFSET: usize = 5;
    pub const VERSION_PATCH_OFFSET: usize = 6;
    
    // Root term offset (4 bytes starting at offset 8)
    pub const ROOT_TERM_OFFSET: usize = 8;
    
    // Total header size: 12 bytes (4 magic + 4 version + 4 root offset)
    pub const HEADER_SIZE: usize = 12;
}

/// Value type tags (for CEK machine values)
pub mod value_tag {
    pub const INTEGER: u8 = 0x00;
    pub const BYTESTRING: u8 = 0x01;
    pub const STRING: u8 = 0x02;
    pub const UNIT: u8 = 0x03;
    pub const BOOLEAN: u8 = 0x04;
    pub const DATA: u8 = 0x05;
    pub const CLOSURE: u8 = 0x06;
    pub const DELAYED: u8 = 0x07;
    pub const BUILTIN: u8 = 0x08;
    pub const CONSTRUCTOR: u8 = 0x09;
} 