/// Term type tags (1 byte)
pub mod term_tag {
    pub const VARIABLE: u32 = 0x00;
    pub const DELAY: u32 = 0x01;
    pub const LAMBDA: u32 = 0x02;
    pub const APPLY: u32 = 0x03;
    pub const CONSTANT: u32 = 0x04;
    pub const FORCE: u32 = 0x05;
    pub const ERROR: u32 = 0x06;
    pub const BUILTIN: u32 = 0x07;
    pub const CONSTRUCTOR: u32 = 0x08;
    pub const CASE: u32 = 0x09;
}

/// Constant type tags (1 byte)
pub mod const_tag {
    pub const INTEGER: u32 = 0x00;
    pub const BYTESTRING: u32 = 0x01;
    pub const STRING: u32 = 0x02;
    pub const UNIT: u32 = 0x03;
    pub const BOOL: u32 = 0x04;
    pub const DATA: u32 = 0x05;
}

/// Data variant tags (1 byte)
pub mod data_tag {
    pub const CONSTR: u32 = 0x00;
    pub const MAP: u32 = 0x01;
    pub const LIST: u32 = 0x02;
    pub const INTEGER: u32 = 0x03;
    pub const BYTESTRING: u32 = 0x04;
}

/// Boolean values
pub mod bool_val {
    pub const FALSE: u32 = 0x00;
    pub const TRUE: u32 = 0x01;
}

pub mod value_tag {
    pub const CONSTANT: u32 = 0x00;
}
