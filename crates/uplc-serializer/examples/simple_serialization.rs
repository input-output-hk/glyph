use std::fs;
use std::io::Write;
use uplc_serializer::parse_and_serialize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A simple UPLC program: (program 1.0.0 [(lam x [(builtin addInteger) x (con integer 1)]) (con integer 41)])
    // This computes 41 + 1 = 42
    let uplc_text = r#"
    (program 1.0.0
      [
        (lam x
          [
            (builtin addInteger)
            x
            (con integer 1)
          ]
        )
        (con integer 41)
      ]
    )
    "#;

    // Parse and serialize the UPLC program
    let binary = parse_and_serialize(uplc_text)?;
    
    // Write the binary to a file
    fs::write("simple_program.bin", &binary)?;
    
    // Print some information about the serialized program
    println!("Successfully serialized UPLC program:");
    println!("  - Size: {} bytes", binary.len());
    println!("  - Magic bytes: {:?}", &binary[0..4]);
    println!("  - Version: {}.{}.{}", binary[4], binary[5], binary[6]);
    println!("  - Root term address: 0x{:08x}", 
             u32::from_le_bytes([binary[8], binary[9], binary[10], binary[11]]));
    
    // Print the binary representation in a readable hex format
    println!("\nBinary representation:");
    
    // Print in chunks of 16 bytes
    for (i, chunk) in binary.chunks(16).enumerate() {
        print!("{:08x}  ", i * 16);
        
        for b in chunk {
            print!("{:02x} ", b);
        }
        
        // Pad if less than 16 bytes
        for _ in chunk.len()..16 {
            print!("   ");
        }
        
        print!(" ");
        
        // Print ASCII representation
        for b in chunk {
            if *b >= 32 && *b <= 126 {
                print!("{}", *b as char);
            } else {
                print!(".");
            }
        }
        
        println!();
    }
    
    println!("\nOutput written to simple_program.bin");
    
    Ok(())
} 