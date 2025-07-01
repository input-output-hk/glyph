use std::fs;
use uplc_serializer::parse_and_serialize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define a more complex UPLC program that includes various term types:
    // - Lambda, Apply, Delay, Force
    // - Multiple constants (integer, bytestring, string, bool)
    // - Builtins
    // - Error handling
    let uplc_text = r#"
    (program 1.0.0
      [
        (lam f
          (lam x
            [
              [
                (force f)
                x
              ]
              [
                (builtin addInteger)
                (con integer 40)
                (con integer 2)
              ]
            ]
          )
        )
        (delay
          (lam arg1
            (lam arg2
              [
                (lam cond
                  [
                    [
                      (builtin ifThenElse)
                      cond
                      (delay arg1)
                      (delay arg2)
                    ]
                  ]
                )
                [
                  (builtin lessThanInteger)
                  arg1
                  arg2
                ]
              ]
            )
          )
        )
      ]
    )
    "#;

    // Parse and serialize the UPLC program
    let binary = parse_and_serialize(uplc_text)?;

    // Write the binary to a file
    fs::write("complex_program.bin", &binary)?;

    // Print some information about the serialized program
    println!("Successfully serialized complex UPLC program:");
    println!("  - Size: {} bytes", binary.len());
    println!("  - Magic bytes: {:?}", &binary[0..4]);
    println!("  - Version: {}.{}.{}", binary[4], binary[5], binary[6]);
    println!(
        "  - Root term address: 0x{:08x}",
        u32::from_le_bytes([binary[8], binary[9], binary[10], binary[11]])
    );

    // Also try a program with different constant types
    let uplc_text_constants = r#"
    (program 1.0.0
      [
        (lam x
          [
            (lam y
              [
                (lam z
                  [
                    (lam b
                      [
                        [
                          (builtin ifThenElse)
                          b
                          (delay x)
                          (delay
                            [
                              [
                                (builtin appendString)
                                y
                              ]
                              z
                            ]
                          )
                        ]
                      ]
                    )
                    (con bool True)
                  ]
                )
                (con byteString #"48656c6c6f")  // "Hello" in hex
              ]
            )
            (con string "World")
          ]
        )
        (con integer 42)
      ]
    )
    "#;

    // Parse and serialize the second UPLC program
    let binary_constants = parse_and_serialize(uplc_text_constants)?;

    // Write the binary to a file
    fs::write("constants_program.bin", &binary_constants)?;

    println!("\nSuccessfully serialized UPLC program with various constants:");
    println!("  - Size: {} bytes", binary_constants.len());

    // Print a hex dump of the first 128 bytes (or less if smaller)
    println!("\nFirst 128 bytes of binary (hex dump):");
    let display_size = std::cmp::min(binary_constants.len(), 128);

    // Print in chunks of 16 bytes
    for (i, chunk) in binary_constants[..display_size].chunks(16).enumerate() {
        print!("{:08x}  ", i * 16);

        for b in chunk {
            print!("{b:02x} ");
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

    println!("\nOutputs written to complex_program.bin and constants_program.bin");

    Ok(())
}
