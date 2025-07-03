use std::path::Path;
use std::process::Command;

fn main() {
    // Get the project root directory (two levels up from crates/glyph/)
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let runtime_dir = project_root.join("runtime");

    // Tell Cargo to rerun this build script if any files in the runtime dir change
    println!("cargo:rerun-if-changed={}", runtime_dir.display());

    // Run zig build in the runtime directory
    let output = Command::new("zig")
        .arg("build")
        .current_dir(&runtime_dir)
        .output()
        .expect("Failed to execute zig build - make sure zig is installed and in PATH");

    // Print stdout and stderr for debugging
    if !output.stdout.is_empty() {
        println!(
            "zig build stdout: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    if !output.stderr.is_empty() {
        eprintln!(
            "zig build stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Fail the build if zig build failed
    if !output.status.success() {
        panic!(
            "zig build failed with exit code: {:?}",
            output.status.code()
        );
    }

    println!("zig build completed successfully");
}
