use std::{fs, path::Path};
use std::process::Command;

fn main() {
    // Get the project root directory (two levels up from crates/glyph/)
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let runtime_dir = project_root.join("runtime");
    let cargo_lock = project_root.join("Cargo.lock");

    // Tell Cargo to rerun this build script if any files in the runtime dir change
    println!("cargo:rerun-if-changed={}", runtime_dir.display());
    println!("cargo:rerun-if-changed={}", cargo_lock.display());

    if let Ok(lock_contents) = fs::read_to_string(&cargo_lock) {
        if let Some((version, git_sha)) = extract_emulator_version(&lock_contents) {
            println!("cargo:rustc-env=GLYPH_BITVMX_CPU_VERSION={}", version);
            if let Some(git_sha) = git_sha {
                println!("cargo:rustc-env=GLYPH_BITVMX_CPU_GIT_SHA={}", git_sha);
            }
        }
    }

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

fn extract_emulator_version(lock_contents: &str) -> Option<(String, Option<String>)> {
    let mut in_emulator = false;
    let mut version = None;
    let mut source = None;

    for line in lock_contents.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            in_emulator = false;
            continue;
        }

        if line.starts_with("name = ") {
            in_emulator = line == "name = \"emulator\"";
            continue;
        }

        if !in_emulator {
            continue;
        }

        if line.starts_with("version = ") {
            version = extract_quoted_value(line);
        } else if line.starts_with("source = ") {
            source = extract_quoted_value(line);
        }

        if version.is_some() && source.is_some() {
            break;
        }
    }

    let version = version?;
    let git_sha = source.and_then(|value| value.split('#').nth(1).map(|v| v.to_string()));
    Some((version, git_sha))
}

fn extract_quoted_value(line: &str) -> Option<String> {
    let mut parts = line.split('"');
    parts.next()?;
    parts.next().map(|value| value.to_string())
}
