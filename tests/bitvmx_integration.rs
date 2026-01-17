use std::path::PathBuf;

use emulator::ExecutionResult;
use glyph::bitvmx::{BundleManifest, BundleOptions, MANIFEST_FILE_NAME, MAPPING_FILE_NAME, ROM_COMMITMENT_FILE_NAME};
use std::fs;

#[test]
fn instruction_mapping_is_non_empty() {
    let output = glyph::bitvmx::instruction_mapping().unwrap();
    assert!(!output.trim().is_empty());
}

#[test]
fn rom_commitment_is_non_empty() {
    let elf_path = PathBuf::from("tests/fixtures/mul.elf");
    let output = glyph::bitvmx::rom_commitment(&elf_path, false).unwrap();
    assert!(!output.trim().is_empty());
}

#[test]
fn execute_simple_elf_halts() {
    let elf_path = PathBuf::from("tests/fixtures/simple.elf");
    let mut opts = glyph::bitvmx::ExecuteOptions::default();
    opts.elf_path = Some(elf_path);
    opts.limit = Some(1000);

    let result = glyph::bitvmx::execute_elf(opts).unwrap();
    match result.execution {
        ExecutionResult::Halt(code, _) => assert_eq!(code, 0),
        other => panic!("expected halt execution result, got {other:?}"),
    }
}

#[test]
fn bundle_writes_artifacts() {
    let elf_path = PathBuf::from("tests/fixtures/mul.elf");
    let temp = tempfile::tempdir().unwrap();
    let out_dir = temp.path().to_path_buf();

    let opts = BundleOptions {
        elf_path,
        out_dir: out_dir.clone(),
        sections: false,
    };

    let _manifest = glyph::bitvmx::bundle(opts).unwrap();

    let mapping_path = out_dir.join(MAPPING_FILE_NAME);
    let commitment_path = out_dir.join(ROM_COMMITMENT_FILE_NAME);
    let manifest_path = out_dir.join(MANIFEST_FILE_NAME);

    assert!(mapping_path.exists());
    assert!(commitment_path.exists());
    assert!(manifest_path.exists());

    let manifest_contents = fs::read_to_string(manifest_path).unwrap();
    let manifest: BundleManifest = serde_json::from_str(&manifest_contents).unwrap();
    assert_eq!(manifest.mapping_file, MAPPING_FILE_NAME);
    assert_eq!(manifest.commitment_file, ROM_COMMITMENT_FILE_NAME);
}
