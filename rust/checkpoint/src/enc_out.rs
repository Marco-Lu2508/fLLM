//! Shared tail of the encoder converters: write the GGUF atomically and build the report.
use crate::{
    gguf::{self, Meta, OutTensor},
    CheckpointError, ConversionReport, ConvertOptions,
};
use std::{fs, path::Path};

/// What an encoder converter prepares: the metadata, the output tensors and the names of the skipped source tensors.
pub(crate) type Prepared = (Vec<(String, Meta)>, Vec<OutTensor>, Vec<String>);

#[allow(dead_code)]
pub(crate) fn write(
    input: &Path,
    output: &Path,
    options: &ConvertOptions,
    meta: &[(String, Meta)],
    mut tensors: Vec<OutTensor>,
    skipped: Vec<String>,
) -> Result<ConversionReport, CheckpointError> {
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    if output.exists() && !options.overwrite {
        return Err(format!("output already exists: {}", output.display()).into());
    }
    if input.is_file() && fs::canonicalize(input).ok() == fs::canonicalize(output).ok() {
        return Err("input and output must differ".into());
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut tmp = output.as_os_str().to_owned();
    tmp.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let tmp = std::path::PathBuf::from(tmp);
    let result = (|| -> Result<u64, CheckpointError> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        let size = gguf::write_to(&mut file, meta, &tensors)?;
        file.sync_all()?;
        drop(file);
        if options.overwrite {
            fs::rename(&tmp, output)?;
        } else {
            fs::hard_link(&tmp, output)?;
            fs::remove_file(&tmp)?;
        }
        Ok(size)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    Ok(ConversionReport {
        architecture: "bert".into(),
        tensors_written: tensors.len(),
        bytes_written: result?,
        skipped_tensors: skipped,
        output: output.to_owned(),
    })
}
