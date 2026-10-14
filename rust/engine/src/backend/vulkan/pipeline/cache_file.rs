//! On-disk pipeline cache: location, file name, container format and header validation. Pure functions, unit tested.
//!
//! Container (little endian): magic `FLLMVKPC`, `u32` format version, `u32` payload length, `u64` FNV-1a of the payload, payload. The
//! payload is the data of `vkGetPipelineCacheData`. Driver caches are not hardened against corrupt input, so a truncated or damaged file
//! must never reach `vkCreatePipelineCache`; the length and checksum guard against that, and the Vulkan header inside the payload (vendor,
//! device, `pipelineCacheUUID`) is compared with the live device before the data is handed over.

use std::ffi::OsString;
use std::path::PathBuf;

const MAGIC: &[u8; 8] = b"FLLMVKPC";
const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 24;
/// `VkPipelineCacheHeaderVersionOne` up to and including the UUID.
const VK_HEADER_LEN: usize = 32;

/// 64-bit FNV-1a (corruption detection, not security).
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Directory of the cache files: `FLLM_VK_CACHE_DIR`, else `$XDG_CACHE_HOME/fllm`, else `~/.cache/fllm`
/// (`%LOCALAPPDATA%\fllm` on Windows). `env` is injectable for tests.
pub fn cache_dir_with(env: &dyn Fn(&str) -> Option<OsString>, windows: bool) -> Option<PathBuf> {
    let non_empty = |name: &str| env(name).filter(|v| !v.is_empty());
    if let Some(dir) = non_empty("FLLM_VK_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    if windows {
        if let Some(dir) = non_empty("LOCALAPPDATA") {
            return Some(PathBuf::from(dir).join("fllm"));
        }
    }
    if let Some(dir) = non_empty("XDG_CACHE_HOME") {
        return Some(PathBuf::from(dir).join("fllm"));
    }
    let home = non_empty("HOME").or_else(|| non_empty("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".cache").join("fllm"))
}

/// [`cache_dir_with`] for the real environment. `None` when the cache is disabled with `FLLM_VK_PIPELINE_CACHE=0` or no home exists.
pub fn cache_dir() -> Option<PathBuf> {
    if std::env::var_os("FLLM_VK_PIPELINE_CACHE").is_some_and(|v| v == "0") {
        return None;
    }
    cache_dir_with(&|name| std::env::var_os(name), cfg!(windows))
}

/// File name for one device: vendor, device id and the pipeline-cache UUID (which also changes with the driver version).
pub fn cache_file_name(vendor_id: u32, device_id: u32, pipeline_cache_uuid: &[u8; 16]) -> String {
    let uuid: String = pipeline_cache_uuid.iter().map(|b| format!("{b:02x}")).collect();
    format!("vk-pipeline-v{FORMAT_VERSION}-{vendor_id:04x}-{device_id:04x}-{uuid}.bin")
}

/// Wrap the driver data in the container.
pub fn encode(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&fnv1a64(payload).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// The payload of a container, or `None` when the magic, version, length or checksum do not match.
pub fn decode(file: &[u8]) -> Option<&[u8]> {
    if file.len() < HEADER_LEN || &file[..8] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(file[8..12].try_into().ok()?);
    let len = u32::from_le_bytes(file[12..16].try_into().ok()?) as usize;
    let hash = u64::from_le_bytes(file[16..24].try_into().ok()?);
    let payload = file.get(HEADER_LEN..HEADER_LEN.checked_add(len)?)?;
    (version == FORMAT_VERSION && file.len() == HEADER_LEN + len && fnv1a64(payload) == hash).then_some(payload)
}

/// The Vulkan header at the start of `payload` (`VkPipelineCacheHeaderVersionOne`) belongs to this device.
pub fn vulkan_header_matches(payload: &[u8], vendor_id: u32, device_id: u32, pipeline_cache_uuid: &[u8; 16]) -> bool {
    if payload.len() < VK_HEADER_LEN {
        return false;
    }
    let word = |i: usize| u32::from_le_bytes([payload[i], payload[i + 1], payload[i + 2], payload[i + 3]]);
    let header_size = word(0) as usize;
    let header_version = word(4);
    header_size >= VK_HEADER_LEN
        && header_size <= payload.len()
        && header_version == 1
        && word(8) == vendor_id
        && word(12) == device_id
        && &payload[16..32] == pipeline_cache_uuid
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = pairs.iter().map(|(k, v)| ((*k).to_owned(), OsString::from(v))).collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn cache_dir_resolution_order() {
        let uuid_dir = |pairs: &[(&str, &str)], windows| cache_dir_with(&env_of(pairs), windows);
        assert_eq!(
            uuid_dir(&[("FLLM_VK_CACHE_DIR", "/x"), ("XDG_CACHE_HOME", "/y")], false),
            Some(PathBuf::from("/x"))
        );
        assert_eq!(
            uuid_dir(&[("XDG_CACHE_HOME", "/y"), ("HOME", "/h")], false),
            Some(PathBuf::from("/y").join("fllm"))
        );
        assert_eq!(
            uuid_dir(&[("HOME", "/h")], false),
            Some(PathBuf::from("/h").join(".cache").join("fllm"))
        );
        // an empty XDG_CACHE_HOME counts as unset
        assert_eq!(
            uuid_dir(&[("XDG_CACHE_HOME", ""), ("HOME", "/h")], false),
            Some(PathBuf::from("/h").join(".cache").join("fllm"))
        );
        assert_eq!(
            uuid_dir(&[("LOCALAPPDATA", "C:\\L"), ("USERPROFILE", "C:\\U")], true),
            Some(PathBuf::from("C:\\L").join("fllm"))
        );
        assert_eq!(
            uuid_dir(&[("USERPROFILE", "C:\\U")], true),
            Some(PathBuf::from("C:\\U").join(".cache").join("fllm"))
        );
        assert_eq!(uuid_dir(&[], false), None);
    }

    #[test]
    fn file_name_carries_vendor_device_and_uuid() {
        let uuid = [0xAB, 0xCD, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0xFF];
        assert_eq!(
            cache_file_name(0x10DE, 0x2684, &uuid),
            "vk-pipeline-v1-10de-2684-abcd000102030405060708090a0b0cff.bin"
        );
        let other = [1u8; 16];
        assert_ne!(cache_file_name(0x10DE, 0x2684, &uuid), cache_file_name(0x10DE, 0x2684, &other));
        assert_ne!(cache_file_name(0x10DE, 0x2684, &uuid), cache_file_name(0x10DE, 0x2685, &uuid));
    }

    #[test]
    fn container_round_trip_and_corruption() {
        let payload: Vec<u8> = (0..200u32).map(|i| (i * 7) as u8).collect();
        let file = encode(&payload);
        assert_eq!(decode(&file), Some(payload.as_slice()));
        assert_eq!(decode(&encode(&[])), Some(&[][..]));

        let mut flipped = file.clone();
        flipped[40] ^= 1;
        assert_eq!(decode(&flipped), None, "checksum");
        assert_eq!(decode(&file[..file.len() - 1]), None, "truncated");
        let mut longer = file.clone();
        longer.push(0);
        assert_eq!(decode(&longer), None, "trailing bytes");
        let mut bad_magic = file.clone();
        bad_magic[0] = b'X';
        assert_eq!(decode(&bad_magic), None);
        let mut bad_version = file;
        bad_version[8] = 9;
        assert_eq!(decode(&bad_version), None);
        assert_eq!(decode(&[1, 2, 3]), None);
    }

    fn vulkan_payload(vendor: u32, device: u32, uuid: &[u8; 16]) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(&32u32.to_le_bytes());
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&vendor.to_le_bytes());
        p.extend_from_slice(&device.to_le_bytes());
        p.extend_from_slice(uuid);
        p.extend_from_slice(&[9; 64]);
        p
    }

    #[test]
    fn vulkan_header_check() {
        let uuid = [7u8; 16];
        let good = vulkan_payload(0x1002, 0x744c, &uuid);
        assert!(vulkan_header_matches(&good, 0x1002, 0x744c, &uuid));
        assert!(!vulkan_header_matches(&good, 0x10de, 0x744c, &uuid));
        assert!(!vulkan_header_matches(&good, 0x1002, 0x1, &uuid));
        assert!(!vulkan_header_matches(&good, 0x1002, 0x744c, &[8u8; 16]));
        assert!(!vulkan_header_matches(&good[..20], 0x1002, 0x744c, &uuid), "too short");
        let mut bad_version = good.clone();
        bad_version[4] = 2;
        assert!(!vulkan_header_matches(&bad_version, 0x1002, 0x744c, &uuid));
        let mut bad_size = good;
        bad_size[0] = 8;
        assert!(!vulkan_header_matches(&bad_size, 0x1002, 0x744c, &uuid));
    }

    #[test]
    fn fnv_reference_values() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
