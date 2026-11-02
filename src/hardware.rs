use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::info;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu_cores: usize,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub free_disk_gb: f64,
    pub storage_profile: String,
    pub recommended_mmap_page_size: usize,
    pub recommended_batch_size: usize,
    pub mode_name: String,
}

#[cfg(windows)]
#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[cfg(windows)]
#[repr(C)]
struct StoragePropertyQuery {
    property_id: u32,
    query_type: u32,
    additional_parameters: [u8; 1],
}

#[cfg(windows)]
#[repr(C)]
struct DeviceSeekPenaltyDescriptor {
    version: u32,
    size: u32,
    incurs_seek_penalty: u8,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    fn CreateFileW(
        lpFileName: *const u16,
        dwDesiredAccess: u32,
        dwShareMode: u32,
        lpSecurityAttributes: *mut std::ffi::c_void,
        dwCreationDisposition: u32,
        dwFlagsAndAttributes: u32,
        hTemplateFile: *mut std::ffi::c_void,
    ) -> isize;
    fn DeviceIoControl(
        hDevice: isize,
        dwIoControlCode: u32,
        lpInBuffer: *const std::ffi::c_void,
        nInBufferSize: u32,
        lpOutBuffer: *mut std::ffi::c_void,
        nOutBufferSize: u32,
        lpBytesReturned: *mut u32,
        lpOverlapped: *mut std::ffi::c_void,
    ) -> i32;
    fn CloseHandle(hObject: isize) -> i32;
}

#[cfg(windows)]
fn probe_seek_penalty(path: &Path) -> Option<bool> {

    let full = std::fs::canonicalize(path).ok().unwrap_or_else(|| path.to_path_buf());
    let path_str = full.to_string_lossy();
    let clean = path_str.trim_start_matches(r"\\?\");
    let drive_letter = if clean.len() >= 2 && clean.as_bytes()[1] == b':' {
        Some(clean.chars().next().unwrap())
    } else {
        None
    };

    let dl = drive_letter?;
    let volume_path = format!(r"\\.\{}:", dl);
    let wide: Vec<u16> = volume_path.encode_utf16().chain(std::iter::once(0)).collect();

    const FILE_SHARE_READ: u32 = 0x00000001;
    const FILE_SHARE_WRITE: u32 = 0x00000002;
    const OPEN_EXISTING: u32 = 3;
    const IOCTL_STORAGE_QUERY_PROPERTY: u32 = 0x002D1400;
    const STORAGE_DEVICE_SEEK_PENALTY_PROPERTY: u32 = 7;
    const PROPERTY_STANDARD_QUERY: u32 = 0;

    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            0, // Query access only (unprivileged)
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null_mut(),
            OPEN_EXISTING,
            0,
            std::ptr::null_mut(),
        )
    };

    if handle == -1 || handle == 0 {
        return None;
    }

    let query = StoragePropertyQuery {
        property_id: STORAGE_DEVICE_SEEK_PENALTY_PROPERTY,
        query_type: PROPERTY_STANDARD_QUERY,
        additional_parameters: [0],
    };

    let mut desc = DeviceSeekPenaltyDescriptor {
        version: 0,
        size: 0,
        incurs_seek_penalty: 0,
    };
    let mut bytes_returned = 0u32;

    let ok = unsafe {
        DeviceIoControl(
            handle,
            IOCTL_STORAGE_QUERY_PROPERTY,
            &query as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<StoragePropertyQuery>() as u32,
            &mut desc as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of::<DeviceSeekPenaltyDescriptor>() as u32,
            &mut bytes_returned,
            std::ptr::null_mut(),
        )
    };

    unsafe { CloseHandle(handle) };

    if ok != 0 {
        Some(desc.incurs_seek_penalty != 0)
    } else {
        None
    }
}

#[cfg(not(windows))]
fn probe_seek_penalty(_path: &Path) -> Option<bool> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/block") {
            for entry in entries.flatten() {
                let rot_path = entry.path().join("queue/rotational");
                if let Ok(val) = std::fs::read_to_string(&rot_path) {
                    let trimmed = val.trim();
                    if trimmed == "0" {
                        return Some(false); // SSD / NVMe
                    } else if trimmed == "1" {
                        return Some(true); // Rotational HDD
                    }
                }
            }
        }
    }
    None
}

impl HardwareProfile {
    pub fn detect<P: AsRef<Path>>(data_dir: P) -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let (total_ram_mb, available_ram_mb) = Self::detect_ram();
        let free_disk_gb = Self::detect_disk(data_dir.as_ref());
        let seek_penalty = probe_seek_penalty(data_dir.as_ref());

        // Dynamic hardware adaptation logic based on verified seek penalty and physical RAM
        let (storage_profile, recommended_mmap_page_size, recommended_batch_size, mode_name) =
            match seek_penalty {
                Some(false) => {
                    // Confirmed Solid-State Drive (Zero Seek Penalty via OS DeviceIoControl / sysfs)
                    if total_ram_mb >= 16_000 && cpu_cores >= 8 {
                        (
                            "Solid-State Storage (SSD/NVMe, Zero Seek Penalty)".to_string(),
                            64 * 1024, // 64 KB page chunking
                            5000,      // 5,000 slots pre-allocation step
                            "God Reboot // Hyperdrive Edition".to_string(),
                        )
                    } else {
                        (
                            "Solid-State Storage (SSD/NVMe, Zero Seek Penalty)".to_string(),
                            32 * 1024, // 32 KB page chunking
                            2000,      // 2,000 slots pre-allocation step
                            "God Reboot // High-Speed Edition".to_string(),
                        )
                    }
                }
                Some(true) => {
                    // Confirmed Rotational Magnetic Drive (Seek Penalty detected)
                    (
                        "Rotational Magnetic Storage (HDD, Seek Penalty Detected)".to_string(),
                        4 * 1024, // 4 KB conservative page
                        500,      // 500 slots conservative step
                        "God Reboot // Conservative HDD Edition".to_string(),
                    )
                }
                None => {
                    // Unverified or access-restricted: do not invent NVMe, be strictly honest
                    let page_sz = if total_ram_mb >= 8_000 { 32 * 1024 } else { 4 * 1024 };
                    let batch_sz = if total_ram_mb >= 8_000 { 1000 } else { 500 };
                    (
                        "Standard Storage (Unverified)".to_string(),
                        page_sz,
                        batch_sz,
                        "God Reboot // Balanced Edition".to_string(),
                    )
                }
            };

        let profile = Self {
            cpu_cores,
            total_ram_mb,
            available_ram_mb,
            free_disk_gb,
            storage_profile,
            recommended_mmap_page_size,
            recommended_batch_size,
            mode_name,
        };

        info!(
            "⚡ StarkDB Hardware Engine: {} Cores | {:.1} GB RAM (free: {:.1} GB) | Media: {} | Mode: {}",
            profile.cpu_cores,
            profile.total_ram_mb as f64 / 1024.0,
            profile.available_ram_mb as f64 / 1024.0,
            profile.storage_profile,
            profile.mode_name
        );

        profile
    }

    #[cfg(windows)]
    fn detect_ram() -> (u64, u64) {
        let mut status = MemoryStatusEx {
            dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };

        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok != 0 {
            (
                status.ull_total_phys / (1024 * 1024),
                status.ull_avail_phys / (1024 * 1024),
            )
        } else {
            (8192, 4096)
        }
    }

    #[cfg(not(windows))]
    fn detect_ram() -> (u64, u64) {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                let mut total_kb = 0u64;
                let mut avail_kb = 0u64;
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(val) = line.split_whitespace().nth(1) {
                            total_kb = val.parse().unwrap_or(0);
                        }
                    } else if line.starts_with("MemAvailable:") {
                        if let Some(val) = line.split_whitespace().nth(1) {
                            avail_kb = val.parse().unwrap_or(0);
                        }
                    }
                }
                if total_kb > 0 {
                    return (total_kb / 1024, avail_kb / 1024);
                }
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
                if let Ok(bytes_str) = std::str::from_utf8(&output.stdout) {
                    if let Ok(bytes) = bytes_str.trim().parse::<u64>() {
                        let total_mb = bytes / (1024 * 1024);
                        return (total_mb, total_mb / 2);
                    }
                }
            }
        }
        (8192, 4096)
    }

    fn detect_disk(path: &Path) -> f64 {
        if let Ok(space) = fs2::available_space(path) {
            space as f64 / (1024.0 * 1024.0 * 1024.0)
        } else if let Ok(space) = fs2::available_space(".") {
            space as f64 / (1024.0 * 1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let profile = HardwareProfile::detect(".");
        assert!(profile.cpu_cores >= 1);
        assert!(profile.total_ram_mb > 0);
        assert!(!profile.mode_name.is_empty());
        assert!(profile.recommended_mmap_page_size >= 4096);
    }
}
