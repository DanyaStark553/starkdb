use crate::format::{
    DbHeader, DbHeaderV3, ManifestSlot, NodeRecord, Sq8Meta, SynapseRecord, FEATURE_SQ8_READY,
    FLAG_DELETED, HEADER_PAGE_SIZE, PAGE_A_OFFSET, PAGE_B_OFFSET, STARKDB_MAGIC,
    STARKDB_V3_VERSION, STARKDB_VERSION, TOTAL_HEADER_SIZE,
};
use anyhow::{bail, Context, Result};
use fs2::FileExt;
use half::f16;
use hashbrown::HashTable;
use memmap2::MmapMut;
use smallvec::SmallVec;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

#[inline]
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325 ^ (bytes.len() as u64).wrapping_mul(0x9e3779b97f4a7c15);
    let mut i = 0;
    while i + 8 <= bytes.len() {
        let word = u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
        h = (h ^ word).wrapping_mul(0x100000001b3);
        i += 8;
    }
    while i < bytes.len() {
        h = (h ^ (bytes[i] as u64)).wrapping_mul(0x100000001b3);
        i += 1;
    }
    h ^ (h >> 32)
}

pub struct StarkStorage {
    /// Logical database base path (e.g. `.../jarvis.starkdb`), preserved across segment flips.
    pub path: PathBuf,
    /// Physical segment file currently memory-mapped (e.g. `.../jarvis.s000001.starkdb`).
    pub active_segment_path: PathBuf,
    pub active_seq: u64,
    pub rollback_seq: u64,
    file: File,
    mmap: MmapMut,
    current_header: DbHeader,
    active_page: u8,
    nodes_offset: usize,
    synapses_offset: usize,
    vectors_offset: usize,
    sq8_codes_offset: usize,
    sq8_meta_offset: usize,
    strings_offset: usize,
    strings_cap: usize,
    vector_dims: usize,
    string_cache: HashTable<(u32, u32)>,
    pub page_growth_size: usize,
    pub slot_allocation_step: u32,
}

impl StarkStorage {
    pub const INITIAL_MAX_NODES: u32 = 32;              // 32 nodes = 1.5 KB
    pub const INITIAL_MAX_SYNAPSES: u32 = 64;           // 64 synapses = 2.0 KB
    pub const INITIAL_MAX_VECTORS: u32 = 0;             // 0 bytes until vectors are used
    pub const INITIAL_STRING_CAPACITY: u64 = 4 * 1024;  // 4 KB initial string pool

    /// Returns the `<base_path>.manifest` path (128 B = 2 x 64 B `ManifestSlot`).
    pub fn manifest_path_for(base_path: &Path) -> PathBuf {
        PathBuf::from(format!("{}.manifest", base_path.display()))
    }

    /// Returns the physical file path for segment `seq` (`seq == 0` -> legacy `base_path`,
    /// `seq > 0` -> `<stem>.s{:06}.starkdb`).
    pub fn segment_path_for_seq(base_path: &Path, seq: u64) -> PathBuf {
        if seq == 0 {
            return base_path.to_path_buf();
        }
        let parent = base_path.parent().unwrap_or(Path::new("."));
        let stem = base_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("db");
        parent.join(format!("{}.s{:06}.starkdb", stem, seq))
    }

    /// Reads the 128-byte A/B manifest file and returns `(winning_slot, winning_slot_index)` if valid.
    pub fn read_manifest(base_path: &Path) -> Option<(ManifestSlot, u8)> {
        let mpath = Self::manifest_path_for(base_path);
        let bytes = std::fs::read(&mpath).ok()?;
        if bytes.len() < 128 {
            return None;
        }
        let slot_a: ManifestSlot = bytemuck::pod_read_unaligned(&bytes[0..64]);
        let slot_b: ManifestSlot = bytemuck::pod_read_unaligned(&bytes[64..128]);
        let va = slot_a.is_valid();
        let vb = slot_b.is_valid();
        match (va, vb) {
            (true, true) => {
                if slot_a.generation >= slot_b.generation {
                    Some((slot_a, 0))
                } else {
                    Some((slot_b, 1))
                }
            }
            (true, false) => Some((slot_a, 0)),
            (false, true) => Some((slot_b, 1)),
            (false, false) => None,
        }
    }

    /// Atomically updates the inactive 64 B slot of `<base_path>.manifest` via temp-file + rename.
    pub fn write_manifest_flip(
        base_path: &Path,
        active_seq: u64,
        rollback_seq: u64,
        vacuum_epoch: u64,
    ) -> Result<()> {
        let mpath = Self::manifest_path_for(base_path);
        let mut raw = [0u8; 128];
        if let Ok(existing) = std::fs::read(&mpath) {
            let copy_len = existing.len().min(128);
            raw[..copy_len].copy_from_slice(&existing[..copy_len]);
        }
        let (next_gen, target_idx) = match Self::read_manifest(base_path) {
            Some((slot, idx)) => (slot.generation.saturating_add(1), 1 - idx),
            None => (1, 0),
        };
        let new_slot = ManifestSlot::new(next_gen, active_seq, rollback_seq, vacuum_epoch);
        let offset = (target_idx as usize) * 64;
        raw[offset..offset + 64].copy_from_slice(bytemuck::bytes_of(&new_slot));

        let tmp_path = PathBuf::from(format!("{}.tmp", mpath.display()));
        {
            let mut f = File::create(&tmp_path)?;
            f.write_all(&raw)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp_path, &mpath)?;
        Ok(())
    }

    /// Cleans up orphan `*.vacuum_tmp` and `*.manifest.tmp` files for this database stem.
    fn cleanup_orphan_temp_files(base_path: &Path) {
        let m_tmp = PathBuf::from(format!("{}.manifest.tmp", base_path.display()));
        if m_tmp.exists() {
            let _ = std::fs::remove_file(&m_tmp);
        }

        let v_tmp = PathBuf::from(format!("{}.vacuum_tmp", base_path.display()));
        if v_tmp.exists() {
            let _ = std::fs::remove_file(&v_tmp);
        }

        let base_seq = Self::read_manifest(base_path)
            .map(|(s, _)| s.active_seq)
            .unwrap_or(0);
        for offset in 0..=3 {
            let seg = Self::segment_path_for_seq(base_path, base_seq + offset);
            let seg_tmp = PathBuf::from(format!("{}.vacuum_tmp", seg.display()));
            if seg_tmp.exists() {
                let _ = std::fs::remove_file(&seg_tmp);
            }
        }
    }

    /// Deletes the previous rollback segment `C` after a healthy read-only scrub pass.
    pub fn cleanup_rollback_segment(&mut self) -> Result<()> {
        if self.rollback_seq != u64::MAX && self.rollback_seq != self.active_seq {
            let old_seg = Self::segment_path_for_seq(&self.path, self.rollback_seq);
            if old_seg.exists() {
                let _ = std::fs::remove_file(&old_seg);
            }
            Self::write_manifest_flip(
                &self.path,
                self.active_seq,
                u64::MAX,
                self.current_header.vacuum_epoch,
            )?;
            self.rollback_seq = u64::MAX;
        }
        Ok(())
    }

    pub fn open_or_create<P: AsRef<Path>>(path: P, vector_dims: u16) -> Result<Self> {
        let parent = path.as_ref().parent().unwrap_or(Path::new("."));
        let hw = crate::hardware::HardwareProfile::detect(parent);
        Self::open_or_create_with_hardware(path, vector_dims, &hw)
    }

    pub fn open_or_create_with_hardware<P: AsRef<Path>>(
        path: P,
        vector_dims: u16,
        hw: &crate::hardware::HardwareProfile,
    ) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        if let Some(parent) = base_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        // 1. Clean up orphan *.vacuum_tmp / *.manifest.tmp files
        Self::cleanup_orphan_temp_files(&base_path);

        // 2. Recover from .compact_bak if an offline compaction crashed mid-rename
        let compact_bak = PathBuf::from(format!("{}.compact_bak", base_path.display()));
        if compact_bak.exists() {
            let base_valid = base_path
                .metadata()
                .map(|m| m.len() >= TOTAL_HEADER_SIZE as u64)
                .unwrap_or(false);
            if !base_valid {
                warn!(
                    "Recovering {:?} from interrupted compaction backup {:?}",
                    base_path, compact_bak
                );
                let _ = std::fs::rename(&compact_bak, &base_path);
            }
        }

        // 3. Check if A/B/C Manifest exists
        let manifest_path = Self::manifest_path_for(&base_path);
        if manifest_path.exists() {
            let Some((slot, _)) = Self::read_manifest(&base_path) else {
                bail!(
                    "Corrupted manifest at {:?}: neither Slot A nor Slot B has a valid CRC32 (refusing to overwrite database)",
                    manifest_path
                );
            };
            let active_seg_path = Self::segment_path_for_seq(&base_path, slot.active_seq);
            match Self::open_raw_segment_file(
                &active_seg_path,
                &base_path,
                slot.active_seq,
                slot.rollback_seq,
                vector_dims,
                hw,
                false,
            ) {
                Ok(storage) => return Ok(storage),
                Err(active_err) => {
                    if slot.rollback_seq != u64::MAX && slot.rollback_seq != slot.active_seq {
                        let rollback_seg_path =
                            Self::segment_path_for_seq(&base_path, slot.rollback_seq);
                        warn!(
                            "Active segment {:?} (seq={}) failed validation ({}); rolling back to C-segment {:?} (seq={})",
                            active_seg_path,
                            slot.active_seq,
                            active_err,
                            rollback_seg_path,
                            slot.rollback_seq
                        );
                        let storage = Self::open_raw_segment_file(
                            &rollback_seg_path,
                            &base_path,
                            slot.rollback_seq,
                            u64::MAX,
                            vector_dims,
                            hw,
                            false,
                        )
                        .with_context(|| {
                            format!(
                                "Both active segment (seq={}) and C-rollback segment (seq={}) failed validation",
                                slot.active_seq, slot.rollback_seq
                            )
                        })?;
                        let _ = Self::write_manifest_flip(
                            &base_path,
                            slot.rollback_seq,
                            u64::MAX,
                            storage.current_header.vacuum_epoch,
                        );
                        let _ = std::fs::remove_file(&active_seg_path);
                        return Ok(storage);
                    }
                    return Err(active_err);
                }
            }
        }

        // 4. Legacy / seq=0 path (no manifest yet)
        Self::open_raw_segment_file(&base_path, &base_path, 0, u64::MAX, vector_dims, hw, true)
    }

    /// Opens or creates a specific physical segment file (`seg_path`) bound to `base_path`.
    /// When `allow_create` is `false` (manifest-directed open), missing or truncated files return `Err`
    /// instead of silently initializing an empty database.
    pub fn open_raw_segment_file(
        seg_path: &Path,
        base_path: &Path,
        active_seq: u64,
        rollback_seq: u64,
        vector_dims: u16,
        hw: &crate::hardware::HardwareProfile,
        allow_create: bool,
    ) -> Result<Self> {
        let exists = seg_path.exists();
        if !exists && !allow_create {
            bail!("Segment file {:?} does not exist", seg_path);
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(allow_create)
            .truncate(false)
            .open(seg_path)
            .with_context(|| format!("Failed to open DB segment file at {:?}", seg_path))?;

        // 1. Cross-process exclusive file locking to prevent dual-opening corruption
        file.try_lock_exclusive().with_context(|| {
            format!(
                "Database file {:?} is locked by another running StarkDB instance or daemon. Concurrent writes are prevented to avoid data corruption.",
                seg_path
            )
        })?;

        let file_len = file.metadata()?.len();

        // 2. Check if this is a legacy v3 file (starts with STARKDB_MAGIC and version 3)
        if exists && file_len >= 128 && file_len < TOTAL_HEADER_SIZE as u64 {
            let temp_mmap = unsafe { MmapMut::map_mut(&file)? };
            let v3_hdr = bytemuck::from_bytes::<DbHeaderV3>(&temp_mmap[0..128]);
            if v3_hdr.magic == STARKDB_MAGIC && v3_hdr.version == STARKDB_V3_VERSION {
                info!(
                    "Upgrading legacy database {:?} from v3 (Wing Accel) to v5 (Ultimate Legacy)",
                    seg_path
                );
                drop(temp_mmap);
                Self::migrate_v3_to_v5(&file, seg_path, vector_dims)?;
            }
        }

        // 3. If file is brand new or smaller than total header size, initialize it if allowed
        if !exists || file.metadata()?.len() < TOTAL_HEADER_SIZE as u64 {
            if !allow_create {
                bail!(
                    "Segment file {:?} is truncated ({} bytes < {})",
                    seg_path,
                    file_len,
                    TOTAL_HEADER_SIZE
                );
            }
            Self::initialize_db_file(&file, vector_dims, Self::INITIAL_STRING_CAPACITY)?;
        }

        // 4. Read A/B headers and validate CRC32 (supporting v5 and in-place v4 -> v5 upgrade)
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        let page_a =
            *bytemuck::from_bytes::<DbHeader>(&mmap[PAGE_A_OFFSET..PAGE_A_OFFSET + HEADER_PAGE_SIZE]);
        let page_b =
            *bytemuck::from_bytes::<DbHeader>(&mmap[PAGE_B_OFFSET..PAGE_B_OFFSET + HEADER_PAGE_SIZE]);

        let valid_a = page_a.is_valid();
        let valid_b = page_b.is_valid();

        let (current_header, active_page) = if valid_a && valid_b {
            if page_a.generation >= page_b.generation {
                (page_a, 0)
            } else {
                (page_b, 1)
            }
        } else if valid_a {
            info!(
                "Recovered database {:?} using Header Page A (Page B had invalid CRC/state)",
                seg_path
            );
            (page_a, 0)
        } else if valid_b {
            info!(
                "Recovered database {:?} using Header Page B (Page A had invalid CRC/state)",
                seg_path
            );
            (page_b, 1)
        } else {
            let v4_a = page_a.is_valid_v4();
            let v4_b = page_b.is_valid_v4();
            if v4_a || v4_b {
                let base_v4 = if v4_a && v4_b {
                    if page_a.generation >= page_b.generation {
                        page_a
                    } else {
                        page_b
                    }
                } else if v4_a {
                    page_a
                } else {
                    page_b
                };
                info!(
                    "Upgrading database {:?} in-place from v4 to v5 Section Directory format",
                    seg_path
                );
                let upgraded = Self::upgrade_v4_header_in_place(&mut mmap, base_v4)?;
                (upgraded, 0)
            } else {
                let v3_hdr = bytemuck::from_bytes::<DbHeaderV3>(&mmap[0..128]);
                if v3_hdr.magic == STARKDB_MAGIC && v3_hdr.version == STARKDB_V3_VERSION {
                    info!("Upgrading legacy database {:?} from v3 to v5 format", seg_path);
                    drop(mmap);
                    Self::migrate_v3_to_v5(&file, seg_path, vector_dims)?;
                    mmap = unsafe { MmapMut::map_mut(&file)? };
                    let fresh_hdr = *bytemuck::from_bytes::<DbHeader>(
                        &mmap[PAGE_A_OFFSET..PAGE_A_OFFSET + HEADER_PAGE_SIZE],
                    );
                    (fresh_hdr, 0)
                } else {
                    bail!(
                        "Corrupted database file at {:?}: neither Header Page A nor B has a valid CRC32",
                        seg_path
                    );
                }
            }
        };

        let nodes_offset = current_header.nodes_offset as usize;
        let synapses_offset = current_header.synapses_offset as usize;
        let vectors_offset = current_header.vectors_offset as usize;
        let sq8_codes_offset = current_header.sq8_codes_offset as usize;
        let sq8_meta_offset = current_header.sq8_meta_offset as usize;
        let strings_offset = current_header.strings_offset as usize;

        // Validate basic section bounds against physical mmap length
        let nodes_end = nodes_offset
            .saturating_add(current_header.node_count as usize * std::mem::size_of::<NodeRecord>());
        let synapses_end = synapses_offset.saturating_add(
            current_header.synapse_count as usize * std::mem::size_of::<SynapseRecord>(),
        );
        let strings_end = strings_offset.saturating_add(current_header.strings_bytes as usize);
        if nodes_end > mmap.len() || synapses_end > mmap.len() || strings_end > mmap.len() {
            bail!(
                "Segment {:?} section bounds exceed physical file length ({} bytes)",
                seg_path,
                mmap.len()
            );
        }

        let strings_cap = Self::compute_strings_cap(&current_header, mmap.len());

        let page_growth_size = hw.recommended_mmap_page_size.max(4096);
        let slot_allocation_step = (hw.recommended_batch_size as u32).clamp(64, 10000);

        let mut storage = Self {
            path: base_path.to_path_buf(),
            active_segment_path: seg_path.to_path_buf(),
            active_seq,
            rollback_seq,
            file,
            mmap,
            current_header,
            active_page,
            nodes_offset,
            synapses_offset,
            vectors_offset,
            sq8_codes_offset,
            sq8_meta_offset,
            strings_offset,
            strings_cap,
            vector_dims: current_header.vector_dims as usize,
            string_cache: HashTable::new(),
            page_growth_size,
            slot_allocation_step,
        };

        storage.rebuild_string_cache();

        if storage.current_header.vector_count > 0 && !storage.has_sq8() {
            storage.backfill_sq8()?;
        }

        Ok(storage)
    }

    fn compute_strings_cap(hdr: &DbHeader, mmap_len: usize) -> usize {
        let str_off = hdr.strings_offset as usize;
        let mut end = mmap_len;
        let candidates = [
            (hdr.nodes_offset as usize, hdr.max_nodes > 0),
            (hdr.synapses_offset as usize, hdr.max_synapses > 0),
            (hdr.vectors_offset as usize, hdr.max_vectors > 0),
            (hdr.sq8_codes_offset as usize, hdr.sq8_codes_offset > 0 && hdr.max_vectors > 0),
            (hdr.sq8_meta_offset as usize, hdr.sq8_meta_offset > 0 && hdr.max_vectors > 0),
        ];
        for (off, active) in candidates {
            if active && off > str_off && off < end {
                end = off;
            }
        }
        end.saturating_sub(str_off)
    }

    fn rebuild_string_cache(&mut self) {
        self.string_cache.clear();
        let str_bytes_len = self.current_header.strings_bytes as usize;
        let str_off = self.strings_offset;
        if str_bytes_len == 0 || str_off + str_bytes_len > self.mmap.len() {
            return;
        }

        let node_record_size = std::mem::size_of::<NodeRecord>();
        let nodes_bytes_end = self.nodes_offset + (self.current_header.node_count as usize * node_record_size);
        let mut entries: Vec<(u32, u32)> = Vec::new();

        if nodes_bytes_end <= self.mmap.len() {
            let nodes_slice: &[NodeRecord] = bytemuck::cast_slice(&self.mmap[self.nodes_offset..nodes_bytes_end]);
            for n in nodes_slice {
                if (n.flags & FLAG_DELETED) != 0 {
                    continue;
                }
                if n.name_len > 0 && (n.name_offset as usize + n.name_len as usize) <= str_bytes_len {
                    entries.push((n.name_offset, n.name_len as u32));
                }
                if n.tags_len > 0 && (n.tags_offset as usize + n.tags_len as usize) <= str_bytes_len {
                    entries.push((n.tags_offset, n.tags_len as u32));
                }
            }
        }

        let synapse_record_size = std::mem::size_of::<SynapseRecord>();
        let syn_bytes_end = self.synapses_offset + (self.current_header.synapse_count as usize * synapse_record_size);
        if syn_bytes_end <= self.mmap.len() {
            let syn_slice: &[SynapseRecord] = bytemuck::cast_slice(&self.mmap[self.synapses_offset..syn_bytes_end]);
            for s in syn_slice {
                if (s.flags & FLAG_DELETED) != 0 {
                    continue;
                }
                if s.rel_len > 0 && (s.rel_offset as usize + s.rel_len as usize) <= str_bytes_len {
                    entries.push((s.rel_offset, s.rel_len as u32));
                }
            }
        }

        let pool = &self.mmap[str_off..str_off + str_bytes_len];
        for (off, len) in entries {
            let slice = &pool[off as usize..off as usize + len as usize];
            let h = hash_bytes(slice);
            if self
                .string_cache
                .find(h, |&(o, l)| {
                    l == len && &pool[o as usize..o as usize + l as usize] == slice
                })
                .is_none()
            {
                self.string_cache.insert_unique(h, (off, len), |&(o, l)| {
                    hash_bytes(&pool[o as usize..o as usize + l as usize])
                });
            }
        }
    }

    fn upgrade_v4_header_in_place(mmap: &mut MmapMut, mut hdr: DbHeader) -> Result<DbHeader> {
        let nodes_offset = TOTAL_HEADER_SIZE as u64;
        let synapses_offset =
            nodes_offset + (hdr.max_nodes as u64 * std::mem::size_of::<NodeRecord>() as u64);
        let vectors_offset =
            synapses_offset + (hdr.max_synapses as u64 * std::mem::size_of::<SynapseRecord>() as u64);

        hdr.version = STARKDB_VERSION;
        hdr.free_vectors_head = u32::MAX;
        hdr.nodes_offset = nodes_offset;
        hdr.synapses_offset = synapses_offset;
        hdr.vectors_offset = vectors_offset;
        hdr.sq8_codes_offset = 0;
        hdr.sq8_meta_offset = 0;
        hdr.dead_string_bytes = 0;
        hdr.vacuum_epoch = 0;
        hdr.dead_vector_slots = 0;
        hdr.feature_flags = 0;
        hdr._padding = [0; 11];
        hdr.active_page = 0;
        hdr.generation = hdr.generation.saturating_add(1);
        hdr.update_crc();

        let hdr_bytes = bytemuck::bytes_of(&hdr);
        mmap[PAGE_A_OFFSET..PAGE_A_OFFSET + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);
        mmap[PAGE_B_OFFSET..PAGE_B_OFFSET + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);
        mmap.flush()?;
        Ok(hdr)
    }

    fn migrate_v3_to_v5(file: &File, path: &Path, vector_dims: u16) -> Result<()> {
        let temp_mmap = unsafe { MmapMut::map_mut(file)? };
        let v3_hdr = *bytemuck::from_bytes::<DbHeaderV3>(&temp_mmap[0..128]);

        let v3_nodes_offset = 128;
        let v3_synapses_offset = v3_nodes_offset + (v3_hdr.max_nodes as usize * std::mem::size_of::<NodeRecord>());
        let v3_vectors_offset = v3_synapses_offset + (v3_hdr.max_synapses as usize * std::mem::size_of::<SynapseRecord>());
        let v3_strings_offset = v3_hdr.strings_offset as usize;

        let node_bytes = temp_mmap[v3_nodes_offset..v3_synapses_offset].to_vec();
        let syn_bytes = temp_mmap[v3_synapses_offset..v3_vectors_offset].to_vec();
        let vec_bytes = temp_mmap[v3_vectors_offset..v3_strings_offset].to_vec();
        let str_bytes = temp_mmap[v3_strings_offset..v3_strings_offset + v3_hdr.strings_bytes as usize].to_vec();

        drop(temp_mmap);

        let new_nodes_offset = TOTAL_HEADER_SIZE;
        let new_synapses_offset = new_nodes_offset + (v3_hdr.max_nodes as usize * std::mem::size_of::<NodeRecord>());
        let new_vectors_offset = new_synapses_offset + (v3_hdr.max_synapses as usize * std::mem::size_of::<SynapseRecord>());
        let new_strings_offset = new_vectors_offset + vec_bytes.len();
        let new_total_len = new_strings_offset + str_bytes.len() + 4096;

        file.set_len(new_total_len as u64)?;
        let mut new_mmap = unsafe { MmapMut::map_mut(file)? };
        new_mmap.fill(0);

        let mut v5_hdr = DbHeader {
            magic: STARKDB_MAGIC,
            version: STARKDB_VERSION,
            vector_dims: if v3_hdr.vector_dims > 0 { v3_hdr.vector_dims } else { vector_dims },
            vector_format: 0,
            generation: 1,
            active_page: 0,
            _reserved1: [0; 3],
            header_crc32: 0,
            max_nodes: v3_hdr.max_nodes,
            node_count: v3_hdr.node_count,
            max_synapses: v3_hdr.max_synapses,
            synapse_count: v3_hdr.synapse_count,
            max_vectors: v3_hdr.max_vectors,
            vector_count: v3_hdr.vector_count,
            strings_offset: new_strings_offset as u64,
            strings_bytes: v3_hdr.strings_bytes,
            created_at: v3_hdr.created_at,
            last_consolidated: v3_hdr.last_consolidated,
            data_crc32: 0,
            free_nodes_head: u32::MAX,
            free_synapses_head: u32::MAX,
            free_vectors_head: u32::MAX,
            nodes_offset: new_nodes_offset as u64,
            synapses_offset: new_synapses_offset as u64,
            vectors_offset: new_vectors_offset as u64,
            sq8_codes_offset: 0,
            sq8_meta_offset: 0,
            dead_string_bytes: 0,
            vacuum_epoch: 0,
            dead_vector_slots: 0,
            feature_flags: 0,
            _padding: [0; 11],
        };
        v5_hdr.update_crc();

        let hdr_bytes = bytemuck::bytes_of(&v5_hdr);
        new_mmap[PAGE_A_OFFSET..PAGE_A_OFFSET + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);
        new_mmap[PAGE_B_OFFSET..PAGE_B_OFFSET + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);

        new_mmap[new_nodes_offset..new_nodes_offset + node_bytes.len()].copy_from_slice(&node_bytes);
        new_mmap[new_synapses_offset..new_synapses_offset + syn_bytes.len()].copy_from_slice(&syn_bytes);
        if !vec_bytes.is_empty() {
            new_mmap[new_vectors_offset..new_vectors_offset + vec_bytes.len()].copy_from_slice(&vec_bytes);
        }
        if !str_bytes.is_empty() {
            new_mmap[new_strings_offset..new_strings_offset + str_bytes.len()].copy_from_slice(&str_bytes);
        }

        new_mmap.flush()?;
        info!("Successfully upgraded {:?} to v5 Ultimate Legacy layout", path);
        Ok(())
    }

    fn initialize_db_file(file: &File, vector_dims: u16, initial_str_cap: u64) -> Result<()> {
        let max_nodes = Self::INITIAL_MAX_NODES;
        let max_synapses = Self::INITIAL_MAX_SYNAPSES;
        let max_vectors = Self::INITIAL_MAX_VECTORS;
        let vector_bytes = max_vectors as u64 * vector_dims as u64 * 2;

        let nodes_bytes = max_nodes as u64 * std::mem::size_of::<NodeRecord>() as u64;
        let synapses_bytes = max_synapses as u64 * std::mem::size_of::<SynapseRecord>() as u64;

        let nodes_offset = TOTAL_HEADER_SIZE as u64;
        let synapses_offset = nodes_offset + nodes_bytes;
        let vectors_offset = synapses_offset + synapses_bytes;
        let strings_offset = vectors_offset + vector_bytes;
        let str_cap = initial_str_cap.max(Self::INITIAL_STRING_CAPACITY);
        let total_file_size = strings_offset + str_cap;

        file.set_len(0)?;
        file.set_len(total_file_size)?;

        let mut header = DbHeader {
            magic: STARKDB_MAGIC,
            version: STARKDB_VERSION,
            vector_dims,
            vector_format: 1, // 1 = f16 default
            generation: 1,
            active_page: 0,
            _reserved1: [0; 3],
            header_crc32: 0,
            max_nodes,
            node_count: 0,
            max_synapses,
            synapse_count: 0,
            max_vectors,
            vector_count: 0,
            strings_offset,
            strings_bytes: 0,
            created_at: current_timestamp(),
            last_consolidated: current_timestamp(),
            data_crc32: 0,
            free_nodes_head: u32::MAX,
            free_synapses_head: u32::MAX,
            free_vectors_head: u32::MAX,
            nodes_offset,
            synapses_offset,
            vectors_offset,
            sq8_codes_offset: 0,
            sq8_meta_offset: 0,
            dead_string_bytes: 0,
            vacuum_epoch: 0,
            dead_vector_slots: 0,
            feature_flags: 0,
            _padding: [0; 11],
        };
        header.update_crc();

        let mut mmap = unsafe { MmapMut::map_mut(file)? };
        mmap.fill(0);

        let hdr_bytes = bytemuck::bytes_of(&header);
        mmap[PAGE_A_OFFSET..PAGE_A_OFFSET + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);
        mmap[PAGE_B_OFFSET..PAGE_B_OFFSET + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);
        mmap.flush()?;
        Ok(())
    }

    #[inline]
    pub fn bytes_per_vector(&self) -> usize {
        match self.current_header.vector_format {
            1 => self.vector_dims * 2,
            _ => self.vector_dims * 4,
        }
    }

    #[inline]
    pub fn has_sq8(&self) -> bool {
        (self.current_header.feature_flags & FEATURE_SQ8_READY) != 0
            && self.sq8_codes_offset > 0
            && self.sq8_meta_offset > 0
    }

    /// Atomically commits updated header to inactive page with monotonic generation and CRC32
    pub fn commit_header(&mut self) -> Result<()> {
        self.mmap.flush()?;

        let next_page = 1 - self.active_page;
        let target_offset = if next_page == 0 { PAGE_A_OFFSET } else { PAGE_B_OFFSET };

        self.current_header.generation = self.current_header.generation.saturating_add(1);
        self.current_header.active_page = next_page;
        self.current_header.update_crc();

        let hdr_bytes = bytemuck::bytes_of(&self.current_header);
        self.mmap[target_offset..target_offset + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);
        self.mmap.flush_range(target_offset, HEADER_PAGE_SIZE)?;
        let _ = self.file.sync_data();

        self.active_page = next_page;
        Ok(())
    }

    /// Commits header update to the inactive A/B page after out-of-place section growth
    /// without overwriting the previous valid header page (preserving crash rollback).
    fn commit_header_after_grow(&mut self) -> Result<()> {
        // 1. Flush modified section data and newly expanded ranges to disk first,
        // ensuring physical disk blocks are durable BEFORE the new header points to them.
        self.mmap.flush()?;

        let next_page = 1 - self.active_page;
        let target_offset = if next_page == 0 { PAGE_A_OFFSET } else { PAGE_B_OFFSET };

        self.current_header.generation = self.current_header.generation.saturating_add(1);
        self.current_header.active_page = next_page;
        self.current_header.update_crc();

        let hdr_bytes = bytemuck::bytes_of(&self.current_header);
        self.mmap[target_offset..target_offset + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);
        self.mmap.flush_range(target_offset, HEADER_PAGE_SIZE)?;
        self.file.sync_data()?;

        self.active_page = next_page;
        Ok(())
    }

    pub fn verify_dual_headers(&self) -> (bool, bool) {
        let page_a = *bytemuck::from_bytes::<DbHeader>(&self.mmap[PAGE_A_OFFSET..PAGE_A_OFFSET + HEADER_PAGE_SIZE]);
        let page_b = *bytemuck::from_bytes::<DbHeader>(&self.mmap[PAGE_B_OFFSET..PAGE_B_OFFSET + HEADER_PAGE_SIZE]);
        (page_a.is_valid(), page_b.is_valid())
    }

    pub fn synchronize_headers(&mut self) -> Result<()> {
        let mut hdr = self.current_header;
        hdr.update_crc();
        let hdr_bytes = bytemuck::bytes_of(&hdr);
        self.mmap[PAGE_A_OFFSET..PAGE_A_OFFSET + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);
        self.mmap[PAGE_B_OFFSET..PAGE_B_OFFSET + HEADER_PAGE_SIZE].copy_from_slice(hdr_bytes);
        self.mmap.flush()?;
        let _ = self.file.sync_data();
        Ok(())
    }

    /// Returns sorted list of currently active (start, end) byte intervals in the mmap file.
    fn active_regions(&self) -> SmallVec<[(usize, usize); 6]> {
        let mut regions: SmallVec<[(usize, usize); 6]> = SmallVec::new();
        let h = &self.current_header;
        if h.max_nodes > 0 {
            let len = h.max_nodes as usize * std::mem::size_of::<NodeRecord>();
            regions.push((self.nodes_offset, self.nodes_offset + len));
        }
        if h.max_synapses > 0 {
            let len = h.max_synapses as usize * std::mem::size_of::<SynapseRecord>();
            regions.push((self.synapses_offset, self.synapses_offset + len));
        }
        if h.max_vectors > 0 {
            let vlen = h.max_vectors as usize * self.bytes_per_vector();
            regions.push((self.vectors_offset, self.vectors_offset + vlen));
            if self.sq8_codes_offset > 0 {
                let clen = h.max_vectors as usize * self.vector_dims;
                regions.push((self.sq8_codes_offset, self.sq8_codes_offset + clen));
            }
            if self.sq8_meta_offset > 0 {
                let mlen = h.max_vectors as usize * std::mem::size_of::<Sq8Meta>();
                regions.push((self.sq8_meta_offset, self.sq8_meta_offset + mlen));
            }
        }
        if self.strings_cap > 0 {
            regions.push((self.strings_offset, self.strings_offset + self.strings_cap));
        }
        regions.sort_unstable_by_key(|&(start, _)| start);
        regions
    }

    /// Finds a disjoint free byte region of size `required_bytes` aligned to `align`
    /// that does not overlap any currently active section (including the old location of the growing section).
    fn find_free_region(&self, required_bytes: usize, align: usize) -> (usize, usize) {
        let regions = self.active_regions();
        let max_active_end = regions.last().map(|&(_, end)| end).unwrap_or(TOTAL_HEADER_SIZE);
        let mut prev_end = TOTAL_HEADER_SIZE;
        for &(start, end) in &regions {
            let cand = prev_end.div_ceil(align) * align;
            if start > cand && (start - cand) >= required_bytes {
                return (cand, max_active_end);
            }
            prev_end = prev_end.max(end);
        }
        let cand = prev_end.div_ceil(align) * align;
        (cand, max_active_end)
    }

    /// If `[old_offset .. old_offset + new_cap)` does not collide with any other active section,
    /// extends the section in-place at `old_offset` (0 bytes copied, 0 dead holes).
    /// Otherwise allocates a disjoint free region via `find_free_region`.
    fn allocate_or_extend_region(
        &self,
        old_offset: usize,
        old_cap: usize,
        new_cap: usize,
        align: usize,
    ) -> usize {
        if old_cap > 0 {
            let new_end = old_offset + new_cap;
            let overlaps_other = self
                .active_regions()
                .into_iter()
                .any(|(s, e)| s != old_offset && s < new_end && e > old_offset);
            if !overlaps_other {
                return old_offset;
            }
        }
        self.find_free_region(new_cap, align).0
    }

    fn ensure_file_len(&mut self, required_end: usize) -> Result<()> {
        if required_end > self.mmap.len() {
            self.file.set_len(required_end as u64)?;
            self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        }
        Ok(())
    }

    fn fixed_regions_except(&self, exclude_offset: usize) -> SmallVec<[(usize, usize); 6]> {
        let mut regions: SmallVec<[(usize, usize); 6]> = SmallVec::new();
        let h = &self.current_header;
        if h.max_nodes > 0 && self.nodes_offset != exclude_offset {
            let len = h.max_nodes as usize * std::mem::size_of::<NodeRecord>();
            regions.push((self.nodes_offset, self.nodes_offset + len));
        }
        if h.max_synapses > 0 && self.synapses_offset != exclude_offset {
            let len = h.max_synapses as usize * std::mem::size_of::<SynapseRecord>();
            regions.push((self.synapses_offset, self.synapses_offset + len));
        }
        if h.max_vectors > 0 && self.vectors_offset != exclude_offset {
            let vlen = h.max_vectors as usize * self.bytes_per_vector();
            regions.push((self.vectors_offset, self.vectors_offset + vlen));
            if self.sq8_codes_offset > 0 {
                let clen = h.max_vectors as usize * self.vector_dims;
                regions.push((self.sq8_codes_offset, self.sq8_codes_offset + clen));
            }
            if self.sq8_meta_offset > 0 {
                let mlen = h.max_vectors as usize * std::mem::size_of::<Sq8Meta>();
                regions.push((self.sq8_meta_offset, self.sq8_meta_offset + mlen));
            }
        }
        regions.sort_unstable_by_key(|&(start, _)| start);
        regions
    }

    /// Maintains the invariant that `strings` is the tail section of the file so that:
    /// 1) `grow_strings` always extends in-place at EOF with 0 copies and 0 dead holes;
    /// 2) fixed-record sections (`nodes`, `synapses`) grow in-place in the arena before `strings`.
    fn slide_tail_strings_if_needed(
        &mut self,
        old_offset: usize,
        old_cap_bytes: usize,
        new_cap_bytes: usize,
    ) -> Result<()> {
        if self.strings_cap == 0 {
            return Ok(());
        }
        let all_regions = self.active_regions();
        let max_active_end = all_regions.last().map(|&(_, e)| e).unwrap_or(TOTAL_HEADER_SIZE);
        if self.strings_offset + self.strings_cap != max_active_end {
            return Ok(());
        }

        let other_fixed = self.fixed_regions_except(old_offset);
        let can_extend_in_fixed = old_cap_bytes > 0
            && !other_fixed
                .iter()
                .any(|&(s, e)| s < old_offset + new_cap_bytes && e > old_offset);

        let target_end = if can_extend_in_fixed {
            old_offset + new_cap_bytes
        } else {
            // Find where this fixed section will be placed among fixed sections (including old_offset)
            let mut fixed_all = other_fixed;
            if old_cap_bytes > 0 {
                fixed_all.push((old_offset, old_offset + old_cap_bytes));
                fixed_all.sort_unstable_by_key(|&(s, _)| s);
            }
            let mut prev_end = TOTAL_HEADER_SIZE;
            let mut chosen = None;
            for &(s, e) in &fixed_all {
                let cand = prev_end.div_ceil(64) * 64;
                if s > cand && (s - cand) >= new_cap_bytes {
                    chosen = Some(cand);
                    break;
                }
                prev_end = prev_end.max(e);
            }
            let cand = chosen.unwrap_or_else(|| prev_end.div_ceil(64) * 64);
            cand + new_cap_bytes
        };

        if target_end <= self.strings_offset {
            return Ok(());
        }

        let active_str = self.current_header.strings_bytes as usize;
        let min_disjoint = self.strings_offset + active_str;
        let headroom = (new_cap_bytes * 7 / 4 + 8192).min(512 * 1024);
        let arena_target = (target_end + headroom).max(49_152);
        let new_str_off = min_disjoint.max(arena_target).div_ceil(4096) * 4096;
        let new_str_cap = self.strings_cap.max((active_str + 16 * 1024).div_ceil(4096) * 4096);

        self.ensure_file_len(new_str_off + new_str_cap)?;
        if active_str > 0 {
            self.mmap
                .copy_within(self.strings_offset..self.strings_offset + active_str, new_str_off);
        }
        self.mmap[new_str_off + active_str..new_str_off + new_str_cap].fill(0);

        self.strings_offset = new_str_off;
        self.strings_cap = new_str_cap;
        self.current_header.strings_offset = new_str_off as u64;
        self.commit_header_after_grow()
    }

    fn grow_nodes(&mut self, new_max_nodes: u32) -> Result<()> {
        let old_max = self.current_header.max_nodes;
        if new_max_nodes <= old_max {
            return Ok(());
        }
        let rec_size = std::mem::size_of::<NodeRecord>();
        let old_offset = self.nodes_offset;
        let old_cap_bytes = old_max as usize * rec_size;
        let active_bytes = self.current_header.node_count as usize * rec_size;
        let new_cap_bytes = new_max_nodes as usize * rec_size;

        self.slide_tail_strings_if_needed(old_offset, old_cap_bytes, new_cap_bytes)?;
        let new_offset = self.allocate_or_extend_region(old_offset, old_cap_bytes, new_cap_bytes, 64);

        self.ensure_file_len(new_offset + new_cap_bytes)?;
        if new_offset != old_offset && active_bytes > 0 {
            self.mmap.copy_within(old_offset..old_offset + active_bytes, new_offset);
        }
        self.mmap[new_offset + active_bytes..new_offset + new_cap_bytes].fill(0);

        self.nodes_offset = new_offset;
        self.current_header.nodes_offset = new_offset as u64;
        self.current_header.max_nodes = new_max_nodes;
        self.commit_header_after_grow()
    }

    fn grow_synapses(&mut self, new_max_synapses: u32) -> Result<()> {
        let old_max = self.current_header.max_synapses;
        if new_max_synapses <= old_max {
            return Ok(());
        }
        let rec_size = std::mem::size_of::<SynapseRecord>();
        let old_offset = self.synapses_offset;
        let old_cap_bytes = old_max as usize * rec_size;
        let active_bytes = self.current_header.synapse_count as usize * rec_size;
        let new_cap_bytes = new_max_synapses as usize * rec_size;

        self.slide_tail_strings_if_needed(old_offset, old_cap_bytes, new_cap_bytes)?;
        let new_offset = self.allocate_or_extend_region(old_offset, old_cap_bytes, new_cap_bytes, 64);

        self.ensure_file_len(new_offset + new_cap_bytes)?;
        if new_offset != old_offset && active_bytes > 0 {
            self.mmap.copy_within(old_offset..old_offset + active_bytes, new_offset);
        }
        self.mmap[new_offset + active_bytes..new_offset + new_cap_bytes].fill(0);

        self.synapses_offset = new_offset;
        self.current_header.synapses_offset = new_offset as u64;
        self.current_header.max_synapses = new_max_synapses;
        self.commit_header_after_grow()
    }

    fn grow_strings(&mut self, new_str_cap: usize) -> Result<()> {
        if new_str_cap <= self.strings_cap {
            return Ok(());
        }
        let old_offset = self.strings_offset;
        let old_cap = self.strings_cap;
        let active_bytes = self.current_header.strings_bytes as usize;

        let new_offset = self.allocate_or_extend_region(old_offset, old_cap, new_str_cap, 64);

        self.ensure_file_len(new_offset + new_str_cap)?;
        if new_offset != old_offset && active_bytes > 0 {
            self.mmap.copy_within(old_offset..old_offset + active_bytes, new_offset);
        }
        self.mmap[new_offset + active_bytes..new_offset + new_str_cap].fill(0);

        self.strings_offset = new_offset;
        self.strings_cap = new_str_cap;
        self.current_header.strings_offset = new_offset as u64;
        self.commit_header_after_grow()
    }

    fn grow_vectors(&mut self, new_max_vectors: u32) -> Result<()> {
        let old_max = self.current_header.max_vectors;
        if new_max_vectors <= old_max && self.sq8_codes_offset > 0 && self.sq8_meta_offset > 0 {
            return Ok(());
        }
        let target_max = new_max_vectors.max(old_max).max(1);
        let vec_stride = self.bytes_per_vector();
        let dims = self.vector_dims;
        let meta_size = std::mem::size_of::<Sq8Meta>();

        let active_vecs = self.current_header.vector_count as usize;
        let active_vec_bytes = active_vecs * vec_stride;
        let active_code_bytes = active_vecs * dims;
        let active_meta_bytes = active_vecs * meta_size;

        let old_vec_bytes = (old_max as usize * vec_stride).div_ceil(64) * 64;
        let old_code_bytes = (old_max as usize * dims).div_ceil(64) * 64;
        let old_meta_bytes = (old_max as usize * meta_size).div_ceil(64) * 64;
        let old_total_bytes = old_vec_bytes + old_code_bytes + old_meta_bytes;

        let new_vec_bytes = (target_max as usize * vec_stride).div_ceil(64) * 64;
        let new_code_bytes = (target_max as usize * dims).div_ceil(64) * 64;
        let new_meta_bytes = (target_max as usize * meta_size).div_ceil(64) * 64;
        let total_block_bytes = new_vec_bytes + new_code_bytes + new_meta_bytes;

        let can_extend_in_place = old_max > 0
            && self.vectors_offset > 0
            && self.sq8_codes_offset == self.vectors_offset + old_vec_bytes
            && self.sq8_meta_offset == self.sq8_codes_offset + old_code_bytes
            && !self.active_regions().into_iter().any(|(s, e)| {
                let in_old_vec_block =
                    s >= self.vectors_offset && e <= self.vectors_offset + old_total_bytes;
                !in_old_vec_block
                    && s < self.vectors_offset + total_block_bytes
                    && e > self.vectors_offset
            });

        let base_offset = if can_extend_in_place {
            self.vectors_offset
        } else {
            self.find_free_region(total_block_bytes, 64).0
        };
        let new_vec_offset = base_offset;
        let new_code_offset = new_vec_offset + new_vec_bytes;
        let new_meta_offset = new_code_offset + new_code_bytes;

        self.ensure_file_len(base_offset + total_block_bytes)?;

        if can_extend_in_place {
            if self.sq8_meta_offset > 0 && active_meta_bytes > 0 {
                self.mmap.copy_within(
                    self.sq8_meta_offset..self.sq8_meta_offset + active_meta_bytes,
                    new_meta_offset,
                );
            }
            self.mmap[new_meta_offset + active_meta_bytes..new_meta_offset + new_meta_bytes]
                .fill(0);

            if self.sq8_codes_offset > 0 && active_code_bytes > 0 {
                self.mmap.copy_within(
                    self.sq8_codes_offset..self.sq8_codes_offset + active_code_bytes,
                    new_code_offset,
                );
            }
            self.mmap[new_code_offset + active_code_bytes..new_code_offset + new_code_bytes]
                .fill(0);

            self.mmap[new_vec_offset + active_vec_bytes..new_vec_offset + new_vec_bytes].fill(0);
        } else {
            if active_vec_bytes > 0 {
                self.mmap.copy_within(
                    self.vectors_offset..self.vectors_offset + active_vec_bytes,
                    new_vec_offset,
                );
            }
            self.mmap[new_vec_offset + active_vec_bytes..new_vec_offset + new_vec_bytes].fill(0);

            if self.sq8_codes_offset > 0 && active_code_bytes > 0 {
                self.mmap.copy_within(
                    self.sq8_codes_offset..self.sq8_codes_offset + active_code_bytes,
                    new_code_offset,
                );
            }
            self.mmap[new_code_offset + active_code_bytes..new_code_offset + new_code_bytes]
                .fill(0);

            if self.sq8_meta_offset > 0 && active_meta_bytes > 0 {
                self.mmap.copy_within(
                    self.sq8_meta_offset..self.sq8_meta_offset + active_meta_bytes,
                    new_meta_offset,
                );
            }
            self.mmap[new_meta_offset + active_meta_bytes..new_meta_offset + new_meta_bytes]
                .fill(0);
        }

        self.vectors_offset = new_vec_offset;
        self.sq8_codes_offset = new_code_offset;
        self.sq8_meta_offset = new_meta_offset;

        self.current_header.vectors_offset = new_vec_offset as u64;
        self.current_header.sq8_codes_offset = new_code_offset as u64;
        self.current_header.sq8_meta_offset = new_meta_offset as u64;
        self.current_header.max_vectors = target_max;
        self.current_header.feature_flags |= FEATURE_SQ8_READY;

        self.commit_header_after_grow()
    }

    pub fn backfill_sq8(&mut self) -> Result<()> {
        let count = self.current_header.vector_count;
        if count == 0 {
            return Ok(());
        }
        let max_v = self.current_header.max_vectors.max(count);
        self.grow_vectors(max_v)?;

        let dims = self.vector_dims;
        let mut buf_f32 = vec![0.0f32; dims];
        let mut buf_i8 = vec![0i8; dims];

        for vid in 0..count {
            if self.get_vector_into(vid, &mut buf_f32).is_ok() {
                let meta = crate::vectors::quantize_sq8(&buf_f32, &mut buf_i8);
                let code_off = self.sq8_codes_offset + (vid as usize * dims);
                let meta_off = self.sq8_meta_offset + (vid as usize * std::mem::size_of::<Sq8Meta>());
                self.mmap[code_off..code_off + dims]
                    .copy_from_slice(bytemuck::cast_slice(&buf_i8));
                self.mmap[meta_off..meta_off + std::mem::size_of::<Sq8Meta>()]
                    .copy_from_slice(bytemuck::bytes_of(&meta));
            }
        }
        self.current_header.feature_flags |= FEATURE_SQ8_READY;
        self.commit_header()
    }

    /// Dynamically grows any arena section using v5 Section Directory & out-of-place `copy_within`.
    pub fn grow_sections(
        &mut self,
        new_max_nodes: u32,
        new_max_synapses: u32,
        new_max_vectors: u32,
        new_string_cap: usize,
    ) -> Result<()> {
        if new_max_nodes > self.current_header.max_nodes {
            self.grow_nodes(new_max_nodes)?;
        }
        if new_max_synapses > self.current_header.max_synapses {
            self.grow_synapses(new_max_synapses)?;
        }
        if new_max_vectors > self.current_header.max_vectors {
            self.grow_vectors(new_max_vectors)?;
        }
        if new_string_cap > self.strings_cap {
            self.grow_strings(new_string_cap)?;
        }
        Ok(())
    }

    pub fn ensure_node_capacity(&mut self, additional: u32) -> Result<()> {
        let h = self.current_header;
        if h.node_count + additional > h.max_nodes {
            let min_needed = h.node_count + additional;
            let step = (self.slot_allocation_step / 16).clamp(64, 128);
            let cur = h.max_nodes;
            let grown = (cur * 3 / 2).max(cur + step).max(min_needed);
            let needed = grown.div_ceil(64) * 64;
            self.grow_nodes(needed)?;
        }
        Ok(())
    }

    pub fn ensure_synapse_capacity(&mut self, additional: u32) -> Result<()> {
        let h = self.current_header;
        if h.synapse_count + additional > h.max_synapses {
            let min_needed = h.synapse_count + additional;
            let step = (self.slot_allocation_step / 8).clamp(128, 256);
            let cur = h.max_synapses;
            let grown = (cur * 3 / 2).max(cur + step).max(min_needed);
            let needed = grown.div_ceil(128) * 128;
            self.grow_synapses(needed)?;
        }
        Ok(())
    }

    pub fn ensure_vector_capacity(&mut self, additional: u32) -> Result<()> {
        let h = self.current_header;
        if h.vector_count + additional > h.max_vectors || self.sq8_codes_offset == 0 {
            let min_needed = h.vector_count + additional;
            let step = (self.slot_allocation_step / 4).clamp(64, 512);
            let cur = h.max_vectors;
            let grown = (cur * 3 / 2).max(cur + step).max(min_needed);
            let needed = grown.div_ceil(step) * step;
            self.grow_vectors(needed)?;
        }
        Ok(())
    }

    pub fn file_size_bytes(&self) -> usize {
        self.mmap.len()
    }

    pub fn header(&self) -> &DbHeader {
        &self.current_header
    }

    #[inline]
    pub fn safe_slice(&self, start: usize, len: usize) -> Result<&[u8]> {
        let end = start.checked_add(len).context("Slice offset overflow")?;
        self.mmap
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("Memory bounds violation: range {}..{} exceeds mmap length {}", start, end, self.mmap.len()))
    }

    #[inline]
    pub fn safe_slice_mut(&mut self, start: usize, len: usize) -> Result<&mut [u8]> {
        let end = start.checked_add(len).context("Slice offset overflow")?;
        let mmap_len = self.mmap.len();
        self.mmap
            .get_mut(start..end)
            .ok_or_else(|| anyhow::anyhow!("Memory bounds violation: mutable range {}..{} exceeds mmap length {}", start, end, mmap_len))
    }

    pub fn header_mut(&mut self) -> &mut DbHeader {
        &mut self.current_header
    }

    pub fn nodes(&self) -> &[NodeRecord] {
        let count = self.current_header.node_count as usize;
        let size = std::mem::size_of::<NodeRecord>();
        let byte_len = count.saturating_mul(size);
        let end = self.nodes_offset.saturating_add(byte_len);
        if let Some(slice) = self.mmap.get(self.nodes_offset..end) {
            bytemuck::cast_slice(slice)
        } else {
            &[]
        }
    }

    pub fn nodes_mut(&mut self) -> &mut [NodeRecord] {
        let max_nodes = self.current_header.max_nodes as usize;
        let size = std::mem::size_of::<NodeRecord>();
        let byte_len = max_nodes.saturating_mul(size);
        let end = self.nodes_offset.saturating_add(byte_len);
        if end <= self.mmap.len() {
            bytemuck::cast_slice_mut(&mut self.mmap[self.nodes_offset..end])
        } else {
            &mut []
        }
    }

    pub fn synapses(&self) -> &[SynapseRecord] {
        let count = self.current_header.synapse_count as usize;
        let size = std::mem::size_of::<SynapseRecord>();
        let byte_len = count.saturating_mul(size);
        let end = self.synapses_offset.saturating_add(byte_len);
        if let Some(slice) = self.mmap.get(self.synapses_offset..end) {
            bytemuck::cast_slice(slice)
        } else {
            &[]
        }
    }

    pub fn synapses_mut(&mut self) -> &mut [SynapseRecord] {
        let max_synapses = self.current_header.max_synapses as usize;
        let size = std::mem::size_of::<SynapseRecord>();
        let byte_len = max_synapses.saturating_mul(size);
        let end = self.synapses_offset.saturating_add(byte_len);
        if end <= self.mmap.len() {
            bytemuck::cast_slice_mut(&mut self.mmap[self.synapses_offset..end])
        } else {
            &mut []
        }
    }

    pub fn get_string(&self, offset: u32, len: u32) -> Result<&str> {
        if len == 0 {
            return Ok("");
        }
        let start = self
            .strings_offset
            .checked_add(offset as usize)
            .context("String offset overflow")?;
        let bytes = self.safe_slice(start, len as usize)?;
        std::str::from_utf8(bytes).context("Invalid UTF-8 in string pool")
    }

    fn ensure_string_capacity(&mut self, additional_bytes: usize) -> Result<()> {
        let current_used = self.current_header.strings_bytes as usize;
        if current_used + additional_bytes <= self.strings_cap {
            return Ok(());
        }

        let step = self.page_growth_size.clamp(4096, 16 * 1024);
        let min_needed = current_used + additional_bytes;
        let delta = if self.strings_cap >= 256 * 1024 {
            (self.strings_cap / 2).max(step)
        } else {
            (self.strings_cap / 2).clamp(step, 32 * 1024)
        };
        let grown = (self.strings_cap + delta).max(min_needed);
        let new_str_cap = grown.div_ceil(4096) * 4096;
        self.grow_strings(new_str_cap)
    }

    pub fn append_string(&mut self, text: &str) -> Result<(u32, u16)> {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return Ok((0, 0));
        }
        let len = bytes.len().min(u16::MAX as usize) as u16;
        let target = &bytes[..len as usize];
        let hash = hash_bytes(target);

        // 1. O(1) zero-allocation lookup comparing directly against mmap string pool
        {
            let str_off = self.strings_offset;
            let mmap_ref = &self.mmap[..];
            if let Some(&(pos, _)) = self.string_cache.find(hash, |&(off, l)| {
                l == len as u32 && {
                    let s = str_off + off as usize;
                    let e = s + l as usize;
                    mmap_ref.get(s..e) == Some(target)
                }
            }) {
                return Ok((pos, len));
            }
        }

        self.ensure_string_capacity(len as usize)?;

        let current_offset = self.current_header.strings_bytes;
        let write_start = self.strings_offset + current_offset as usize;

        let dest = self.safe_slice_mut(write_start, len as usize)?;
        dest.copy_from_slice(target);
        self.current_header.strings_bytes += len as u64;

        let pos = current_offset as u32;
        let str_off = self.strings_offset;
        let mmap_ref = &self.mmap[..];
        self.string_cache
            .insert_unique(hash, (pos, len as u32), |&(off, l)| {
                let s = str_off + off as usize;
                hash_bytes(&mmap_ref[s..s + l as usize])
            });

        Ok((pos, len))
    }

    pub fn append_payload(&mut self, text: &str) -> Result<(u32, u32)> {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return Ok((0, 0));
        }
        let len = bytes.len() as u32;
        let hash = if bytes.len() <= 1024 {
            let h = hash_bytes(bytes);
            let str_off = self.strings_offset;
            let mmap_ref = &self.mmap[..];
            if let Some(&(pos, _)) = self.string_cache.find(h, |&(off, l)| {
                l == len && {
                    let s = str_off + off as usize;
                    let e = s + l as usize;
                    mmap_ref.get(s..e) == Some(bytes)
                }
            }) {
                return Ok((pos, len));
            }
            Some(h)
        } else {
            None
        };

        self.ensure_string_capacity(bytes.len())?;

        let current_offset = self.current_header.strings_bytes;
        let write_start = self.strings_offset + current_offset as usize;

        let dest = self.safe_slice_mut(write_start, bytes.len())?;
        dest.copy_from_slice(bytes);
        self.current_header.strings_bytes += bytes.len() as u64;

        let pos = current_offset as u32;
        if let Some(h) = hash {
            let str_off = self.strings_offset;
            let mmap_ref = &self.mmap[..];
            self.string_cache
                .insert_unique(h, (pos, len), |&(off, l)| {
                    let s = str_off + off as usize;
                    hash_bytes(&mmap_ref[s..s + l as usize])
                });
        }

        Ok((pos, len))
    }

    #[inline(always)]
    pub fn get_vector_f16(&self, vector_id: u32) -> Result<&[f16]> {
        if vector_id == u32::MAX {
            bail!("Node has no associated vector");
        }
        let dims = self.vector_dims;
        let byte_len = dims * 2;
        let byte_offset = self.vectors_offset + (vector_id as usize) * byte_len;
        if let Some(slice) = self.mmap.get(byte_offset..byte_offset + byte_len) {
            Ok(bytemuck::cast_slice(slice))
        } else {
            bail!("Vector range out of bounds for slot {}", vector_id)
        }
    }

    #[inline(always)]
    pub fn get_vector_f32(&self, vector_id: u32) -> Result<&[f32]> {
        if vector_id == u32::MAX {
            bail!("Node has no associated vector");
        }
        let dims = self.vector_dims;
        let byte_len = dims * 4;
        let byte_offset = self.vectors_offset + (vector_id as usize) * byte_len;
        if let Some(slice) = self.mmap.get(byte_offset..byte_offset + byte_len) {
            Ok(bytemuck::cast_slice(slice))
        } else {
            bail!("Vector range out of bounds for slot {}", vector_id)
        }
    }

    #[inline(always)]
    pub fn get_vector_sq8(&self, vector_id: u32) -> Result<(&[i8], Sq8Meta)> {
        if vector_id == u32::MAX || self.sq8_codes_offset == 0 {
            bail!("SQ8 vector unavailable for slot {}", vector_id);
        }
        let dims = self.vector_dims;
        let idx = vector_id as usize;
        let code_off = self.sq8_codes_offset + idx * dims;
        let meta_off = self.sq8_meta_offset + idx * std::mem::size_of::<Sq8Meta>();
        if let (Some(code_slice), Some(meta_slice)) = (
            self.mmap.get(code_off..code_off + dims),
            self.mmap.get(meta_off..meta_off + std::mem::size_of::<Sq8Meta>()),
        ) {
            Ok((bytemuck::cast_slice(code_slice), *bytemuck::from_bytes(meta_slice)))
        } else {
            bail!("SQ8 vector range out of bounds for slot {}", vector_id)
        }
    }

    /// Reads vector directly into caller-provided `out: &mut [f32]` buffer with ZERO heap allocations.
    pub fn get_vector_into(&self, vector_id: u32, out: &mut [f32]) -> Result<()> {
        let dims = self.vector_dims;
        if out.len() < dims {
            bail!("Output buffer too small: {} < {}", out.len(), dims);
        }
        if self.current_header.vector_format == 1 {
            let f16_slice = self.get_vector_f16(vector_id)?;
            for (dst, &src) in out[..dims].iter_mut().zip(f16_slice.iter()) {
                *dst = src.to_f32();
            }
        } else {
            let f32_slice = self.get_vector_f32(vector_id)?;
            out[..dims].copy_from_slice(f32_slice);
        }
        Ok(())
    }

    pub fn get_vector(&self, vector_id: u32) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.vector_dims];
        self.get_vector_into(vector_id, &mut out)?;
        Ok(out)
    }

    /// Releases a vector slot onto the `free_vectors_head` linked list for O(1) reuse.
    pub fn release_vector(&mut self, vector_id: u32) -> Result<()> {
        if vector_id == u32::MAX || vector_id >= self.current_header.vector_count {
            return Ok(());
        }
        let stride = self.bytes_per_vector();
        if stride < 4 {
            bail!("Vector stride {} < 4 bytes", stride);
        }
        let byte_offset = self
            .vectors_offset
            .checked_add(vector_id as usize * stride)
            .context("Vector offset overflow")?;
        let prev_head = self.current_header.free_vectors_head;
        let dest = self.safe_slice_mut(byte_offset, 4)?;
        dest.copy_from_slice(&prev_head.to_le_bytes());
        self.current_header.free_vectors_head = vector_id;
        self.current_header.dead_vector_slots =
            self.current_header.dead_vector_slots.saturating_add(1);
        Ok(())
    }

    pub fn write_vector(&mut self, vector: &[f32]) -> Result<u32> {
        let dims = self.vector_dims;
        if vector.len() != dims {
            bail!(
                "Vector length mismatch: expected {}, got {}",
                dims,
                vector.len()
            );
        }

        // L2 unit normalization at write time (L2 = 1.0)
        let norm_sq: f32 = vector.iter().map(|&x| x * x).sum();
        let inv_norm = if norm_sq > 0.0 { 1.0 / norm_sq.sqrt() } else { 0.0 };

        let vector_id = if self.current_header.free_vectors_head != u32::MAX {
            let recycled_id = self.current_header.free_vectors_head;
            let stride = self.bytes_per_vector();
            let byte_offset = self.vectors_offset + (recycled_id as usize * stride);
            let next_bytes = self.safe_slice(byte_offset, 4)?;
            let next_free = u32::from_le_bytes(next_bytes.try_into().unwrap());
            self.current_header.free_vectors_head = next_free;
            self.current_header.dead_vector_slots =
                self.current_header.dead_vector_slots.saturating_sub(1);
            if self.sq8_codes_offset == 0 || self.sq8_meta_offset == 0 {
                let max_v = self.current_header.max_vectors.max(self.current_header.vector_count);
                self.grow_vectors(max_v)?;
            }
            recycled_id
        } else {
            self.ensure_vector_capacity(1)?;
            let id = self.current_header.vector_count;
            self.current_header.vector_count += 1;
            id
        };

        let stride = self.bytes_per_vector();
        let byte_offset = self.vectors_offset + (vector_id as usize * stride);

        // Compute normalized values & SQ8 quantization in one pass
        let mut max_abs = 0.0f32;
        if self.current_header.vector_format == 1 {
            let byte_len = dims * 2;
            let dest = self.safe_slice_mut(byte_offset, byte_len)?;
            let dest_f16: &mut [f16] = bytemuck::cast_slice_mut(dest);
            for (d, &val) in dest_f16.iter_mut().zip(vector.iter()) {
                let nv = val * inv_norm;
                let a = nv.abs();
                if a > max_abs {
                    max_abs = a;
                }
                *d = f16::from_f32(nv);
            }
        } else {
            let byte_len = dims * 4;
            let dest = self.safe_slice_mut(byte_offset, byte_len)?;
            let dest_f32: &mut [f32] = bytemuck::cast_slice_mut(dest);
            for (d, &val) in dest_f32.iter_mut().zip(vector.iter()) {
                let nv = val * inv_norm;
                let a = nv.abs();
                if a > max_abs {
                    max_abs = a;
                }
                *d = nv;
            }
        }

        // Write SQ8 codes and Sq8Meta directly into mmap
        if self.sq8_codes_offset > 0 && self.sq8_meta_offset > 0 {
            let scale = if max_abs > 1e-12 { max_abs / 127.0 } else { 0.0 };
            let inv_scale = if max_abs > 1e-12 { 127.0 / max_abs } else { 0.0 };
            let code_off = self.sq8_codes_offset + (vector_id as usize * dims);
            let meta_off =
                self.sq8_meta_offset + (vector_id as usize * std::mem::size_of::<Sq8Meta>());

            let mut sum_q: i32 = 0;
            {
                let code_dest = self.safe_slice_mut(code_off, dims)?;
                let code_i8: &mut [i8] = bytemuck::cast_slice_mut(code_dest);
                for (q_out, &val) in code_i8.iter_mut().zip(vector.iter()) {
                    let nv = val * inv_norm;
                    let q = (nv * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    *q_out = q;
                    sum_q += q as i32;
                }
            }
            let meta = Sq8Meta { scale, sum_q };
            let meta_dest = self.safe_slice_mut(meta_off, std::mem::size_of::<Sq8Meta>())?;
            meta_dest.copy_from_slice(bytemuck::bytes_of(&meta));
            self.current_header.feature_flags |= FEATURE_SQ8_READY;
        }

        Ok(vector_id)
    }

    /// Copies a vector slot verbatim (raw f16/f32 bytes + SQ8 codes & metadata) from `src`
    /// without re-normalizing or losing f16 precision.
    pub fn copy_raw_vector_from(&mut self, src: &StarkStorage, old_vid: u32) -> Result<u32> {
        let new_vid = if self.current_header.free_vectors_head != u32::MAX {
            let recycled_id = self.current_header.free_vectors_head;
            let stride = self.bytes_per_vector();
            let byte_offset = self.vectors_offset + (recycled_id as usize * stride);
            let next_bytes = self.safe_slice(byte_offset, 4)?;
            let next_free = u32::from_le_bytes(next_bytes.try_into().unwrap());
            self.current_header.free_vectors_head = next_free;
            self.current_header.dead_vector_slots =
                self.current_header.dead_vector_slots.saturating_sub(1);
            recycled_id
        } else {
            self.ensure_vector_capacity(1)?;
            let id = self.current_header.vector_count;
            self.current_header.vector_count += 1;
            id
        };
        self.overwrite_raw_vector_from(new_vid, src, old_vid)?;
        Ok(new_vid)
    }

    /// Overwrites an existing vector slot `dst_vid` verbatim from `src` slot `src_vid`.
    pub fn overwrite_raw_vector_from(
        &mut self,
        dst_vid: u32,
        src: &StarkStorage,
        src_vid: u32,
    ) -> Result<()> {
        let dims = self.vector_dims;
        let stride = self.bytes_per_vector();
        let src_off = src.vectors_offset + (src_vid as usize * stride);
        let dst_off = self.vectors_offset + (dst_vid as usize * stride);
        let src_bytes = src.safe_slice(src_off, stride)?;
        let dst_bytes = self.safe_slice_mut(dst_off, stride)?;
        dst_bytes.copy_from_slice(src_bytes);

        if self.sq8_codes_offset > 0
            && self.sq8_meta_offset > 0
            && src.sq8_codes_offset > 0
            && src.sq8_meta_offset > 0
        {
            let src_code = src.safe_slice(src.sq8_codes_offset + (src_vid as usize * dims), dims)?;
            let src_meta = *bytemuck::from_bytes::<Sq8Meta>(src.safe_slice(
                src.sq8_meta_offset + (src_vid as usize * std::mem::size_of::<Sq8Meta>()),
                std::mem::size_of::<Sq8Meta>(),
            )?);
            let dst_code =
                self.safe_slice_mut(self.sq8_codes_offset + (dst_vid as usize * dims), dims)?;
            dst_code.copy_from_slice(src_code);
            let dst_meta = self.safe_slice_mut(
                self.sq8_meta_offset + (dst_vid as usize * std::mem::size_of::<Sq8Meta>()),
                std::mem::size_of::<Sq8Meta>(),
            )?;
            dst_meta.copy_from_slice(bytemuck::bytes_of(&src_meta));
            self.current_header.feature_flags |= FEATURE_SQ8_READY;
        }
        Ok(())
    }

    /// Toggles SQ8 quantization mode (`enable = true` -> backfills SQ8 and sets `FEATURE_SQ8_READY`;
    /// `enable = false` -> clears `FEATURE_SQ8_READY`).
    pub fn set_sq8_mode(&mut self, enable: bool) -> Result<()> {
        if enable {
            self.backfill_sq8()?;
        } else {
            self.current_header.feature_flags &= !FEATURE_SQ8_READY;
            self.commit_header()?;
        }
        Ok(())
    }

    pub fn set_dead_string_bytes(&mut self, dead_bytes: u64) {
        self.current_header.dead_string_bytes = dead_bytes;
    }

    pub fn set_vacuum_epoch(&mut self, epoch: u64) {
        self.current_header.vacuum_epoch = epoch;
    }

    pub fn flush(&mut self) -> Result<()> {
        self.commit_header()
    }
}

pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
