use bytemuck::{Pod, Zeroable};

pub const STARKDB_MAGIC: [u8; 8] = *b"STARK_DB";
pub const STARKDB_VERSION: u32 = 5; // Ultimate Legacy v1.0 Edition (Section Directory + SQ8 + Vacuum)
pub const STARKDB_V4_VERSION: u32 = 4; // Victory Boost Robust A/B Edition
pub const STARKDB_V3_VERSION: u32 = 3; // Wing Accel Legacy Edition

pub const HEADER_PAGE_SIZE: usize = 256;
pub const PAGE_A_OFFSET: usize = 0;
pub const PAGE_B_OFFSET: usize = 256;
pub const TOTAL_HEADER_SIZE: usize = 512;

pub const FLAG_DELETED: u16 = 1 << 0;
pub const FLAG_PINNED: u16 = 1 << 1;

pub const FEATURE_SQ8_READY: u32 = 1 << 0;
pub const FEATURE_BM25_CACHED: u32 = 1 << 1;
pub const FEATURE_INT4_RESERVED: u32 = 1 << 2;

pub const MANIFEST_MAGIC: [u8; 8] = *b"STARKMAN";
pub const MANIFEST_VERSION: u32 = 1;

/// Dual A/B Header Page (exactly 256 bytes, 64-byte cacheline aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DbHeader {
    pub magic: [u8; 8],            // b"STARK_DB" (8 bytes, offset 0..8)
    pub version: u32,              // DB format version = 5 (4 bytes, offset 8..12)
    pub vector_dims: u16,          // e.g. 768 or 1024 (2 bytes, offset 12..14)
    pub vector_format: u16,        // 0 = f32, 1 = f16, 2 = int8 (2 bytes, offset 14..16)

    pub generation: u64,           // Monotonically increasing generation / tx ID (8 bytes, offset 16..24)
    pub active_page: u8,           // 0 = Page A, 1 = Page B (1 byte, offset 24..25)
    pub _reserved1: [u8; 3],       // Alignment padding (3 bytes, offset 25..28)
    pub header_crc32: u32,         // CRC32 of bytes 0..28 and 32..256 (4 bytes, offset 28..32)

    pub max_nodes: u32,            // Current allocated node slots (4 bytes, offset 32..36)
    pub node_count: u32,           // Allocated node high-water count (4 bytes, offset 36..40)

    pub max_synapses: u32,         // Current allocated synapse slots (4 bytes, offset 40..44)
    pub synapse_count: u32,        // Allocated synapse high-water count (4 bytes, offset 44..48)

    pub max_vectors: u32,          // Current allocated vector slots (4 bytes, offset 48..52)
    pub vector_count: u32,         // Allocated vector high-water count (4 bytes, offset 52..56)

    pub strings_offset: u64,       // Byte offset where String Blob Pool begins (8 bytes, offset 56..64)
    pub strings_bytes: u64,        // Active bytes written into String Blob Pool (8 bytes, offset 64..72)

    pub created_at: u64,           // Unix epoch timestamp (8 bytes, offset 72..80)
    pub last_consolidated: u64,    // Last memory consolidation timestamp (8 bytes, offset 80..88)

    pub data_crc32: u32,           // Optional CRC32 of active data block snapshot (4 bytes, offset 88..92)
    pub free_nodes_head: u32,      // Free-list head for deleted node slots (u32::MAX if empty) (4 bytes, offset 92..96)
    pub free_synapses_head: u32,   // Free-list head for deleted synapse slots (u32::MAX if empty) (4 bytes, offset 96..100)
    pub free_vectors_head: u32,    // Free-list head for deleted vector slots (u32::MAX if empty) (4 bytes, offset 100..104)

    pub nodes_offset: u64,         // Section directory: NodeRecord table byte offset (8 bytes, offset 104..112)
    pub synapses_offset: u64,      // Section directory: SynapseRecord table byte offset (8 bytes, offset 112..120)
    pub vectors_offset: u64,       // Section directory: Vector arena byte offset (8 bytes, offset 120..128)
    pub sq8_codes_offset: u64,     // Section directory: SQ8 codes byte offset (0 = absent) (8 bytes, offset 128..136)
    pub sq8_meta_offset: u64,      // Section directory: SQ8 metadata byte offset (0 = absent) (8 bytes, offset 136..144)
    pub dead_string_bytes: u64,    // Dead string pool bytes tracked for Vacuum (8 bytes, offset 144..152)
    pub vacuum_epoch: u64,         // Monotonic vacuum epoch counter (8 bytes, offset 152..160)
    pub dead_vector_slots: u32,    // Number of recycled vector slots in free list (4 bytes, offset 160..164)
    pub feature_flags: u32,        // Bitmask of FEATURE_* flags (4 bytes, offset 164..168)

    pub _padding: [u64; 11],       // 11 * 8 = 88 bytes (offset 168..256)
}

impl DbHeader {
    pub fn compute_crc(&self) -> u32 {
        let mut clone = *self;
        clone.header_crc32 = 0;
        let bytes = bytemuck::bytes_of(&clone);
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(bytes);
        hasher.finalize()
    }

    pub fn update_crc(&mut self) {
        self.header_crc32 = self.compute_crc();
    }

    pub fn is_valid(&self) -> bool {
        self.magic == STARKDB_MAGIC
            && self.version == STARKDB_VERSION
            && self.compute_crc() == self.header_crc32
    }

    pub fn is_valid_v4(&self) -> bool {
        self.magic == STARKDB_MAGIC
            && self.version == STARKDB_V4_VERSION
            && self.compute_crc() == self.header_crc32
    }
}

/// Per-vector symmetric SQ8 quantization metadata (8 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Sq8Meta {
    pub scale: f32, // max_abs / 127.0
    pub sum_q: i32, // sum of quantized i8 elements (used for VNNI u8*i8 offset correction)
}

/// A/B/C Manifest Slot for Windows-safe atomic segment swap (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ManifestSlot {
    pub magic: [u8; 8],       // b"STARKMAN"
    pub version: u32,         // 1
    pub crc32: u32,           // CRC32 of slot with crc32 = 0
    pub generation: u64,      // Monotonic manifest generation
    pub active_seq: u64,      // Active segment number
    pub rollback_seq: u64,    // Rollback segment number (u64::MAX = none)
    pub vacuum_epoch: u64,    // Vacuum epoch
    pub _pad: [u64; 2],       // 16 bytes padding -> 64 bytes total
}

impl ManifestSlot {
    pub fn new(
        generation: u64,
        active_seq: u64,
        rollback_seq: u64,
        vacuum_epoch: u64,
    ) -> Self {
        let mut slot = Self {
            magic: *b"STARKMAN",
            version: 1,
            crc32: 0,
            generation,
            active_seq,
            rollback_seq,
            vacuum_epoch,
            _pad: [0; 2],
        };
        slot.crc32 = crc32fast::hash(bytemuck::bytes_of(&slot));
        slot
    }

    pub fn is_valid(&self) -> bool {
        if &self.magic != b"STARKMAN" || self.version != 1 {
            return false;
        }
        let mut copy = *self;
        copy.crc32 = 0;
        crc32fast::hash(bytemuck::bytes_of(&copy)) == self.crc32
    }
}

/// Legacy Header from Wing Accel v3 (128 bytes). Used for auto-migration.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DbHeaderV3 {
    pub magic: [u8; 8],            // b"STARK_DB" (8 bytes, offset 0..8)
    pub version: u32,              // DB format version = 3 (4 bytes, offset 8..12)
    pub vector_dims: u16,          // e.g. 768 or 1024 (2 bytes, offset 12..14)
    pub _reserved1: u16,           // (2 bytes, offset 14..16)

    pub max_nodes: u32,            // Current allocated node slots (4 bytes, offset 16..20)
    pub node_count: u32,           // Used node slots (4 bytes, offset 20..24)

    pub max_synapses: u32,         // Current allocated synapse slots (4 bytes, offset 24..28)
    pub synapse_count: u32,        // Used synapse slots (4 bytes, offset 28..32)

    pub max_vectors: u32,          // Current allocated vector slots (4 bytes, offset 32..36)
    pub vector_count: u32,         // Used vector slots (4 bytes, offset 36..40)

    pub strings_offset: u64,       // Byte offset where String Blob Pool begins (8 bytes, offset 40..48)
    pub strings_bytes: u64,        // Active bytes written into String Blob Pool (8 bytes, offset 48..56)

    pub created_at: u64,           // Unix epoch timestamp (8 bytes, offset 56..64)
    pub last_consolidated: u64,    // Last memory consolidation timestamp (8 bytes, offset 64..72)

    pub _reserved2: [u32; 2],      // 2 * 4 = 8 bytes, offset 72..80
    pub _padding: [u64; 6],        // 6 * 8 = 48 bytes, offset 80..128
}

/// Node (Concept / Memory document) record with fixed binary size (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct NodeRecord {
    pub id: u32,                  // 0..4: Node ID (0, 1, 2...)
    pub node_type: u16,           // 4..6: 0=Entity, 1=Fact, 2=Config, 3=Document, 4=Hardware, 5=Software, 6=Concept
    pub flags: u16,               // 6..8: Bit 0 = deleted, Bit 1 = pinned (immune to decay)

    pub name_offset: u32,         // 8..12: Byte offset in String Pool for node name
    pub name_len: u16,            // 12..14: Length of node name in bytes
    pub tags_len: u16,            // 14..16: Length of tags string in bytes

    pub tags_offset: u32,         // 16..20: Byte offset in String Pool for comma-separated tags
    pub content_offset: u32,      // 20..24: Byte offset in String Pool for JSON/text payload
    pub content_len: u32,         // 24..28: Length of payload in bytes
    pub vector_id: u32,           // 28..32: Slot in Vector Pool (u32::MAX if no vector)

    pub created_at: u32,          // 32..36: Timestamp of creation
    pub last_activated: u32,      // 36..40: Timestamp of last recall/activation
    pub base_weight: f32,         // 40..44: Baseline importance (0.0 .. 1.0)
    pub activation_count: u32,    // 44..48: Total times this node was activated
}

/// Synapse (Directed associative link) with fixed binary size (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SynapseRecord {
    pub id: u32,                  // 0..4: Synapse ID (0, 1, 2...)
    pub source_id: u32,           // 4..8: Origin Node ID
    pub target_id: u32,           // 8..12: Destination Node ID
    pub rel_offset: u32,          // 12..16: Offset in String Pool

    pub rel_len: u16,             // 16..18: Length of relation name
    pub flags: u16,               // 18..20: Bit 0 = deleted, Bit 1 = bidirectional
    pub weight: f32,              // 20..24: Strength of association (0.0 .. 1.0)
    pub last_used: u32,           // 24..28: Timestamp of last traversal
    pub activation_count: u32,    // 28..32: How many times this synapse fired
}

pub fn node_type_to_str(t: u16) -> &'static str {
    match t {
        0 => "Entity",
        1 => "Fact",
        2 => "Config",
        3 => "Document",
        4 => "Hardware",
        5 => "Software",
        6 => "Concept",
        _ => "Entity",
    }
}

pub fn str_to_node_type(s: &str) -> u16 {
    match s.trim().to_lowercase().as_str() {
        "entity" | "person" | "user" | "сутність" => 0,
        "fact" | "факт" => 1,
        "config" | "setting" | "конфіг" => 2,
        "document" | "doc" | "json" | "документ" => 3,
        "hardware" | "gpu" | "device" | "залізо" => 4,
        "software" | "os" | "system" | "daemon" | "софт" => 5,
        "concept" | "idea" | "концепт" => 6,
        _ => 0,
    }
}

const _: () = assert!(std::mem::size_of::<DbHeader>() == 256);
const _: () = assert!(std::mem::size_of::<DbHeaderV3>() == 128);
const _: () = assert!(std::mem::size_of::<NodeRecord>() == 48);
const _: () = assert!(std::mem::size_of::<SynapseRecord>() == 32);
const _: () = assert!(std::mem::size_of::<Sq8Meta>() == 8);
const _: () = assert!(std::mem::size_of::<ManifestSlot>() == 64);
