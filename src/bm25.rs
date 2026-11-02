use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use hashbrown::HashTable;
use smallvec::SmallVec;
use std::path::Path;

pub const BM25_K1: f32 = 1.2;
pub const BM25_B: f32 = 0.75;
pub const FIELD_W_NAME: u16 = 3;
pub const FIELD_W_TAGS: u16 = 2;
pub const FIELD_W_CONTENT: u16 = 1;
pub const MAX_TOKEN_BYTES: usize = 64;
pub const TAG_TERM_PREFIX: u8 = 0x01;

pub const BM25_CACHE_MAGIC: &[u8; 8] = b"STARKBM2";
pub const BM25_CACHE_VERSION: u32 = 1;

/// Sorted static list of common English and Ukrainian stop-words for binary_search filtering.
static STOP_WORDS: &[&str] = &[
    "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "was", "with", "або", "але", "бо", "вже", "від",
    "він", "вона", "вони", "воно", "все", "всі", "для", "до", "дуже", "за", "зі", "із", "ін",
    "її", "їх", "його", "коли", "лише", "має", "ми", "між", "на", "навіть", "над", "нам", "нас",
    "не", "ні", "ній", "однак", "під", "після", "по", "при", "про", "саме", "собі", "та", "так",
    "також", "там", "те", "ти", "то", "того", "тоді", "той", "тому", "тут", "усі", "хоча", "це",
    "цей", "цих", "цю", "ця", "через", "чи", "ची", "що", "щоб", "як", "яка", "який", "які",
    "якого", "якої", "якщо",
];

#[inline]
fn hash_term_bytes(bytes: &[u8]) -> u64 {
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TermMeta {
    pub bytes_off: u32, // offset in term_bytes (folded token, UTF-8)
    pub bytes_len: u16,
    pub max_tf: u16,    // for upper-bound / MaxScore
    pub df: u32,        // number of live documents containing term
    pub block_off: u32, // offset in post_doc / post_tf
    pub block_len: u32,
    pub block_cap: u32, // doubled on overflow, relocated to end of arena
}

const _: () = assert!(std::mem::size_of::<TermMeta>() == 24);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Bm25CacheHeader {
    pub magic: [u8; 8],
    pub generation: u64,
    pub version: u32,
    pub n_docs: u32,
    pub total_len: u64,
    pub avgdl_at_norm: f32,
    pub dead_slots: u32,
    pub term_bytes_len: u32,
    pub terms_len: u32,
    pub post_len: u32,
    pub doc_len_len: u32,
    pub payload_crc32: u32,
    pub _reserved: u32,
}

const _: () = assert!(std::mem::size_of::<Bm25CacheHeader>() == 64);

/// Zero-allocation tokenizer over `&str` using an internal stack buffer `[u8; MAX_TOKEN_BYTES]`.
/// Folds both ASCII and Cyrillic/Unicode characters via `char::to_lowercase`.
pub struct TokenIter<'a> {
    chars: std::str::Chars<'a>,
    buf: [u8; MAX_TOKEN_BYTES],
}

impl<'a> TokenIter<'a> {
    #[inline]
    pub fn new(src: &'a str) -> Self {
        Self {
            chars: src.chars(),
            buf: [0u8; MAX_TOKEN_BYTES],
        }
    }

    #[inline]
    pub fn next_token(&mut self) -> Option<&[u8]> {
        loop {
            let mut byte_len = 0usize;
            let mut char_count = 0usize;
            let mut overflowed = false;

            for c in self.chars.by_ref() {
                if c.is_alphanumeric() || c == '_' {
                    if !overflowed {
                        for lc in c.to_lowercase() {
                            let need = lc.len_utf8();
                            if byte_len + need <= MAX_TOKEN_BYTES {
                                lc.encode_utf8(&mut self.buf[byte_len..]);
                                byte_len += need;
                                char_count += 1;
                            } else {
                                overflowed = true;
                                break;
                            }
                        }
                    }
                } else if byte_len > 0 || overflowed {
                    break;
                }
            }

            if byte_len == 0 && !overflowed {
                return None;
            }
            if overflowed || char_count < 2 {
                continue;
            }

            let is_stop = std::str::from_utf8(&self.buf[..byte_len])
                .map(|s| STOP_WORDS.binary_search(&s).is_ok())
                .unwrap_or(false);
            if is_stop {
                continue;
            }
            return Some(&self.buf[..byte_len]);
        }
    }
}

/// Folds a single tag into `\x01<lowercase_tag>` inside caller's stack buffer.
#[inline]
pub fn encode_tag_token<'a>(raw_tag: &str, buf: &'a mut [u8; MAX_TOKEN_BYTES]) -> Option<&'a [u8]> {
    let trimmed = raw_tag.trim();
    if trimmed.is_empty() {
        return None;
    }
    buf[0] = TAG_TERM_PREFIX;
    let mut byte_len = 1usize;
    for c in trimmed.chars() {
        for lc in c.to_lowercase() {
            let need = lc.len_utf8();
            if byte_len + need > MAX_TOKEN_BYTES {
                return None;
            }
            lc.encode_utf8(&mut buf[byte_len..]);
            byte_len += need;
        }
    }
    if byte_len <= 1 {
        None
    } else {
        Some(&buf[..byte_len])
    }
}

#[derive(Clone)]
pub struct Bm25Index {
    pub term_bytes: Vec<u8>,
    pub terms: Vec<TermMeta>,
    pub dict: HashTable<u32>,
    pub post_doc: Vec<u32>,
    pub post_tf: Vec<u16>,
    pub doc_len: Vec<u32>,
    pub doc_norm: Vec<f32>,
    pub n_docs: u32,
    pub total_len: u64,
    pub avgdl_at_norm: f32,
    pub dead_slots: u32,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            term_bytes: Vec::new(),
            terms: Vec::new(),
            dict: HashTable::new(),
            post_doc: Vec::new(),
            post_tf: Vec::new(),
            doc_len: Vec::new(),
            doc_norm: Vec::new(),
            n_docs: 0,
            total_len: 0,
            avgdl_at_norm: 0.0,
            dead_slots: 0,
        }
    }

    pub fn clear(&mut self) {
        self.term_bytes.clear();
        self.terms.clear();
        self.dict.clear();
        self.post_doc.clear();
        self.post_tf.clear();
        self.doc_len.clear();
        self.doc_norm.clear();
        self.n_docs = 0;
        self.total_len = 0;
        self.avgdl_at_norm = 0.0;
        self.dead_slots = 0;
    }

    #[inline]
    pub fn active_terms_count(&self) -> usize {
        self.terms.iter().filter(|t| t.df > 0).count()
    }

    #[inline]
    pub fn active_postings_count(&self) -> usize {
        self.terms.iter().map(|t| t.block_len as usize).sum()
    }

    #[inline]
    pub fn avgdl(&self) -> f32 {
        if self.n_docs == 0 {
            1.0
        } else {
            (self.total_len as f32 / self.n_docs as f32).max(1.0)
        }
    }

    #[inline]
    pub fn idf(&self, df: u32) -> f32 {
        if df == 0 || self.n_docs == 0 {
            return 0.0;
        }
        let n = self.n_docs as f32;
        let df_f = df as f32;
        let num = (n - df_f + 0.5).max(0.0);
        let den = df_f + 0.5;
        (1.0 + num / den).ln()
    }

    #[inline]
    pub fn find_term_id(&self, token: &[u8]) -> Option<u32> {
        let h = hash_term_bytes(token);
        let tb = &self.term_bytes;
        let terms = &self.terms;
        self.dict
            .find(h, |&tid| {
                let m = &terms[tid as usize];
                let s = m.bytes_off as usize;
                let e = s + m.bytes_len as usize;
                &tb[s..e] == token
            })
            .copied()
    }

    #[inline]
    fn get_or_create_term_id(&mut self, token: &[u8]) -> u32 {
        if let Some(id) = self.find_term_id(token) {
            return id;
        }
        let h = hash_term_bytes(token);
        let bytes_off = self.term_bytes.len() as u32;
        let bytes_len = token.len() as u16;
        self.term_bytes.extend_from_slice(token);

        let block_off = self.post_doc.len() as u32;
        let initial_cap = 4u32;
        self.post_doc.resize(block_off as usize + initial_cap as usize, 0);
        self.post_tf.resize(block_off as usize + initial_cap as usize, 0);

        let tid = self.terms.len() as u32;
        self.terms.push(TermMeta {
            bytes_off,
            bytes_len,
            max_tf: 0,
            df: 0,
            block_off,
            block_len: 0,
            block_cap: initial_cap,
        });

        let tb = &self.term_bytes;
        let terms = &self.terms;
        self.dict.insert_unique(h, tid, |&existing_tid| {
            let m = &terms[existing_tid as usize];
            let s = m.bytes_off as usize;
            let e = s + m.bytes_len as usize;
            hash_term_bytes(&tb[s..e])
        });
        tid
    }

    #[inline]
    fn append_posting(&mut self, term_id: u32, doc_id: u32, tf: u16) {
        let meta = &mut self.terms[term_id as usize];
        let b_off = meta.block_off as usize;
        let b_len = meta.block_len as usize;
        if b_len > 0 && self.post_doc[b_off + b_len - 1] == doc_id {
            self.post_tf[b_off + b_len - 1] = tf;
            if tf > meta.max_tf {
                meta.max_tf = tf;
            }
            return;
        }

        if meta.block_len == meta.block_cap {
            let old_off = meta.block_off as usize;
            let old_cap = meta.block_cap as usize;
            let new_cap = (old_cap * 2).max(4);
            let new_off = self.post_doc.len();

            self.post_doc.resize(new_off + new_cap, 0);
            self.post_tf.resize(new_off + new_cap, 0);

            self.post_doc.copy_within(old_off..old_off + b_len, new_off);
            self.post_tf.copy_within(old_off..old_off + b_len, new_off);

            self.dead_slots = self.dead_slots.saturating_add(old_cap as u32);
            meta.block_off = new_off as u32;
            meta.block_cap = new_cap as u32;
        }

        let write_pos = meta.block_off as usize + meta.block_len as usize;
        self.post_doc[write_pos] = doc_id;
        self.post_tf[write_pos] = tf;
        meta.block_len += 1;
        meta.df += 1;
        if tf > meta.max_tf {
            meta.max_tf = tf;
        }
    }

    #[inline]
    fn remove_posting(&mut self, term_id: u32, doc_id: u32) {
        let meta = &mut self.terms[term_id as usize];
        let b_off = meta.block_off as usize;
        let b_len = meta.block_len as usize;
        for i in 0..b_len {
            if self.post_doc[b_off + i] == doc_id {
                let last = b_off + b_len - 1;
                self.post_doc[b_off + i] = self.post_doc[last];
                self.post_tf[b_off + i] = self.post_tf[last];
                meta.block_len -= 1;
                meta.df = meta.df.saturating_sub(1);
                return;
            }
        }
    }

    fn update_doc_norms(&mut self, updated_node_id: u32) {
        let cur_avgdl = self.avgdl();
        let drift = if self.avgdl_at_norm <= 0.0 {
            1.0
        } else {
            (cur_avgdl - self.avgdl_at_norm).abs() / self.avgdl_at_norm
        };

        if self.n_docs <= 32 || drift > 0.05 {
            self.recompute_all_norms();
        } else if let Some(&dl) = self.doc_len.get(updated_node_id as usize) {
            if dl > 0 {
                self.doc_norm[updated_node_id as usize] =
                    BM25_K1 * (1.0 - BM25_B + BM25_B * (dl as f32) / cur_avgdl);
            } else {
                self.doc_norm[updated_node_id as usize] = 0.0;
            }
        }
    }

    pub fn recompute_all_norms(&mut self) {
        let cur_avgdl = self.avgdl();
        self.avgdl_at_norm = cur_avgdl;
        if self.doc_norm.len() < self.doc_len.len() {
            self.doc_norm.resize(self.doc_len.len(), 0.0);
        }
        for (i, &dl) in self.doc_len.iter().enumerate() {
            if dl > 0 {
                self.doc_norm[i] = BM25_K1 * (1.0 - BM25_B + BM25_B * (dl as f32) / cur_avgdl);
            } else {
                self.doc_norm[i] = 0.0;
            }
        }
    }

    /// Indexes a document (`node_id`) with BM25F field weights (`3x name`, `2x tags`, `1x content`)
    /// plus exact tag filter tokens (`\x01<tag>`).
    pub fn index_doc(&mut self, node_id: u32, name: &str, tags: &str, content: &str) {
        let idx = node_id as usize;
        if idx >= self.doc_len.len() {
            let new_len = idx + 1;
            self.doc_len.resize(new_len, 0);
            self.doc_norm.resize(new_len, 0.0);
        }

        let mut acc_terms: SmallVec<[(u32, u16); 64]> = SmallVec::new();
        let mut dl: u32 = 0;

        let mut add_field = |text: &str, weight: u16, bm25: &mut Self| {
            let mut iter = TokenIter::new(text);
            while let Some(tok) = iter.next_token() {
                let tid = bm25.get_or_create_term_id(tok);
                dl = dl.saturating_add(weight as u32);
                if let Some(entry) = acc_terms.iter_mut().find(|(id, _)| *id == tid) {
                    entry.1 = entry.1.saturating_add(weight);
                } else {
                    acc_terms.push((tid, weight));
                }
            }
        };

        add_field(name, FIELD_W_NAME, self);
        add_field(tags, FIELD_W_TAGS, self);
        add_field(content, FIELD_W_CONTENT, self);

        for (tid, tf) in acc_terms {
            self.append_posting(tid, node_id, tf);
        }

        // Index exact tag tokens `\x01<tag>` for O(1) tag filtering in Bound Attack
        if !tags.is_empty() {
            let mut tag_ids: SmallVec<[u32; 8]> = SmallVec::new();
            let mut tag_buf = [0u8; MAX_TOKEN_BYTES];
            for raw_tag in tags.split(',') {
                if let Some(tag_tok) = encode_tag_token(raw_tag, &mut tag_buf) {
                    let tid = self.get_or_create_term_id(tag_tok);
                    if !tag_ids.contains(&tid) {
                        tag_ids.push(tid);
                    }
                }
            }
            for tid in tag_ids {
                self.append_posting(tid, node_id, 1);
            }
        }

        if dl > 0 {
            if self.doc_len[idx] == 0 {
                self.n_docs += 1;
            } else {
                self.total_len = self.total_len.saturating_sub(self.doc_len[idx] as u64);
            }
            self.doc_len[idx] = dl;
            self.total_len += dl as u64;
            self.update_doc_norms(node_id);
        }
    }

    /// Removes a document (`node_id`) from the BM25 index by re-tokenizing its old text from mmap
    /// and performing O(df) `swap_remove` on each posting block.
    pub fn remove_doc(&mut self, node_id: u32, old_name: &str, old_tags: &str, old_content: &str) {
        let idx = node_id as usize;
        let old_dl = self.doc_len.get(idx).copied().unwrap_or(0);
        if old_dl == 0 && old_tags.is_empty() && old_name.is_empty() && old_content.is_empty() {
            return;
        }

        let mut tids: SmallVec<[u32; 64]> = SmallVec::new();
        let mut collect_field = |text: &str, bm25: &Self| {
            let mut iter = TokenIter::new(text);
            while let Some(tok) = iter.next_token() {
                if let Some(tid) = bm25.find_term_id(tok) {
                    if !tids.contains(&tid) {
                        tids.push(tid);
                    }
                }
            }
        };

        collect_field(old_name, self);
        collect_field(old_tags, self);
        collect_field(old_content, self);

        if !old_tags.is_empty() {
            let mut tag_buf = [0u8; MAX_TOKEN_BYTES];
            for raw_tag in old_tags.split(',') {
                if let Some(tag_tok) = encode_tag_token(raw_tag, &mut tag_buf) {
                    if let Some(tid) = self.find_term_id(tag_tok) {
                        if !tids.contains(&tid) {
                            tids.push(tid);
                        }
                    }
                }
            }
        }

        for tid in tids {
            self.remove_posting(tid, node_id);
        }

        if old_dl > 0 {
            self.doc_len[idx] = 0;
            self.doc_norm[idx] = 0.0;
            self.n_docs = self.n_docs.saturating_sub(1);
            self.total_len = self.total_len.saturating_sub(old_dl as u64);
            self.update_doc_norms(node_id);
        }
    }

    /// Returns the posting slice of `node_id`s for a given exact tag (`\x01<tag>`).
    #[inline]
    pub fn postings_for_tag(&self, raw_tag: &str) -> &[u32] {
        let mut tag_buf = [0u8; MAX_TOKEN_BYTES];
        if let Some(tag_tok) = encode_tag_token(raw_tag, &mut tag_buf) {
            if let Some(tid) = self.find_term_id(tag_tok) {
                let m = &self.terms[tid as usize];
                let s = m.block_off as usize;
                let e = s + m.block_len as usize;
                return &self.post_doc[s..e];
            }
        }
        &[]
    }

    /// Computes exact analytical upper bound $UB(q) = \sum_{t \in q} \mathrm{idf}(t) \cdot (k_1 + 1)$.
    pub fn upper_bound(&self, query: &str) -> f32 {
        let mut seen: SmallVec<[u32; 16]> = SmallVec::new();
        let mut ub = 0.0f32;
        let mut iter = TokenIter::new(query);
        while let Some(tok) = iter.next_token() {
            if let Some(tid) = self.find_term_id(tok) {
                if !seen.contains(&tid) {
                    seen.push(tid);
                    let df = self.terms[tid as usize].df;
                    if df > 0 {
                        ub += self.idf(df) * (BM25_K1 + 1.0);
                    }
                }
            }
        }
        ub
    }

    /// Term-at-a-time BM25F evaluation writing raw scores into `scratch.acc[doc_id][1]`.
    /// Returns `(matched_query_terms, ub_q)` where `ub_q = \sum idf(t) * (k1 + 1)`.
    #[inline]
    pub fn search_into(
        &self,
        query: &str,
        scratch: &mut crate::hybrid::QueryScratch,
        top_k: usize,
    ) -> (usize, f32) {
        self.search_into_filtered(query, scratch, top_k, |_| true)
    }

    /// Filtered Term-at-a-Time BM25F evaluation with ascending-DF ordering and MaxScore upper-bound pruning.
    pub fn search_into_filtered<F>(
        &self,
        query: &str,
        scratch: &mut crate::hybrid::QueryScratch,
        top_k: usize,
        filter: F,
    ) -> (usize, f32)
    where
        F: Fn(u32) -> bool,
    {
        let mut q_terms: SmallVec<[u32; 16]> = SmallVec::new();
        let mut iter = TokenIter::new(query);
        while let Some(tok) = iter.next_token() {
            if let Some(tid) = self.find_term_id(tok) {
                let m = &self.terms[tid as usize];
                if m.df > 0 && m.block_len > 0 && !q_terms.contains(&tid) {
                    q_terms.push(tid);
                }
            }
        }
        if q_terms.is_empty() {
            return (0, 0.0);
        }

        // Order query terms by ascending DF so selective, high-IDF terms execute first
        q_terms.sort_unstable_by_key(|&tid| self.terms[tid as usize].df);

        let k1_plus_1 = BM25_K1 + 1.0;
        let min_norm = (BM25_K1 * (1.0 - BM25_B)).max(0.1);
        let mut ub_q = 0.0f32;
        let mut term_max_score: SmallVec<[f32; 16]> = SmallVec::with_capacity(q_terms.len());
        let mut term_idfs: SmallVec<[f32; 16]> = SmallVec::with_capacity(q_terms.len());

        for &tid in &q_terms {
            let meta = &self.terms[tid as usize];
            let term_idf = self.idf(meta.df);
            ub_q += term_idf * k1_plus_1;
            let mtf = (meta.max_tf.max(1)) as f32;
            let max_s = term_idf * (mtf * k1_plus_1) / (mtf + min_norm);
            term_idfs.push(term_idf);
            term_max_score.push(max_s);
        }

        let mut suffix_ub: SmallVec<[f32; 16]> = SmallVec::with_capacity(q_terms.len());
        suffix_ub.resize(q_terms.len(), 0.0);
        let mut running_ub = 0.0f32;
        for idx in (0..q_terms.len()).rev() {
            running_ub += term_max_score[idx];
            suffix_ub[idx] = running_ub;
        }

        let mut threshold = 0.0f32;

        for (t_idx, &tid) in q_terms.iter().enumerate() {
            let meta = &self.terms[tid as usize];
            let term_idf = term_idfs[t_idx];
            let b_off = meta.block_off as usize;
            let b_len = meta.block_len as usize;
            let docs = &self.post_doc[b_off..b_off + b_len];
            let tfs = &self.post_tf[b_off..b_off + b_len];

            // MaxScore check: if we already have >= top_k candidates and a high-DF term's remaining
            // suffix_ub cannot reach threshold on its own, only score already-touched candidates.
            if top_k > 0
                && t_idx > 0
                && b_len > 256
                && scratch.touched.len() >= top_k
                && threshold == 0.0
            {
                // Compute current top_k threshold from touched candidates
                let mut scores: SmallVec<[f32; 128]> = scratch
                    .touched
                    .iter()
                    .map(|&id| scratch.acc[id as usize][1])
                    .collect();
                if scores.len() >= top_k {
                    scores.select_nth_unstable_by(top_k - 1, |a, b| {
                        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    threshold = scores[top_k - 1];
                }
            }

            let non_essential = top_k > 0
                && t_idx > 0
                && b_len > 256
                && scratch.touched.len() >= top_k
                && suffix_ub[t_idx] < threshold;

            if non_essential {
                for &doc_id in &scratch.touched {
                    let _ = doc_id;
                }
                for i in 0..b_len {
                    let doc_id = docs[i];
                    if !scratch.is_touched(doc_id) {
                        continue;
                    }
                    let tf = tfs[i] as f32;
                    let norm = self
                        .doc_norm
                        .get(doc_id as usize)
                        .copied()
                        .unwrap_or(BM25_K1);
                    if norm > 0.0 {
                        scratch.acc[doc_id as usize][1] +=
                            term_idf * (tf * k1_plus_1) / (tf + norm);
                    }
                }
            } else {
                for i in 0..b_len {
                    let doc_id = docs[i];
                    if !filter(doc_id) {
                        continue;
                    }
                    let tf = tfs[i] as f32;
                    let norm = self
                        .doc_norm
                        .get(doc_id as usize)
                        .copied()
                        .unwrap_or(BM25_K1);
                    if norm > 0.0 {
                        let score = term_idf * (tf * k1_plus_1) / (tf + norm);
                        scratch.touch(doc_id);
                        scratch.acc[doc_id as usize][1] += score;
                    }
                }
            }
        }

        (q_terms.len(), ub_q)
    }

    /// Standalone BM25F top-k search returning `(score, node_id)` sorted by descending score.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(f32, u32)> {
        if top_k == 0 || self.n_docs == 0 {
            return Vec::new();
        }
        let max_nodes = self.doc_len.len().max(1);
        crate::hybrid::with_scratch(max_nodes, |scratch| {
            let (n_terms, _) = self.search_into(query, scratch, top_k);
            if n_terms == 0 {
                return Vec::new();
            }
            let mut hits: Vec<(f32, u32)> = scratch
                .touched
                .iter()
                .filter_map(|&doc_id| {
                    let s = scratch.acc[doc_id as usize][1];
                    if s > 0.0 {
                        Some((s, doc_id))
                    } else {
                        None
                    }
                })
                .collect();
            if hits.len() > top_k {
                hits.select_nth_unstable_by(top_k, |a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                hits.truncate(top_k);
            }
            hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            hits
        })
    }

    /// Compacts the posting arena in-place (`dead_slots -> 0`), recalculating `df`, `max_tf`, and `doc_norm`.
    pub fn compact_arena(&mut self) {
        let total_live: usize = self.terms.iter().map(|t| t.block_len as usize).sum();
        let mut new_doc = Vec::with_capacity(total_live);
        let mut new_tf = Vec::with_capacity(total_live);

        for meta in &mut self.terms {
            let b_off = meta.block_off as usize;
            let b_len = meta.block_len as usize;
            let new_off = new_doc.len() as u32;
            let mut max_tf = 0u16;
            for i in 0..b_len {
                let d = self.post_doc[b_off + i];
                let t = self.post_tf[b_off + i];
                new_doc.push(d);
                new_tf.push(t);
                if t > max_tf {
                    max_tf = t;
                }
            }
            meta.block_off = new_off;
            meta.block_cap = meta.block_len;
            meta.df = meta.block_len;
            meta.max_tf = max_tf;
        }

        self.post_doc = new_doc;
        self.post_tf = new_tf;
        self.dead_slots = 0;
        self.recompute_all_norms();
    }

    pub fn serialize_cache(&self, generation: u64) -> Vec<u8> {
        let tb_bytes = &self.term_bytes;
        let terms_bytes: &[u8] = bytemuck::cast_slice(&self.terms);
        let post_doc_bytes: &[u8] = bytemuck::cast_slice(&self.post_doc);
        let post_tf_bytes: &[u8] = bytemuck::cast_slice(&self.post_tf);
        let doc_len_bytes: &[u8] = bytemuck::cast_slice(&self.doc_len);
        let doc_norm_bytes: &[u8] = bytemuck::cast_slice(&self.doc_norm);

        let payload_len = tb_bytes.len()
            + terms_bytes.len()
            + post_doc_bytes.len()
            + post_tf_bytes.len()
            + doc_len_bytes.len()
            + doc_norm_bytes.len();
        let mut payload = Vec::with_capacity(payload_len);
        payload.extend_from_slice(tb_bytes);
        payload.extend_from_slice(terms_bytes);
        payload.extend_from_slice(post_doc_bytes);
        payload.extend_from_slice(post_tf_bytes);
        payload.extend_from_slice(doc_len_bytes);
        payload.extend_from_slice(doc_norm_bytes);

        let crc = crc32fast::hash(&payload);
        let hdr = Bm25CacheHeader {
            magic: *BM25_CACHE_MAGIC,
            generation,
            version: BM25_CACHE_VERSION,
            n_docs: self.n_docs,
            total_len: self.total_len,
            avgdl_at_norm: self.avgdl_at_norm,
            dead_slots: self.dead_slots,
            term_bytes_len: self.term_bytes.len() as u32,
            terms_len: self.terms.len() as u32,
            post_len: self.post_doc.len() as u32,
            doc_len_len: self.doc_len.len() as u32,
            payload_crc32: crc,
            _reserved: 0,
        };

        let mut out = Vec::with_capacity(std::mem::size_of::<Bm25CacheHeader>() + payload.len());
        out.extend_from_slice(bytemuck::bytes_of(&hdr));
        out.extend_from_slice(&payload);
        out
    }

    pub fn deserialize_cache(&mut self, bytes: &[u8], expected_generation: u64) -> Result<bool> {
        let hdr_size = std::mem::size_of::<Bm25CacheHeader>();
        if bytes.len() < hdr_size {
            return Ok(false);
        }
        let hdr: Bm25CacheHeader = bytemuck::pod_read_unaligned(&bytes[..hdr_size]);
        if &hdr.magic != BM25_CACHE_MAGIC
            || hdr.version != BM25_CACHE_VERSION
            || hdr.generation != expected_generation
        {
            return Ok(false);
        }

        let payload = &bytes[hdr_size..];
        let tb_len = hdr.term_bytes_len as usize;
        let terms_bytes_len = hdr.terms_len as usize * std::mem::size_of::<TermMeta>();
        let post_doc_bytes_len = hdr.post_len as usize * 4;
        let post_tf_bytes_len = hdr.post_len as usize * 2;
        let doc_len_bytes_len = hdr.doc_len_len as usize * 4;
        let doc_norm_bytes_len = hdr.doc_len_len as usize * 4;

        let expected_payload_len = tb_len
            + terms_bytes_len
            + post_doc_bytes_len
            + post_tf_bytes_len
            + doc_len_bytes_len
            + doc_norm_bytes_len;
        if payload.len() != expected_payload_len {
            return Ok(false);
        }
        if crc32fast::hash(payload) != hdr.payload_crc32 {
            return Ok(false);
        }

        let mut off = 0usize;
        let term_bytes = payload[off..off + tb_len].to_vec();
        off += tb_len;

        let mut terms = Vec::with_capacity(hdr.terms_len as usize);
        for _ in 0..hdr.terms_len as usize {
            let m: TermMeta = bytemuck::pod_read_unaligned(&payload[off..off + 24]);
            if (m.bytes_off as usize) + (m.bytes_len as usize) > term_bytes.len()
                || (m.block_off as usize) + (m.block_len as usize) > hdr.post_len as usize
            {
                return Ok(false);
            }
            terms.push(m);
            off += 24;
        }

        let mut post_doc = Vec::with_capacity(hdr.post_len as usize);
        for _ in 0..hdr.post_len as usize {
            post_doc.push(u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()));
            off += 4;
        }

        let mut post_tf = Vec::with_capacity(hdr.post_len as usize);
        for _ in 0..hdr.post_len as usize {
            post_tf.push(u16::from_le_bytes(payload[off..off + 2].try_into().unwrap()));
            off += 2;
        }

        let mut doc_len = Vec::with_capacity(hdr.doc_len_len as usize);
        for _ in 0..hdr.doc_len_len as usize {
            doc_len.push(u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()));
            off += 4;
        }

        let mut doc_norm = Vec::with_capacity(hdr.doc_len_len as usize);
        for _ in 0..hdr.doc_len_len as usize {
            doc_norm.push(f32::from_le_bytes(payload[off..off + 4].try_into().unwrap()));
            off += 4;
        }

        let mut dict = HashTable::with_capacity(terms.len());
        for (tid, m) in terms.iter().enumerate() {
            let s = m.bytes_off as usize;
            let e = s + m.bytes_len as usize;
            let h = hash_term_bytes(&term_bytes[s..e]);
            dict.insert_unique(h, tid as u32, |&existing_tid| {
                let em = &terms[existing_tid as usize];
                let es = em.bytes_off as usize;
                let ee = es + em.bytes_len as usize;
                hash_term_bytes(&term_bytes[es..ee])
            });
        }

        self.term_bytes = term_bytes;
        self.terms = terms;
        self.dict = dict;
        self.post_doc = post_doc;
        self.post_tf = post_tf;
        self.doc_len = doc_len;
        self.doc_norm = doc_norm;
        self.n_docs = hdr.n_docs;
        self.total_len = hdr.total_len;
        self.avgdl_at_norm = hdr.avgdl_at_norm;
        self.dead_slots = hdr.dead_slots;

        Ok(true)
    }

    pub fn save_to_file(&self, path: &Path, generation: u64) -> Result<()> {
        let bytes = self.serialize_cache(generation);
        std::fs::write(path, bytes)
            .with_context(|| format!("Failed to write BM25 cache to {}", path.display()))
    }

    pub fn load_from_file(&mut self, path: &Path, expected_generation: u64) -> Result<bool> {
        if !path.exists() {
            return Ok(false);
        }
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => return Ok(false),
        };
        self.deserialize_cache(&bytes, expected_generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_exact_scoring() {
        let mut bm25 = Bm25Index::new();
        // 3 documents with distinct name (3x), tags (2x), and content (1x) weights
        // Doc 0: name="stark server" (2 tokens * 3 = 6), tags="rust" (1 token * 2 = 2), content="fast engine" (2 tokens * 1 = 2) -> dl_0 = 10
        // Doc 1: name="tauri ui" (2 * 3 = 6), tags="rust,gui" (2 * 2 = 4), content="desktop window" (2 * 1 = 2) -> dl_1 = 12
        // Doc 2: name="vector index" (2 * 3 = 6), tags="simd" (1 * 2 = 2), content="rust engine stark" (3 * 1 = 3) -> dl_2 = 11
        bm25.index_doc(0, "Stark Server", "rust", "fast engine");
        bm25.index_doc(1, "Tauri UI", "rust,gui", "desktop window");
        bm25.index_doc(2, "Vector Index", "simd", "rust engine stark");

        assert_eq!(bm25.n_docs, 3);
        assert_eq!(bm25.doc_len[0], 10);
        assert_eq!(bm25.doc_len[1], 12);
        assert_eq!(bm25.doc_len[2], 11);

        let avgdl = (10.0 + 12.0 + 11.0) / 3.0; // 11.0
        assert!((bm25.avgdl() - avgdl).abs() < 1e-6);

        // Query: "stark engine"
        // Term "stark": appears in Doc 0 (name -> tf=3) and Doc 2 (content -> tf=1), df = 2
        // Term "engine": appears in Doc 0 (content -> tf=1) and Doc 2 (content -> tf=1), df = 2
        let idf_stark = (1.0f32 + (3.0 - 2.0 + 0.5) / (2.0 + 0.5)).ln();
        let idf_engine = idf_stark;

        let norm0 = BM25_K1 * (1.0 - BM25_B + BM25_B * (10.0 / avgdl));
        let norm2 = BM25_K1 * (1.0 - BM25_B + BM25_B * (11.0 / avgdl));

        let expected_doc0 = idf_stark * (3.0 * (BM25_K1 + 1.0)) / (3.0 + norm0)
            + idf_engine * (1.0 * (BM25_K1 + 1.0)) / (1.0 + norm0);
        let expected_doc2 = idf_stark * (1.0 * (BM25_K1 + 1.0)) / (1.0 + norm2)
            + idf_engine * (1.0 * (BM25_K1 + 1.0)) / (1.0 + norm2);

        let hits = bm25.search("Stark Engine", 10);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].1, 0);
        assert_eq!(hits[1].1, 2);
        assert!(
            (hits[0].0 - expected_doc0).abs() < 1e-5,
            "Doc 0 score {} != expected {}",
            hits[0].0,
            expected_doc0
        );
        assert!(
            (hits[1].0 - expected_doc2).abs() < 1e-5,
            "Doc 2 score {} != expected {}",
            hits[1].0,
            expected_doc2
        );

        let expected_ub = (idf_stark + idf_engine) * (BM25_K1 + 1.0);
        assert!((bm25.upper_bound("Stark Engine") - expected_ub).abs() < 1e-5);
    }
}
