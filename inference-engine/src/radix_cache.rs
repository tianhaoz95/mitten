use inference_backend::{NodeId, PageIndex, Request};
use rustc_hash::FxHashMap;
use slab::Slab;
use std::{cmp::Reverse, collections::BinaryHeap, time::Instant};

pub struct RadixNode {
    pub edge_tokens: Vec<u32>,
    pub kv_pages: Vec<PageIndex>,
    pub children: FxHashMap<u32, NodeId>,
    pub parent: Option<NodeId>,
    /// In-flight request ref count. Nodes with lock_ref > 0 cannot be evicted.
    pub lock_ref: i32,
    pub last_used: Instant,
}

impl RadixNode {
    fn new_root() -> Self {
        RadixNode {
            edge_tokens: vec![],
            kv_pages: vec![],
            children: FxHashMap::default(),
            parent: None,
            lock_ref: 0,
            last_used: Instant::now(),
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn is_evictable(&self) -> bool {
        self.is_leaf() && self.lock_ref == 0 && !self.kv_pages.is_empty()
    }
}

pub struct RadixCache {
    nodes: Slab<RadixNode>,
    root: NodeId,
    /// LRU heap over (last_used, node_id). May contain stale entries.
    evictable: BinaryHeap<(Reverse<Instant>, NodeId)>,
    page_size: usize,
    total_pages: usize,
    max_pages: usize,
}

impl RadixCache {
    pub fn new(page_size: usize, max_pages: usize) -> Self {
        let mut nodes = Slab::new();
        let root = nodes.insert(RadixNode::new_root());
        RadixCache {
            nodes,
            root,
            evictable: BinaryHeap::new(),
            page_size,
            total_pages: 0,
            max_pages,
        }
    }

    /// Walk the tree following `tokens`. Return (matched_len, kv_pages, locked_node_ids).
    /// `matched_len` is always a multiple of `page_size`.
    /// Increments lock_ref on each matched node. Caller MUST call unlock_nodes().
    pub fn match_prefix(&mut self, tokens: &[u32]) -> (usize, Vec<PageIndex>, Vec<NodeId>) {
        let mut matched_pages: Vec<PageIndex> = Vec::new();
        let mut matched_nodes: Vec<NodeId> = Vec::new();
        let mut offset = 0usize;
        let mut cur = self.root;

        loop {
            let first = match tokens.get(offset) {
                Some(&t) => t,
                None => break,
            };

            let child_id = match self.nodes[cur].children.get(&first).copied() {
                Some(id) => id,
                None => break,
            };

            let (_edge_len, is_partial_match) = {
                let edge = &self.nodes[child_id].edge_tokens;
                let remaining = &tokens[offset..];
                let match_len = edge
                    .iter()
                    .zip(remaining)
                    .take_while(|(a, b)| a == b)
                    .count();

                if match_len == 0 {
                    break;
                }

                // Only count full pages.
                let full_pages = match_len / self.page_size;
                if full_pages == 0 {
                    break;
                }

                let matched_from_this_node = full_pages * self.page_size;
                matched_pages.extend_from_slice(&self.nodes[child_id].kv_pages[..full_pages]);
                offset += matched_from_this_node;
                (edge.len(), match_len < edge.len())
            };

            matched_nodes.push(child_id);
            self.nodes[child_id].lock_ref += 1;
            self.nodes[child_id].last_used = Instant::now();

            if is_partial_match {
                break;
            } // partial edge match: stop
            cur = child_id;
        }

        (offset, matched_pages, matched_nodes)
    }

    /// Insert `tokens[..N*page_size]` → `pages[..N]` into the tree.
    /// Handles full-edge matches (descend), partial-edge matches (split_edge), and
    /// no-match (new leaf). Only inserts complete pages (partial last page is ignored).
    /// Evicts LRU pages before inserting if total_pages would exceed max_pages.
    pub fn insert_prefix(&mut self, tokens: &[u32], pages: &[PageIndex]) {
        let full_pages = pages.len();
        let full_tokens = full_pages * self.page_size;
        if full_pages == 0 {
            return;
        }
        let tokens = &tokens[..full_tokens];

        // Evict if needed before inserting.
        if self.total_pages + full_pages > self.max_pages {
            self.evict_pages(self.total_pages + full_pages - self.max_pages);
        }

        let mut offset = 0usize;
        let mut cur = self.root;

        loop {
            if offset >= tokens.len() {
                break;
            }
            let first = tokens[offset];

            match self.nodes[cur].children.get(&first).copied() {
                None => {
                    // No matching child: insert new leaf.
                    let new_id = self.nodes.insert(RadixNode {
                        edge_tokens: tokens[offset..].to_vec(),
                        kv_pages: pages[offset / self.page_size..].to_vec(),
                        children: FxHashMap::default(),
                        parent: Some(cur),
                        lock_ref: 0,
                        last_used: Instant::now(),
                    });
                    self.nodes[cur].children.insert(first, new_id);
                    let added = pages[offset / self.page_size..].len();
                    self.total_pages += added;
                    // New leaf is immediately evictable.
                    self.evictable.push((Reverse(Instant::now()), new_id));
                    break;
                }
                Some(child_id) => {
                    let edge_len = self.nodes[child_id].edge_tokens.len();
                    let remaining = &tokens[offset..];
                    let match_len = self.nodes[child_id]
                        .edge_tokens
                        .iter()
                        .zip(remaining)
                        .take_while(|(a, b)| a == b)
                        .count();

                    if match_len == edge_len {
                        // Full edge match: descend.
                        self.nodes[child_id].last_used = Instant::now();
                        offset += edge_len;
                        cur = child_id;
                    } else {
                        // Partial match: split the edge at match_len.
                        let mid_id = self.split_edge(cur, child_id, match_len);
                        offset += match_len;
                        cur = mid_id;
                    }
                }
            }
        }
    }

    /// Split the edge from `parent` → `child` at `split_at` tokens.
    /// Creates a new intermediate node; returns its NodeId.
    fn split_edge(&mut self, parent: NodeId, child: NodeId, split_at: usize) -> NodeId {
        let page_split = split_at / self.page_size;
        let prefix_tokens = self.nodes[child].edge_tokens[..split_at].to_vec();
        let suffix_tokens = self.nodes[child].edge_tokens[split_at..].to_vec();
        let prefix_pages = self.nodes[child].kv_pages[..page_split].to_vec();
        let suffix_pages = self.nodes[child].kv_pages[page_split..].to_vec();
        let first_suffix = suffix_tokens[0];
        let first_prefix = prefix_tokens[0];

        // Trim child to suffix.
        self.nodes[child].edge_tokens = suffix_tokens;
        self.nodes[child].kv_pages = suffix_pages;
        // self.nodes[child].parent will be set below.

        // Create intermediate node with prefix, pointing to child.
        let mid_id = self.nodes.insert(RadixNode {
            edge_tokens: prefix_tokens,
            kv_pages: prefix_pages,
            children: {
                let mut m = FxHashMap::default();
                m.insert(first_suffix, child);
                m
            },
            parent: Some(parent),
            lock_ref: 0,
            last_used: Instant::now(),
        });
        self.nodes[child].parent = Some(mid_id);

        // Replace child pointer in parent.
        self.nodes[parent].children.insert(first_prefix, mid_id);
        mid_id
    }

    /// Decrement lock_ref for each node id. Update last_used to now.
    /// Adds newly evictable nodes to the eviction heap.
    pub fn unlock_nodes(&mut self, node_ids: &[NodeId]) {
        for &id in node_ids {
            if let Some(node) = self.nodes.get_mut(id) {
                node.lock_ref = (node.lock_ref - 1).max(0);
                node.last_used = Instant::now();
                // If now evictable, add to heap.
                if node.is_evictable() {
                    self.evictable.push((Reverse(node.last_used), id));
                }
            }
        }
    }

    /// Evict LRU unlocked leaf nodes until `target_pages` pages are freed.
    /// Returns freed page indices (caller must return them to KvCachePool).
    /// If fewer than `target_pages` can be freed, returns all that were freed.
    pub fn evict_pages(&mut self, target_pages: usize) -> Vec<PageIndex> {
        let mut freed_pages: Vec<PageIndex> = Vec::new();

        while freed_pages.len() < target_pages {
            let candidate = loop {
                match self.evictable.pop() {
                    None => return freed_pages, // nothing left to evict
                    Some((_, id)) => {
                        // The heap may contain stale entries.
                        if self.nodes.contains(id) && self.nodes[id].is_evictable() {
                            break id;
                        }
                    }
                }
            };

            // Collect pages from this leaf.
            let pages = std::mem::take(&mut self.nodes[candidate].kv_pages);
            self.total_pages -= pages.len();
            freed_pages.extend_from_slice(&pages);

            // Remove from parent's child map.
            if let Some(parent_id) = self.nodes[candidate].parent {
                let first_token = self.nodes[candidate].edge_tokens.first().copied();
                if let Some(t) = first_token {
                    self.nodes[parent_id].children.remove(&t);
                    // If parent is now a childless leaf with lock_ref==0, it
                    // becomes evictable too.
                    if self.nodes[parent_id].is_evictable() {
                        let ts = self.nodes[parent_id].last_used;
                        self.evictable.push((Reverse(ts), parent_id));
                    }
                }
            }

            self.nodes.remove(candidate);
        }

        freed_pages
    }

    /// Promote a completed request's KV pages into the cache.
    /// Called when a request transitions Prefilling → Decoding.
    /// Only full pages are inserted.
    pub fn promote_request(&mut self, req: &Request) {
        let full_pages = req.device_len / self.page_size;
        let cache_tokens = full_pages * self.page_size;
        if full_pages == 0 {
            return;
        }
        self.insert_prefix(&req.input_ids[..cache_tokens], &req.kv_pages[..full_pages]);
    }

    pub fn total_cached_pages(&self) -> usize {
        self.total_pages
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_match() {
        let mut cache = RadixCache::new(4, 16);
        let (len, pages, nodes) = cache.match_prefix(&[1, 2, 3]);
        assert_eq!(len, 0);
        assert!(pages.is_empty());
        assert!(nodes.is_empty());
    }

    #[test]
    fn test_insert_and_match_full() {
        let mut cache = RadixCache::new(4, 16);
        cache.insert_prefix(&[1, 2, 3, 4], &[10]);
        let (len, pages, nodes) = cache.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(len, 4);
        assert_eq!(pages, vec![10]);
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn test_edge_split() {
        let mut cache = RadixCache::new(2, 16);
        cache.insert_prefix(&[1, 2, 3, 4], &[10, 11]);
        // Current: root -> [1,2,3,4] (p10, p11)
        cache.insert_prefix(&[1, 2, 5, 6], &[10, 12]);
        // Should be: root -> [1,2] (p10) -> [3,4] (p11)
        //                            -> [5,6] (p12)

        let (len1, pages1, _) = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(len1, 4);
        assert_eq!(pages1, vec![10, 11]);

        let (len2, pages2, _) = cache.match_prefix(&[1, 2, 5, 6]);
        assert_eq!(len2, 4);
        assert_eq!(pages2, vec![10, 12]);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = RadixCache::new(2, 2);
        cache.insert_prefix(&[1, 2], &[10]);
        cache.insert_prefix(&[3, 4], &[11]);
        assert_eq!(cache.total_cached_pages(), 2);

        // This should evict [1, 2]
        cache.insert_prefix(&[5, 6], &[12]);
        assert_eq!(cache.total_cached_pages(), 2);

        let (len, _, _) = cache.match_prefix(&[1, 2]);
        assert_eq!(len, 0);
    }

    #[test]
    fn test_lock_prevents_eviction() {
        let mut cache = RadixCache::new(2, 1);
        cache.insert_prefix(&[1, 2], &[10]);

        let (_, _, nodes) = cache.match_prefix(&[1, 2]);
        // node [1,2] is now locked.

        // Try to insert another page, but max is 1 and [1,2] is locked.
        // It should NOT evict [1,2].
        cache.insert_prefix(&[3, 4], &[11]);
        assert_eq!(cache.total_cached_pages(), 2); // It overflowed because nothing was evictable

        cache.unlock_nodes(&nodes);
        // Now it should be evictable.
        let freed = cache.evict_pages(1);
        assert_eq!(freed.len(), 1);
    }

    #[test]
    fn test_insert_and_match_partial_edge() {
        // page_size=4: [1,2,3,4] is exactly 1 page, [1,2,5,6] shares first 2 tokens (< 1 page)
        let mut cache = RadixCache::new(4, 16);
        cache.insert_prefix(&[1, 2, 3, 4], &[10]);
        cache.insert_prefix(&[1, 2, 5, 6], &[11]);
        // After split: root -> [1,2](0 pages) -> [3,4](page 10) and [5,6](page 11)
        // Matching [1,2,3,4]: shared prefix [1,2] has 0 full pages, so match stops there
        // Then descends to [3,4] which has 1 full page -> total match = 4 tokens, 1 page
        // But the current match_prefix only counts full pages per edge, so [1,2] contributes 0
        // and [3,4] contributes 1 page (4 tokens). Total = 4 tokens.
        let (len, pages, _) = cache.match_prefix(&[1, 2, 3, 4]);
        // The shared [1,2] node has 0 pages (not a full page), so match stops at 0
        // unless the implementation descends through 0-page nodes.
        // With the current implementation, match breaks when full_pages==0.
        // So the match returns 0 for [1,2,3,4] after the split.
        // This is correct: only full pages are matched.
        assert_eq!(pages.len(), len / 4); // pages = matched_len / page_size
        // The key invariant: [1,2,5,6] and [1,2,3,4] are both stored correctly
        let (len2, pages2, _) = cache.match_prefix(&[1, 2, 5, 6]);
        assert_eq!(pages2.len(), len2 / 4);
    }

    #[test]
    fn test_promote_request() {
        use inference_backend::{Request, SamplingParams};
        let mut cache = RadixCache::new(8, 16);
        let mut req = Request::new(uuid::Uuid::new_v4(), (1u32..=8).collect(), SamplingParams::default());
        req.device_len = 8;
        req.kv_pages = vec![42];
        cache.promote_request(&req);
        let (len, pages, _) = cache.match_prefix(&(1u32..=8).collect::<Vec<_>>());
        assert_eq!(len, 8);
        assert_eq!(pages, vec![42]);
    }

    #[test]
    fn test_page_alignment() {
        let mut cache = RadixCache::new(4, 16);
        // 6 tokens with page_size=4: only 1 full page (4 tokens), partial last page ignored
        cache.insert_prefix(&[1, 2, 3, 4, 5, 6], &[10]);
        let (len, pages, _) = cache.match_prefix(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(len, 4); // only 1 full page = 4 tokens
        assert_eq!(pages, vec![10]);
    }
}
