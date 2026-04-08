use inference_backend::{PageIndex, RequestId};
use std::collections::HashMap;

pub struct KvCachePool {
    pub num_pages: usize,
    pub page_size: usize,
    free_pages: Vec<PageIndex>,
    page_states: Vec<PageState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageState {
    Free,
    InUse(RequestId),
}

impl KvCachePool {
    pub fn new(num_pages: usize, page_size: usize) -> Self {
        let mut free_pages: Vec<PageIndex> = (0..num_pages as PageIndex).collect();
        free_pages.reverse(); // Use as a stack
        Self {
            num_pages,
            page_size,
            free_pages,
            page_states: vec![PageState::Free; num_pages],
        }
    }

    /// Allocate one page. Returns None if the pool is exhausted.
    pub fn allocate(&mut self, owner: RequestId) -> Option<PageIndex> {
        if let Some(page) = self.free_pages.pop() {
            self.page_states[page as usize] = PageState::InUse(owner);
            Some(page)
        } else {
            None
        }
    }

    /// Free a page and return it to the pool.
    pub fn free(&mut self, page: PageIndex) {
        debug_assert!(
            matches!(self.page_states[page as usize], PageState::InUse(_)),
            "Attempted to free a page that is not in use"
        );
        self.page_states[page as usize] = PageState::Free;
        self.free_pages.push(page);
    }

    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub fn total_pages(&self) -> usize {
        self.num_pages
    }

    /// Compute bytes required for a pool of the given dimensions.
    pub fn required_bytes(
        num_pages: usize,
        page_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype_bytes: usize,
    ) -> usize {
        // 2 = K + V
        2 * num_layers * num_pages * page_size * num_kv_heads * head_dim * dtype_bytes
    }
}

pub struct PageTable {
    inner: HashMap<(RequestId, usize), PageIndex>,
}

impl PageTable {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn insert(&mut self, req_id: RequestId, slot: usize, page: PageIndex) {
        self.inner.insert((req_id, slot), page);
    }

    pub fn lookup(&self, req_id: RequestId, slot: usize) -> Option<PageIndex> {
        self.inner.get(&(req_id, slot)).copied()
    }

    /// Remove all entries for req_id and return the freed page indices.
    pub fn remove_request(&mut self, req_id: RequestId) -> Vec<PageIndex> {
        let keys: Vec<_> = self
            .inner
            .keys()
            .filter(|(r, _)| *r == req_id)
            .cloned()
            .collect();
        keys.into_iter()
            .filter_map(|k| self.inner.remove(&k))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_allocate_and_free() {
        let mut pool = KvCachePool::new(16, 16);
        let id = Uuid::new_v4();
        let p1 = pool.allocate(id).unwrap();
        assert_eq!(pool.free_pages(), 15);
        pool.free(p1);
        assert_eq!(pool.free_pages(), 16);
    }

    #[test]
    fn test_allocate_exhaustion() {
        let mut pool = KvCachePool::new(2, 16);
        let id = Uuid::new_v4();
        pool.allocate(id).unwrap();
        pool.allocate(id).unwrap();
        assert_eq!(pool.allocate(id), None);
        assert_eq!(pool.free_pages(), 0);
    }

    #[test]
    fn test_required_bytes() {
        // Simple case: 1 page, 1 layer, 1 head, dim 1, dtype 1 byte
        // 2 (K+V) * 1 * 1 * 16 * 1 * 1 * 1 = 32
        assert_eq!(KvCachePool::required_bytes(1, 16, 1, 1, 1, 1), 32);
        // Spec example: required_bytes(512, 16, 28, 4, 128, 2)
        assert_eq!(KvCachePool::required_bytes(512, 16, 28, 4, 128, 2), 2 * 512 * 16 * 28 * 4 * 128 * 2);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_double_free_panics() {
        let mut pool = KvCachePool::new(4, 16);
        let id = Uuid::new_v4();
        let p = pool.allocate(id).unwrap();
        pool.free(p);
        // Freeing a Free page should panic in debug builds
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.free(p);
        }));
        assert!(result.is_err(), "Expected panic on double-free");
    }

    #[test]
    fn test_page_table_roundtrip() {
        let mut pt = PageTable::new();
        let id = Uuid::new_v4();
        pt.insert(id, 0, 7);
        assert_eq!(pt.lookup(id, 0), Some(7));
        let freed = pt.remove_request(id);
        assert_eq!(freed, vec![7]);
        assert_eq!(pt.lookup(id, 0), None);
    }
}
