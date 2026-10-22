// src/core/kv_pool.cpp - see include/strata/core/kv_pool.hpp.
#include "strata/core/kv_pool.hpp"

#include "strata/core/layout.hpp"
#include "strata/core/on_device.hpp"

#include <cuda_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::core {

bool KvPool::init(const ModelGeometry& g, const std::vector<StageRef>& stages, int64_t tokens, std::string& error,
                  bool lazy) {
    if (stages.empty() || stages[0].ss == nullptr) { error = "kv pool: no session to size the pool by"; return false; }
    page_size_ = strata::kernels::qsa_real_shapes().page_size;
    const int64_t chunk = chunk_cells();
    const int64_t n = (tokens + chunk - 1) / chunk;
    if (n <= 0) { error = "kv pool: no tokens"; return false; }
    const int64_t pages = (n + 1) << kChunkShift;   // + the trash chunk
    const uint64_t before = qsa_kv_host_bytes();
    stages_.clear();
    lanes_.clear();
    regions_.clear();
    lazy_ = lazy;
    const strata::kernels::QsaShapes qs = qsa_shapes(g);
    // lazy: K and V of a layer in their own address space, so a chunk is page aligned in both
    const uint64_t array_chunk = (uint64_t) (1ll << kChunkShift) * qs.n_head_kv * qs.page_size * qs.head_dim * 2;
    for (const StageRef& sr : stages) {
        const SessionState& ref = *sr.ss;
        StageLayers sl;
        sl.qsa_ord0 = ref.qsa_ord0;
        sl.device = sr.device;
        sl.layers.assign((size_t) ref.qsa_alloc, strata::kernels::KvHostPools{});
        const OnDevice on(sr.device);   // the stage's own context pins its arrays (portable: every GPU reaches them)
        for (int64_t j = 0; j < ref.qsa_alloc; ++j) {
            const QsaState& st = ref.qsa_states[ref.qsa_ord0 + j];
            if (st.kv_mode != 1) continue;   // a layer kept whole in VRAM has no host copy
            if (lazy) {
                int host_ptr = 0;
                if (st.kv_int8 || st.kv_q4 || st.kv_hybrid) { error = "kv pool: lazy pinning is fp16 K/V only"; return false; }
                if (cudaDeviceGetAttribute(&host_ptr, cudaDevAttrCanUseHostPointerForRegisteredMem,
                                           sr.device < 0 ? 0 : sr.device) != cudaSuccess || host_ptr == 0) {
                    error = "kv pool: lazy pinning needs the GPU to use registered host pointers as they are";
                    return false;
                }
                strata::kernels::KvHostPools& hp = sl.layers[(size_t) j];
                for (uint16_t** arr : {&hp.k_pool, &hp.v_pool}) {
                    const uint64_t bytes = array_chunk * (uint64_t) (n + 1);
                    void* m = mmap(nullptr, (size_t) bytes, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                    if (m == MAP_FAILED) { error = "kv pool: cannot map the pool's address space"; return false; }
                    *arr = (uint16_t*) m;
                    regions_.push_back({(uint8_t*) m, array_chunk});
                }
                continue;
            }
            if (!qsa_host_alloc(g, st, pages, sl.layers[(size_t) j])) {
                error = "kv pool: cannot pin the pool's K/V";
                return false;
            }
        }
        stages_.push_back(std::move(sl));
    }
    map_.init(n);
    if (!lazy) {
        pinned_bytes_ = qsa_kv_host_bytes() - before;
        chunk_bytes_ = pinned_bytes_ / (uint64_t) (n + 1);
        return true;
    }
    for (const Region& r : regions_) chunk_bytes_ += r.chunk;
    pinned_.assign((size_t) n + 1, 0);
    if (!pin_chunk(map_.trash())) { error = "kv pool: cannot pin the trash chunk"; return false; }
    return true;
}

bool KvPool::pin_chunk(int32_t c) {
    if (!lazy_ || pinned_[(size_t) c]) return true;
    for (size_t i = 0; i < regions_.size(); ++i) {
        uint8_t* at = regions_[i].base + (uint64_t) c * regions_[i].chunk;
        if (cudaHostRegister(at, regions_[i].chunk, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
            (void) cudaGetLastError();
            for (size_t k = 0; k < i; ++k) {
                uint8_t* back = regions_[k].base + (uint64_t) c * regions_[k].chunk;
                cudaHostUnregister(back);
                madvise(back, regions_[k].chunk, MADV_DONTNEED);
            }
            madvise(at, regions_[i].chunk, MADV_DONTNEED);
            return false;
        }
    }
    pinned_[(size_t) c] = 1;
    pinned_bytes_ += chunk_bytes_;
    return true;
}

void KvPool::unpin_chunk(int32_t c) {
    if (!lazy_ || !pinned_[(size_t) c] || c == map_.trash()) return;
    for (const Region& r : regions_) {
        uint8_t* at = r.base + (uint64_t) c * r.chunk;
        cudaHostUnregister(at);
        madvise(at, r.chunk, MADV_DONTNEED);   // the pages back to the kernel; a later pin maps zeroed ones
    }
    pinned_[(size_t) c] = 0;
    pinned_bytes_ -= chunk_bytes_;
}

uint64_t KvPool::pin_bytes_for(const SessionState& ss, int64_t cells) const {
    const Lane* lane = find(ss);
    if (!lazy_ || lane == nullptr) return 0;
    const int64_t chunk = chunk_cells();
    const int64_t need = std::min<int64_t>((std::max<int64_t>(cells, 0) + chunk - 1) / chunk, map_.logical(lane->id));
    const int64_t more = need - map_.held(lane->id);
    return more > 0 ? (uint64_t) more * chunk_bytes_ : 0;
}

KvPool::Lane* KvPool::find(const SessionState& ss) {
    for (auto& l : lanes_)
        for (const Part& p : l->parts)
            if (p.ss == &ss) return l.get();
    return nullptr;
}
const KvPool::Lane* KvPool::find(const SessionState& ss) const {
    for (const auto& l : lanes_)
        for (const Part& p : l->parts)
            if (p.ss == &ss) return l.get();
    return nullptr;
}

bool KvPool::upload(const Lane& lane) {
    // From pageable memory the copy may still be landing when cudaMemcpy returns, and the kernels run on
    // non-blocking streams that do not wait for it: the table is complete before the next launch reads it.
    // An upload is rare: once per chunk (4096 cells) a session grows by, and when it gives chunks back.
    const int32_t* table = map_.table(lane.id);
    const size_t bytes = (size_t) map_.logical(lane.id) * sizeof(int32_t);
    for (const Part& p : lane.parts) {
        const OnDevice on(p.device);
        if (cudaMemcpy(p.dev, table, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess)
            return false;
    }
    return true;
}

void KvPool::uploaded_or_die(const Lane& lane) {
    if (upload(lane)) return;
    std::fprintf(stderr, "strata: kv pool: chunk table upload failed\n");
    std::exit(1);   // the kernels would write through a table the movers disagree with
}

bool KvPool::bind(Lane& lane, SessionState& ss, size_t stage, std::string& error) {
    if (stage >= stages_.size()) { error = "kv pool: no such stage"; return false; }
    const StageLayers& sl = stages_[stage];
    if (ss.qsa_ord0 != sl.qsa_ord0 || ss.qsa_alloc != (int64_t) sl.layers.size()) {
        error = "kv pool: the session's QSA layers differ from the pool's";
        return false;
    }
    if (lane.parts.size() != stage) { error = "kv pool: a lane's stages attach in order"; return false; }
    Part part;
    part.ss = &ss;
    part.device = sl.device;
    {
        const OnDevice on(sl.device);
        const size_t bytes = (size_t) map_.logical(lane.id) * sizeof(int32_t);
        if (cudaMalloc((void**) &part.dev, bytes) != cudaSuccess) {
            error = "kv pool: cannot allocate a chunk table";
            return false;
        }
    }
    for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
        QsaState& st = ss.qsa_states[ss.qsa_ord0 + j];
        if (st.kv_mode != 1) continue;
        if (!sl.layers[(size_t) j].present()) { error = "kv pool: a streamed layer the pool has no K/V for"; return false; }
        st.host = sl.layers[(size_t) j];
        st.host.chunk = part.dev;
        st.host.chunk_host = map_.table(lane.id);
        st.host.chunk_shift = kChunkShift;
    }
    lane.parts.push_back(part);
    if (!upload(lane)) {
        lane.parts.pop_back();
        error = "kv pool: cannot allocate a chunk table";
        return false;
    }
    return true;
}

bool KvPool::attach(SessionState& ss, std::string& error) {
    if (!active() || find(ss) != nullptr) { error = "kv pool: not initialized, or the session is attached already"; return false; }
    auto lane = std::make_unique<Lane>();
    lane->id = map_.add_lane((ss.max_cells + chunk_cells() - 1) / chunk_cells());
    if (!bind(*lane, ss, 0, error)) return false;
    lanes_.push_back(std::move(lane));
    return true;
}

bool KvPool::attach_stage(const SessionState& head, SessionState& ss, size_t stage, std::string& error) {
    Lane* lane = find(head);
    if (lane == nullptr || find(ss) != nullptr) { error = "kv pool: no such lane, or the session is attached already"; return false; }
    return bind(*lane, ss, stage, error);
}

void KvPool::detach(const SessionState& ss) {
    shrink(ss, 0);
    for (auto it = lanes_.begin(); it != lanes_.end(); ++it)
        for (const Part& p : (*it)->parts)
            if (p.ss == &ss) {
                for (const Part& q : (*it)->parts) {   // the lane's tables are not needed any more
                    const OnDevice on(q.device);
                    cudaFree(q.dev);
                }
                lanes_.erase(it);
                return;
            }
}

bool KvPool::reserve(const SessionState& ss, int64_t cells) {
    Lane* lane = find(ss);
    if (lane == nullptr) return false;
    const int64_t chunk = chunk_cells();
    const int64_t held = map_.held(lane->id);
    bool changed = false;
    if (!map_.reserve(lane->id, (std::max<int64_t>(cells, 0) + chunk - 1) / chunk, &changed)) return false;
    if (!changed) return true;
    const int32_t* table = map_.table(lane->id);
    for (int64_t c = held; c < map_.held(lane->id); ++c) {
        if (pin_chunk(table[c])) continue;
        for (int64_t u = held; u < c; ++u) unpin_chunk(table[u]);
        map_.shrink(lane->id, held);   // nothing changed: the chunks go back as they came
        return false;
    }
    uploaded_or_die(*lane);
    return true;
}

void KvPool::shrink(const SessionState& ss, int64_t cells) {
    Lane* lane = find(ss);
    if (lane == nullptr) return;
    const int64_t chunk = chunk_cells();
    const int64_t keep = (std::max<int64_t>(cells, 0) + chunk - 1) / chunk;
    const int64_t held = map_.held(lane->id);
    std::vector<int32_t> freed;
    for (int64_t c = keep; c < held; ++c) freed.push_back(map_.table(lane->id)[c]);
    if (!map_.shrink(lane->id, keep)) return;
    uploaded_or_die(*lane);   // the tables name the trash chunk before the pages go
    for (int32_t c : freed) unpin_chunk(c);
}

bool KvPool::swap(const SessionState& a, const SessionState& b) {
    Lane* la = find(a);
    Lane* lb = find(b);
    if (la == nullptr || lb == nullptr || la == lb || la->parts.size() != lb->parts.size() ||
        !map_.swap(la->id, lb->id))
        return false;
    uploaded_or_die(*la);
    uploaded_or_die(*lb);
    return true;
}

int64_t KvPool::reserved_cells(const SessionState& ss) const {
    const Lane* lane = find(ss);
    return lane != nullptr ? map_.held(lane->id) * chunk_cells() : 0;
}

}  // namespace strata::core
