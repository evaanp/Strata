// Host-only validation fixtures. CUDA is linked but must never be initialized:
// invalid restores must return before the first CUDA call or destination write.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_disk.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <cstring>

#if defined(CONVERSATION_TEST_TRANSFERS)
// GNU/ELF link wrapping exercises the actual restore control flow without
// initializing CUDA or deliberately poisoning a real device context. This is
// host-backend fault injection, not a claim of hardware-failure recovery.
namespace {
int copy_calls = 0, sync_calls = 0, fail_copy = 0, fail_sync = 0;
size_t copied_bytes = 0;
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (++copy_calls == fail_copy) return cudaErrorInvalidValue;
    copied_bytes += n;
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    return ++sync_calls == fail_sync ? cudaErrorUnknown : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }
// The disk park's bounce buffer and staging come from pinned host memory; here they are plain malloc.
extern "C" cudaError_t __wrap_cudaHostAlloc(void** p, size_t n, unsigned) {
    *p = std::malloc(n);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
extern "C" cudaError_t __wrap_cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
struct Pools {
    QsaState st;
    std::array<std::vector<uint8_t>, 8> data;
    Pools(const ModelGeometry& g, int format) {
        st.max_cells = 96; st.n_pages = st.n_slots = 24; st.idx_pooled_rows = 26;
        st.kv_int8 = format == 1; st.kv_q4 = format == 2;
        const size_t per = format == 2 ? strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                      : g.head_dim * (format == 1 ? 1 : 2);
        data[0].resize(96 * g.n_head_kv * per, 0xa5); data[1] = data[0];
        data[2].resize(format == 1 ? 96 * g.n_head_kv * (g.head_dim / 64) * 2 : 0, 0xa5); data[3] = data[2];
        data[4].resize(26 * g.idx_key_dim * 4, 0xa5);
        data[5].resize(3 * g.idx_key_dim * 4, 0xa5);
        data[6].resize(g.idx_key_dim * 4, 0xa5); data[7].resize(4, 0xa5);
        if (format == 2) { st.k_q4 = data[0].data(); st.v_q4 = data[1].data(); }
        else if (format == 1) {
            st.k_q = (int8_t*) data[0].data(); st.v_q = (int8_t*) data[1].data();
            st.k_scale = (uint16_t*) data[2].data(); st.v_scale = (uint16_t*) data[3].data();
        } else { st.k_pool = (uint16_t*) data[0].data(); st.v_pool = (uint16_t*) data[1].data(); }
        st.idx_pooled = (float*) data[4].data(); st.idx_tail = (float*) data[5].data();
        st.idx_dead = (float*) data[6].data(); st.idx_block_pos = (int32_t*) data[7].data();
    }
    ConversationKv image(const ModelGeometry& g, bool index) const {
        ConversationKv k;
        k.format = qsa_kv_format(st); k.cells = 12; k.page_size = 4;
        k.heads = g.n_head_kv; k.head_dim = g.head_dim; k.idx_dim = g.idx_key_dim;
        k.pooled_rows = index ? 3 : 0;
        k.k.resize(data[0].size() / 8, 13); k.v = k.k;
        k.k_scale.resize(data[2].size() / 8, 13); k.v_scale = k.k_scale;
        k.pooled.resize(k.pooled_rows * g.idx_key_dim * 4, 13);
        return k;
    }
};

void fixture(int format, int experts, bool zero_qsa, bool ple) {
    ModelGeometry g;
    g.n_layers = zero_qsa ? 3 : 8; g.n_expert = experts;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::string error;
    ConversationStateSizes z;
    check(conversation_state_sizes(g, z, error), "checked canonical/pruned/zero-QSA sizing");
    Pools first(g, format), last(g, format), draft(g, format);
    std::array<QsaState, 2> layers{first.st, last.st};
    std::vector<uint8_t> gdn(z.gdn, 0xa5), history(ple ? z.ple : 0, 0xa5);
    SessionState ss;
    ss.max_cells = 96; ss.gdn_state = (float*) gdn.data();
    // the whole-model carve, as session_init leaves a one-GPU session
    ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
    ss.ple_hist = ple ? (float*) history.data() : nullptr;
    ss.qsa_states = zero_qsa ? nullptr : layers.data();
    SavedConversation image;
    image.geometry = {g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                      g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                      g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    image.layer_hi = g.n_layers;
    auto checkpoint = [&](size_t tokens) {
        ConversationCheckpoint c;
        c.ids.resize(tokens);
        for (size_t i = 0; i < tokens; ++i) c.ids[i] = (int32_t) i + 1;
        c.imgs = {{1, 44}};
        c.gdn.resize(z.gdn, 13); c.ple.resize(ple ? z.ple : 0, 13);
        c.tails.resize(g.n_qsa_layers() * z.tail, 13); c.dead.resize(g.n_qsa_layers() * z.dead, 13);
        c.block_pos.resize(g.n_qsa_layers() * z.block_pos, 13);
        return c;
    };
    image.live = checkpoint(9); image.checkpoints.push_back(checkpoint(5));
    image.kv.reserve((size_t) g.n_qsa_layers() + 1);
    if (!zero_qsa) { image.kv.push_back(first.image(g, true)); image.kv.push_back(last.image(g, true)); }
    image.kv.push_back(draft.image(g, false));
    check(conversation_snapshot_validate(image, ss, g, draft.st, error), "complete image validates without CUDA");
    size_t estimate = 0;
    check(conversation_snapshot_bytes({image.live.ids,image.live.imgs,image.checkpoints,true},ss,g,draft.st,estimate,error),
          "capture estimate works without CUDA");
    check(estimate == image.bytes(), "estimate covers checkpoint/indexer/spare payloads");
    auto unchanged = [&] {
        auto pristine = [](const auto& bytes) { return std::all_of(bytes.begin(),bytes.end(),[](uint8_t b){return b==0xa5;}); };
        if (!pristine(gdn) || !pristine(history) || ss.ple_prev[0] != -1 || ss.ple_prev[1] != -1) return false;
        for (const auto* p : {&first,&last,&draft}) for (const auto& bytes : p->data) if (!pristine(bytes)) return false;
        return true;
    };
    auto reject = [&](const std::function<void(SavedConversation&)>& mutate, const char* label) {
        auto bad = image; mutate(bad);
        check(conversation_snapshot_restore(bad,ss,g,draft.st,error) == ConversationRestore::invalid,label);
        check(unchanged(), "invalid restore did not touch any layer or running-state buffer");
    };
    reject([](auto& s){s.kv.back().k.pop_back();}, "late draft corruption rejected before first layer write");
    reject([](auto& s){s.geometry[16] = 128;}, "different expert geometry rejected");
    reject([](auto& s){s.live.stage_parts.emplace_back();}, "layer-split live state rejected before writes");
    reject([](auto& s){s.checkpoints[0].stage_parts.emplace_back();}, "layer-split checkpoint rejected before writes");
    reject([](auto& s){s.kv.pop_back();}, "missing KV layer rejected");
    reject([](auto& s){s.live.gdn.pop_back();}, "bad live recurrence rejected");
    reject([](auto& s){s.checkpoints.back().gdn.pop_back();}, "bad retained checkpoint rejected");
    reject([](auto& s){s.checkpoints.back().ids[0] = 99;}, "foreign checkpoint prefix rejected");
    reject([](auto& s){s.checkpoints.back().imgs[0].hash++;}, "foreign checkpoint image rejected");
    reject([](auto& s){s.live.imgs[0].start = -1;}, "negative image position rejected");
    reject([](auto& s){s.live.ids[0] = -1;}, "negative live token rejected");
    reject([](auto& s){s.live.ple.push_back(0);}, "PLE size/enabled mismatch rejected");
    reject([](auto& s){s.live.tails.push_back(0);}, "tail size/zero-QSA mismatch rejected");
    if (!zero_qsa) {
        reject([](auto& s){s.kv[1].pooled.pop_back();}, "late indexer spare row corruption rejected");
        reject([](auto& s){s.live.dead.pop_back();}, "missing indexer spare key rejected");
        reject([](auto& s){s.live.block_pos.pop_back();}, "missing indexer metadata rejected");
        ss.qsa_states = nullptr;
        check(!conversation_snapshot_validate(image,ss,g,draft.st,error), "missing QSA targets rejected");
        ss.qsa_states = layers.data();
        reject([](auto& s){s.layer_lo = 4;}, "snapshot from another layer range rejected");
        // #216's carve: a split stage owning layers [4, 8) holds GDN rows 3..5 and QSA ordinal 1 only
        SessionState stage = ss;
        stage.layer_lo = 4; stage.gdn_alloc = 3; stage.qsa_ord0 = 1; stage.qsa_alloc = 1;
        ConversationStateSizes zs;
        check(conversation_session_sizes(g,stage,zs,error) && zs.gdn == z.gdn / 6 * 3 && zs.tail == z.tail,
              "carved session sizes cover the owned rows only");
        check(!conversation_checkpoint_validate(image.live,stage,g,error), "whole-model checkpoint rejected by a carve");
        ConversationCheckpoint part = image.live;
        part.gdn.resize(zs.gdn); part.tails.resize(zs.tail); part.dead.resize(zs.dead);
        part.block_pos.resize(zs.block_pos);
        check(conversation_checkpoint_validate(part,stage,g,error), "carve-sized checkpoint validates");
        check(!conversation_snapshot_validate(image,stage,g,draft.st,error), "whole-model snapshot rejected by a carve");
        stage.qsa_alloc = 2;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the last QSA ordinal rejected");
        stage.qsa_alloc = 1; stage.gdn_alloc = 7;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the GDN rows rejected");
    }
    auto bad_geometry = g; bad_geometry.ssm_state_size = std::numeric_limits<int64_t>::max();
    check(!conversation_state_sizes(bad_geometry,z,error), "running-state arithmetic overflow rejected");
    bad_geometry = g; bad_geometry.qsa_interval = 0;
    check(!conversation_state_sizes(bad_geometry,z,error), "zero layer interval rejected before division");
    bad_geometry = g; bad_geometry.n_head_kv = std::numeric_limits<int64_t>::max();
    check(conversation_kv_bytes(draft.st,bad_geometry,9,false)==0, "KV byte overflow rejected");
#if defined(CONVERSATION_TEST_TRANSFERS)
    auto reset = [&] {
        for (auto* p : {&first,&last,&draft}) for (auto& bytes : p->data) std::fill(bytes.begin(),bytes.end(),0xa5);
        std::fill(gdn.begin(),gdn.end(),0xa5); std::fill(history.begin(),history.end(),0xa5);
        ss.ple_prev[0] = ss.ple_prev[1] = -1;
        copy_calls = sync_calls = fail_copy = fail_sync = 0;
    };
    reset(); fail_sync = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "pre-transfer CUDA synchronization failure is fatal");
    check(copy_calls==0 && unchanged(), "pre-sync failure did not mutate state");
    reset(); fail_copy = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "first CUDA transfer failure is fatal");
    check(unchanged(), "first-copy failure did not mutate state");
    reset(); fail_copy = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "partial CUDA transfer failure is fatal, not an invalid-image fallback");
    check(!unchanged(), "fault fixture genuinely produced partial state");
    reset(); fail_sync = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "post-transfer CUDA synchronization failure is fatal");
    check(copy_calls>0 && !unchanged(), "post-sync failure occurred after state writes");
    reset();
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::restored,
          "injected host transfer backend completes a valid restore");
    check(gdn==image.live.gdn && history==image.live.ple && ss.ple_prev[0]==8 && ss.ple_prev[1]==9,
          "successful restore publishes correct running state and PLE window");
    uint64_t fingerprint = 0;
    check(conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back verifies complete draft payload");
    draft.data[0][0] ^= 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back detects corrupted draft byte");
    draft.data[0][0] ^= 1;
    fail_copy = copy_calls + 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back transfer failure is not a successful fingerprint");
    reset(); copied_bytes = 0;
    const ConversationView view{image.live.ids,image.live.imgs,image.checkpoints,true};
    SavedConversation full,incremental;
    check(conversation_snapshot_save(full,view,ss,g,draft.st,error), "full capture through host transfer backend");
    const size_t full_copies = copied_bytes;
    ConversationKvReuse reuse{full.kv,9,9};
    size_t peak = 0, reused = 0;
    check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "incremental capture admission without transfers");
    copied_bytes = 0;
    const bool saved_incrementally = conversation_snapshot_save(incremental,view,ss,g,draft.st,error,std::move(reuse),&reused);
    if (!saved_incrementally) std::fprintf(stderr,"capture error: %s\n",error.c_str());
    check(saved_incrementally, "incremental capture through host backend");
    check(reused > 0 && copied_bytes + reused == full_copies, "reused byte count measures transfers actually omitted");
    check(incremental.bytes() <= peak && incremental.live.gdn == full.live.gdn, "running state refreshed within admitted allocation");
    for (size_t i=0;i<full.kv.size();++i)
        check(incremental.kv[i].k==full.kv[i].k && incremental.kv[i].v==full.kv[i].v &&
              incremental.kv[i].k_scale==full.kv[i].k_scale && incremental.kv[i].v_scale==full.kv[i].v_scale &&
              incremental.kv[i].pooled==full.kv[i].pooled, "incremental and full capture agree across every payload");
    reuse = {full.kv,9,9};
    reuse.kv.back().k.pop_back();
    copy_calls = sync_calls = 0;
    check(!conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "malformed retained draft rejected before admission");
    check(!copy_calls && !sync_calls, "invalid retained storage performs no CUDA calls");
    reuse = {full.kv,9,9};
    fail_copy = copy_calls + 1;
    SavedConversation unpublished;
    check(!conversation_snapshot_save(unpublished,view,ss,g,draft.st,error,std::move(reuse)), "incremental capture copy failure is reported");
    check(unpublished.kv.empty() && unpublished.live.ids.empty(), "failed incremental capture cannot publish a partial image");
#endif
}

// A two-stage split: stage 0 owns layers [0,4) (GDN rows 0..2, QSA ordinal 0), stage 1 owns [4,8)
// (rows 3..5, ordinal 1). Parking composes one image from per-stage parts: `live` holds stage 0's running
// state and `stage_images` one SavedConversation per later stage; the K/V is copied by the flat entry list
// (stage 0's owned layers, then stage 1's, then the drafter on the last stage). `dev` stays -1 outside the
// transfer build so no CUDA is touched; there each stage gets its own card to show the OnDevice scoping.
void split_fixture(int format) {
    ModelGeometry g;
    g.n_layers = 8; g.n_expert = 256;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::string error;
    ConversationStateSizes z;
    check(conversation_state_sizes(g, z, error), "split fixture sizing");
    Pools first(g, format), second(g, format), draft(g, format);
    std::vector<uint8_t> gdn0(z.gdn / 6 * 3, 0xa5), hist0(z.ple, 0xa5);
    std::vector<uint8_t> gdn1(z.gdn / 6 * 3, 0xa5), hist1(z.ple, 0xa5);
    SessionState stage0, stage1;
    stage0.max_cells = stage1.max_cells = 96;
    // The QSA table is whole-model: each stage owns a slice of it by ordinal, not its own array.
    std::array<QsaState, 2> qsa{first.st, second.st};
    stage0.gdn_state = (float*) gdn0.data(); stage0.ple_hist = (float*) hist0.data(); stage0.qsa_states = qsa.data();
    stage1.gdn_state = (float*) gdn1.data(); stage1.ple_hist = (float*) hist1.data(); stage1.qsa_states = qsa.data();
    stage0.layer_hi = 4; stage0.gdn_alloc = 3; stage0.qsa_alloc = 1;                            // QSA ordinal 0
    stage1.layer_lo = 4; stage1.layer_hi = 8; stage1.gdn_alloc = 3; stage1.qsa_ord0 = 1; stage1.qsa_alloc = 1;   // ordinal 1
    ConversationStageSet set{{{&stage0, -1}, {&stage1, -1}}};
    set.draft = &draft.st;
    const auto entries = conversation_kv_entries(set);
    check(entries.size() == 3 && entries[0].state == &qsa[0] && entries[1].state == &qsa[1] &&
          entries[2].state == &draft.st, "flat K/V order: stage 0's layers, then stage 1's, then the drafter");
    check(entries[0].index && entries[1].index && !entries[2].index,
          "the drafter is the entry without an indexer; both stages' layers have one");
    std::vector<int32_t> ids9;
    for (int32_t i = 1; i <= 9; ++i) ids9.push_back(i);
    ConversationCheckpoint cp5;   // a retained checkpoint at token 5, as the cache would hold it
    cp5.ids.assign(ids9.begin(), ids9.begin() + 5);
    cp5.imgs = {{1, 44}};
    ConversationCheckpoint part5;   // stage 1's part of that checkpoint: the merged chain shape serve holds
    part5.ids = cp5.ids; part5.imgs = cp5.imgs;
    part5.layer_lo = 4; part5.layer_hi = 8;
    ConversationStateSizes z1;
    check(conversation_session_sizes(g, stage1, z1, error), "stage 1 carve sizing");
    part5.gdn.assign(z1.gdn, 0x2a); part5.ple.assign(z1.ple, 0x2b);
    part5.tails.assign(z1.tail, 0x2c); part5.dead.assign(z1.dead, 0x2d); part5.block_pos.assign(z1.block_pos, 0x2e);
    cp5.stage_parts.push_back(std::move(part5));
    const ConversationView view{ids9, {{1, 44}}, {cp5}, true};
    SavedConversation short_image; short_image.stage_images.emplace_back();   // one part for two stages
    check(!conversation_live_restore(short_image, set, g, error), "a park missing a stage's live state is refused");
    ConversationStageSet no_draft = set; no_draft.draft = nullptr;
    SavedConversation none;
    check(!conversation_live_save(none, view, no_draft, g, error), "a park without a drafter is refused");
#if defined(CONVERSATION_TEST_TRANSFERS)
    // Real device numbers here: a -1 device cannot show which card a copy was made on, and the point is that
    // each stage is drained on its own card before anything there is read or written.
    set.stages = {{&stage0, 0}, {&stage1, 1}};
    auto on_stage1 = [&](const void* p) {
        auto inside = [&](const std::vector<uint8_t>& v) {
            return p >= (const void*) v.data() && p < (const void*) (v.data() + v.size());
        };
        if (inside(gdn1) || inside(hist1)) return true;
        for (const auto& bytes : second.data) if (inside(bytes)) return true;
        return false;
    };
    reset_counters();
    SavedConversation image;
    check(conversation_live_save(image, view, set, g, error), "two-stage live capture through the host backend");
    check(image.live.ids == ids9 && image.stage_images.size() == 1 && image.stage_images[0].live.ids == ids9,
          "each stage's live part carries the same conversation");
    check(image.layer_lo == 0 && image.layer_hi == 4, "the image range is the first stage's carve");
    bool stage1_read = false, read_after_drain = true;
    for (const CopyEvent& e : copy_order) {
        if (e.drains < 1) read_after_drain = false;
        if (!on_stage1(e.src)) continue;
        stage1_read = true;
        if (e.device != 1) read_after_drain = false;   // the copy was made on another card's scope
    }
    check(read_after_drain && stage1_read,
          "stage 1's state was read on its own device after that device was drained");
    reset_counters();
    for (auto* p : {&first,&second,&draft}) for (auto& bytes : p->data) std::fill(bytes.begin(),bytes.end(),0x5a);
    std::fill(gdn0.begin(),gdn0.end(),0x5a); std::fill(hist0.begin(),hist0.end(),0x5a);
    std::fill(gdn1.begin(),gdn1.end(),0x5a); std::fill(hist1.begin(),hist1.end(),0x5a);
    stage0.ple_prev[0] = stage0.ple_prev[1] = stage1.ple_prev[0] = stage1.ple_prev[1] = -1;
    check(conversation_live_restore(image, set, g, error), "both stages' live state restores through the host backend");
    check(gdn0 == image.live.gdn && hist0 == image.live.ple, "stage 0's running state landed in its own buffers");
    check(gdn1 == image.stage_images[0].live.gdn && hist1 == image.stage_images[0].live.ple,
          "stage 1's running state landed in its own buffers");
    check(first.data[5] == image.live.tails && first.data[6] == image.live.dead &&
          first.data[7] == image.live.block_pos, "stage 0's indexer payload landed");
    check(second.data[5] == image.stage_images[0].live.tails &&
          second.data[6] == image.stage_images[0].live.dead &&
          second.data[7] == image.stage_images[0].live.block_pos, "stage 1's indexer payload landed");
    check(stage0.ple_prev[0] == 8 && stage0.ple_prev[1] == 9 &&
          stage1.ple_prev[0] == 8 && stage1.ple_prev[1] == 9, "both stages got the PLE window");
    bool stage1_written = false, write_after_drain = true;
    for (const CopyEvent& e : copy_order) {
        if (e.drains < 1) write_after_drain = false;
        if (!on_stage1(e.dst)) continue;
        stage1_written = true;
        if (e.device != 1) write_after_drain = false;
    }
    check(write_after_drain && stage1_written,
          "stage 1's buffers were written on its own device after that device was drained");
    reset_counters();
    auto bad = image; bad.stage_images.pop_back();
    check(!conversation_live_restore(bad, set, g, error) && copy_calls == 0,
          "a park missing a stage part is refused before any transfer");
    // ---- disk parking through the same host backend (pla-0139 regression: park wrote only stage 0's live
    // state into live.bin, so under a split every load failed with "the running state does not match the
    // stage set" and every returning conversation re-prefilled). Seed every buffer with distinct bytes, park
    // to a scratch directory, load back, compare, then restore into pools scrubbed to another value: a stage
    // whose K/V or running state was skipped stays wrong.
    auto fill = [](std::vector<uint8_t>& v, size_t tag) {
        for (size_t i = 0; i < v.size(); ++i) v[i] = (uint8_t) (i * 29 + tag);
    };
    fill(gdn0, 1); fill(hist0, 3); fill(gdn1, 5); fill(hist1, 7);
    for (size_t b = 0; b < first.data.size(); ++b) {
        fill(first.data[b], 11 + b); fill(second.data[b], 31 + b); fill(draft.data[b], 51 + b);
    }
    const auto want_first = first.data, want_second = second.data, want_draft = draft.data;
    SavedConversation parked_ref;
    check(conversation_live_save(parked_ref, view, set, g, error), "live reference for the disk park");
    namespace fs = std::filesystem;
    const std::string dir = (fs::temp_directory_path() / ("strata-disk-test-" + std::to_string(format))).string();
    fs::remove_all(dir);
    ConversationDisk disk;
    error.clear();
    check(disk.open(dir, 1u << 30, error), "disk parking opens its scratch directory");
    ConversationDisk::Stats stats;
    check(disk.park("c1", view, set, g, 0, stats, error), "the two-stage conversation parks to disk");
    check(stats.written > 0, "the park wrote K/V, checkpoints and running state");
    for (auto* p : {&first,&second,&draft}) for (auto& bytes : p->data) std::fill(bytes.begin(),bytes.end(),0x5a);
    std::fill(gdn0.begin(),gdn0.end(),0x5a); std::fill(hist0.begin(),hist0.end(),0x5a);
    std::fill(gdn1.begin(),gdn1.end(),0x5a); std::fill(hist1.begin(),hist1.end(),0x5a);
    SavedConversation loaded;
    const bool load_ok = disk.load("c1", set, g, loaded, error);
    if (!load_ok) std::fprintf(stderr, "disk load of the split park failed: %s\n", error.c_str());
    check(load_ok, "the two-stage park loads back from disk");
    check(loaded.live.ids == ids9 && loaded.live.imgs == view.images, "the loaded live end matches the parked conversation");
    if (loaded.stage_images.size() != 1) {
        check(false, "stage 1's live state comes back from disk too");
    } else {
        check(loaded.stage_images[0].live.ids == ids9 && loaded.stage_images[0].live.imgs == view.images,
              "stage 1's live part carries the same conversation");
        check(loaded.live.gdn == parked_ref.live.gdn && loaded.live.ple == parked_ref.live.ple &&
              loaded.live.tails == parked_ref.live.tails && loaded.live.dead == parked_ref.live.dead &&
              loaded.live.block_pos == parked_ref.live.block_pos, "stage 0's running-state payload round-trips");
        check(loaded.stage_images[0].live.gdn == parked_ref.stage_images[0].live.gdn &&
              loaded.stage_images[0].live.ple == parked_ref.stage_images[0].live.ple &&
              loaded.stage_images[0].live.tails == parked_ref.stage_images[0].live.tails &&
              loaded.stage_images[0].live.dead == parked_ref.stage_images[0].live.dead &&
              loaded.stage_images[0].live.block_pos == parked_ref.stage_images[0].live.block_pos,
              "stage 1's running-state payload round-trips");
    }
    check(loaded.checkpoints.size() == 1 && loaded.checkpoints[0].ids == cp5.ids &&
          loaded.checkpoints[0].imgs == cp5.imgs && loaded.checkpoints[0].stage_parts.size() == 1 &&
          loaded.checkpoints[0].stage_parts[0].gdn == cp5.stage_parts[0].gdn &&
          loaded.checkpoints[0].stage_parts[0].tails == cp5.stage_parts[0].tails,
          "the mid-prompt checkpoint keeps its stage part through the disk round-trip");
    uint64_t moved = 0;
    const auto restored = disk.restore("c1", loaded, set, g, moved, error);
    if (restored != ConversationRestore::restored) std::fprintf(stderr, "disk restore failed: %s\n", error.c_str());
    check(restored == ConversationRestore::restored, "the split park restores into both stages");
    size_t extent_bytes = 0;
    bool all_back = true;
    for (size_t k = 0; k < entries.size() && all_back; ++k) {
        ConversationKvExtent ext;
        if (!conversation_kv_extent(*entries[k].state, g, (int64_t) ids9.size(), entries[k].index, ext, error)) {
            all_back = false;
            break;
        }
        const auto buffers = conversation_kv_buffers(*entries[k].state);
        const auto& want = k == 0 ? want_first : (k == 1 ? want_second : want_draft);
        for (size_t b = 0; b < ext.sizes.size(); ++b) {
            if (ext.sizes[b] > want[b].size()) { all_back = false; break; }
            const auto* got = (const uint8_t*) buffers[b];
            all_back = all_back && std::equal(want[b].begin(), want[b].begin() + ext.sizes[b], got);
            extent_bytes += ext.sizes[b];
        }
    }
    check(all_back, "every stage's K/V came back byte for byte over its whole extent");
    check(moved == extent_bytes, "the restore moved every entry's full extent");
    ConversationDisk::Stats again;
    check(disk.park("c1", view, set, g, (int64_t) ids9.size(), again, error),
          "re-parking the restored conversation works");
    check(again.kept > 0 && again.written < stats.written, "the re-park keeps the K/V and checkpoint it need not rewrite");
    {
        fs::path entry;   // the hashed directory name is ConversationDisk's own: it is the only child
        for (const auto& child : fs::directory_iterator(dir)) entry = child.path();
        const fs::path live = entry / "live.bin";
        std::ifstream in(live, std::ios::binary);
        const std::vector<uint8_t> raw{(std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()};
        in.close();
        check(raw.size() > 8, "the running-state file carries its format word");
        std::ofstream out(live, std::ios::binary | std::ios::trunc);   // strip the word: what an older binary wrote
        out.write((const char*) raw.data() + 8, (std::streamsize) (raw.size() - 8));
        out.close();
        SavedConversation half;
        check(!disk.load("c1", set, g, half, error), "an entry from an older binary is refused");
        check(error.find("older strata") != std::string::npos, "the refusal names the running-state format");
        check(half.live.ids.empty() && half.stage_images.empty(), "a refused entry is not half-loaded");
        disk.drop("c1");
        SavedConversation gone;
        check(!disk.load("c1", set, g, gone, error) && error.find("no parked") != std::string::npos,
              "the refused entry is removed");
    }
    fs::remove_all(dir);
#endif
}
}

int main() {
    for (int format : {0,1,2}) for (int experts : {256,512})
        for (bool zero_qsa : {false,true}) for (bool ple : {false,true}) fixture(format,experts,zero_qsa,ple);
    std::printf("conversation_validation_test: %d host-only checks passed\n", checks);
}
