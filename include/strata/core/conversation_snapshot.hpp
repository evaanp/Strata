#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

// Caller synchronizes the device before saving, and after restoring all layers.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);
// Diagnostic read-back after a synchronized restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error);
// Disk save without a host copy: the layer's authoritative K/V as a streamed source (same bytes and metadata as
// conversation_kv_save).  Caller synchronizes the device first and keeps the state untouched while it is read.
bool conversation_kv_source(SessionKvSource& source, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, std::string& error);
// The bytes of each K/V part (k, v, k_scale, v_scale, pooled) a snapshot of `upto` tokens holds: no state read.
bool conversation_kv_part_sizes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                                std::array<uint64_t, 5>& sizes, std::string& error);
// Disk sessions: the read limits this session can ever restore - its geometry and layer range, at most
// min(max_tokens, its cells) tokens, `max_checkpoints` checkpoints, the exact running-state sizes and the K/V part
// sizes at that many tokens - so session_file_read refuses an oversized or foreign file before it allocates.
bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& session, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error);

// What a save of one layer up to `upto` produces: its K/V geometry and the byte size of each of the five buffers
// (K, V, K scales, V scales, pooled indexer rows). A disk image is checked against it before anything is written.
struct ConversationKvExtent {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<uint64_t, 5> sizes{};
    bool operator==(const ConversationKvExtent&) const = default;
};
bool conversation_kv_extent(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                            ConversationKvExtent& extent, std::string& error);
// The leading bytes of each buffer a save with `unchanged_tokens` keeps instead of copying (conversation_kv_save's rule).
std::array<uint64_t, 5> conversation_kv_kept(const ConversationKvExtent& extent, const ModelGeometry& g,
                                             int64_t unchanged_tokens, bool include_index);
// The five authoritative buffers in extent order: the host pool for streamed and ring layers, VRAM otherwise.
std::array<void*, 5> conversation_kv_buffers(const QsaState& state);
// After the authoritative buffers were written: refill the VRAM slots or the draft ring from them.
bool conversation_kv_restored(const QsaState& state, const ModelGeometry& g, int64_t upto, std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for one session's layer carve (#216): `gdn` covers its `gdn_alloc` rows; the per-QSA-layer sizes
/// apply to each owned state [qsa_ord0, qsa_ord0 + qsa_alloc).  Rejects an inconsistent carve.  Checkpoints,
/// snapshots and their validation all use this: a session saves and restores only the state it owns.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& session, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
};
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
// Disk save without capturing the K/V on the host: `meta` gets everything but the K/V (running state copied,
// checkpoints as given by the view), `sources` one streamed source per QSA layer then the draft.  Caller has
// synchronized and must not run the session until the file is written.
bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& session,
                                   const ModelGeometry& g, const QsaState& draft, std::string& error);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
// The sessions a parked conversation is captured from or restored into: the first stage's session (device -1: the
// caller's current device is that session's), then each later layer-split stage's with the GPU it runs on, in
// stage order.  `draft` is the drafter, which lives on the last stage's device.
struct ConversationStageSet {
    std::vector<std::pair<SessionState*, int>> stages;
    QsaState* draft = nullptr;
};
// The K/V entries in a snapshot's flat order (each stage's owned layers, then the drafter), with their device.
struct ConversationKvEntry {
    const QsaState* state;
    int dev;
    bool index;   // the drafter's attention has no indexer
};
std::vector<ConversationKvEntry> conversation_kv_entries(const ConversationStageSet& set);
// The live running state of every stage (image.live, one SavedConversation per later stage) without K/V or
// checkpoints: what disk parking rewrites on every park.
bool conversation_live_save(SavedConversation& image, const ConversationView& view, const ConversationStageSet& set,
                            const ModelGeometry& g, std::string& error);
bool conversation_live_restore(const SavedConversation& image, const ConversationStageSet& set,
                               const ModelGeometry& g, std::string& error);
enum class ConversationRestore { restored, invalid, transfer_failed };
// Invalid images are rejected before any CUDA call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);

// The same with `draft == nullptr`: an image WITHOUT the draft layer's K/V (kv holds the session's own QSA layers
// only).  A layer split's later stages park this way; the draft ring is saved once, with the first stage's image.
// An image is validated and restored with the same kind of call it was saved with (a K/V layer count mismatch is
// rejected as invalid).
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session, const ModelGeometry& g,
                                 const QsaState* draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                         size_t& bytes, std::string& error);
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& session,
                                const ModelGeometry& g, const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error);

} // namespace strata::core
