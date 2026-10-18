// include/strata/core/penalty_hist.hpp - the penalty-history bookkeeping of the serve decode loops.
//
// A window row's penalty history is the last `hist_n` tokens its pick follows: the tokens the session has
// consumed, then `window[0]` (the fed-back head) and the drafts before that row - `kernels::penalty_rows`
// builds those rows.  Both decode loops keep one running tail, `consumed`, that ends at the token BEFORE the
// current window's `window[0]`: staging on it before the window runs gives every row exactly the history
// plain decode would have counted there, and a verdict advances the tail by exactly the kept tokens
// `window[0..a]` - the row-a pick (the bonus) starts the next window and joins the tail only as its `window[0]`.
//
// The pipelined loop (#859, --pipeline-windows) launches window K+1 on stage 0 while stage 1 still verifies
// K, on the chain's assumption that K is accepted whole; but stage 1 never runs a speculative window.  A
// window reaches `stage` - and so the sampler's history buffer - only once the assumption it was built on is
// proven, at which point the real `consumed` already equals the assumed one, and it reaches it before its own
// sampling launches on that stage's stream (pl_finish samples and syncs before its picks are read, so no other
// sampling is in flight).  A speculative window rejected before stage 1 never samples: the history it WOULD
// have had never enters the buffer, and the rebuild a rejection forces (the GDN snapshot replay) is, for the
// penalty history, just the next on-path window's `stage` - it reads the real `consumed`, from which the
// rejected tokens were never committed.  `penalty_hist_test` pins that this pair of calls gives the pipelined
// loop the serial loop's rows, window for window, across rejected speculative windows.
#pragma once

#include "strata/kernels/sampler.hpp"

#include <cstdint>
#include <vector>

namespace strata::core {

struct PipePenalty {
    std::vector<int32_t>& consumed;   // the tail: ends at the token before the next window's window[0]
    int hist_n = 0;                   // penalty_last_n at the serve cap; 0: no penalties, nothing staged
    std::vector<int32_t>& rows;       // the shared staging buffer (cap * kVerifyMaxT slots, -1 padded)

    // The T rows to upload for an on-path window, enqueued before its verify (and its sampler reads of them)
    // on the stage's stream.
    const int32_t* stage(const int32_t* window, int T) {
        kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window, T, hist_n, rows.data());
        return rows.data();
    }
    // A verdict: the window keeps window[0..a]; its row-a pick becomes the next window's window[0].
    void commit(const int32_t* window, int a) {
        consumed.insert(consumed.end(), window, window + (size_t) a + 1);
    }
};

}  // namespace strata::core
