// src/core/penalty_hist_test.cpp - the penalty-history bookkeeping of the pipelined decode loop
// (include/strata/core/penalty_hist.hpp), checked against the serial loop on the same tokens.
//
// No GPU.  The serve pipelined loop (#859, --pipeline-windows) runs window K+1 on stage 0 while stage 1
// verifies K, on the drafter chain's guess that K is accepted whole and that its bonus is the chain's first
// output.  The history the sampler reads (the `penalty_rows` buffer) must be, for every window whose picks
// the session emits, exactly what the serial loop would have staged for it.  This test replays the loop's
// decisions and calls the production `PipePenalty` at exactly the points the loop calls it:
//   - the chain builds speculative window B while A is still in flight: from the tail as it stands, plus
//     the ASSUMED commit of all of A (its rows are computed here only to be compared against - they never
//     reach the buffer, because stage 1 never runs a speculative window);
//   - `stage` runs before the stage-1 launch of each ON-PATH window, from the real consumed tail;
//   - `commit` runs after each verdict with the kept count.
// A speculative window whose guess fails is rejected (the GDN snapshot replay): its tokens never enter the
// tail, and the next on-path window re-stages from the real tail - the penalty half of that rebuild.
//
// Checks, per scenario:
//   1. before every stage, the pipelined tail equals an independent serial ledger's tail (token for token);
//   2. the staged rows equal the rows the serial ledger gives the same window (`penalty_rows` on the
//      ledger's tail, bookkept without PipePenalty);
//   3. a proven speculative window's staged rows equal the rows its chain assumed at build time - at that
//      point the assumption is the truth, so no rebuild was needed;
//   4. a rejected speculative window's tokens never appear in any later staged row (ids are unique, so a
//      missed re-stage or a wrongly-advanced tail shows up immediately).
#include "strata/core/penalty_hist.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::core;

namespace {

int g_fail = 0;
int64_t g_checks = 0;
void check(bool ok, const char* what, long long a = 0, long long b = 0) {
    ++g_checks;
    if (!ok) {
        if (++g_fail <= 20) std::printf("  FAIL %s (%lld vs %lld)\n", what, a, b);
    }
}

constexpr int kMaxT = 8;   // the widest verify window (kernels::kVerifyMaxT)

// One decode: `rounds` windows, sizes 1..kMaxT, random accepted counts; after every verdict a speculative
// window is built on the chain's guess and proven or rejected.  An independent serial ledger over the same
// emitted tokens is the ground truth for every staged buffer.
void scenario(unsigned seed, int hist_n, int rounds) {
    std::mt19937 rng(seed);
    const size_t slots = static_cast<size_t>(hist_n) * kMaxT;

    // the prompt: `consumed` holds all but its last token, which is the first window's window[0]
    std::vector<int32_t> consumed, serial;
    const int prompt = 1 + static_cast<int>(rng() % 50);
    for (int i = 0; i < prompt; ++i) consumed.push_back(1000 + i);
    serial = consumed;

    std::vector<int32_t> rows(slots, -1);
    PipePenalty book{consumed, hist_n, rows};

    // the serial ledger: rows for a window, then advance it past the kept tokens
    std::vector<int32_t> ground(slots);
    auto serial_rows = [&](const std::vector<int32_t>& w, int T) {
        strata::kernels::penalty_rows(serial.data(), static_cast<int64_t>(serial.size()), w.data(), T, hist_n,
                                      ground.data());
        return ground;
    };
    auto serial_commit = [&](const std::vector<int32_t>& w, int a) {
        for (int i = 0; i <= a; ++i) serial.push_back(w[static_cast<size_t>(i)]);
    };

    int32_t next_id = 1;
    auto fresh_token = [&] { return 5000 + next_id++; };

    std::vector<int32_t> never;                          // tokens of rejected windows (check 4)
    std::vector<int32_t> pending;                        // a proven speculative window: next A
    std::vector<int32_t> pending_assumed(slots);         // the rows its chain assumed (check 3)
    bool pending_proven = false;
    int32_t bonus = 999;                                 // last verdict's pick: next window[0]

    for (int r = 0; r < rounds; ++r) {
        const int T = pending_proven ? static_cast<int>(pending.size())
                                     : 1 + static_cast<int>(rng() % kMaxT);
        std::vector<int32_t> w;
        if (pending_proven) {
            w = pending;
        } else {
            w.resize(static_cast<size_t>(T));
            w[0] = bonus;
            for (int t = 1; t < T; ++t) w[static_cast<size_t>(t)] = fresh_token();
        }

        // ---- stage 1: the on-path window stages from the real tail (checks 1, 2, 3)
        check(consumed == serial, "pipelined tail == serial ledger before the stage", r, 0);
        const int32_t* staged = book.stage(w.data(), T);
        std::vector<int32_t> upload(staged, staged + static_cast<size_t>(T) * hist_n);
        if (pending_proven)
            check(std::memcmp(upload.data(), pending_assumed.data(), upload.size() * sizeof(int32_t)) == 0,
                  "a proven speculative window stages exactly what its chain assumed", r, 0);
        const std::vector<int32_t>& g = serial_rows(w, T);
        check(std::memcmp(upload.data(), g.data(), upload.size() * sizeof(int32_t)) == 0,
              "staged rows == the serial loop's rows", r, 0);
        pending_proven = false;

        // ---- the chain builds the next speculative window while A is still in flight: from the tail as
        // it stands PLUS the assumed full commit of A (row t's history is built as if A had been kept
        // whole and its bonus were the chain's first output).
        const int32_t real_bonus = fresh_token();
        const int TB = 1 + static_cast<int>(rng() % kMaxT);
        std::vector<int32_t> b(static_cast<size_t>(TB));
        b[0] = rng() % 2 == 0 ? real_bonus : fresh_token();   // the chain's guess at A's bonus
        for (int t = 1; t < TB; ++t) b[static_cast<size_t>(t)] = fresh_token();
        std::vector<int32_t> assumed_tail = consumed;
        assumed_tail.insert(assumed_tail.end(), w.begin(), w.end());   // A kept WHOLE: the assumption
        pending_assumed.assign(slots, -1);
        strata::kernels::penalty_rows(assumed_tail.data(), static_cast<int64_t>(assumed_tail.size()),
                                      b.data(), TB, hist_n, pending_assumed.data());

        // ---- the verdict
        const int a = static_cast<int>(rng() % T);
        book.commit(w.data(), a);
        serial_commit(w, a);
        bonus = real_bonus;
        if (a == T - 1 && b[0] == real_bonus) {   // the loop's on-path test
            pending = std::move(b);
            pending_proven = true;
        } else {
            // rejected before stage 1.  Only the tokens the session will never hold are the chain's
            // drafts and, when its guess missed, the guessed bonus itself: a GUESS THAT MATCHED is the
            // real bonus - a legal token, about to become the next window's window[0].
            if (b[0] != real_bonus) never.push_back(b[0]);
            for (int t = 1; t < TB; ++t) never.push_back(b[static_cast<size_t>(t)]);
        }
        // check 4: no rejected window's token has ever entered a staged row
        for (const int32_t t : never)
            check(std::find(upload.begin(), upload.end(), t) == upload.end(),
                  "a rejected speculative token never reaches the history", r, t);
    }
}

}  // namespace

int main() {
    std::printf("penalty_hist_test: the pipelined loop's penalty history vs the serial loop's\n");
    for (const int hist_n : {1, 3, 17, 64, 512})
        for (unsigned s = 0; s < 40; ++s)
            scenario(s * 7919u + static_cast<unsigned>(hist_n), hist_n, 60);
    std::printf("penalty_hist_test: %lld checks, %d failures\n", (long long) g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
