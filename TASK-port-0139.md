# Task: move our Strata patches from 0.1.38 to upstream v0.1.39, drop what upstream now covers, and judge `parallel`

You are working in `~/strata-0139`, a git worktree of the Strata engine on branch `pla-0139`, created at upstream
tag `v0.1.39` (6f32ec0, released 2026-10-04). Our production branch is `prefix-park-analysis` (worktree
`~/strata-prefix`, HEAD a5309b0, image `strata:0.1.38-pla1-prefix`): upstream v0.1.38 plus 53 of our commits
(`git log --oneline v0.1.38..prefix-park-analysis`). Upstream is remote `upstream` (Niko1221/Strata), fetched.
Release notes: `gh release view v0.1.39 --repo Niko1221/Strata`.

A trial `git merge-tree` of our branch onto v0.1.39 gives 11 conflicted files: conversation_cache.hpp,
native_dense.hpp/.cpp, remote_experts.hpp/.cpp, serve/server.py, serve/test_server.py, serve/web/app.js,
src/core/conversation_state.cpp, src/program/generate.cpp, tools/make_profile.py. Part of that is because upstream
took several of our things in its own form:

- Our PR #716 (conversation parking with `--layer-split`) was **closed**: the owner says parking under a layer split
  is in 0.1.39 via #559 ("a parked conversation holds one image per stage"). Read
  `gh pr view 716 --repo Niko1221/Strata --comments`.
- Our PR #715 (`Connection: close` on every response) and #718 (disk parking keyed by `X-Strata-Conversation`,
  keyless requests in a RAM cache) are still **open**. PR texts in `~/strata-build/pr-texts/`.
- Our branch merged upstream PRs #390 (layer-split dense-weight carve), #385, #439, #453 before they were released;
  0.1.39 has `STRATA_STAGE_TRIM=1` (#639) for the carve.

## Deliverables

1. `~/strata-0139/PORT-0139.md`: one row per commit (or per coherent group) of `v0.1.38..prefix-park-analysis`, with
   a verdict **drop** (upstream covers it - name the upstream commit/PR and file:line that proves it), **port**
   (still needed - say what changed to make it fit) or **adapt** (upstream has a different version; say which wins
   and why). Merge commits and docs/notes files get one line each.
2. Branch `pla-0139` with the patches we keep applied on top of v0.1.39 (cherry-pick or re-implement; small,
   reviewable commits). It must build (see Build) and its relevant unit tests must pass.
3. A section in `PORT-0139.md` on **`parallel` (#465) for us** (see below).
4. A short "what to measure on the box" list at the end: which A/B runs prove the port is at least as good as
   `0.1.38-pla1-prefix` (parity of parking/disk parking/keyless prefix cache, decode tok/s, prefill tok/s on a ~70K
   prompt), and which new 0.1.39 switches are worth a sweep (`STRATA_RING_BYTES`, `STRATA_PREFILL_HELP`,
   `STRATA_STAGE_TRIM`, THP arena, `--remote-expert-opt` - say if it even applies to a pure layer split).

## What our patches do (so you know what must survive)

- **Disk parking** keyed by the `X-Strata-Conversation` header (`--conversation-disk-dir`): our client PLA switches
  between conversations constantly; a parked conversation comes back in ~1.4 s instead of 90-215 s of prefill.
  Must keep working **with `--layer-split`**. Where upstream's per-stage parking (#559) and ours overlap, prefer one
  implementation - upstream's where it does the same job - and keep ours only for what upstream lacks (disk, the
  key, logging where a returning conversation stops matching its live end).
- **Keyless RAM cache + hybrid prefix parking** (`--conversation-cache-mib 512 --conversation-cache-min-free-mib
  2048`; commits 9825b49..a5309b0, docs in `IMPL-prefix-park.md` / `ANALYSIS-prefix-park.md` on the old branch):
  for PLA's short background requests without a key.
- **Sampled tails** (60b8ea3, 4c2ae1e): keep the sampled token ids of a conversation's live end, because the model
  samples non-canonical tokenisations (`-b`+`udget`) and re-tokenising breaks the prefix. Plus `/tokenize`,
  `/detokenize`, `/apply-template`. Check whether 0.1.39 solved this differently.
- **`--lock-experts-gib`** (d074baa, 9e3c3dc): mlock the experts no GPU slot holds, following the adaptive tier.
  Check how it interacts with the 0.1.39 arena changes (THP #650, `STRATA_ARENA_MMAP`, #577's RAM-budget choice).
- `bd69e87` (max_new for splice ids), `fb63f69` (lone surrogate in /tokenize), `7fbbcba` (Connection: close),
  `c57f58d` (image inside a tool_result), the parity tool (031322d, 31aa4ef), monitor cards (4309a4c),
  `8a9c179` (four-way placement), `efd79ba`, `cc9a9d1`/`2a3a99b` (batched gather), `c55854c`, `a084595`.
- Our earlier merge notes: `MERGE_0138_NOTES.md` on the old branch. Known open issue there:
  `conversation_transfer_test` (drain order across stages) failed before the 0.1.38 merge too - check if it passes
  on v0.1.39 alone, and with our port.

## `parallel` (#465) for us

We postponed batching yesterday because upstream was already building it. Earlier analysis of the PR version:
`~/strata-batch/ANALYSIS-batch-559.md` (penalties forced to 0 in batch windows, slots not parked at BDONE, its
split-parking clashing with ours, ~3 GiB extra pinned RAM for `--batch 2`). Re-check every one of those claims
against the **released** code with file:line, and answer:

- Does a batch slot honour `presence_penalty` (we need 1.0; without it the model loops while thinking)? MTP drafts
  in batch windows (`--batch-spec`)?
- Does a slot's conversation go through our parking (keyed and keyless) on entry and get parked when it finishes?
- Cost on our setup: VRAM per slot (release notes: 0.56 GiB per slot out of the expert cache), pinned host RAM per
  slot at 150K context with int8 KV, and whether `parallel: 2` fits in the RAM we have (see below).
- Expected throughput for 2 concurrent PLA conversations vs today's serialized ~47 tok/s (estimate, say so).
- Verdict: turn it on, turn it on after a fix (which), or not for us (why).

## How we run (production)

- Hardware: RTX 5060 Ti 16 GB (CUDA0, gen5 x8) and RTX 2000 Ada 16 GB (CUDA1, gen4 x4 behind the chipset), no P2P;
  Ryzen 7 9700X (8 cores, AVX-512); the LXC has ~58 GiB RAM and in arena mode only **~2.9 GB MemAvailable** is
  left. The host has nothing spare to give.
- Model: Qwen3.8-Flash-Next GSQ-RCO IQ3_S, arena mode (all experts pinned in RAM), `layer_split 30` (layers 0-29
  CUDA0, 30-47 CUDA1), `--spec 4` with the MTP draft head, `--kv int8`, 150K context (153600), vision on CUDA0.
  Sampling defaults temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.0.
- **The exact production run config is `PROD-strata-iq3_s.arena-0138-prefix.json`** (copied from the box) and the
  container definition `PROD-docker-compose-excerpt.yml` in this directory. These are the source of truth over the
  summary above: every flag in there must still be accepted and mean the same on `pla-0139`, or the port notes say
  what replaces it.
- Measured: decode ~47-50 tok/s single stream; per draft window ~21 ms on stage 0 (dense ~47%, waiting on misses
  ~28%, experts ~22%), the two GPUs work one after the other.

## Rules

- Measure, don't reason: claims about code cite file:line. Estimates are labelled as estimates.
- This machine has no GPU. **Do not touch the inference box (172.16.98.21)**: no ssh, no docker there. Do not push,
  do not build images, do not open or comment on PRs.
- Do **not** use the `task` tool / sub-agents (they share this session's parking key and force a full re-prefill on
  every switch).
- Build: `BUILD=build-0139 ~/strata-build/build-0139.sh` (builds `~/strata-0139`; `TESTS=ON TARGET=<test>` for a
  test binary). One build at a time, `JOBS=3` at most. CPU-only tests can run in the same `strata-builder` image.
- Don't run `git checkout` to undo experiments in this worktree; use a scratch branch.
- Commit on `pla-0139` as you go; commit `PORT-0139.md` too. Write all code, comments and docs in English.
- Stop when the deliverables are done; no deployment.
