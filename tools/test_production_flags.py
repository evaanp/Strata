#!/usr/bin/env python3
"""The box's production start line must still parse - every flag on it, no exceptions.

0140a failed to start on the GPU machine six seconds in: the merge kept every use of `--prompt-cache-tail`
(the field, the gates, the trace line) but lost the one parser line that sets it, so the engine answered
"unknown argument" and the server gave up.  A merge can silently drop any flag's parse line while its
semantics survive everywhere else, and no unit test builds the real argv.  This does: it runs the real
binary with the production flag list (paths are placeholders - the parser never opens them before the
unknown-arg check) and fails on the first "unknown argument".

  python3 tools/test_production_flags.py --exe ~/strata-build/cache/build-0140/strata
"""
import argparse
import os
import subprocess
import sys
import tempfile
import unittest

EXE = None

# The 0140b engine argv: 0139b's line plus --prompt-cache-tail (see PORT-0140.md's config diff).
# STRATA_CACHE_MESSAGE_BOUNDARY=1 / STRATA_STAGE_CACHE_SLOTS are env, not argv.  The parity tool's
# extra flags (--pipeline-windows, --expert-cache-gguf) are on the same line.
PRODUCTION = [
    '--serve',
    '--pack', '/data/packs/qwen3-next-80b-a3b.arena',
    '--max-context', '196608',
    '--kv', 'int8',
    '--prompt-cache', '6',
    '--prompt-cache-every', '16384',
    '--prompt-cache-tail',
    '--layer-split', '17',
    '--conversation-disk-dir', '/data/conversations',
    '--conversation-disk-gib', '96',
    '--conversation-cache-mib', '0',
    '--conversation-cache-slots', '0',
    '--conversation-cache-min-free-mib', '2560',
    '--lock-experts-gib', '24',
    '--no-remote-auto',
    '--adapt-async', '1',
    '--mtp', '/data/packs/mtp-draft.gguf',
    '--mtp-max-t', '4',
    '--spec', '4',
    '--vision',
    '--pool-workers', '6',
    '--expert-cache', 'auto',
    '--pipeline-windows', '1',
]


def engine_stderr(args):
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([EXE, *args], capture_output=True, text=True, timeout=120,
                              env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, cwd=tmp)
    return proc.stderr


class ProductionFlags(unittest.TestCase):
    def test_every_production_flag_parses(self):
        err = engine_stderr(PRODUCTION)
        unknown = [ln for ln in err.splitlines() if 'unknown argument' in ln]
        self.assertEqual([], unknown, "the engine rejects flags the box passes:\n" + "\n".join(unknown))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True, help='the built strata engine binary')
    EXE = ap.parse_args().exe
    unittest.main(argv=[sys.argv[0]])
