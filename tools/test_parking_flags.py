#!/usr/bin/env python3
"""The engine's parking flags side by side: --conversation-disk-dir AND --conversation-cache-mib.

A conversation with a `conv=` id parks on disk; a request without one parks in the RAM cache (cd61c48).  Before that,
the disk replaced the RAM cache and the engine said so and zeroed --conversation-cache-mib, so PLA's keyless requests
got no cache at all.  This runs the real engine binary: the start line is what production reads, and a build that
re-introduces the replacement silently loses the keyless cache.

  python3 tools/test_parking_flags.py --exe ~/strata-build/cache/build-0139/strata
"""
import argparse
import os
import subprocess
import sys
import tempfile
import unittest

EXE = None


def engine_stderr(args):
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([EXE, '--serve', *args], capture_output=True, text=True, timeout=60,
                              env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, cwd=tmp)
    return proc.stderr


class ParkingFlags(unittest.TestCase):
    def test_disk_and_ram_park_side_by_side(self):
        err = engine_stderr(['--conversation-disk-dir', 'conv-disk', '--conversation-cache-mib', '512'])
        self.assertNotIn('unknown argument', err)                     # both flags still parse
        self.assertNotIn('ignoring --conversation-cache-mib', err)    # the old replacement is gone
        self.assertIn('park on disk', err)
        self.assertIn('without one in 512 MiB of RAM', err)

    def test_ram_cache_alone_still_says_nothing_about_disk(self):
        err = engine_stderr(['--conversation-cache-mib', '512'])
        self.assertNotIn('ignoring --conversation-cache-mib', err)
        self.assertNotIn('park on disk', err)

    def test_disk_alone_still_warns_only_about_prompt_cache(self):
        err = engine_stderr(['--conversation-disk-dir', 'conv-disk'])
        self.assertNotIn('park on disk', err)
        self.assertNotIn('ignoring', err)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True, help='the built strata engine binary')
    EXE = ap.parse_args().exe
    unittest.main(argv=[sys.argv[0]])
