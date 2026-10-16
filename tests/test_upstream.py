import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from synthetic_media import upstream
import numpy as np


class UpstreamIsolationTests(unittest.TestCase):
    def test_odd_dimensions_round_trip_without_resampling(self):
        original=np.random.default_rng(12).integers(0,256,(321,423,3),dtype=np.uint8)
        padded=upstream.prepare_target(original)
        self.assertEqual(padded.shape,(322,424,3))
        np.testing.assert_array_equal(upstream.restore_target(padded,original),original)
        with self.assertRaisesRegex(ValueError,'dimensions'):
            upstream.restore_target(padded[:-2],original)

    def test_timeout_reaps_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(upstream, 'VENDOR', root):
                with self.assertRaisesRegex(RuntimeError, 'timed out'):
                    upstream.execute([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     root / 'worker.log', timeout=.1)

    def test_failure_retains_diagnostic_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'worker.log'
            with patch.object(upstream, 'VENDOR', root):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    upstream.execute([sys.executable, '-c', 'raise RuntimeError("fixture failure")'], log)
            self.assertIn('fixture failure', log.read_text())
