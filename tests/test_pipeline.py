import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import imageio_ffmpeg
from unittest.mock import patch
from synthetic_media.components import Artifact
from synthetic_media.pipeline import run, registry

class PipelineTests(unittest.TestCase):
    def test_technical_pass_does_not_mark_unreviewed_portrait_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.png'
            cv2.imwrite(str(source),np.full((128,128,3),127,np.uint8))
            artifact=Artifact(source,'synthetic_image',{'quality_report':{
                'status':'technical_checks_passed','visual_acceptance':'unverified'}})
            with patch('synthetic_media.pipeline.registry') as registered:
                registered.return_value.require.return_value.process.return_value={'rendered':artifact}
                manifest=run(source,root/'jobs',True,'Owned test fixture','face_swapping',source)
            self.assertEqual(manifest['status'],'needs_review')

    def test_authorization_and_unavailable_stage(self):
        with self.assertRaisesRegex(ValueError, 'Authorization'):
            run(Path('missing'), Path('/tmp'), False, '')
        with self.assertRaisesRegex(ValueError, 'no installed model'):
            run(Path('missing'), Path('/tmp'), True, 'owned', 'voice_cloning')
        self.assertFalse(registry().capabilities()['voice_cloning']['available'])

    def test_image_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.png'
            cv2.imwrite(str(source), np.full((128, 256, 3), 127, np.uint8))
            result = run(source, root / 'jobs', True, 'Test fixture created by author')
            job = root / 'jobs' / result['job_id']
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['analysis']['faces']['samples'][0]['faces'], [])
            preview = cv2.imread(str(job / result['outputs']['labeled_preview']))
            np.testing.assert_array_equal(preview[:128], cv2.imread(str(source)))
            self.assertGreater(preview.shape[0],128)
            self.assertTrue(json.loads((job / 'manifest.json').read_text())['synthetic'])

    def test_invalid_media_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'invalid.bin'
            source.write_bytes(b'not media')
            with self.assertRaises(ValueError):
                run(source, root / 'jobs', True, 'Owned test data')
            manifests = list((root / 'jobs').glob('*/manifest.json'))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(json.loads(manifests[0].read_text())['status'], 'failed')

    def test_video_timing_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mp4'
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([ffmpeg, '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=128x96:r=24:d=1', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-c:v', 'libx264', '-c:a', 'aac', str(source)], check=True)
            result = run(source, root / 'jobs', True, 'Generated test fixture')
            output = root / 'jobs' / result['job_id'] / result['outputs']['labeled_preview']
            capture = cv2.VideoCapture(str(output))
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 24)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 24)
            capture.release()
            # Mapping required audio fails if the output lost its audio stream.
            subprocess.run([ffmpeg, '-loglevel', 'error', '-i', str(output), '-map', '0:a:0', '-f', 'null', '-'], check=True)

if __name__ == '__main__':
    unittest.main()
