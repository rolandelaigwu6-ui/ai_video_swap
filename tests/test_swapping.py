import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from synthetic_media.swapping import LandmarkDetector, blend_face, model_path, read_working_image, MODEL_SHA256
from synthetic_media.pipeline import run, digest


def face():
    return np.array([30, 20, 80, 100, 52, 52, 88, 52, 70, 72, 55, 92, 85, 92, .99], np.float32)


class SwappingTests(unittest.TestCase):
    def test_bundled_model_checksum_and_real_inference(self):
        self.assertEqual(digest(model_path()), MODEL_SHA256)
        detector = LandmarkDetector(model_path())
        with self.assertRaisesRegex(ValueError, 'found 0'):
            detector.single_face(np.zeros((160, 160, 3), np.uint8), 'Reference')

    def test_affine_blend_changes_face_preserves_background(self):
        source = np.full((160, 160, 3), 100, np.uint8)
        cv2.circle(source, (70, 70), 20, (240, 20, 180), -1)
        target = np.full_like(source, 70)
        result, metadata = blend_face(source, target, face(), face())
        self.assertGreater(np.abs(result.astype(float) - target).sum(), 1000)
        np.testing.assert_array_equal(result[:10], target[:10])
        np.testing.assert_allclose(metadata['affine_matrix'], [[1, 0, 0], [0, 1, 0]], atol=1e-4)

    def test_swap_job_records_both_inputs_and_label(self):
        # Controlled landmarks test orchestration; actual detector is exercised above.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.full((160, 160, 3), 100, np.uint8)
            cv2.circle(source, (70, 70), 20, (240, 20, 180), -1)
            target = np.full_like(source, 70)
            reference_path, target_path = root / 'reference.png', root / 'target.png'
            cv2.imwrite(str(reference_path), source)
            cv2.imwrite(str(target_path), target)
            with patch.object(LandmarkDetector, 'single_face', return_value=face()):
                manifest = run(reference_path, root / 'jobs', True, 'Authorized fixtures', 'face_swapping', target_path, backend='baseline')
            job = root / 'jobs' / manifest['job_id']
            saved = json.loads((job / 'manifest.json').read_text())
            self.assertEqual(saved['links'], manifest['links'])
            self.assertEqual(saved['target_sha256'], digest(target_path))
            self.assertFalse(saved['analysis']['rendered']['generative'])
            output = cv2.imread(str(job / saved['outputs']['labeled_preview']))
            raw, _ = blend_face(source, target, face(), face())
            self.assertFalse(np.array_equal(output[-30:], raw[-30:]))

    def test_memory_resize_and_missing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'large.png'
            cv2.imwrite(str(path), np.zeros((1000, 2000, 3), np.uint8))
            image, original = read_working_image(path)
            self.assertEqual(image.shape[:2], (640, 1280))
            self.assertEqual(original, [2000, 1000])
            with self.assertRaisesRegex(ValueError, 'requires a target'):
                run(path, Path(directory), True, 'Owned fixture', 'face_swapping')

    def test_blend_retains_broad_source_detail(self):
        source = np.full((160, 160, 3), 70, np.uint8)
        for x in range(160):
            source[:, x] = 30 + x
        target = np.full_like(source, 70)
        result, metadata = blend_face(source, target, face(), face())
        # A broad illumination gradient disappears with Poisson-only cloning.
        self.assertGreater(int(result[70, 85, 0]) - int(result[70, 55, 0]), 15)
        self.assertEqual(metadata['source_weight'], .85)
        np.testing.assert_array_equal(result[0], target[0])

    def test_small_face_detection_retries_without_lowering_threshold(self):
        detector = LandmarkDetector(model_path())
        with patch.object(detector, 'detector') as model:
            upscaled = face().copy()
            upscaled[:14] *= 3
            model.detect.side_effect = [(None, None), (None, np.array([upscaled]))]
            result = detector.single_face(np.zeros((160,160,3), np.uint8), 'Reference')
            np.testing.assert_allclose(result, face())
            self.assertEqual(model.detect.call_count, 2)

    def test_native_target_scene_is_locked_even_if_backend_changes_whole_image(self):
        from synthetic_media.swapping import ImageFaceSwapper
        from synthetic_media.components import Context,Artifact
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=np.full((160,160,3),100,np.uint8)
            target=np.random.default_rng(7).integers(0,256,(1000,2000,3),dtype=np.uint8)
            source_path,target_path=root/'reference.png',root/'target.png'
            cv2.imwrite(str(source_path),source);cv2.imwrite(str(target_path),target)
            with patch.object(LandmarkDetector,'single_face',return_value=face()),patch.object(ImageFaceSwapper,'transform',return_value=(np.zeros_like(target),{})):
                artifact=ImageFaceSwapper().process({'reference':Artifact(source_path,'reference'),'target':Artifact(target_path,'target')},Context('test',root))['rendered']
            exported=cv2.imread(str(artifact.path))
            self.assertEqual(exported.shape[1],2000)
            self.assertGreater(exported.shape[0],1000)
            content=exported[:1000]
            allowed=np.zeros((1000,2000),bool);allowed[20:120,30:110]=True
            np.testing.assert_array_equal(content[~allowed],target[~allowed])
            self.assertTrue((content[allowed]==0).all())
            self.assertEqual(artifact.metadata['content_size'],[2000,1000])
            self.assertTrue(artifact.metadata['scene_integrity']['outside_face_pixels_preserved'])
