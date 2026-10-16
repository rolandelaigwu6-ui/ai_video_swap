import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib
import cv2
import numpy as np
from synthetic_media.neural import align, paste_prediction, run_worker, status
from synthetic_media.neural_install import install


class NeuralContractTests(unittest.TestCase):
    def test_alignment_matches_canonical_coordinates(self):
        landmarks = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                              [41.5493,92.3655],[70.7299,92.2041]], np.float32)
        image = np.zeros((160,160,3), np.uint8)
        crop, matrix = align(image, landmarks, 112)
        np.testing.assert_allclose(matrix, [[1,0,0],[0,1,0]], atol=1e-5)
        self.assertEqual(crop.shape, (112,112,3))
        _, matrix = align(image, landmarks, 128)
        np.testing.assert_allclose(matrix, [[1,0,8],[0,1,0]], atol=1e-5)

    def test_paste_preserves_pixels_outside_generated_crop(self):
        target = np.full((200,200,3), 70, np.uint8)
        prediction = np.ones((1,3,128,128), np.float32) * .8
        result = paste_prediction(target, prediction, np.array([[1,0,0],[0,1,0]],np.float32))
        self.assertGreater(int(result[64,64,0]), 150)
        np.testing.assert_array_equal(result[140:], target[140:])

    def test_installer_validates_before_marking_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'test.onnx.part'
            model.write_bytes(b'test-model')
            (root / 'test.hash').write_text(format(zlib.crc32(b'test-model'), '08x'))
            with patch('synthetic_media.neural_install.MODELS', {'test.onnx': 10}):
                config = install(root, verify_only=True)
            self.assertEqual(config['models']['test.onnx']['size'], 10)
            self.assertTrue((root / 'ready.json').is_file())
            (root / 'ready.json').unlink()
            (root / 'test.onnx').write_bytes(b'bad-model!')
            with patch('synthetic_media.neural_install.MODELS', {'test.onnx': 10}):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    install(root, verify_only=True)
            self.assertFalse((root / 'ready.json').exists())

    def test_uninstalled_status_and_explicit_neural_selection(self):
        from synthetic_media.pipeline import registry
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'SYNTHETIC_MEDIA_NEURAL_MODELS':directory}):
            self.assertFalse(status()['available'])
            with self.assertRaisesRegex(ValueError, 'not installed'):
                registry('neural')
            self.assertFalse(registry('baseline').require('face_swapping').generative)

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'Optional ONNX dependencies absent')
    def test_isolated_embedding_worker_with_generated_model(self):
        # Tiny generated graph validates actual runtime/process I/O, not portrait quality.
        import onnx
        from onnx import helper, TensorProto
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = helper.make_tensor('value', TensorProto.FLOAT, [1,512], [1.] * 512)
            graph = helper.make_graph([helper.make_node('Constant', [], ['embedding'], value=value)], 'fixture',
                    [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,3,112,112])],
                    [helper.make_tensor_value_info('embedding', TensorProto.FLOAT, [1,512])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',13)])
            model.ir_version = 8
            path = root / 'fixture.onnx'
            onnx.save(model, path)
            np.save(root / 'source.npy', np.zeros((1,3,112,112),np.float32))
            measurement = run_worker('embed',path,root,timeout=20)
            embedding = np.load(root / 'embedding.npy')
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1, places=5)
            self.assertEqual(measurement['stage'],'embed')

    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'), 'Optional ONNX dependencies absent')
    def test_hyperswap_worker_normalization_with_generated_graph(self):
        import onnx
        from onnx import helper,TensorProto
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            graph=helper.make_graph([helper.make_node('Identity',['target'],['result'])],'hyperswap_fixture',
                    [helper.make_tensor_value_info('target',TensorProto.FLOAT,[1,3,256,256]),
                     helper.make_tensor_value_info('source',TensorProto.FLOAT,[1,512])],
                    [helper.make_tensor_value_info('result',TensorProto.FLOAT,[1,3,256,256])])
            model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]);model.ir_version=8
            path=root/'fixture.onnx';onnx.save(model,path)
            np.save(root/'target.npy',np.zeros((1,3,256,256),np.float32))
            np.save(root/'embedding.npy',np.ones((1,512),np.float32))
            run_worker('hyperswap',path,root,timeout=20)
            np.testing.assert_allclose(np.load(root/'prediction.npy'),.5)
