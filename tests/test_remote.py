import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
import cv2
import numpy as np
from synthetic_media import remote
from synthetic_media.media import mark_footer
from synthetic_media.scene_edit import composite, head_mask, pixel_digest
from synthetic_media.pipeline import run
from synthetic_media.swapping import LandmarkDetector


class RemoteTests(unittest.TestCase):
    def test_free_budget_rejects_paid_hardware_and_credit_capable_tokens(self):
        api=Mock()
        api.get_space_runtime.return_value=SimpleNamespace(hardware='zero-a10g',requested_hardware=None)
        api.whoami.return_value={'type':'user','isPro':False}
        remote.verify_free_runtime('owner/server','fixture',api)
        api.get_space_runtime.return_value=SimpleNamespace(hardware='zero-a10g',requested_hardware='a100-large')
        with self.assertRaisesRegex(ValueError,'Free-only'):
            remote.verify_free_runtime('owner/server','fixture',api)
        api.get_space_runtime.return_value=SimpleNamespace(hardware='zero-a10g',requested_hardware=None)
        api.whoami.return_value={'type':'user','isPro':True}
        with self.assertRaisesRegex(ValueError,'Free-only'):
            remote.verify_free_runtime('owner/server','fixture',api)

    def test_no_server_configuration_never_enables_remote(self):
        with patch.dict(os.environ, {'SYNTHETIC_MEDIA_REMOTE_SPACE': ''}):
            self.assertFalse(remote.status()['available'])
            with self.assertRaisesRegex(ValueError, 'no server is active'):
                from synthetic_media.pipeline import registry
                registry('remote')

    def test_semantic_head_mask_excludes_background_and_clothing(self):
        def worker(mode, model, folder):
            labels=np.zeros((512,512),np.uint8)
            labels[50:180,100:400]=17  # hair
            labels[180:330,100:400]=1  # face
            labels[330:400,180:320]=14 # neck
            labels[400:]=16           # clothing
            np.save(folder/'parse.npy',labels)
            return {}
        face=np.array([40,40,60,80],np.float32)
        with tempfile.TemporaryDirectory() as directory, patch('synthetic_media.quality_models.model_info'):
            mask,meta=head_mask(np.zeros((240,180,3),np.uint8),face,Path(directory),worker)
        self.assertGreater(np.count_nonzero(mask),100)
        self.assertEqual(float(mask[0,0]),0)
        self.assertEqual(float(mask[-1].sum()),0)
        self.assertIn(17,meta['labels'])
        self.assertNotIn(16,meta['labels'])

    def test_head_composite_preserves_original_scene_exactly(self):
        target=np.random.default_rng(4).integers(0,256,(101,151,3),dtype=np.uint8)
        mask=np.zeros(target.shape[:2],np.float32);mask[10:80,40:100]=.6
        result,meta=composite(target,np.zeros_like(target),mask)
        np.testing.assert_array_equal(result[mask==0],target[mask==0])
        self.assertGreater(np.count_nonzero(result!=target),0)
        self.assertEqual(meta['native_size'],[151,101])
        with self.assertRaisesRegex(ValueError,'preserve scene'):
            composite(target,target,np.ones_like(mask))

    def test_remote_response_rejects_wrong_target_and_missing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            target=np.full((100,160,3),100,np.uint8)
            path=Path(directory)/'output.png'
            metadata={'protocol':1,'synthetic':True,'model':remote.MODEL,
                      'model_revision':remote.REVISION,'content_size':[160,100],
                      'target_pixel_sha256':pixel_digest(target)}
            cv2.imwrite(str(path),mark_footer(target))
            np.testing.assert_array_equal(remote.validate_output(path,metadata,target),target)
            with self.assertRaisesRegex(ValueError,'different target'):
                remote.validate_output(path,dict(metadata,target_pixel_sha256='wrong'),target)
            cv2.imwrite(str(path),target)
            with self.assertRaisesRegex(ValueError,'disclosure'):
                remote.validate_output(path,metadata,target)

    def test_remote_head_edit_survives_face_guard_and_remains_unreviewed(self):
        face=np.array([30,20,80,100,52,52,88,52,70,72,55,92,85,92,.99],np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=np.full((160,160,3),100,np.uint8)
            source=np.full_like(target,80)
            source_path,target_path=root/'reference.png',root/'target.png'
            cv2.imwrite(str(source_path),source);cv2.imwrite(str(target_path),target)
            mask=np.zeros((160,160),np.float32);mask[10:140,20:120]=1
            def generate(reference_path,submitted_target,folder,note):
                self.assertEqual(note,'Authorized media on my selected server')
                np.testing.assert_array_equal(cv2.imread(str(reference_path)),source)
                np.testing.assert_array_equal(cv2.imread(str(submitted_target)),target)
                path=folder/'generated.png';cv2.imwrite(str(path),mark_footer(np.full_like(target,180)))
                return str(path),{'protocol':1,'synthetic':True,'model':remote.MODEL,
                       'model_revision':remote.REVISION,'content_size':[160,160],
                       'target_pixel_sha256':pixel_digest(target)}
            with patch.object(remote,'status',return_value={'available':True}),patch.object(LandmarkDetector,'single_face',return_value=face),patch.object(remote,'head_mask',return_value=(mask,{})),patch.object(remote,'request_generation',side_effect=generate):
                manifest=run(source_path,root/'jobs',True,'Authorized media on my selected server',
                             'face_swapping',target_path,'remote')
            result=cv2.imread(str(root/'jobs'/manifest['job_id']/'swapped.png'))[:160]
            np.testing.assert_array_equal(result[mask==0],target[mask==0])
            self.assertEqual(int(result[12,22,0]),180) # hair region outside old face box
            self.assertEqual(manifest['status'],'needs_review')
            self.assertNotIn('_edit_mask',json.dumps(manifest))

    @unittest.skipUnless(importlib.util.find_spec('gradio_client'),'Optional remote client absent')
    def test_queue_timeout_cancels_and_closes_connection(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'SYNTHETIC_MEDIA_REMOTE_PASSWORD':'fixture'}):
            root=Path(directory);reference=root/'reference.png';target=root/'target.png'
            reference.write_bytes(b'fixture');target.write_bytes(b'fixture')
            with patch.object(remote,'status',return_value={'available':True,'space':'owner/server'}),patch.object(remote,'verify_free_runtime'),patch('gradio_client.Client',autospec=True) as client:
                job=client.return_value.submit.return_value
                job.result.side_effect=TimeoutError
                with self.assertRaisesRegex(RuntimeError,'cancellation requested'):
                    remote._request_generation(reference,target,root,'Authorized fixture')
                job.cancel.assert_called_once()
                client.return_value.close.assert_called_once()
