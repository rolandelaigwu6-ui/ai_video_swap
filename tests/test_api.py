import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from synthetic_media import api

class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_manifest_preview_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, encoded = cv2.imencode('.png', np.zeros((96, 128, 3), dtype=np.uint8))
            upload = UploadFile(filename='../../untrusted.png', file=io.BytesIO(encoded.tobytes()))
            with patch.object(api, 'ROOT', root):
                result = await api.create_job(upload, True, 'Owned generated fixture')
                self.assertEqual(api.get_job(result['job_id'])['status'], 'completed')
                self.assertTrue(Path(api.preview(result['job_id']).path).is_file())
                self.assertEqual(list((root / 'uploads').iterdir()), [])
                with self.assertRaises(HTTPException):
                    api.get_job('../outside')

    async def test_authorization_rejected(self):
        upload = UploadFile(filename='a.png', file=io.BytesIO(b''))
        with self.assertRaises(HTTPException) as error:
            await api.create_job(upload, False, 'no consent')
        self.assertEqual(error.exception.status_code, 422)
        await upload.close()

    async def test_resource_lock_and_upload_limit(self):
        upload = UploadFile(filename='a.png', file=io.BytesIO(b'oversized'))
        with patch.object(api, 'JOB_LOCK') as lock:
            lock.acquire.return_value = False
            with self.assertRaises(HTTPException) as error:
                await api.create_job(upload, True, 'Owned fixture')
            self.assertEqual(error.exception.status_code, 429)
            lock.release.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            upload = UploadFile(filename='a.png', file=io.BytesIO(b'oversized'))
            with patch.object(api, 'ROOT', Path(directory)), patch.object(api, 'MAX_BYTES', 2):
                with self.assertRaises(HTTPException) as error:
                    await api.create_job(upload, True, 'Owned fixture')
                self.assertEqual(error.exception.status_code, 413)
                self.assertFalse(api.JOB_LOCK.locked())
                self.assertEqual(list((Path(directory) / 'uploads').iterdir()), [])

    async def test_swap_uploads_with_baseline_explicitly_selected(self):
        from synthetic_media.swapping import LandmarkDetector
        with tempfile.TemporaryDirectory() as directory:
            image = np.full((160,160,3), 100, np.uint8)
            cv2.circle(image, (70,70),20,(240,20,180),-1)
            _, source = cv2.imencode('.png',image)
            _, target = cv2.imencode('.png',np.full_like(image,70))
            landmark = np.array([30,20,80,100,52,52,88,52,70,72,55,92,85,92,.99],np.float32)
            with patch.object(api, 'ROOT', Path(directory)), patch.object(LandmarkDetector, 'single_face', return_value=landmark):
                result = await api.swap_job(UploadFile(filename='source.png',file=io.BytesIO(source.tobytes())),
                        UploadFile(filename='target.png',file=io.BytesIO(target.tobytes())), True, 'Generated authorized fixtures', 'baseline')
                self.assertEqual(result['stage'],'face_swapping')
                self.assertFalse(result['analysis']['rendered']['generative'])
                self.assertTrue(Path(api.diagnostics(result['job_id']).path).is_file())
                self.assertEqual(list((Path(directory)/'uploads').iterdir()),[])
