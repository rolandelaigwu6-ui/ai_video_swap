import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import HTTPException, UploadFile
from synthetic_media import api, shared
from synthetic_media.service_errors import service_error, public_manifest

QUOTA='Upstream processor failed (1): Traceback: You have exceeded your ZeroGPU runs limit. Authenticate with a Hugging Face token for more quota'

class ServiceErrorTests(unittest.IsolatedAsyncioTestCase):
    def test_old_manifest_hides_traceback_and_identifies_quota(self):
        result=public_manifest({'error':QUOTA})
        self.assertEqual(result['error_code'],'shared_quota_exhausted')
        self.assertNotIn('Traceback',result['error'])
        self.assertIn('hf auth login',result['error'])

    def test_worker_failure_preserves_log_and_safe_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); folder=root/'worker';folder.mkdir()
            (folder/'network.log').write_text(QUOTA)
            with patch('synthetic_media.upstream.execute',side_effect=RuntimeError(QUOTA)):
                with self.assertRaises(RuntimeError) as error:
                    shared.request(root/'reference.png',root/'target.png',folder,
                        SimpleNamespace(instruction='swap face',edit_scope='face',output_dir=root),'shared-flux')
            self.assertNotIn('Traceback',str(error.exception))
            self.assertEqual((root/'shared-service.log').read_text(),QUOTA)

    async def test_api_returns_actionable_429_and_cleans_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(api,'ROOT',root),patch.object(api,'run',side_effect=service_error(QUOTA)):
                with self.assertRaises(HTTPException) as error:
                    await api.create_job(UploadFile(filename='a.png',file=io.BytesIO(b'fixture')),True,'Owned fixture')
            self.assertEqual(error.exception.status_code,429)
            self.assertEqual(error.exception.detail['code'],'shared_quota_exhausted')
            self.assertNotIn('Traceback',str(error.exception.detail))
            self.assertEqual(list((root/'uploads').iterdir()),[])
            self.assertFalse(api.JOB_LOCK.locked())
