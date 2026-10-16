import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from synthetic_media.quality import assess
from synthetic_media.masking import semantic_mask


class QualityTests(unittest.TestCase):
    def test_semantic_compositor_excludes_accessories_and_background(self):
        def worker(mode, model, folder):
            if mode=='parse':
                labels=np.zeros((512,512),np.uint8)
                labels[80:430,100:430]=1
                labels[200:240,100:430]=6 # glasses must stay from target
                np.save(folder/'parse.npy',labels)
            else:
                np.save(folder/'occlude.npy',np.ones((256,256),np.float32))
            return {'stage':mode}
        with tempfile.TemporaryDirectory() as directory, patch('synthetic_media.masking.model_info', return_value={'test':True}):
            crop=np.full((128,128,3),127,np.uint8)
            mask,metadata=semantic_mask(crop,crop,Path(directory),worker)
            self.assertEqual(float(mask[54,60]),0)
            self.assertEqual(float(mask[0,0]),0)
            self.assertGreater(float(mask[80,60]),.9)
            self.assertEqual(len(metadata['worker_measurements']),3)

    def test_mask_failure_does_not_silently_use_box_mask(self):
        def worker(mode, model, folder):
            if mode=='parse':
                np.save(folder/'parse.npy',np.zeros((512,512),np.uint8))
            else:
                np.save(folder/'occlude.npy',np.ones((256,256),np.float32))
            return {}
        with tempfile.TemporaryDirectory() as directory, patch('synthetic_media.masking.model_info', return_value={}):
            with self.assertRaisesRegex(ValueError,'implausible face coverage'):
                semantic_mask(np.zeros((128,128,3),np.uint8),np.zeros((128,128,3),np.uint8),Path(directory),worker)

    def test_quality_flags_poor_identity_and_insufficient_detail(self):
        face=np.array([30,20,80,100,52,52,88,52,70,72,55,92,85,92,.99],np.float32)
        image=np.full((160,160,3),70,np.uint8)
        reference=np.zeros((1,512),np.float32);reference[0,0]=1
        other=np.zeros_like(reference);other[0,1]=1
        report=assess(image,image,image,face,face,reference,other,other,None)
        self.assertEqual(report['status'],'needs_review')
        self.assertFalse(report['flawless_certified'])
        self.assertGreaterEqual(len(report['issues']),4)
        self.assertEqual(report['visual_acceptance'],'unverified')

    def test_profile_parser_rejection_uses_reported_occlusion_mask(self):
        def worker(mode,model,folder):
            if mode=='parse':
                np.save(folder/'parse.npy',np.zeros((512,512),np.uint8))
            else:
                visible=np.zeros((256,256),np.float32);visible[40:220,50:210]=1
                np.save(folder/'occlude.npy',visible)
            return {}
        with tempfile.TemporaryDirectory() as directory, patch('synthetic_media.masking.model_info',return_value={}):
            image=np.zeros((128,128,3),np.uint8)
            mask,metadata=semantic_mask(image,image,Path(directory),worker)
            self.assertGreater(mask[64,64],.9)
            self.assertEqual(metadata['method'],'occlusion-only-parser-rejected')
            self.assertEqual(len(metadata['limitations']),2)

    def test_candidate_selection_records_failure_and_keeps_neural_result(self):
        from synthetic_media.neural import BestNeuralFaceSwapper
        from synthetic_media.components import Context
        image=np.zeros((128,128,3),np.uint8)
        metadata={'backend':'test-neural','quality_report':{'issues':['low detail'],'identity_cosine':.8},'compositing':{},'worker_measurements':[]}
        with tempfile.TemporaryDirectory() as directory, patch('synthetic_media.neural.NeuralFaceSwapper.transform',return_value=(image,metadata)), patch('synthetic_media.neural.HyperSwapFaceSwapper.transform',side_effect=RuntimeError('memory limit')):
            result,report=BestNeuralFaceSwapper().transform(image,image,None,None,Context('test',Path(directory)))
            self.assertEqual(report['selected_backend'],'test-neural')
            self.assertEqual(report['failed_candidates'][0]['error'],'memory limit')
            self.assertIs(result,image)

    def test_small_reference_is_processed_with_warning(self):
        import cv2
        from synthetic_media.neural import NeuralFaceSwapper
        from synthetic_media.swapping import LandmarkDetector
        from synthetic_media.components import Context,Artifact
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'fixture.png'
            image=np.full((160,160,3),100,np.uint8)
            cv2.imwrite(str(path),image)
            face=np.array([30,20,80,100,52,52,70,52,63,72,55,92,72,92,.99],np.float32)
            with patch.object(LandmarkDetector,'single_face',return_value=face),patch.object(NeuralFaceSwapper,'transform',return_value=(image.copy(),{})) as transform:
                artifacts=NeuralFaceSwapper().process({'reference':Artifact(path,'reference'),'target':Artifact(path,'target')},Context('test',root))
                transform.assert_called_once()
                self.assertTrue(artifacts['rendered'].metadata['warnings'])
