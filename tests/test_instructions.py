import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from synthetic_media.instructions import interpret,analyze_job,execute_job,person_mask
from synthetic_media.swapping import LandmarkDetector


class InstructionTests(unittest.TestCase):
    def test_face_of_person_is_not_whole_person_replacement(self):
        self.assertEqual(interpret('Swap just the face')['operation'],'face')
        self.assertEqual(interpret('Swap the face of the person')['operation'],'face')
        self.assertEqual(interpret('Replace the person with this reference')['operation'],'person')
        self.assertEqual(interpret('Replace the whole person')['operation'],'person')
        self.assertEqual(interpret('Replace the head and hair')['operation'],'head')
        self.assertEqual(interpret('Swap face but preserve the hair')['operation'],'face')

    def test_contradictory_or_negated_instructions_do_not_execute(self):
        for text,scope in [("Don't swap the face",'auto'),('Replace the person, swap only the face','auto'),
                           ('Swap the face','person'),('Make it better','auto'),('Swap face and change the background','auto')]:
            with self.subTest(text=text),self.assertRaises(ValueError):interpret(text,scope)

    def test_actual_instruction_reaches_generated_prompt(self):
        text='Replace the person with the reference, preserving the target sitting posture.'
        plan=interpret(text)
        self.assertIn(text,plan['generation_prompt'])
        self.assertIn('Image 1 is the target',plan['generation_prompt'])

    def test_analysis_creates_inspectable_region_and_warns_about_detail(self):
        face=np.array([30,20,80,100,52,52,88,52,70,72,55,92,85,92,.99],np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.png';target=root/'target.png'
            cv2.imwrite(str(source),np.full((160,160,3),80,np.uint8));cv2.imwrite(str(target),np.full((160,160,3),120,np.uint8))
            with patch.object(LandmarkDetector,'single_face',return_value=face):
                result=analyze_job(source,target,root/'jobs',True,'Owned fixtures','Swap just face')
            self.assertEqual(result['plan']['operation'],'face')
            self.assertTrue(result['plan']['can_generate'])
            self.assertTrue(result['plan']['warnings'])
            overlay=cv2.imread(str(root/'jobs'/result['job_id']/'edit-region.png'))
            np.testing.assert_array_equal(overlay[:10],cv2.imread(str(target))[:10])
            self.assertGreater(overlay.shape[0],160)

    def test_unsupported_scope_returns_blocked_without_running_face_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=root/'analysis';job.mkdir()
            analysis={'job_id':'analysis','plan':{'can_generate':False,'operation':'person'},'status':'analyzed'}
            with patch('synthetic_media.instructions.analyze_job',return_value=analysis),patch('synthetic_media.pipeline.run') as generate:
                result=execute_job(root/'ref',root/'target',root,True,'Owned fixtures','Replace the person')
            self.assertEqual(result['status'],'blocked');generate.assert_not_called()

    def test_external_edit_requires_explicit_service_opt_in(self):
        from synthetic_media.instructions import inspect
        face=np.array([30,20,80,100,52,52,88,52,70,72,55,92,85,92,.99],np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.png'
            cv2.imwrite(str(source),np.full((160,160,3),100,np.uint8))
            with patch.object(LandmarkDetector,'single_face',return_value=face):
                plan,*_=inspect(source,source,'Swap face','face',root,'shared-flux',False)
            self.assertFalse(plan['can_generate'])
            self.assertIn('external service',plan['blockers'][0])

    @unittest.skipUnless((Path(__file__).resolve().parents[1]/'.lab/models/person/pphumanseg.onnx').exists(),'Optional person model absent')
    def test_real_person_segmenter_rejects_empty_scene(self):
        with self.assertRaisesRegex(ValueError,'reliable editable region'):
            person_mask(np.zeros((160,160,3),np.uint8),np.array([30,20,80,100],np.float32))
