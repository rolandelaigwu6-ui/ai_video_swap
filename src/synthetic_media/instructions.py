"""Explicit scope routing plus measured image analysis; not a foundation model."""
import json
from pathlib import Path
import re
import tempfile
from uuid import uuid4
import hashlib
import cv2
import numpy as np
from .swapping import LandmarkDetector, model_path, read_working_image
from .scene_edit import head_mask
from .media import mark_footer

PERSON_SHA = '552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24'


def interpret(instruction, scope='auto'):
    text=' '.join(instruction.lower().replace("don't",'do not').split())
    if not text or len(text)>2000:raise ValueError('Describe the edit in 1–2000 characters')
    if scope not in ('auto','face','head','person'):raise ValueError('Unknown edit scope')
    if re.search(r'\b(change|replace|remove)\b.{0,15}\bbackground\b',text):
        raise ValueError('This workflow preserves the background; remove the background-change request')
    person=bool(re.search(r'\b(replace|swap|change)\s+(?:(?:the|this|entire|whole|full|target)\s+)*(person|body|subject)\b',text) or re.search(r'\b(whole|entire|full)\s+(person|body)\b',text))
    face=bool(re.search(r'\b(face|facial)\b',text))
    head=bool(re.search(r'\b(head|hair|hairstyle)\b',text))
    if re.search(r'\b(do not|don t|never)\s+(swap|replace|change)',text):
        raise ValueError('The instruction negates the edit; specify the part you want replaced')
    if person and re.search(r'\b(only|just)\s+(the\s+)?face\b',text):
        raise ValueError('Choose face-only replacement or whole-person replacement; the instruction requests both')
    detected='person' if person else 'head' if head and not re.search(r'\b(keep|preserve)\b.{0,15}\b(hair|head|hairstyle)\b',text) else 'face' if face else None
    if scope!='auto':
        if detected is not None and detected!=scope:
            raise ValueError('The selected scope conflicts with the instruction')
        detected=scope
    if detected is None:raise ValueError('Specify whether to swap the face, replace the head, or replace the person')
    return {'operation':detected,'instruction':instruction.strip(),
            'interpreter':'constrained rule-based scope routing; not general language understanding',
            'preserve':['background','camera framing','target pose'],
            'generation_prompt':f'Image 1 is the target scene. Image 2 is the authorized reference identity. {instruction.strip()} Preserve the target camera framing, background, pose, expression and lighting. Produce one natural photograph.'}


def person_mask(target, face):
    path=Path(__file__).resolve().parents[2]/'.lab/models/person/pphumanseg.onnx'
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=PERSON_SHA:
        raise ValueError('The verified person-segmentation model is not installed')
    rgb=cv2.resize(target,(192,192))[:,:,::-1].astype(np.float32)/127.5-1
    net=cv2.dnn.readNet(str(path));net.setInput(rgb.transpose(2,0,1)[None])
    scores=net.forward()[0].transpose(1,2,0)
    scores=cv2.resize(scores,target.shape[:2][::-1])
    binary=(scores.argmax(axis=2)==1).astype(np.uint8)
    coverage=float(binary.mean())
    x,y,w,h=face[:4];cx,cy=int(x+w/2),int(y+h/2)
    if not .02<coverage<.98 or not binary[min(cy,target.shape[0]-1),min(cx,target.shape[1]-1)]:
        raise ValueError('Person segmentation did not produce a reliable editable region')
    distance=cv2.distanceTransform(binary,cv2.DIST_L2,3)
    mask=np.clip(distance/3,0,1)
    return mask,{'method':'PPHumanSeg person silhouette','coverage':coverage,'model_sha256':PERSON_SHA}


def edit_mask(target,face,scope,folder):
    if scope=='person':return person_mask(target,face)
    if scope=='head':
        from .neural import run_worker
        return head_mask(target,face,folder,run_worker)
    mask=np.zeros(target.shape[:2],np.float32)
    x,y,w,h=face[:4]
    mask[max(0,int(y)):min(target.shape[0],int(np.ceil(y+h))),
         max(0,int(x)):min(target.shape[1],int(np.ceil(x+w)))]=1
    return mask,{'method':'detected face rectangle'}


def inspect(reference_path,target_path,instruction,scope,folder,backend='auto',allow_external=False):
    plan=interpret(instruction,scope)
    reference,_=read_working_image(reference_path)
    target,_=read_working_image(target_path,max_edge=None)
    detector=LandmarkDetector(model_path())
    source_face=detector.single_face(reference,'Reference image')
    target_face=detector.single_face(target,'Target image')
    mask,segmentation=edit_mask(target,target_face,plan['operation'],folder)
    measures={};warnings=[]
    for role,face in [('reference',source_face),('target',target_face)]:
        eyes=float(np.linalg.norm(face[4:6]-face[6:8]))
        eye_vector=face[6:8]-face[4:6]
        measures[role]={'face_box':face[:4].tolist(),'detection_confidence':float(face[14]),'eye_distance_pixels':eyes,
                        'eye_line_roll_degrees':float(np.degrees(np.arctan2(eye_vector[1],eye_vector[0]))),
                        'nose_offset_proxy':float((face[8]-(face[4]+face[6])/2)/max(eyes,1))}
        if eyes<40:warnings.append(f'{role}: face detail is limited ({eyes:.1f} pixels between eyes)')
    blockers=[]
    local=backend in ('auto','neural','baseline','best','facefusion','hyperswap')
    if local and re.search(r'\b(remove|add|recolor|colour|colorize|make|beautify)\b',instruction.lower()):
        blockers.append('The local swapper cannot execute extra appearance commands; use the shared instruction editor')
    if backend in ('auto','neural','baseline','best','facefusion','hyperswap') and plan['operation']!='face':
        blockers.append('Head/person replacement requires a reference-conditioned editor; select a shared GPU editor')
    if backend in ('shared','shared-flux','shared-head','remote') and not allow_external:
        blockers.append('Enable sending these images to the selected external service before generation')
    if backend=='shared-head' and plan['operation']!='head':
        blockers.append('The trained head-swap service supports head replacement only')
    if backend=='shared-head' and re.search(r'\b(remove|add|recolor|make|change)\b',instruction.lower()):
        blockers.append('The head preset cannot follow extra appearance instructions; select an instruction editor')
    if backend=='shared':
        from .shared import login_available
        if not login_available():blockers.append('Sign in locally with .venv/bin/hf auth login for the free instruction editor GPU quota')
    if backend=='remote':
        blockers.append('Use the shared instruction editor for text-controlled edits; the private server template is head-only')
    plan.update(analysis=measures,warnings=warnings,blockers=blockers,can_generate=not blockers,
                instruction_execution='Face-swap preset selected by scope; not arbitrary text editing' if local else 'Full instruction sent to editor' if backend in ('shared','shared-flux') else 'Fixed trained head-replacement preset; not arbitrary text editing',
                backend=backend,segmentation=segmentation,editable_pixels=int((mask>0).sum()),
                analysis_methods=['YuNet face landmarks',segmentation['method']],
                limitations=['Scope routing is limited; the shared editor receives the actual instruction',
                             'Segmentation does not prove identity or pose fidelity; inspect the proposed region'])
    return plan,reference,target,source_face,target_face,mask


def analyze_job(reference,target,root,authorized,note,instruction,scope='auto',backend='auto',allow_external=False):
    if not authorized or not note.strip():raise ValueError('Authorization covering both inputs is required')
    job_id=uuid4().hex;folder=root/job_id;folder.mkdir(parents=True)
    try:
        plan,_,image,_,_,mask=inspect(reference,target,instruction,scope,folder,backend,allow_external)
        overlay=image.copy();selected=mask>0
        overlay[selected]=(.55*image[selected]+.45*np.array([210,90,160])).astype(np.uint8)
        cv2.imwrite(str(folder/'edit-region.png'),mark_footer(overlay))
        cv2.imwrite(str(folder/'edit-mask.png'),(mask*255).astype(np.uint8))
        manifest={'job_id':job_id,'stage':'instruction_analysis','status':'analyzed',
                  'synthetic':True,'authorization':{'attested':True,'note':note},'plan':plan,
                  'outputs':{'labeled_preview':'edit-region.png','edit_mask':'edit-mask.png'},
                  'links':{'manifest':f'/jobs/{job_id}','preview':f'/jobs/{job_id}/preview'}}
    except Exception as error:
        manifest={'job_id':job_id,'stage':'instruction_analysis','status':'failed','error':str(error),'outputs':{}}
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));raise
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def execute_job(reference,target,root,authorized,note,instruction,scope='auto',backend='auto',allow_external=False):
    from .pipeline import run
    analysis=analyze_job(reference,target,root,authorized,note,instruction,scope,backend,allow_external)
    plan=analysis['plan']
    if not plan['can_generate']:
        analysis['status']='blocked'
        (root/analysis['job_id']/'manifest.json').write_text(json.dumps(analysis,indent=2))
        return analysis
    result=run(reference,root,authorized,note,'face_swapping',target,backend,False,instruction,plan['operation'])
    result['plan']=plan;result['links']['edit_region']=analysis['links']['preview']
    (root/result['job_id']/'manifest.json').write_text(json.dumps(result,indent=2))
    return result
