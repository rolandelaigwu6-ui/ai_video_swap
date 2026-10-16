"""Transparent diagnostics, never a claim of perceptual perfection."""
import cv2
import numpy as np


def evaluate_identity(reference,target,result,source_face,target_face,folder):
    from .neural import align,run_worker,model_dir,status,verify_models
    from .swapping import LandmarkDetector,model_path
    if not status()['available']:
        return {'status':'needs_review','visual_acceptance':'unverified','flawless_certified':False,
                'issues':['Local identity evaluator is unavailable; likeness has not been measured']}
    verify_models()
    embeddings=[]
    for image,face in [(reference,source_face),(target,target_face),(result,target_face)]:
        crop,_=align(image,face[4:14].reshape(5,2),112)
        np.save(folder/'source.npy',((crop[:,:,::-1].astype(np.float32)/127.5-1).transpose(2,0,1))[None])
        run_worker('embed',model_dir()/'arcface_w600k_r50.onnx',folder)
        embeddings.append(np.load(folder/'embedding.npy',allow_pickle=False).copy())
    try:detected=LandmarkDetector(model_path()).single_face(result,'Generated image')
    except ValueError:detected=None
    return assess(reference,target,result,source_face,target_face,*embeddings,detected)


def assess(reference, target, result, source_face, target_face, source_embedding,
           target_embedding, result_embedding, result_face=None):
    cosine = lambda a,b: float(np.sum(a*b))
    source_result = cosine(source_embedding,result_embedding)
    source_target = cosine(source_embedding,target_embedding)
    issues = []
    if source_result < .65:
        issues.append('Reference-identity similarity is below the research threshold')
    if source_target < .65 and source_result - source_target < .15:
        issues.append('Identity transfer did not improve enough over the original target')
    detail = {}
    for role, face, image in [('reference',source_face,reference),('target',target_face,target)]:
        eye_distance = float(np.linalg.norm(face[4:6]-face[6:8]))
        x,y,w,h=face[:4]
        crop=image[max(0,int(y)):min(image.shape[0],int(y+h)),max(0,int(x)):min(image.shape[1],int(x+w))]
        sharpness = float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()) if crop.size else 0
        detail[role]={'eye_distance_pixels':eye_distance,'laplacian_variance':sharpness}
        if eye_distance < 40:
            issues.append(f'{role} face has fewer than 40 pixels between eyes; fine identity detail is unreliable')
        if sharpness < 15:
            issues.append(f'{role} face has low measured sharpness')
    landmark_drift = None
    if result_face is None:
        issues.append('Generated face could not be independently redetected')
    else:
        target_points = target_face[4:14].reshape(5,2)
        output_points = result_face[4:14].reshape(5,2)
        eye_distance = max(float(np.linalg.norm(target_points[0]-target_points[1])),1)
        landmark_drift = float(np.linalg.norm(output_points-target_points,axis=1).mean()/eye_distance)
        if landmark_drift > .25:
            issues.append('Generated face landmarks drifted substantially from the target')
    if not np.isfinite(result).all() or result.shape != target.shape:
        issues.append('Invalid output dimensions or pixel values')
    critical_failure=(source_result<.65 or (source_target<.65 and source_result-source_target<.15)
                      or result_face is None or (landmark_drift is not None and landmark_drift>.25))
    return {'status':'needs_review' if issues else 'technical_checks_passed',
            'automatic_acceptance':'rejected' if critical_failure else 'requires_visual_review',
            'visual_acceptance':'unverified', 'flawless_certified':False,
            'identity_cosine':source_result,'original_identity_cosine':source_target,
            'identity_gain':source_result-source_target, 'normalized_landmark_drift':landmark_drift,'input_detail':detail,'issues':issues,
            'limitations':'Same recognizer conditions and evaluates the generator; these heuristic checks do not establish flawless output.'}
