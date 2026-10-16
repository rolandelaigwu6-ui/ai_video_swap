"""Run the unmodified upstream CLI with a bounded, single-module inference cache.

All content checks, thresholds, model validation and CLI processing remain upstream.
Only ONNX CPU session settings and session-cache residency are constrained.
"""
import gc
import ctypes
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['PATH']=str(ROOT/'vendor/ffmpeg')+os.pathsep+os.environ.get('PATH','')
sys.path.insert(0,str(ROOT/'vendor/facefusion'))


def configure():
    os.chdir(ROOT/'vendor/facefusion')
    import onnxruntime as ort
    original_session=ort.InferenceSession
    class CpuBudgetSession(original_session):
        def __init__(self,*args,**kwargs):
            options=kwargs.pop('sess_options',None) or ort.SessionOptions()
            options.intra_op_num_threads=1
            options.inter_op_num_threads=1
            options.enable_cpu_mem_arena=False
            options.enable_mem_pattern=False
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            super().__init__(*args,sess_options=options,**kwargs)
    ort.InferenceSession=CpuBudgetSession
    from facefusion import inference_manager
    original_pool=inference_manager.get_inference_pool
    active=[None]
    def bounded_pool(module_name,model_names,model_sources):
        key=(module_name,tuple(model_names))
        if active[0]!=key:
            for cache in inference_manager.INFERENCE_POOL_SET.values():
                cache.clear()
            gc.collect()
            # Return freed model/session heap pages to Linux between modules.
            trim=getattr(ctypes.CDLL(None),'malloc_trim',None)
            if trim is not None:trim(0)
            active[0]=key
        return original_pool(module_name,model_names,model_sources)
    inference_manager.get_inference_pool=bounded_pool


if __name__=='__main__':
    configure()
    from facefusion import conda,core
    conda.setup()
    core.cli()
