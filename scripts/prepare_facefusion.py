"""Create and verify only the models needed by the chosen upstream processor."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import facefusion_runtime
facefusion_runtime.configure()
from facefusion import state_manager
from facefusion.args import apply_args
from facefusion.program import create_program
from facefusion.processors.modules.face_swapper import core as swapper

args=['headless-run','--processors','face_swapper','--face-swapper-model','hyperswap_1a_256',
      '--face-detector-model','yolo_face','--face-landmarker-model','peppa_wutz',
      '--execution-providers','cpu','--execution-thread-count','1','--video-memory-strategy','strict']
sys.argv=['facefusion.py']+args
program=create_program()
values=vars(program.parse_args(args))
apply_args(values,state_manager.init_item)
plan={}
for module in swapper.get_common_modules()+[swapper]:
    if hasattr(module,'collect_model_downloads'):
        hashes,sources=module.collect_model_downloads()
    else:
        options=module.get_model_options()
        hashes,sources=options['hashes'],options['sources']
    for key,source in sources.items():
        path=Path(source['path']).resolve()
        plan[path.name]={'url':source['url'],'path':str(path),'hash_url':hashes[key]['url'],
                         'hash_path':str(Path(hashes[key]['path']).resolve())}
path=ROOT/'vendor/facefusion/model-plan.json'
path.write_text(json.dumps(plan,indent=2))
print(json.dumps({'required_models':list(plan),'plan':str(path)},indent=2))
