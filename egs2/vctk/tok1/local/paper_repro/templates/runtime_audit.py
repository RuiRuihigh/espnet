import json,os,sys,subprocess,importlib.metadata,time
from pathlib import Path
R=Path(__file__).resolve().parent
print('RUNTIME_IMPORT_START',flush=True)
import torch,numpy,soundfile,kaldiio,sentencepiece,sklearn
import espnet2.tasks.gan_tok as task
assert str(Path(task.__file__).resolve()).startswith(str(R/'espnet'))
parser=task.GANSpeechTokenizerTask.get_parser()
args=parser.parse_args(['--config',str(R/'espnet/egs2/vctk/tok1/conf/train_paper.yaml')])
versions={k:importlib.metadata.version(k) for k in ['torch','torchaudio','s3prl','numpy','scikit-learn','sentencepiece']}
report={'versions':versions,'python':sys.version,'official_task_import':task.__file__,'config_parser_passed':True,
 'centroids_not_ready_yet':True,'sox':subprocess.check_output(['sox','--version'],text=True).strip()}
(R/'SETUP_OK.json').write_text(json.dumps(report,indent=2));print('RUNTIME_OK',json.dumps(report),flush=True)
