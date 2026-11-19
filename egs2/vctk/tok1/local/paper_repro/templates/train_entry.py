"""Unmodified official training; atomically replace checkpoint files only."""
import os,runpy
from pathlib import Path
import torch
save=torch.save
def atomic_save(obj,path,*args,**kwargs):
    if isinstance(path,(str,Path)) and str(path).endswith('.pth'):
        path=Path(path);tmp=path.with_name(path.name+'.writing')
        save(obj,tmp,*args,**kwargs);os.replace(tmp,path)
    else:save(obj,path,*args,**kwargs)
torch.save=atomic_save
runpy.run_module('espnet2.bin.gan_tok_train',run_name='__main__')
