"""Unmodified official training; atomically replace checkpoint files only."""
import os
from pathlib import Path
import torch
save=torch.save
def atomic_save(obj,path,*args,**kwargs):
    if isinstance(path,(str,Path)) and str(path).endswith('.pth'):
        path=Path(path);tmp=path.with_name(path.name+'.writing')
        save(obj,tmp,*args,**kwargs);os.replace(tmp,path)
    else:save(obj,path,*args,**kwargs)
torch.save=atomic_save
# Spawned DDP workers import this file as __mp_main__. Install atomic saves
# in each worker, but never start a second trainer while importing.
if __name__ == '__main__':
    from espnet2.bin.gan_tok_train import main
    main()
