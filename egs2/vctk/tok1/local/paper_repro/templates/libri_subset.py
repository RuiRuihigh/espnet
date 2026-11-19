"""Seeded 30h subset; archive exact IDs because paper's list is unavailable."""
import hashlib,json,random,time
from pathlib import Path
import soundfile as sf
R=Path(__file__).resolve().parent
source=Path('@LIBRI@')
out=R/'libri_data/libri30h';out.mkdir(parents=True,exist_ok=True)
if (out/'COMPLETE.json').exists():
    print('LIBRI30_REUSED',flush=True);raise SystemExit(0)
paths=sorted(source.glob('*/*/*.flac'));assert len(paths)>10000
random.Random(0).shuffle(paths);seconds=0;rows=[];transcripts={}
for p in paths:
    info=sf.info(p);assert info.samplerate==16000 and info.channels==1
    u=p.stem;spk,chapter,_=u.split('-');t=p.parent/f'{spk}-{chapter}.trans.txt'
    if t not in transcripts:transcripts[t]=dict(line.split(' ',1) for line in t.read_text().splitlines())
    rows.append({'utt':u,'path':str(p),'samples':info.frames,'seconds':info.duration,'speaker':spk,'text':transcripts[t][u]})
    seconds+=info.duration
    if len(rows)%500==0:print('LIBRI30_SELECT',len(rows),seconds/3600,flush=True)
    if seconds>=30*3600:break
assert 30*3600<=seconds<30*3600+60
rows.sort(key=lambda x:x['utt']);spk2utt={}
for r in rows:spk2utt.setdefault(r['speaker'],[]).append(r['utt'])
contents={'wav.scp':'\n'.join(f"{r['utt']} {r['path']}" for r in rows),
          'text':'\n'.join(f"{r['utt']} {r['text']}" for r in rows),
          'utt2spk':'\n'.join(f"{r['utt']} {r['speaker']}" for r in rows),
          'spk2utt':'\n'.join(f"{s} {' '.join(us)}" for s,us in sorted(spk2utt.items())),
          'utt2num_samples':'\n'.join(f"{r['utt']} {r['samples']}" for r in rows)}
for name,txt in contents.items():(out/name).write_text(txt+'\n')
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
(out/'COMPLETE.json').write_text(json.dumps({'seed':0,'source':str(source),'hours':seconds/3600,'utterances':len(rows),
 'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
 'original_author_subset_unknown':True,'includes_test_or_dev':False},indent=2))
print('LIBRI30_COMPLETE',len(rows),seconds/3600,flush=True)
