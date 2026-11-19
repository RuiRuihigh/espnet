import json
from pathlib import Path
R=Path(__file__).resolve().parent;report={}
for s in ['tr_no_dev','tr_no_dev_sp','dev','eval1']:
    d=R/'dump/raw'/s
    scp=dict(l.split(maxsplit=1) for l in (d/'wav.scp').read_text().splitlines())
    text=dict(l.split(maxsplit=1) for l in (d/'text').read_text().splitlines())
    assert set(scp)==set(text)
    nums=dict(l.split() for l in (d/'utt2num_samples').read_text().splitlines())
    assert set(scp)==set(nums)
    report[s]={'utterances':len(scp),'hours':sum(int(v) for v in nums.values())/16000/3600}
base=set(dict(l.split(maxsplit=1) for l in (R/'dump/raw/tr_no_dev/wav.scp').read_text().splitlines()))
dev=set(dict(l.split(maxsplit=1) for l in (R/'dump/raw/dev/wav.scp').read_text().splitlines()))
test=set(dict(l.split(maxsplit=1) for l in (R/'dump/raw/eval1/wav.scp').read_text().splitlines()))
assert not (base&dev or base&test or dev&test)
tokens=R/'espnet/egs2/vctk/tok1/data/noinfo_token_list/bpe_unigram5000/tokens.txt'
assert len(tokens.read_text().splitlines())>=4999
report['split_overlap']=False;report['token_list_size']=len(tokens.read_text().splitlines())
(R/'DATA_READY.json').write_text(json.dumps(report,indent=2));print('DATA_AUDIT',json.dumps(report),flush=True)
