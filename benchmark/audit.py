#!/usr/bin/env python3
"""Audit benchmark JSONL. Standard-library checks; sklearn probes run when installed."""
import argparse, collections, json, re, string, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = [x['intent'] for x in json.loads((ROOT.parent/'bge-intent/intents.json').read_text())]
SLOT_LABELS = {'Get_coordinates','Set_navigation_target','Add_waypoint','Delete_waypoint','Complete_task'}
FILLERS = {'uh','um','erm','er','well','okay','ok'}

def norm(s):
    s=s.lower()
    s=re.sub(r'\b(uh|um|erm|er|well|okay|ok)\b',' ',s)
    return ' '.join(re.sub(r'[^a-z0-9 ]',' ',s).split())
def grams(s):
    s='  '+norm(s)+'  '
    return set(s[i:i+3] for i in range(max(0,len(s)-2)))
def jac(a,b):
    x,y=grams(a),grams(b)
    return len(x&y)/len(x|y) if x|y else 1

def read(paths):
    out=[]
    for p in paths:
        for n,line in enumerate(p.read_text(encoding='utf8').splitlines(),1):
            if not line.strip(): continue
            try: r=json.loads(line)
            except Exception as e: print(f'ERROR {p}:{n}: invalid JSON ({e})'); continue
            r['_file']=str(p); r['_line']=n; out.append(r)
    return out

def style(r):
    s=r.get('text',''); words=s.split()
    has_punctuation=any(c in string.punctuation for c in s)
    lower=s==s.lower()
    asr=bool(re.search(r'\b(?:oh two|see oh two|c o 2|c o two)\b',s.lower())) or bool(re.search(r'\b[a-z]+\s+[a-z]+\s+\d+\b',s.lower()))
    return [len(words), int('?' in s), sum(w.strip(string.punctuation).lower() in FILLERS for w in words), int(not has_punctuation), int(lower), int(asr), sum(c.isdigit() for c in s)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('paths',nargs='*'); ap.add_argument('--split-check',action='store_true'); ap.add_argument('--partial',action='store_true',help='audit a generation round before all 100 rows per label exist'); a=ap.parse_args()
    paths=[Path(x) for x in a.paths] or sorted((ROOT/'parts').glob('*.jsonl'))
    rows=read(paths); errors=[]; counts=collections.Counter(); seen={}; openings=collections.defaultdict(collections.Counter); lengths=collections.defaultdict(collections.Counter); stylev=collections.defaultdict(list); allstyle=[]; texts=[]; ys=[]
    names=set(json.loads((ROOT/'names.json').read_text())); name_usage=collections.defaultdict(collections.Counter)
    for i,r in enumerate(rows):
        t=r.get('text'); it=r.get('intents')
        if not isinstance(t,str) or not t.strip(): errors.append(f'row {i}: empty text'); continue
        if not isinstance(it,list) or len(it)!=1 or it[0] not in LABELS: errors.append(f'row {i}: invalid intents {it!r}'); continue
        if r.get('source')!='llm': errors.append(f'row {i}: source must be "llm"')
        y=it[0]; counts[y]+=1; texts.append(t); ys.append(y); st=style(r); stylev[y].append(st); allstyle.append(st)
        nt=norm(t)
        if nt in seen:
            old=rows[seen[nt]]
            errors.append(f'duplicate {old["_file"]}:{old["_line"]} <> {r["_file"]}:{r["_line"]}')
        seen[nt]=i
        w=nt.split(); openings[y][' '.join(w[:2])]+=1; lengths[y][0 if len(w)<=3 else 1 if len(w)<=7 else 2 if len(w)<=12 else 3]+=1
        if y in SLOT_LABELS:
            slot=r.get('slot')
            if slot not in names or slot not in t: errors.append(f'row {i}: invalid/missing verbatim slot {slot!r}')
            elif not a.partial: name_usage[slot][y]+=1
    for y in LABELS:
        if not a.partial and counts[y]!=100: errors.append(f'{y}: count {counts[y]} != 100')
        if not a.partial and len(openings[y])<40: errors.append(f'{y}: only {len(openings[y])} distinct openings')
        if openings[y] and max(openings[y].values())>8: errors.append(f'{y}: opening frequency >8')
        if not a.partial and any(lengths[y][b]==0 for b in range(4)): errors.append(f'{y}: missing length bucket')
    if not a.partial:
        for name in names:
            if len(name_usage[name])<2: errors.append(f'name {name!r}: used in fewer than two name intents')
            if any(n>3 for n in name_usage[name].values()): errors.append(f'name {name!r}: used more than three times in one name intent')
    # Similarity check is quadratic but manageable at 3600 rows.
    gram_cache=[grams(t) for t in texts]
    for i in range(len(rows)):
        if i>=len(ys): continue
        for j in range(i):
            if j>=len(ys): continue
            x,y=gram_cache[i],gram_cache[j]
            sim=len(x&y)/len(x|y) if x|y else 1
            if sim>=.8 and ys[i]==ys[j]: errors.append(f'near duplicate same label {rows[j]["_file"]}:{rows[j]["_line"]} <> {rows[i]["_file"]}:{rows[i]["_line"]}: {sim:.3f}')
            elif sim>=.7 and ys[i]!=ys[j]: errors.append(f'cross-label near duplicate {rows[j]["_file"]}:{rows[j]["_line"]} <> {rows[i]["_file"]}:{rows[i]["_line"]}: {sim:.3f}')
    mean_len=sum(st[0] for st in allstyle)/max(1,len(allstyle))
    if not a.partial:
        for y in LABELS:
            local=stylev[y]
            if local and abs(sum(x[0] for x in local)/len(local)-mean_len)>2: errors.append(f'{y}: mean length outside ±2 words')
    rates=[]
    features=[(lambda st:st[1]>0,'question'),(lambda st:st[2]>0,'filler'),
              (lambda st:st[3]>0 or st[4]>0,'no_punctuation_or_lowercase'),(lambda st:st[5]>0,'asr')]
    for feature,name in features:
        glob=sum(feature(st) for st in allstyle)/max(1,len(allstyle))
        diffs={y:sum(feature(st) for st in stylev[y])/max(1,len(stylev[y]))-glob for y in LABELS}
        rates.append((name,glob,max((abs(d) for d in diffs.values()),default=0)))
        if not a.partial:
            for y,d in diffs.items():
                if abs(d)>.15: errors.append(f'{y}: {name} rate differs by {d:+.3f}')
    print(f'rows={len(rows)} labels={len(counts)} mean_words={mean_len:.2f}')
    for y in LABELS: print(f'COUNT {y}: {counts[y]}')
    print('STYLE_RATES '+json.dumps(rates))
    print('INTENT_STYLE '+json.dumps({
        label: {
            name: round(sum(feature(st) for st in stylev[label]) / max(1, len(stylev[label])), 3)
            for feature, name in features
        }
        for label in LABELS if stylev[label]
    }))
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        if len(set(ys))>=2 and min(collections.Counter(ys).values())>=5:
            X=np.array([style(r) for r in rows if isinstance(r.get('intents'),list) and r['intents']]); Y=np.array([r['intents'][0] for r in rows if isinstance(r.get('intents'),list) and r['intents']])
            cv=StratifiedKFold(5,shuffle=True,random_state=42)
            model=LogisticRegression(max_iter=1500)
            score=float(cross_val_score(model,X,Y,cv=cv).mean()); print(f'STYLE_PROBE accuracy={score:.4f}')
            if not a.partial and score>=.15: errors.append(f'style probe accuracy {score:.4f} >= .15')
            pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1),LogisticRegression(max_iter=1500))
            sc=float(cross_val_score(pipe,texts,ys,cv=cv).mean()); print(f'CONTENT_PROBE accuracy={sc:.4f}')
            if not a.partial and sc>.99: errors.append(f'content probe accuracy {sc:.4f} > .99')
            from sklearn.model_selection import cross_val_predict
            pred=cross_val_predict(pipe,texts,ys,cv=cv); pairs=collections.Counter((a,b) for a,b in zip(ys,pred) if a!=b)
            print('TOP_CONFUSIONS '+json.dumps(pairs.most_common(10)))
        else: print('PROBES deferred: each present class needs at least five rows')
    except ImportError: print('PROBES unavailable: scikit-learn/numpy not installed')
    if a.split_check and len(paths)==3:
        splitrows=[read([p]) for p in paths]; expected=[80,10,10]
        for part,need,rs in zip(('train','val','test'),expected,splitrows):
            c=collections.Counter(r.get('intents',[''])[0] for r in rs)
            if any(c[y]!=need for y in LABELS): errors.append(f'{part}: expected {need} per intent')
        if len(splitrows)==3:
            train=splitrows[0]
            for part,rs in zip(('val','test'),splitrows[1:]):
                for r in rs:
                    same=[t for t in train if t.get('intents')==r.get('intents')]
                    if any(jac(r['text'],t['text'])>=.65 for t in same): errors.append(f'{part} leakage: {r["text"]!r}')
    print(f'GATES: {len(errors)} issue(s)')
    for e in errors: print('FAIL '+e)
    return bool(errors)
if __name__=='__main__': sys.exit(main())
