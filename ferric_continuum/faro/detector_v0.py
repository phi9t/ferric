"""Historical v0 offline detector, not the later validated host-progress detector.
No elapsed-time gate; streak can bridge missing windows. Use complete timelines.
Fixed exploratory settings, not production calibration.
Input flat records: case,rank,step,region,kind,duration_ms. Baseline case=clean.
No treatment labels used in scoring. Pairwise contrast is not causal attribution.
"""
import argparse,json,math,statistics,itertools

def quantile(xs,q):
    a=sorted(xs);return a[min(len(a)-1,int(math.ceil(q*len(a)))-1)]

def detect(rows,baseline='clean',window=8,occupancy=.75,onset=2,relative=.15,absolute_ms=1.,min_span_ms=0):
    groups={}
    for r in rows:
        x=r['duration_ms']
        if not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 or r.get('valid',True) is False:continue
        k=(r['case'],r['kind'],r['region'])
        z=groups.setdefault(k,{})
        identity=(r['rank'],r['step'])
        if identity in z:raise ValueError('duplicate rank/step in semantic region')
        z[identity]=float(x)
    out=[]
    for (case,kind,region),vals in sorted(groups.items()):
        if case==baseline:continue
        clean=groups.get((baseline,kind,region),{})
        ranks=sorted({r for r,s in vals})
        for r,q in itertools.permutations(ranks,2):
            bs=sorted({s for rr,s in clean if rr==r}&{s for rr,s in clean if rr==q})
            if len(bs)<window:continue
            bl=[math.log(clean[r,s]/clean[q,s]) for s in bs];bd=[clean[r,s]-clean[q,s] for s in bs]
            b=statistics.median(bl);a=statistics.median(bd)
            tl=max(math.log1p(relative),quantile([v-b for v in bl],.95));td=max(absolute_ms,quantile([v-a for v in bd],.95))
            steps=sorted({s for rr,s in vals if rr==r}&{s for rr,s in vals if rr==q})
            windows=[];streak=0;opened=None
            for start in range(0,len(steps),window):
                ss=steps[start:start+window]
                if len(ss)<window:break
                # Require consecutive logical steps; missing observations cannot masquerade as persistence.
                valid=ss[-1]-ss[0]==window-1
                es=[math.log(vals[r,s]/vals[q,s])-b for s in ss];hs=[vals[r,s]-vals[q,s]-a for s in ss]
                occ=sum(e>tl and h>td for e,h in zip(es,hs))/window
                passed=valid and occ>=occupancy and statistics.median(es)>tl and statistics.median(hs)>td
                if min_span_ms:raise ValueError('elapsed span requires availability timestamps, unavailable in flat schema')
                streak=streak+1 if passed else 0
                if streak>=onset and opened is None:opened=ss[-1]
                windows.append(dict(steps=[ss[0],ss[-1]],occupancy=occ,median_ratio_residual=math.exp(statistics.median(es)),median_excess_ms=statistics.median(hs),passed=passed))
            if opened is not None:
                out.append(dict(case=case,kind=kind,region=region,comparison_ranks=[r,q],long_duration_rank=r,reference_rank=q,candidate_origin_ranks=[r,q],observed_affected_ranks=[r] if kind=='comm' else [],classification=kind+'-path disparity; causal origin unresolved',onset_step=opened,threshold_relative=math.expm1(tl),threshold_ms=td,windows=windows))
    return dict(status='exploratory offline scoring; not held-out calibrated',settings=dict(window=window,occupancy=occupancy,onset_windows=onset,relative_floor=relative,absolute_floor_ms=absolute_ms),limitations=['no elapsed-time persistence gate: consecutive steps only','clean baseline reused for noise estimation, not independent FP qualification','candidate rank pair not causal fault proof','no host subtype attribution without direct hooks'],findings=out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output');p.add_argument('--absolute-ms',type=float,default=1.)
    a=p.parse_args();data=json.load(open(a.input));result=detect(data,absolute_ms=a.absolute_ms);s=json.dumps(result,indent=2)
    if a.output:open(a.output,'w').write(s+'\n')
    else:print(s)
