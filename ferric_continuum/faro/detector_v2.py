"""V2: current absolute disparity AND residual gates. Fixed exploratory settings, not production calibration.
Input flat records: case,rank,step,region,kind,duration_ms. Baseline case=clean.
No treatment labels used in scoring. Pairwise contrast is not causal attribution.
"""
import argparse,json,math,statistics,itertools

def quantile(xs,q):
    a=sorted(xs);return a[min(len(a)-1,int(math.ceil(q*len(a)))-1)]

def detect(rows,baseline='clean',window=8,occupancy=.75,onset=2,relative=.15,absolute_ms=1.,min_span_ms=1000):
    groups={}; times={}
    for r in rows:
        x=r['duration_ms']
        if not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 or r.get('valid',True) is False:continue
        k=(r['case'],r['kind'],r['region'])
        z=groups.setdefault(k,{})
        identity=(r['rank'],r['step'])
        if identity in z:raise ValueError('duplicate rank/step in semantic region')
        z[identity]=float(x)
        if 'time_ms' in r:times[(k,identity)]=r['time_ms']
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
            windows=[];streak=0;opened=None;streak_start=None;previous_end=None
            for start in range(0,len(steps),window):
                ss=steps[start:start+window]
                if len(ss)<window:break
                # Require consecutive logical steps; missing observations cannot masquerade as persistence.
                valid=ss[-1]-ss[0]==window-1
                es=[math.log(vals[r,s]/vals[q,s])-b for s in ss];hs=[vals[r,s]-vals[q,s]-a for s in ss]
                raw_gaps=[vals[r,s]-vals[q,s] for s in ss]
                occ=sum(e>tl and h>td and g>absolute_ms for e,h,g in zip(es,hs,raw_gaps))/window
                passed=valid and occ>=occupancy and statistics.median(es)>tl and statistics.median(hs)>td and statistics.median(raw_gaps)>absolute_ms
                if previous_end is not None and ss[0]!=previous_end+1:streak=0;streak_start=None
                if passed:
                    if streak==0:streak_start=ss[0]
                    streak+=1
                else:streak=0;streak_start=None
                previous_end=ss[-1]
                span=0.
                if streak_start is not None:
                    kk=(case,kind,region)
                    if all((kk,(rr,st)) in times for rr in (r,q) for st in (streak_start,ss[-1])):
                        span=min(times[kk,(rr,ss[-1])]-times[kk,(rr,streak_start)] for rr in (r,q))
                if streak>=onset and span>=min_span_ms and opened is None:opened=ss[-1]
                windows.append(dict(steps=[ss[0],ss[-1]],occupancy=occ,median_ratio_residual=math.exp(statistics.median(es)),median_excess_ms=statistics.median(hs),passed=passed,consecutive_span_ms=span,current_median_gap_ms=statistics.median(raw_gaps),baseline_median_gap_ms=a,baseline_median_ratio=math.exp(b)))
            if opened is not None:
                out.append(dict(case=case,kind=kind,region=region,comparison_ranks=[r,q],long_duration_rank=r,reference_rank=q,candidate_origin_ranks=[r,q],observed_affected_ranks=[r] if kind=='comm' else [],classification=kind+'-path disparity; causal origin unresolved',onset_step=opened,threshold_relative=math.expm1(tl),threshold_ms=td,windows=windows))
    return dict(status='exploratory offline scoring; not held-out calibrated',settings=dict(window=window,occupancy=occupancy,onset_windows=onset,relative_floor=relative,absolute_floor_ms=absolute_ms,min_consecutive_span_ms=min_span_ms),limitations=['v2 adds current raw gap floor; not a fix for unqualified baseline or host-launch gaps','1second experimental persistence floor is not a production-calibrated threshold','clean baseline reused for noise estimation, not independent FP qualification','candidate rank pair not causal fault proof','no host subtype attribution without direct hooks'],findings=out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output');p.add_argument('--absolute-ms',type=float,default=1.)
    a=p.parse_args();data=json.load(open(a.input));result=detect(data,absolute_ms=a.absolute_ms);s=json.dumps(result,indent=2)
    if a.output:open(a.output,'w').write(s+'\n')
    else:print(s)
