"""Forward-only same-main-thread-fixture host progress. Not GPU readiness."""
import collections

def edges(metadata,step):
 calls=sorted([c for c in metadata['c10d_records'] if c['step']==step and c['phase']=='forward'],key=lambda c:c['host_submit_ns'])
 threads={c.get('native_thread_id') for c in calls}
 if len(threads)>1:raise ValueError('mixed submission threads are unsupported')
 counts=collections.Counter();seq=[]
 for c in calls:
  axis='TP' if c['members']==metadata['tp_members'] else 'DP' if c['members']==metadata['dp_members'] else None
  if axis is None:raise ValueError('unsupported process group')
  key=(axis,c['api']);ordinal=counts[key];counts[key]+=1
  signature=(axis,c['api'],ordinal,tuple((tuple(t['shape']),t['dtype']) for t in c['tensors']))
  seq.append((signature,c))
 out=[]
 for (sa,a),(sb,b) in zip(seq,seq[1:]):
  gap=(b['host_submit_ns']-a['host_return_ns'])/1e6
  if gap<0:raise ValueError('overlapping calls: unsupported thread ordering')
  out.append((repr((sa,sb)),gap))
 return out
