import unittest
from detector_v2 import detect

def rows(kind):
 out=[]
 for case in ['clean','test']:
  for s in range(32):
   for r in range(2):
    x=10.
    if kind=='reversal':x=10.8 if (case=='clean' and r==0) or (case=='test' and r==1) else 10.
    if kind=='strong' and case=='test' and r==1:x=13.
    if kind=='spike' and case=='test' and r==1 and s==10:x=50.
    if kind=='uniform' and case=='test':x=20.
    out.append(dict(case=case,rank=r,step=s,kind='calc',region='same',duration_ms=x,time_ms=s*100))
 return out
class Test(unittest.TestCase):
 def test_reversal(self):self.assertEqual(detect(rows('reversal'),relative=.01)['findings'],[])
 def test_strong(self):self.assertEqual(len(detect(rows('strong'))['findings']),1)
 def test_spike(self):self.assertEqual(detect(rows('spike'))['findings'],[])
 def test_uniform(self):self.assertEqual(detect(rows('uniform'))['findings'],[])
 def test_rank_clock_offsets(self):
  a=rows('strong');expected=detect(a)
  for r in a:r['time_ms']+=100000*r['rank']
  self.assertEqual(detect(a),expected)
 def test_short_span(self):
  a=rows('strong')
  for r in a:r['time_ms']=r['step']
  self.assertEqual(detect(a)['findings'],[])
 def test_missing_timestamps(self):
  a=rows('strong')
  for r in a:del r['time_ms']
  self.assertEqual(detect(a)['findings'],[])
 def test_gap_resets_streak(self):
  a=[r for r in rows('strong') if r['step']<8 or r['step']>=24]
  self.assertEqual(detect(a)['findings'],[])
if __name__=='__main__':unittest.main()
