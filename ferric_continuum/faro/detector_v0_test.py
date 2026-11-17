import unittest
from detector_v0 import detect

class DetectorTest(unittest.TestCase):
    def rows(self):
        return [dict(case=c,rank=r,step=s,kind='calc',region='layer0',duration_ms=(20 if c=='holdout' and r==1 else 10)) for c in ('clean','holdout') for r in (0,1) for s in range(24)]
    def test_persistent_disparity(self):
        f=detect(self.rows())['findings']
        self.assertEqual(len(f),1); self.assertEqual(f[0]['long_duration_rank'],1)
        self.assertEqual(f[0]['onset_step'],15)
        self.assertIn('unresolved',f[0]['classification'])
    def test_equal_ranks(self):
        rows=self.rows()
        for r in rows:r['duration_ms']=10
        self.assertEqual(detect(rows)['findings'],[])
    def test_one_spike(self):
        rows=self.rows()
        for r in rows:r['duration_ms']=20 if r['case']=='holdout' and r['rank']==1 and r['step']==0 else 10
        self.assertEqual(detect(rows)['findings'],[])
    def test_duplicate_refused(self):
        rows=self.rows(); rows.append(rows[0])
        with self.assertRaises(ValueError):detect(rows)
    def test_missing_steps_do_not_create_persistence(self):
        rows=[r for r in self.rows() if r['step']%8!=3]
        self.assertEqual(detect(rows)['findings'],[])
    def test_treatment_labels_irrelevant(self):
        rows=self.rows(); expected=detect(rows)
        for r in rows:r['injected_rank']=99
        self.assertEqual(detect(rows),expected)

if __name__=='__main__':unittest.main()
