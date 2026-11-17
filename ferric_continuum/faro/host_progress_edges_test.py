import unittest
from host_progress_edges import edges
class TestEdges(unittest.TestCase):
 def sample(self):
  return dict(tp_members=[0,1],dp_members=[0,2],c10d_records=[dict(step=0,phase='forward',members=[0,1],api='all_reduce',tensors=[dict(shape=[8],dtype='float32')],host_submit_ns=0,host_return_ns=1000000),dict(step=0,phase='forward',members=[0,2],api='all_gather_single',tensors=[dict(shape=[8],dtype='float32')],host_submit_ns=26000000,host_return_ns=27000000)])
 def test_gap(self):self.assertEqual(edges(self.sample(),0)[0][1],25)
 def test_no_truth(self):
  m=self.sample();expected=edges(m,0);m['GAP_RECORDS']=['false'];m['treatment']='false';self.assertEqual(edges(m,0),expected)
 def test_mixed_threads(self):
  m=self.sample();m['c10d_records'][0]['native_thread_id']=1;m['c10d_records'][1]['native_thread_id']=2
  with self.assertRaises(ValueError):edges(m,0)
 def test_phase(self):
  m=self.sample();m['c10d_records'][1]['phase']='backward';self.assertEqual(edges(m,0),[])
 def test_reject_overlap(self):
  m=self.sample();m['c10d_records'][1]['host_submit_ns']=0
  with self.assertRaises(ValueError):edges(m,0)
if __name__=='__main__':unittest.main()
