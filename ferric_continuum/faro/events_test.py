import unittest
from types import SimpleNamespace
from events import StreamInterval

class Event:
    def __init__(self): self.complete = False; self.records = []
    def record(self, stream): self.records.append(stream.cuda_stream)
    def query(self): return self.complete
    def elapsed_time(self, other): return 3.5

class IntervalsTest(unittest.TestCase):
    def make(self):
        self.stream = SimpleNamespace(cuda_stream=7)
        return StreamInterval(Event, lambda: self.stream)
    def test_deferred_read(self):
        x=self.make(); x.begin(); x.finish()
        with self.assertRaises(RuntimeError): x.elapsed_ms()
        x.end.complete=True
        self.assertEqual(x.elapsed_ms(),3.5)
        self.assertEqual(x.start.records,[7])
    def test_single_use(self):
        x=self.make(); x.begin(); x.finish()
        with self.assertRaises(RuntimeError): x.begin()
    def test_wrong_stream(self):
        x=self.make(); x.begin(); self.stream.cuda_stream=8
        with self.assertRaises(RuntimeError): x.finish()
    def test_missing_begin(self):
        with self.assertRaises(RuntimeError): self.make().finish()
    def test_unique_iterations(self):
        a,b=self.make(),self.make()
        self.assertIsNot(a.start,b.start)
        self.assertIsNot(a.end,b.end)

if __name__=='__main__': unittest.main()
