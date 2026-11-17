"""Retained, single-use stream intervals. Importing does not initialize CUDA."""

class StreamInterval:
    """One pair per semantic region per iteration; never reuse before reading.

    This measures a stream interval, including scheduling gaps and dependencies,
    not exclusive kernel busy time or a timestamp comparable across devices.
    """

    def __init__(self, event_factory=None, current_stream=None):
        if event_factory is None or current_stream is None:
            import torch
            event_factory = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
            current_stream = current_stream or torch.cuda.current_stream
        self.start, self.end = event_factory(), event_factory()
        self.current_stream = current_stream
        self.stream = None
        self.state = 'new'

    def begin(self):
        if self.state != 'new':
            raise RuntimeError('interval is single-use')
        stream = self.current_stream()
        self.stream = stream.cuda_stream
        self.start.record(stream)
        self.state = 'started'

    def finish(self):
        if self.state != 'started':
            raise RuntimeError('begin must precede finish')
        stream = self.current_stream()
        if stream.cuda_stream != self.stream:
            raise RuntimeError('interval endpoints must use the same stream')
        self.end.record(stream)
        self.state = 'recorded'

    def elapsed_ms(self):
        if self.state != 'recorded':
            raise RuntimeError('interval is incomplete')
        # Non-blocking check during offline collection, never a polling loop.
        if not self.end.query():
            raise RuntimeError('interval not complete; collect after the final fence')
        return self.start.elapsed_time(self.end)
