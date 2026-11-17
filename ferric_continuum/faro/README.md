# FARO measurement primitives

These experimental helpers separate three observations that should not be
conflated: a local CUDA stream interval, CPU submission progress, and a
peer-relative duration disparity. They do not identify a faulty rank by
collective duration alone.

## What is included

* `events.StreamInterval` records one start/end CUDA event pair on one stream.
  Retain a distinct object for each region and iteration. Collect after a final
  completion fence; the reader performs one nonblocking completion query, not a
  CPU busy-poll loop. No implicit synchronization occurs when recording.
* `host_progress_edges.edges` extracts forward submission-return to next-submit
  gaps, keyed by process-group axis, API, occurrence and tensor signature. Its
  input contract is a known single host submission thread. Non-overlapping
  timestamps alone cannot prove that contract. A gap is CPU progress, not GPU
  readiness; it may include interpreter, scheduling or synchronous API work.
* `detector_v2.detect` is the current experimental detector: it adds a current
  raw-duration disparity floor, resets streaks across missing windows, and
  requires at least one second of persistence on each rank's local timeline.
  Supply `time_ms` per observation; the pair uses the smaller within-rank delta,
  never a subtraction between different rank clocks. Missing timestamps cannot
  satisfy the default persistence gate. Settings remain exploratory, not
  production-calibrated.
* `detector_v0.detect` preserves the historical v0 exploratory algorithm,
  not the later host-progress detector validated in follow-up experiments:
  eight-step windows, 75% occupancy, two passing windows, a 15% relative floor
  and 1 ms absolute floor above clean-baseline noise. Treatment labels are not
  read. It has no one-second persistence gate. Its streak does not reset across
  missing windows; do not use incomplete timelines for acceptance. It reports a
  candidate pair, not a causal origin. Do not tune it on a
  holdout and continue calling that holdout independent.
* `mlp_fixture.py` contains a portable three-layer SwiGLU model, TP2 autograd
  collectives and a dense reference. Its fixed topology is DP2×TP2, without
  sequence parallelism, attention, normalization, residuals or loss/optimizer.
  It is not a complete transformer or a standalone distributed launcher.
  The caller constructs process groups and supplies the output gradient.

Run the CPU-only tests from the repository root:

```sh
python3 -m unittest discover -s ferric_continuum/faro -p '*_test.py'
# Alternatively, with the repository's Bazel Python toolchain:
bazel test //ferric_continuum/faro:all
```

PyTorch is optional for these tests. Actual CUDA events and the model require a
separately provisioned PyTorch/CUDA runtime. Fake-event tests check API ordering
and refusal behavior, not GPU timing accuracy. The `StreamInterval` API separately passed a four-worker H100 smoke on
PyTorch 2.13/CUDA 12.6: distinct retained events, ordered CUDA operations, one
final completion fence, expected tensor results and duplicate-begin rejection.
That tiny first-use smoke is not a performance measurement. The model and this
package as a distributed end-to-end workflow have not been GPU-qualified. No launcher, cluster credentials, host paths, captured
traces, private run receipts or runtime-specific monkeypatches are included.

## Minimal CUDA API usage

```python
import torch
from ferric_continuum.faro.events import StreamInterval

x = torch.ones(256, device="cuda")
interval = StreamInterval()
interval.begin()
y = x * 2
interval.finish()
final = torch.cuda.Event()
final.record()
final.synchronize()
print(interval.elapsed_ms())  # API smoke only, not a useful benchmark
```

## How to measure without changing every training step into a barrier

Allocate distinct event pairs before the measured loop. Record begin/end around
known model regions, retain detached loss tensors, and record a final completion
event after the final optimizer work. Synchronize once, then read losses and
intervals. For a matched synchronized control, use the same retained-event
harness and additionally synchronize each step's end event. Do not reuse event
pairs across asynchronous steps: later records overwrite the measurements.

A final event only covers work ordered before it on its stream, including other
streams explicitly joined by waits. FSDP communication completion must be
established from the selected implementation and trace, not assumed from Python
`backward()` returning. A later validation `.cpu()` can hide a missing fence;
read timing results before validation copies when qualifying coverage.

Local elapsed time requires no shared device clock. It cannot give absolute
cross-rank arrival order. A long NCCL interval may include late-peer waiting,
transfer, scheduling and contention. A short interval may suggest late arrival,
but collective algorithms are not universal all-rank completion barriers.
Use architecture-specific process-group membership and matched logical events,
then confirm candidates with compute and host progress observations.

## Trace and numerical boundaries

Join collectives by actual process-group membership/name, native sequence,
operation and tensor metadata. Do not assume one Python call equals one native
sequence. Select the Flight Recorder API for the actual backend: an `nccl`
requested alias may resolve to `nccl2`, whose generic recorder is distinct from
the legacy NCCL recorder. Exported profiler thread lanes may be rewritten;
verify runtime thread provenance and correlation IDs before joining CPU scopes
to GPU kernels. Timestamp overlap alone is insufficient.

For FSDP reference checks, parameter gradients are averaged across DP ranks;
local input gradients are not. Optimizer moments must follow the same DTensor
placements as parameters. Stacked gate/up weights `[2,F,D]` can use `Shard(1)`
for a larger DP axis, as in TorchTitan's stacked-linear rule. This DP2 fixture
does not implement the DP4 FSDP adapter or claim TorchTitan end-to-end parity.

FP32 reduction order can change small gradients enough to exceed a relative
threshold even when absolute error is small. Keep thresholds fixed, label the
exact tensor/step, and use a separate FP64 algebra diagnostic rather than
silently relaxing acceptance. Full-size FP32 validation remains a separate test.
