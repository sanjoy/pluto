# Width/depth search scheduling notes

These experiments measure whether a model can memorize the corpus under a
recorded update schedule, not isolated GPU throughput. In particular, the
following scheduling changes must accompany any use of their elapsed times.

## 2026-09-21: long narrow search and width-24 refinement

`width_depth_long_1` started at 02:30:46 UTC. Its one-block width-32 trial
completed before the overlap below. The next one-block width-16 trial was
already running when `width_depth_refine_24` started at 02:44:58 UTC.

The two independent native processes used separate model/optimizer state,
streams, and checkpoint/artifact roots. They shared the same pinned binary,
read-only corpus/tokenizer, seed, batch size, and step-based training schedule.
No learned state was transferred between the trials.

Concurrency substantially reduced aggregate throughput: width 16's interval
between evaluations 128 updates apart increased from roughly 6.5 seconds to
25--29 seconds, while width 24's first such interval after step 128 took about
40 seconds. Therefore the width-24 native process (PID 3266920) was paused with
`SIGSTOP` at **02:47:21 UTC**, retaining its in-memory optimizer state. Its
latest completed-step log entry was step 464; that does not identify the exact
in-flight operation at the instant of the pause. Width 16's evaluation interval
returned to roughly 6.5 seconds after the pause.

At **02:53:19 UTC**, only the `width_depth_long_1` Python coordinator (PID
3264113) was also paused. Its existing width-16 native child (PID 3266171)
continued normally to its stopping condition. This prevented the coordinator
from immediately launching a deeper/narrower trial before the paused width-24
midpoint could run. Width 16 finished all 20,000 updates with 1,833 errors and
`reached_time_limit=0`. After its GPU process exited, the two targeted width-24
attention/GPT-2 test targets passed. The native experiment executable's SHA256
remained unchanged.

The coordinator was briefly resumed to complete the fresh native checkpoint
reload and independent prediction audit for width 16. Both checks passed and
its manifest now records `verified_budget_failure`. At **03:00:20 UTC**, the
coordinator and its newly spawned two-block width-16 child (PID 3275073) were
paused. At **03:00:50 UTC**, the original width-24 process was resumed with
`SIGCONT`, after a roughly 13-minute-29-second pause. Its existing optimizer
state and update schedule were preserved; no checkpoint restart occurred.

Width 24 completed all 20,000 updates with three errors and
`reached_time_limit=0`; its coordinator finished both independent checks at
03:17:32 UTC. All 65 optimized native test targets then passed (63 cached,
the two updated targets executed). At **03:19:22 UTC**, the original two-block
width-16 child (PID 3275073) and coordinator (PID 3264113) were both resumed
with `SIGCONT`. They continue the existing `width_depth_long_1` search; no new
instance was launched and the pinned binary hash is unchanged. Two-block
width 16 subsequently reached 20,000 updates with 454 errors and
`reached_time_limit=0`; both independent checks completed at **03:39:37 UTC**.
The same coordinator then started four-block width 16 normally. That trial
completed all 20,000 updates with 485 errors and `reached_time_limit=0`, and
finished both independent checks at **04:06:23 UTC**. No competing GPU trial
or test was launched during it. The same coordinator then started eight-block
width 16 normally. No process remains deliberately paused.

Pausing the coordinator alone did not pause the earlier one-block child or
alter that child's update schedule. None of these processes has been restarted.

Both native runs have a 10,800-second wall-clock cap, which includes contention,
evaluation/checkpoint overhead, and a scheduling pause. Final evidence must check
`reached_time_limit`: a run stopped by this cap is not a full 20,000-update
failure. The overlap/pause does not change a step-based learning rate, but
elapsed durations from these trials are not isolated performance measurements.
Implementation inspection and existing repeatability tests support independent
per-update trajectories; no concurrent-versus-isolated replay was performed.

## Queued width-24 longer-budget refinement

At **03:59:36 UTC**, a CPU-only coordinator (PID 3290991) was queued for the
planned fresh one-block width-24, one-head, 40,000-update trial. It waits on a
Linux process handle for the existing `width_depth_long_1` coordinator (PID
3264113), not a reused numeric PID or an estimated completion time. It launches
no GPU work until that entire search exits, including all native checkpoint
reloads and independent prediction audits.

Before launching, the coordinator requires the predecessor's final manifest to
say `completed`, runs the evidence-checking reporter on it, and checks the pinned
binary, corpus, and tokenizer hashes again. An interrupted/failed predecessor
or changed input aborts the queued trial. The existing driver also refuses
preexisting output/checkpoint destinations. The new artifact root is
`runs/width_depth_refine_24_long_0/`; its checkpoints will be under
`/home/ubuntu/checkpoints/memorize_general_facts/width_depth_refine_24_long_0/`.
This is a fresh initialization with a longer cosine schedule, not a checkpoint
resume. The actual start and final result will be recorded by its own manifest.
