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

The width-24 process is currently paused. Resume/cancellation and any later
scheduling changes will be recorded here before treating its outcome as final.
The search process waiting for it has not been restarted.

Both native runs have a 10,800-second wall-clock cap, which includes contention,
evaluation/checkpoint overhead, and a scheduling pause. Final evidence must check
`reached_time_limit`: a run stopped by this cap is not a full 20,000-update
failure. The overlap/pause does not change a step-based learning rate, but
elapsed durations from these trials are not isolated performance measurements.
Implementation inspection and existing repeatability tests support independent
per-update trajectories; no concurrent-versus-isolated replay was performed.
