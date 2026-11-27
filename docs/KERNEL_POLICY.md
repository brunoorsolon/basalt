# Host kernel policy

**Accepted decision: develop and validate Redrock on the host's running kernel. Kernel rollback is an exceptional recovery measure, considered only for catastrophic, repeatable failures documented with current evidence.**

## Why the host moves forward

This is a maintained, general-purpose host, not a permanently frozen qualification appliance. Security fixes, distribution updates, hardware support and other applications all require the system to evolve. Expecting it to remain on one kernel indefinitely because an earlier Redrock test passed there is unrealistic and is not Redrock's operating model. Adapt Redrock to the maintained host instead of making the host stay old for Redrock.

A release manifest records the environment that was tested; it does not prescribe the host's future kernel. Reproducible source, build inputs and historical evidence can be preserved without freezing the running operating system. New runs record their actual kernel and results, and support claims remain bounded by that evidence.

## Normal operation

- Discover the running kernel from the host rather than treating a version in documentation as the current target or a permanent pin.
- Investigate and fix problems on that platform. A version difference, missing qualification, compilation error, ordinary model failure or isolated crash is not sufficient grounds to recommend an older kernel.
- Keep historical release records intact. A historical fault report does not establish a current, repeatable failure and must not become a recurring warning, standing blocker or downgrade recommendation. Discuss it when the human asks or fresh evidence makes it directly relevant.
- This policy does not authorize automatic kernel upgrades, downgrades or other host changes. The human owns those decisions.

## Exceptional rollback

A rollback recommendation must meet all of these conditions:

1. **Catastrophic impact:** failure threatens data integrity or host availability, such as corruption, a kernel panic, an unrecoverable machine hang or a GPU/driver failure that takes down the host or desktop. An engine process exiting by itself is not automatically catastrophic.
2. **Repeatability:** documented reproduction steps have produced the catastrophic failure in independent attempts under identified conditions. Preserve attempt counts and outcomes; a single event or an old anecdote is not proof. Never deliberately risk further data loss just to obtain another reproduction; stop unsafe execution and seek the human's recovery decision instead.
3. **Relevant evidence:** current logs and the identified kernel, driver/runtime, engine, model and device configuration support investigating kernel-level recovery. Distinguish observations from hypotheses; repeated failure alone does not prove the kernel caused it or that an older kernel fixes it.
4. **Recorded decision:** retain the incident and evidence in this repository before recommending rollback. Name any proposed recovery kernel, the evidence for that choice and the conditions for returning to a maintained kernel. The human must authorize the actual kernel change.

Stopping a dangerous run is appropriate immediately; it does not require first proving repeatability. A necessary stop is not, by itself, a rollback decision. Any approved rollback is a bounded recovery action, not a new indefinite kernel freeze.

## Record incidents if and when they occur

Create a dated record under `docs/incidents/` only when there is an actual incident; do not invent an incident or promote a historical report into a new one. Link it from the affected issue or qualification record. Include:

- Actual kernel, driver/ROCr/toolkit identities, hardware/device isolation, engine revision and immutable model/input identities.
- Reproduction commands, resource limits, independent attempt counts, successful and failed outcomes, and the catastrophic impact observed.
- Relevant logs or durable evidence links, with credentials and unrelated private data excluded.
- Observed facts, unresolved hypotheses, mitigations tried, and whether the catastrophic/repeatable threshold is established.
- The human's recovery decision, any temporary kernel change, subsequent verification and the forward-fix/requalification outcome as it becomes available.

A new incident changes the current assessment only to the extent its evidence supports. Preserve its history when resolved; report the current outcome rather than carrying an obsolete warning forward forever.
