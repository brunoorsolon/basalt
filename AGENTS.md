# Redrock maintenance

## Current host platform

Work on the host's running kernel; adapt and validate Redrock there. Kernel rollback requires documented catastrophic, repeatable failures and the human's authorization. Before assessing kernel selection, interpreting historical kernel results or handling a kernel/driver incident, read `docs/KERNEL_POLICY.md` for the accepted rationale, evidence threshold and incident-recording requirements. Historical qualification is not a kernel pin or a recurring status warning.

For an upstream update check or incorporation, read `UPSTREAM.md`, including its update procedure, checkpoints and deferred work. A request to check for updates means assess and recommend, not import or release. For release preparation or publication, read `RELEASING.md` and the selected `releases/<version>.md`/JSON records.

Use `python3 tools/redrock.py` for Redrock engine candidates. Keep builds outside the checkout, preserve active installations and model/rollback inputs, and report actual compilation and hardware/model evidence separately. The inherited Makefile and upstream setup prose are not Redrock's installation workflow.
