# Redrock maintenance

## Current host platform

Target the host's running kernel (currently 7.3.4); adapt and validate Redrock on that platform. The 1.0.0 results on 7.3.3 and its earlier fault report are historical evidence, not a kernel pin, downgrade requirement or standing blocker. Keep those release records intact. Mention the historical fault only when the human asks about it or fresh execution evidence makes it relevant to the current task; routine status and handoff replies should report current results instead. Kernel changes belong to the human.

For an upstream update check or incorporation, read `UPSTREAM.md`, including its update procedure, checkpoints and deferred work. A request to check for updates means assess and recommend, not import or release. For release preparation or publication, read `RELEASING.md` and the selected `releases/<version>.md`/JSON records.

Use `python3 tools/redrock.py` for Redrock engine candidates. Keep builds outside the checkout, preserve active installations and model/rollback inputs, and report actual compilation and hardware/model evidence separately. The inherited Makefile and upstream setup prose are not Redrock's installation workflow.
