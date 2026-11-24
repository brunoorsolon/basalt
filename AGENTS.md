# Basalt maintenance

Before updating engine sources, dependency pins or imported optimizations, read `UPSTREAM.md`, including its deferred work and review boundaries. Record immutable source hashes and adaptation provenance in the same candidate; advance review bookmarks only for assessed ranges.

Use `python3 tools/basalt.py` for Basalt engine candidates. Keep builds outside the checkout, preserve active installations and model/rollback inputs, and report actual compilation and hardware/model evidence separately. The inherited Makefile and upstream setup prose are not Basalt's installation workflow.
