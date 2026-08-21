# RL-S4D-015 — Prospective availability firewall

Date: 2026-08-20

The extended-post manifest was built from filenames and frame existence only.
Before any OBJ vertex value was decoded, an availability audit found complete
raw SMPLer-X A0 coverage but occasional missing or failed balanced Ensemble A1
and original-HaMeR A0 fits. The comparator hierarchy was therefore frozen as:

1. balanced Ensemble A1 when its per-frame result exists;
2. original-HaMeR A0 when Ensemble A1 is absent;
3. raw SMPLer-X A0 only when both fitted sources are absent.

This hierarchy is applied per declared frame, never by evaluator score. The
finalizer must record exact source counts, the baseline and candidate must each
cover all 56 clips/769 frames, and the release must be hashed before creation of
the prospective GT cache. The maximum possible claim remains limited to the
temporally disjoint SIGNAL-4D extended-post endpoint because sign identities
overlap and signer IDs are unavailable.
