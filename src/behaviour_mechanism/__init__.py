"""Behaviour–Mechanism: behavioural vs activation-patching measurements during training.

The modules in this package are a faithful, tested extraction of the code in
``notebooks/behaviour_vs_mechanism.ipynb`` (the code that produced the historical
results in ``results/``), plus clearly separated additions:

* ``patching`` exposes per-pair records so that the historical aggregate
  (``legacy_effect``) can be decomposed into "fraction of probe pairs that were
  defined" and "recovery conditional on a defined pair".
* ``probes`` adds diagnostics that were not part of the historical run
  (e.g. shifted-offset repeats that separate token-matching induction from
  fixed-offset positional copying).

Nothing here changes the historical metric definitions silently: the legacy
aggregate is reproduced exactly and kept alongside any new quantity.
"""

__version__ = "0.2.0"
