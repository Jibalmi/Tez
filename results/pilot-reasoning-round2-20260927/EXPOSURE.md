# Training-exposure qualification

Matched inference inputs do not imply matched training exposure. The public comparison measures
the utility of the released checkpoints; it does not isolate architecture or data efficiency.

A read-only inspection of Decider code revision `a5120cce45b9ff70964fac54ea6e8c1ac5b08c7f`
finds English MASSIVE intent, Civil Comments and ToxicChat in its training-task registry. The
loaders separate train and test splits; MASSIVE scenario is separately marked held out. The
reference full-training script builds this task pool and passes it to the mixture builder.
[Pinned registry](https://github.com/Mapika/decider/blob/a5120cce45b9ff70964fac54ea6e8c1ac5b08c7f/decider/data/core.py),
[pinned recipe](https://github.com/Mapika/decider/blob/a5120cce45b9ff70964fac54ea6e8c1ac5b08c7f/scripts/train.sh).

This establishes what the inspected published recipe includes. It does not establish the exact
membership of the released 0.8B checkpoint's training run, nor demonstrate overlap with our test
rows. No release-specific training manifest was reconstructed. The absence of a named
typed-decisions loader in this inspection would not prove absence of semantic or upstream overlap.

The Tez students excluded the entire typed-decisions dataset, while Laya-td used its training
split. The original report already identifies that distinction. The authored confirmation uses
fresh states and domains, but its logical families recur in the rule-supervision training data.
Scores on it are a utility comparison with different training exposure, not an unseen-family test.
