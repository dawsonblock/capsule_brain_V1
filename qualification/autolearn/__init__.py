"""AutoLearn v0.1 qualification harness.

End-to-end pipeline:
  build_dataset.py      <- run_counterfactuals.py <- tasks.py
  train_policy.py       <- build_dataset.py
  evaluate_policy.py    <- train_policy.py
  run_shadow_eval.py    <- train_policy.py
  promote_policy.py     <- evaluate_policy.py
  report.py             <- all of the above

Run order:
  python -m qualification.autolearn.run_counterfactuals
  python -m qualification.autolearn.build_dataset
  python -m qualification.autolearn.train_policy
  python -m qualification.autolearn.evaluate_policy
  python -m qualification.autolearn.run_shadow_eval
  python -m qualification.autolearn.promote_policy
  python -m qualification.autolearn.report
"""
