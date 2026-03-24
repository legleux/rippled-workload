"""Transaction builder modules — grouped by domain.

Each module exports:
  BUILDERS: dict mapping type name → (builder_fn, model_cls)
  ELIGIBILITY: dict mapping type name → per-account eligibility predicate (optional)
  TAINTERS: dict mapping type name → list of tainting functions (optional)
"""
