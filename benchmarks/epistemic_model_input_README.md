# V27.7 — epistemic-aware model input

V27.7 makes PromptForge's epistemic layer visible at the model boundary.

The model packet now contains a compact `epistemic` context block with:

- included and excluded evidence ids
- preserved contradiction ids
- observation/inference/hypothesis coverage
- temporal exclusion metadata
- deterministic selection audit
- compiler-estimated tokens for the metadata representation

The raw evidence remains available as a separate context block. The epistemic
summary is metadata and control information; it does not claim that PromptForge
has established truth or causality.

Pipeline:

    ticket
      -> epistemic compilation
      -> PromptForge context budgeting
      -> model
      -> strict parser
      -> action guard
      -> evaluation

This keeps the control plane explicit in the model input instead of relying only
on downstream action blocking.

The benchmark's evidence objects are metadata-only by design. V27.7 does not invent source prose; the epistemic compiler receives a canonical metadata representation explicitly labeled `metadata_only`.
