# PromptForge V0.8.3 - Parametric Sweep

Task: T003
Provider/model: DeepSeek
Repetitions per condition: 20
Conditions: 6
Total scheduled runs: 120

Structure: noop / representation_A / representation_B
Serialization: pretty / compact

Primary response: paired reasoning-token delta.
Secondary: input, output, total, latency, context chars, prompt chars, quality.

No adaptive repetition.
No automatic retry.
No silent exclusion.
No evaluator changes.
No task changes.
No provider changes.
No protected artifact changes.

Global positions: 0..119.
- transforms.py: d851443f75b02876def4a971b0c97b0a34a6de044ed49326e133a1d421ec8e4a
- v082_hotspot_replication.py: 3fe7fb524feb1e553ce118ff8c13cc1ed8020784de6cf50df2c430a4be530fc6
- deepseek_adapter.py: 4913395597afab924a8fb8927cb61288603b4b87a5b1b0bc47f6959707d0e614
- real_executor.py: 5589F8C777A594E81BBBFA15D14BF181B141957825DB176EAEA1ABB5DD2D9842