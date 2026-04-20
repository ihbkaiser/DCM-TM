# Continual Topic Model

This repo now supports a soft DCM-TM update path. Enable it with:

```yaml
soft_memory:
  enabled: true
```

At each timestamp the pipeline:

1. trains a local ProdLDA/ETM model on the current batch,
2. learns each topic embedding `alpha_k` in a factorized decoder where
   `logit_{kv} = alpha_k^T e_v`,
3. asks the LLM, or the similarity fallback, for soft retain and novelty priors,
4. trains a lightweight gate controller with Bernoulli KL to those priors,
5. updates the fixed-size global topic memory with soft assignment and novelty flow,
6. optionally infers aligned document-topic vectors under the updated memory.

Top words are still used for readable prompts and saved summaries, but nearest
topic retrieval and soft assignment use the learned decoder topic embeddings.

Aligned theta vectors are saved per timestamp as:

```text
outputs/T*/aligned_theta.npy
```

Set `soft_memory.enabled: false` to use the original hard retain/remove and
novel/covered dynamic-K pipeline.
