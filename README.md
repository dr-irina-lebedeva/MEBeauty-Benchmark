# MEBeauty Benchmark

**Dataset, code, models, and reproducible evaluation for facial attractiveness prediction, facial beauty prediction, and personalized attractiveness assessment.**

MEBeauty is a multi-ethnic facial beauty dataset collected in unconstrained, real-world conditions.

## Project scope

- MEBeauty dataset documentation and access
- Reproducible training and evaluation
- Standardized experimental configurations and splits
- Baseline and future model comparisons
- General and personalized prediction
- Pretrained models and benchmark results
- Cross-dataset, subgroup, fairness, and bias evaluation

Large datasets and model weights will be hosted on Hugging Face rather than stored directly in GitHub.

## Maintained repository

This repository is maintained by **Irina Lebedeva, PhD**, first author of the original MEBeauty paper.

- [Original paper](https://link.springer.com/article/10.1007/s00521-021-06535-0)
- [Historical MEBeauty repository](https://github.com/fbplab/MEBeauty-database)

Administrative access to the historical repository is no longer available. Therefore, its existing issues and pull requests cannot be answered, reviewed, or merged.

Please reopen unresolved issues or contributions in this repository and include a link to the original issue or pull request.

### Coming from the original repository?

**Read [`docs/CHANGES_VS_ORIGINAL.md`](docs/CHANGES_VS_ORIGINAL.md)** — a complete, reproducible account of every difference between this dataset and the original release.

Most important: the original `FaceNet_512_features` covered only **8 of 151 `male/indian` images (5.3%)** because of a silent face-detection failure, against ~100% for every other subgroup. If you trained on those embeddings, your subgroup results are affected. All 143 missing embeddings have been recovered.

## Documentation

| Document | What it covers |
|---|---|
| [`CHANGES_VS_ORIGINAL.md`](docs/CHANGES_VS_ORIGINAL.md) | Every difference vs. the original MEBeauty release |
| [`PRE_RELEASE_CHECKLIST.md`](docs/PRE_RELEASE_CHECKLIST.md) | Ordered steps required before publishing to Hugging Face |
| [`DATASET_AUDIT.md`](docs/DATASET_AUDIT.md) | Full findings with evidence and reproduction commands |
| [`DATASET_CARD.md`](docs/DATASET_CARD.md) | Hugging Face-style dataset card |
| [`DATASHEET.md`](docs/DATASHEET.md) | Datasheet for Datasets (Gebru et al.) |
| [`REPRODUCE_LEGACY_BASELINE.md`](docs/REPRODUCE_LEGACY_BASELINE.md) | Regenerating crops, embeddings, and geometric features |
| [`RESTRUCTURE_PROPOSAL.md`](docs/RESTRUCTURE_PROPOSAL.md) | Why the dataset layout changed |

> **Licensing is unresolved.** No licence has been selected for images, annotations, splits, or derived features, and nothing has been published to Hugging Face. Do not treat any of this as licensed for redistribution — see the "Still open" section of `CHANGES_VS_ORIGINAL.md`.

## Maintainer

**Irina Lebedeva, PhD** — AI researcher and engineer

[Website](https://irina-lebedeva.com/) · [LinkedIn](https://www.linkedin.com/in/ailina/) · [Email](mailto:dr.irina.lebedeva@gmail.com)

## Citation

If you use the **MEBeauty dataset, ratings, annotations, code, models, protocols, or benchmark results**, please cite:

```bibtex
@article{lebedeva2022mebeauty,
  title   = {MEBeauty: a multi-ethnic facial beauty dataset in-the-wild},
  author  = {Lebedeva, Irina and Guo, Yi and Ying, Fangli},
  journal = {Neural Computing and Applications},
  volume  = {34},
  number  = {17},
  pages   = {14169--14183},
  year    = {2022},
  doi     = {10.1007/s00521-021-06535-0}
}
```
