---

pretty_name: MEBeauty
annotations_creators:

* crowdsourced
  source_datasets:
* original
  size_categories:
* 1K<n<10K
  license: other
  license_name: MEBeauty Non-Commercial Research Use
  tags:
* image
* facial-attractiveness
* facial-attractiveness-prediction
* facial-beauty-prediction
* facial-attractiveness-assessment
* facial-aesthetics
* face-analysis
* computer-vision
* image-regression
* personalized-ai
* human-preferences
* multi-ethnic
* in-the-wild
* benchmark

---

# MEBeauty

**A multi-ethnic, in-the-wild dataset for general and personalized facial attractiveness prediction.**

MEBeauty contains facial images collected in unconstrained conditions together with aggregate and individual attractiveness ratings.

## Dataset overview

| Property         | Value                                                     |
| ---------------- | --------------------------------------------------------- |
| Images           | 2,550                                                     |
| Face groups      | Female and male                                           |
| Ethnicity groups | Asian, Black, Caucasian, Hispanic, Indian, and Mideastern |
| Rating scale     | 1–10                                                      |
| Raters           | Approximately 300 volunteers overall                      |
| Setting          | Unconstrained, in-the-wild images                         |
| Labels           | Aggregate and individual attractiveness ratings           |
| Usage            | Non-commercial research                                   |

The images include variation in age, gender, ethnicity, pose, expression, illumination, and background. Individual images were rated by subsets of the overall volunteer pool.

## Research uses

MEBeauty supports research in:

* general facial attractiveness prediction;
* personalized attractiveness prediction;
* beauty-score regression and ranking;
* human-preference modelling;
* cross-dataset generalization;
* demographic, fairness, and bias analysis;
* computational facial aesthetics.

## Data access

The dataset is intended for **non-commercial research use**.

The repository is currently private while the original files, annotations, redistribution permissions, and access terms are verified. The planned public release will use gated access where appropriate. Hugging Face gated repositories require users to request access and share their Hub username and email with the dataset owner.

The precise research-use agreement and the licensing status of images, annotations, derived features, and code will be documented separately before release.

## Responsible use

Facial attractiveness is subjective and may reflect individual, cultural, social, and demographic preferences.

MEBeauty and models trained on it must not be used:

* to measure a person’s value, competence, health, or social worth;
* in hiring, education, credit, insurance, or other consequential decisions;
* to rank people on dating or social platforms without informed consent;
* for harassment, discrimination, surveillance, or harmful profiling;
* to claim universal or objective standards of beauty.

Models may reproduce or amplify biases present in the ratings. Researchers should report subgroup performance, uncertainty, limitations, and failure cases.

Individuals who believe that they are depicted in the dataset may contact the maintainer to request review or removal.

## Code and benchmark

Code, evaluation tools, configurations, and benchmark results:

https://github.com/dr-irina-lebedeva/MEBeauty-Benchmark

## Paper

[MEBeauty: a multi-ethnic facial beauty dataset in-the-wild](https://link.springer.com/article/10.1007/s00521-021-06535-0)

The article was published online in 2021 and appears in the 2022 journal volume. The recommended citation uses the final journal year, **2022**.

## Citation

If you use the MEBeauty images, ratings, annotations, protocols, code, models, or benchmark results, please cite:

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

Earlier references may use the key `lebedeva2021mebeauty` because the article was first published online in 2021. Both keys refer to the same work; the final bibliographic year is 2022.

Please also cite the original publication of any method reproduced through the benchmark.

## Maintainer

**Irina Lebedeva, PhD** — first author of the MEBeauty paper

[Website](https://irina-lebedeva.com/) · [LinkedIn](https://www.linkedin.com/in/ailina/) · [Email](mailto:dr.irina.lebedeva@gmail.com)

## Historical repository

The original release is preserved at:

https://github.com/fbplab/MEBeauty-database

Administrative access to the historical repository is no longer available, so its issues and pull requests cannot currently be reviewed or merged. Please reopen unresolved contributions in the maintained GitHub repository.

## Status

Dataset files, schema, validated statistics, access terms, and loading instructions are being prepared.

