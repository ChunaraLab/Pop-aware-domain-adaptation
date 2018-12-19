# Pop-aware-domain-adaptation
Do demographics make prediction better?

### Code
* Plots - plotting scripts
* Processing_Scripts - exploration and preprocessing the data
* Helper code -
 * split data into proportions: split_data.ipynb (was needed for earlier experiments)
 * distance between datasets: Distance metric.ipynb (variance along the features used for the finding the distance)
 * combine pairs of symptoms: Combine Symptoms.ipynb
 * combine the demographics with the symptoms: Combine demographics with symptoms.ipynb
 * Split the data into train and test split: split_train_test.ipynb
 * Balance the data to have equal positive & negative samples: Data balancing.ipynb
 * Transform the covariate shift in the data: Handle Covariate Shift.ipynb
* Models - (6 models)
 * logistic regression combining all the source data but without the target data: Logistic Regression All data.ipynb
 * model is trained only on some portion of the target data and the rest if used for testing: Target only.ipynb

### Submission
The poster submitted to 2018 NIPS Machine Learning for Health workshop can be found under NIPS folder.

The paper 'Population-aware Hierarchical Bayesian Domain Adaptation' can be found at <https://arxiv.org/abs/1811.08579>.
