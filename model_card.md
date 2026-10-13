# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was developed by Basil Atiyeh as part of the Udacity Machine Learning DevOps program. It is a scikit-learn `RandomForestClassifier` (version 1.5) with 100 trees, a minimum of 5 samples per leaf and a fixed random state of 42. The other hyperparameters use the scikit-learn defaults. Categorical features are one-hot encoded with a `OneHotEncoder`, and the label is binarized with a `LabelBinarizer`. The trained model and the fitted encoder are saved as `model/model.pkl` and `model/encoder.pkl`.

## Intended Use

The model predicts whether a person's annual income is above or below $50,000 from demographic and employment attributes in the census data. It is intended for learning purposes, specifically to demonstrate an end-to-end pipeline that trains a model, tests it and serves predictions through a FastAPI REST API. It should not be used to make real decisions about individuals, such as hiring, lending or eligibility for services.

## Training Data

The data comes from the UCI Census Income dataset, which was extracted from the 1994 US Census database. The provided `census.csv` file has 32,561 rows and 15 columns. Of these, 8 are categorical features, 6 are numeric features and 1 is the `salary` label. About 76% of the rows are labeled `<=50K` and 24% are labeled `>50K`, so the classes are imbalanced. The data was split with scikit-learn's `train_test_split`, and 80% of the rows (26,048) were used for training. The split was stratified on `salary`, so both sets keep the same class balance.

## Evaluation Data

The remaining 20% of the rows (6,513) were held out as the test set. The test set was processed with the encoder and label binarizer fitted on the training data, so no information from the test set was used during training.

## Metrics

The model was evaluated with precision, recall and F1 score, where the positive class is `>50K`. Precision measures how many of the predicted high earners were actually high earners. Recall measures how many of the actual high earners the model found. F1 is the harmonic mean of the two.

On the test set, the model scored a precision of 0.7847, a recall of 0.6135 and an F1 score of 0.6886.

Performance was also measured on slices of the test set, with one slice for every value of every categorical feature. The full results are in `slice_output.txt`. For example, the F1 score was 0.6920 for men and 0.6683 for women. By race, it was 0.6937 for White individuals, 0.6447 for Black individuals and 0.4615 for American Indian or Eskimo individuals.

## Ethical Considerations

The dataset contains sensitive attributes, including race, sex and native country. The model uses them as features, so its predictions can reflect the income disparities that existed in the 1994 data. The slice results show uneven performance across groups. Recall is lower for women (0.5429) than for men (0.6266), which means the model misses more high-earning women. Some groups also have very few test rows, such as 73 American Indian or Eskimo individuals, so their metrics are unreliable. For these reasons, the model should not be used for any decision that affects real people.

## Caveats and Recommendations

The data is more than 30 years old, so incomes and job categories have changed since it was collected. The model is also a baseline with no hyperparameter tuning, and the class imbalance lowers its recall on the `>50K` class. Future work could tune the model with cross-validation, adjust the decision threshold to balance precision and recall, and review performance on small slices with more data. Any new version should be checked with the slice metrics before it is deployed.
