# Feature Engineering
Feature engineering involves using the raw data in some way to construct new features, perhaps addind domain knowledge to the dataset where we have it. We'll explore some options we have to improve our models by preprocessing the data.

## Interaction Features
We will explore the idea of *interaction features*, sometimes also called feature crosses [1]. The idea is to create new features that are the product of existing ones, such as a feature representing the value of `V1` multiplied by `V2`. This allows linear models to learn more complex functions of the input data. Some of our other methods such as the random forest classifier might not benefit as much because they can more easily represent features such as `V1 x V2`.

To do this, we can use scikit-learn's `PolynomialFeatures` transformer. It is easy to chain this with a scaler using the `Pipeline` we introduced previously. We'll test this on a linear model as the interaction features are likely to benefit this type of model most.

<small>[1] Google Developers. *Feature Crosses*. In: *Machine Learning Crash Course*. [URL](https://developers.google.com/machine-learning/crash-course/feature-crosses/video-lecture). (2020)</small>

```python
{{#include ../../src/feature_interactions.py}}
```
### Output
```
Average precision score
-------------
Fold   Score
1      0.772

2      0.838

3      0.827

4      0.848

5      0.829
```
This is a good improvement on the linear model that did not contain the feature crosses. While the area under the precision–recall curve does not beat the random forest classifier, we've obtained another much-improved classifier.

## Encoding Time
As discussed in the exploratory data analysis, there appears to be a pattern in the `Time` feature that we would like the model to consider, but it is less likely to be learned due to its complexity. We can try to add the time in seconds modulo 86400 (the number of seconds per day) to see if this allows the model to learn more easily.
```python
{{#include ../../src/time_feature.py}}
```
### Output
```
Average precision score
-------------
Fold   Score
1      0.829

2      0.860

3      0.836

4      0.869

5      0.851
```
The results give a mixed picture. While some folds saw an improvements, others saw a decline, so on balance it doesn't seem to matter too much whether we include the `Time` feature or not.