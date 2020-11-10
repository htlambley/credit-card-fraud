# Feature Selection
Not all the features in our dataset contribute equally to the model. In fact, some features may not be any use to the model at all, and simply add noise which could be misinterpreted and overfit upon by the learner.

Linear models are very straightforward to understand as we can examine the coefficients we have fit. Other models such as neural network classifiers are generally much more difficult to interpret. We will use a variety of methods to determine which features seem to have more predictive power, and try to remove the features which offer little benefit.

## Feature Permutation
We can apply a general method to any model in order to test how important a certain feature is. The idea is as follows:
1. Construct a learner from the training data.
2. Given test data \\((x_1, y_1), \dots, (x_n, y_n)\\) and a chosen feature \\(k\\), permute the \\(k\\)th feature of the samples so that they no longer correspond to the correct sample.
3. Measure the drop in score of the learner between the unpermuted and permuted data. 

We can choose any arbitrary scoring method, but we will choose the average precision score to remain consistent with the previous experiments. A large decrease in score indicates that a feature seems important, and little change implies the feature has a small impact on the model. If the score increases, the learner would be more accurate by not considering that feature at all. That might be because it has overfit and learned noise in that feature; this is an issue we should be aware of to ensure our model can generalise.

A similar approach is discussed in [1], and we use the Python [ELI5](https://eli5.readthedocs.io/en/latest/) package to generate the feature importance values. 

<small>[1] Leo Breiman. *Random Forests*. Machine Learning 45, pp. 5–32. (2001) [URL](https://link.springer.com/article/10.1023/A:1010933404324)</small>

### Applying Feature Permutations to Random Forests
As the random forest models performed best in the baseline model construction, we'll perform feature permutation on these to try and determine which features are proving to be the most influential.

In the previous section, we discussed using `StandardScaler` in a pipeline. We won't do this for the below example because in principle, a random forest should not be affected by scaling the variables: the decision rules can easily be adapted by the model without the need to preprocess.
```python
{{#include ../../src/feature_importances.py}}
```
#### Output
```
feature
V12       3.402100e-01
V14       1.212035e-01
V17       8.958874e-02
V11       6.773449e-02
V10       4.367965e-02
V3        3.511822e-02
V4        3.028638e-02
V9        1.818182e-02
V16       1.515152e-02
V7        7.878788e-03
V13       6.060606e-03
V6        4.545455e-03
V8        3.636364e-03
V15       1.616162e-03
V2        9.090909e-04
V28       1.554312e-16
V1        1.332268e-16
V18       1.110223e-16
V22       8.881784e-17
V19       0.000000e+00
V23       0.000000e+00
V25       0.000000e+00
V21      -3.535354e-03
V24      -4.444444e-03
Time     -5.050505e-03
V5       -6.666667e-03
Amount   -8.080808e-03
V27      -8.888889e-03
V20      -9.595960e-03
V26      -9.595960e-03
Name: weight, dtype: float64
```
We can see that over all the folds, the features `V12`, `V14`, `V17`, `V11` and `V10` seemed to have the most predictive power. The features `V21` and the ones below actively harmed our model, and removing them would improve the area under the precision–recall curve.

## Regularised Models
The idea of regularisation is rooted in [Occam's razor](https://en.wikipedia.org/wiki/Occam%27s_razor), or the principle that a simpler explanation should be favoured over a more complex one. This is highly relevant for machine learning models in order to prevent overfitting and extremely complex models.

We can try to reduce overfitting and unnecessarily complex models by adding a *penalty* to large weights. In other words, we modify the algorithms we use so that a large weight is considered "worse" than a low weight, given equal performance. The challenge is to obtain the right balance between penalising weights and allowing a sufficiently complex model to explain the data well.

Two popular regularisations schemes are to use the \\( \ell^1 \\) and \\( \ell^2 \\) norms, defined by
\\[ \lVert x \rVert_{\ell^1} = \sum_{i = 1}^n |x_i|, \\]
\\[ \lVert x \rVert_{\ell^2} = \Bigg ( \sum_{i = 1}^n x_i^2 \Bigg )^\frac12. \\]

The \\(\ell^1\\) norm can be seen as penalising any weight, and so we tend to find that optimal values are *sparse* and contain many zeros. This idea can be exploited in combination with linear regression to find a linear model using a smaller number of coefficients: in effect, selecting the most powerful explanatory features.

### LASSO
LASSO is a linear regression model which is broadly similar to the ordinary least squares regression method, with the added condition that the coeffients have a bounded \\(\ell^1\\) norm. So, the model looks for coefficients of the form
\\[ Y = \beta_0 + \beta_1 X_1 + \cdots + \beta_n X_n, \\]
and tries to minimise the least squares best fit line with the added condition that \\( \lVert \beta \rVert_{\ell^1} \leq K\\) for some chosen regularisation term \\(K\\). A more detailed derivation of the LASSO method can be found in [2, §3.4.3].

We can use a LASSO regression model to suggest which features are most important in the linear regression model too. 

```python
{{#include ../../src/feature_importances_lasso.py}}
```
#### Output
```
1      0.661
2      0.747
3      0.692
4      0.751
5      0.690

V11       0.007156
V4        0.002706
Amount    0.000000
V15      -0.000000
V1       -0.000000
V2        0.000000
V5       -0.000000
V6       -0.000000
V8        0.000000
V9       -0.000000
V13      -0.000000
V28       0.000000
Time     -0.000000
V22       0.000000
V25       0.000000
V18      -0.000000
V19       0.000000
V27       0.000000
V20       0.000000
V26       0.000000
V21       0.000000
V23      -0.000000
V24      -0.000000
V7       -0.013871
V3       -0.015057
V16      -0.015804
V10      -0.020023
V12      -0.029102
V14      -0.037817
V17      -0.042780
dtype: float64
```

A similar story emerges: `V17`, `V14`, `V11`, `V12` and `V4` seem to be a few of the most significant features. 

<small>[2] Trevor Hastie et al. *The Elements of Statistical Learning: Data Mining, Inference and Prediction*. 1st ed. Springer, New York, NY. (2001)</small>

## Dropping Features
We can try to drop the features that we think are of low importance to see how much this affects the model. If the model improves, or doesn't get much worse, we might decide to choose the simpler model. 

Using scikit-learn's `SelectFromModel` we can programmatically choose which features are worth keeping.
```python
{{#include ../../src/feature_dropping.py }}
```
### Output
```
Average precision score
-------------
Fold   Score
1      0.795
2      0.830
3      0.835
4      0.839
5      0.801
```
The support vector machine doesn't seem to have been affected much by the dropping of low importance features. We might want to keep this simpler model.