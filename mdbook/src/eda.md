# Exploratory Data Analysis
Before we go on to construct models on the dataset, we will first spend some time exploring the dataset to gain some intuition on which approaches ought to work best. This process is often called *exploratory data analysis* as developed by John W. Tukey [1, 2]. It allows us to seek to understand the data without having to develop a hypothesis or model *a priori* and testing if the data support such a hypothesis.

Exploratory data analysis will be particularly important given we do not know much about the features. However, we need to be very careful about the problem of **data leakage** before we go ahead and draw conclusions from the data. 

## Data Leakage
To understand the problem of data leakage, we need to think about how our model will be used in practice:
1. We select a model and fit its parameters using *training data* that we have collected.
2. We then want to maximise its performance on unseen data in the future.

We don't know anything about the data we will encounter in the future: whether our model is for predicting stock prices, supermarket demand or sentiment analysis, we don't have any information to go on other than what we've seen in the past, and what we reasonably expect to happen in the future.

The first step above omits a large number of steps in statistics and data science. We cannot simply pick a model out of thin air in the hope that it will work well on future data: instead we need to have some way of measuring which models work best on the training data. 

This leads us to the idea of *cross-validation*. An excellent reference on this topic is [3, § 7.10]. As discussed in the section *Binary Classification*, so long as we may assume that the training data are independent realisations from the probability distribution, we can use the *empirical risk minimisation* principle to find a good model. To evaluate the performance of the model, we need some additional data independently drawn from the distribution that was not used to fit the model. We should not evaluate the final model performance on the data we trained from, as this will generally provide an estimate of the performance which is higher than the true performance on unseen data.

The simplest solution is to choose a *hold-out* set from the data we have. Given the dataset \\(D \subset \mathcal{X} \times \mathcal{Y}\\), we split \\(D\\) into two smaller sets: a training set \\(T\\) and a validation set \\(V\\). We should take care to ensure that the way we split the data ensures the two sets have a similar class balance. Intuitively, a classifier could learn the wrong *prior probability* for each class if the ratios are different in the training and validation set.

A slightly more sophisticated method splits the data into \\(k\\) *folds*, then trains on \\(k - 1\\) of these folds, while testing on the final fold. This allows us to construct multiple models by trying different combinations of folds: for example, we can split the data into 5 folds, train on 1–4 and test on 5. Or, we could train on 2–5 and test on 1. 

<small>[1] John W. Tukey. *Exploratory Data Analysis*. Addison–Wesley, Reading, MA. (1977)</small><br/>
<small>[2] *Exploratory Data Analysis*. In: *The Concise Encyclopedia of Statistics*. Springer, New York, NY. [DOI](https://doi.org/10.1007/978-0-387-32833-1_136). (2008)</small><br/>
<small>[3] Trevor Hastie et al. *The Elements of Statistical Learning: Data Mining, Inference and Prediction*. 1st ed. Springer, New York, NY. (2001)</small>