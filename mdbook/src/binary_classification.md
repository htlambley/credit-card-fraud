# Binary Classification
We'll briefly review the mathematics of the problem in the context of *statistical learning theory*. This section assumes a basic familiarity with probability theory, distributions and set notation. A thorough reference on the necessary notions in probability is [1].

We have a dataset consisting of transactions, each of which has \\(n\\) features and a label to say if that transaction is fraudulent or not. More formally, we are given training data
\\[ D = \\Big \\{ (x_i, y_i) \\Big \\} \subset \mathcal{X} \times \mathcal{Y}, \\]
where \\(\mathcal{X}\\) is the *space of features*. As we have \\(n\\) real-valued features, \\(\mathcal{X}\\) can be thought of as the space of real-valued vectors, i.e. \\(\mathcal{X} = \mathbb{R}^n\\). The space \\(\mathcal{Y}\\) is the set of possible labels, which for our *binary classification* problem means \\(\mathcal{Y} = \\{0, 1\\}\\). 

Our goal is to find a suitable classifier — in other words, a function — which we denote \\(h \colon \mathcal{X} \to \mathcal{Y}\\), based on the training data \\(D\\). 

We can model these training data as coming from some unknown *probability distribution* on \\(\mathcal{X} \times \mathcal{Y} \\). This means that there is some underlying rule that gives the probability of any \\( (x, y) \in \mathcal{X} \times \mathcal{Y}\\). If we knew this distribution, we could construct a "perfect" classifier, known as the *Bayes classifier*, which minimises the probability of error. 

There are many distributions where even the Bayes classifier is not completely accurate: suppose \\(X \sim \mathrm{Uniform}[0, 1]\\) and \\(Y \sim \mathrm{Bernoulli}(1/2)\\) with \\(X\\) and \\(Y\\) independent. Intuitively, \\(X\\) does not predict \\(Y\\) at all, so no learner can be constructed that will perform any better than random guessing on the distribution. While we can happily try to fit a model on the training data, this will merely "learn" using the noise in the problem. The Bayes classifier in this case can be constructed by just picking 0 or 1 with equal probability. On the entire distribution, we would expect this classifier to be correct 50% of the time. Any other learner will be worse than this on the distribution.

### Risk

We would like this classifier to *generalise well* to new data sampled from the distribution. So, given a sample \\((x, y) \in \mathcal{X} \times \mathcal{Y} \\), we can choose some *loss function* \\( L(h(x), y) \\) which measures how badly incorrect the prediction \\(h(x)\\) is, and define the *empirical risk* of a classifier given \\(m\\) points \\( \\{ (x_i, y_i) \\} \\) to be

\\[ \hat{R}(h) = \frac1m \sum_{i = 1}^m L(h(x_i), y_i). \\]

The empirical loss gives us the loss on our training data, but we really wish to minimise the loss over the entire distribution. This quantity is known as the *(generalisation) risk*, \\(R(h)\\). Let \\((X, Y)\\) be random variables with the distribution on \\(\mathcal{X} \times \mathcal{Y} \\). Then the risk is given by
\\[ R(h) = \mathbb{E}\big[L(h(X), Y)\big]. \\]

### Empirical Risk Minimisation
We will construct classifiers according to the *empirical risk minimisation* principle [2, §1.5]. Since we do not know the distribution on \\(\mathcal{X} \times \mathcal{Y}\\), we cannot directly minimise \\(R(h)\\). But, under the assumption that the training data are independent realisations from this distribution, the expectation of the empirical risk is in fact the generalisation risk.

Therefore, we wish to find parameters for any model we construct which minimise the empirical risk, and hope that this leads to a low generalisation risk. 

## Intuition on Data Analysis
For \\(n \leq 3\\), classification can be understood visually: we can plot the points on a graph with their corresponding labels and try to devise a rule which distinguishes each class. In higher dimensions, the idea is the same, but we cannot just plot a graph to understand the geometry of the feature space. 

Many of the algorithms used in machine learning operate on simple principles. Support vector machines, for example, look for a *hyperplane* to separate the points of each class. If the data are *linearly separable*, in other words it's possible to find a hyperplane which splits the two groups, linear support vector machines can learn a rule with zero empirical risk.

K Nearest Neighbours models are equally intuitive: given a point we want to classify, we simply look for \\(k\\) points nearby, and choose the most common label for all of those points.

Part of the challenge of data science is to gain similar intuititon for data with tens, or hundreds, of dimensions, and understand the relationships, which may be highly non-linear.

An interesting visualisation of the links between neural networks and topology is given in [3] which is well worth reading.

<small>[1] Roman Vershynin. *High-Dimensional Probability: An Introduction with Applications in Data Science*. Cambridge University Press, Cambridge. (2018)</small><br/>
<small>[2] Vladimir N. Vapnik. *The Nature of Statistical Learning Theory*. 2nd ed. Springer, New York, NY. (2000)</small><br/>
<small>[3] Christopher Olah. *Neural Networks, Manifolds, and Topology*. [URL](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/). (2014)</small>