# Frontiers of Machine Learning
We will briefly discuss some of the alternative approaches we could explore with the dataset.

## Oversampling and undersampling
To resolve the imbalanced class issue, we could try *oversampling* or *undersampling*. Undersampling refers to keeping only a fraction of the majority class (genuine transactions) so that the ratio between genuine and fraud is balanced. On our dataset, we would ignore the vast majority of the data if we undersampled. That may not be a bad thing in itself, but we would need to compare undersampling against doing no preprocessing to see which approach worked better.

Oversampling involves adding additional samples to the dataset, which may consist of copies of the fraudulent transactions, or some samples generated from the existing ones. We could simply add random copies of the fraudulent samples to the dataset until we reached a balance, or try and perform some process to generate "new" data. One such approach is known as SMOTE [1], which was shown by the authors to perform better than undersampling on their datasets. More recent techniques include using a *generative adversarial network* to synthetically create new samples, as described in [2].


<small>[1] Nitesh Chawla et al. *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research 16, pp. 321–357. (2002)</small><br/>
<small>[2] Georgios Douzas and Fernando Bacao. *Effective data generation for imbalanced learning using conditional generative adversarial networks*. Expert Systems with Applications 91, pp. 464—471. [DOI](https://doi.org/10.1016/j.eswa.2017.09.030). (2018)</small>

## Autoencoders
Instead of performing a traditional dimensionality reduction algorithm such as prinicpal component analysis, it is possible to use a neural network approach to *learn* an encoding of the features into a lower dimension. This type of network is known as an *autoencoder*, and dimensionality reduction using these is an active research topic. An overview of the topic is given in [3].

<small>[3] Wang et al. *Auto-encoder based dimensionality reduction*. Neurocomputing 184, pp. 232–242. (2016)</small>

## Neural Networks
We tried some simple multi-layer perceptron (MLP) neural networks, but finding a good neural network architecture is an art in itself. 
Using packages such as [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/), we could experiment much further with 
neural networks. It may be the case that a deeper approach would be able to learn patterns that were hard to learn in a single layer.

In principle a single-layer feedforward neural network can approximate any continuous function: this is the celebrated *Universal Approximation Theorem*, derived independently by George Cybenko [4], Kurt Hornik et al. [5] and Ken-Ichi Funahashi [6] in the late 1980s. The results given do not say how well a single-layer network converges, however, and in practice a more reasonable network may need more layers in order to converge in a reasonable time.

<small>[4] George Cybenko. *Approximation by superpositions of a sigmoidal function*. Mathematics of Control, Signals and Systems 2, pp. 303–314. [DOI](https://doi.org/10.1007/BF02551274). (1989)</small><br/>
<small>[5] Kurt Hornik et al. *Multilayer feedforward networks are universal approximators*. Neural networks 2.5, pp. 359–366. [DOI](https://doi.org/10.1016/0893-6080(89)90020-8). (1989)</small><br/>
<small>[6] Ken-Ichi Funahashi. *On the approximate realization of continuous mappings by neural networks*. Neural Networks 2.3, pp. 183–192. [DOI](https://doi.org/10.1016/0893-6080(89)90003-8). (1989) 