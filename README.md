# Detecting Credit Card Fraud with Machine Learning
A classification model for Worldline &amp; Université Libre de Bruxelles' credit card fraud dataset. 

## Installation and Usage
This project requires Python 3 and the following packages:
- `numpy`
- `scikit-learn`
- `pandas`.

Usage of [Anaconda](https://www.anaconda.com/products/individual) is recommended for convenience, but it is also perfectly acceptable
 to install the required packages with `pip` (e.g. using your operating system's package manager).

**The dataset must also be downloaded separately**. This is due to technical constraints on the maximum allowed file size on GitHub.
A zip archive can be downloaded directly from Kaggle by clicking [here](https://www.kaggle.com/mlg-ulb/creditcardfraud/download). Unzip the given CSV file as `creditcard.csv` in the same directory as this `README.md` file.

In order to generate the accompanying book, use the [mdBook](https://github.com/rust-lang/mdBook) package. Binaries are available [here](https://github.com/rust-lang/mdBook/releases). To build the documentation, use:
```shell
mdbook build
```
