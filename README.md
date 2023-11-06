# Mean Robust Optimization
This repository is by
[Irina Wang](https://sites.google.com/view/irina-wang),
Cole Becker,
[Bart Van Parys](https://mitsloan.mit.edu/faculty/directory/bart-p-g-van-parys),
and [Bartolomeo Stellato](https://stellato.io/),
and contains the Python source code to
reproduce the experiments in our paper
"[Mean Robust Optimization](http://arxiv.org/abs/2207.10820)."

If you find this repository helpful in your publications,
please consider citing our paper.

## Introduction
Robust optimization is a tractable and expressive technique for decision-making under uncertainty, but it can lead to overly conservative decisions because of pessimistic assumptions on the uncertain parameters.
Wasserstein distributionally robust optimization can reduce conservatism by being closely data-driven, but it often leads to very large problems with prohibitive solution times.
We introduce mean robust optimization, a general framework that combines the best of both worlds by providing a trade-off between computational effort and conservatism.
Using machine learning, we define uncertainty sets and constraints around clustered data points, undergoing a significant size reduction while preserving key properties.
We show finite-sample performance guarantees and conditions for which clustering does not increase conservatism, and otherwise provide bounds on the effect on conservatism.
Numerical examples illustrate the high efficiency and performance preservation of our approach.

## Dependencies
Install dependencies with
```
pip install -r requirements.txt
```

## Instructions
### Running experiments
Experiments can from the root folder using the commands below.

Facility location:
```
python facility/facility.py --foldername facility/
python facility/plots.py --foldername facility/
```
Capital budgeting
```
python capital_budgeting/capital.py --foldername capital_budgeting/
python capital_budgeting/plots.py --foldername capital_budgeting/
```
Quadratic concave:
```
python quadratic_concave/quadratic.py --foldername quadratic_concave/
python quadratic_concave/plots.py --foldername quadratic_concave/
```
Log-sum-exp concave:
```
python logsumexp/logsumexp.py --foldername logsumexp/
python logsumexp/plots.py --foldername logsumexp/
```
Newsvendor problem:
```
python newsvendor/newsvendor.py --foldername newsvendor/
python newsvendor/plots.py --foldername newsvendor/
```
Sparse portfolio optimization:
You need to run 'generate_synthetic_data.R" using R, to generate the csv file of the dataset. Then, run the python files.
```
python portfolio/portMIP.py --foldername portfolio/
python portfolio/plots.py --foldername portfolio/
```
