# Mean Robust Optimization
This repository is by 
Irina Wang,
[Bartolomeo Stellato](https://stellato.io/),
Cole Becker,
and [Bart Van Parys](https://mitsloan.mit.edu/faculty/directory/bart-p-g-van-parys),
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
+ Python 3.x/numpy/pandas
+ [PyTorch](https://pytorch.org)
+ [Mosek](https://www.mosek.com/)
+ [sklearn](https://scikit-learn.org/stable/modules/clustering.html#clustering)

## Instructions
### Running experiments
Experiments can be run in their respective folders using the commands below.


Continuous portfolio: 
```
python portcont.py
```
Sparse portfolio:
```
python portMIP.py
```
Facility location:
```
python facility.py
```
Continuous newsvendor:
```
python main.py
```
Sparse newsvendor:
```
python newsMIP.py   
```
Quadratic concave:
```
python quadratic.py
```
Log-sum-exp concave:
```
python logsumexp.py 
```

### Generating plots

After running the experiments above, plots can then be generated by running in their respective folders:

```
python plots.py
```
