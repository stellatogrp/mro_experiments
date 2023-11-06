#load packages and returns data
install.packages("Rsafd")
library(Rsafd)
returns = read.csv(file = 'returns.csv',header = FALSE)
n = 20000
K = dim(returns)[2]
rrets = returns

#calculate the Mean and Covariances
Mu = apply(rrets,2,mean)
Sig = var(rrets)

#Fit generalized Pareto distribution
Ret.est = NULL
for (I in 1:K) Ret.est = c(Ret.est,fit.gpd(rrets[,I],plot=FALSE))

#Generate correlated uniform data using a Gaussian Copula
SD = rmvgaussian.copula(n, Sigma = Sig)

#Generate synthetic data using Monte Carlos simulations from the fitted gPd and uniform data
for (I in 1:K) SD[,I] = qgpd(Ret.est[[I]],SD[,I])

write.csv(SD,"sp500_synthetic_returns.csv", row.names = FALSE)
