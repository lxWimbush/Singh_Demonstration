import scipy.stats as sps
import numpy as np
from pba import Interval
import matplotlib.pyplot as plt
from Useful import NormTPivot

# First example, NormTPivot, single parameter. A good result would be for the Singh plot to dominate the uniform.

## Distribution Parameters
N = 10
mu = 3
std = 1

## Singh plot parameters
NN = 2000

## Generate Datasets using known parameters
data = sps.norm.rvs(loc=mu, scale = std, size=[NN, N])

## Loop through datasets and identify minimum confidence required to bound mu for each one. This is calculated here by generating a confidence structure from centred confidence intervals and calculating the possibility of the true mean on this structure.
alphas = [1-NormTPivot(d).possibility(mu) for d in data]

## Plot the ECDF of the alpha levels
plt.plot(sorted(alphas), np.linspace(0,1,NN), 'k', label = 'Alphas')

## Plot the cumulative distribution of the U(0,1) uniform for comparison
plt.plot([0,1], [0,1], 'k:', label = 'U(0,1)')

plt.legend()
plt.show()



# Second Example, Imprecise Singh Plots, Estimating an uknown probability with a known, fixed, sample size. A good result would be for the Upper bound to be dominated by the uniform, and for the lower bound to dominate the uniform.
from Useful import quickclop

## Distribution parameters
N = 10
p = 0.4

## Singh plot parameters
NN = 2000

## Generate data sets
data = sps.binom.rvs(p=p, n=N, size=NN)

## Loop through datasets and identify minimum confidence required to bound p on each bound from a [0,a] one sided interval.
alphas = [quickclop(p, d, N) for d in data]

## Plot the ECDF of the alpha levels for each bound
### Lower Bound
plt.plot(sorted([a.left for a in alphas]), np.linspace(0,1,NN), 'k', label = 'Lower')
### Upper Bound
plt.plot(sorted([a.right for a in alphas]), np.linspace(0,1,NN), 'r', label = 'Upper')

## Plot the cumulative distribution of the U(0,1) uniform for comparison
plt.plot([0,1], [0,1], 'k:', label = 'U(0,1)')

plt.legend()
plt.show()



# Third example, structure Singh plots. Basically a repeat of the above but using a confidence structure instead. A good result would be for the Singh plot to dominate the uniform.
from Useful import BalchStruct

## Distribution parameters
N = 10
p = 0.4

## Singh plot parameters
NN = 2000

## Generate data sets
data = sps.binom.rvs(p=p, n=N, size=NN)

## Loop through datasets and identify minimum confidence required to bound p on each bound from a [0,a] one sided interval.
alphas = [1-BalchStruct(d, N).possibility(p) for d in data]

## Plot the ECDF of the alpha levels for each bound
### Lower Bound
plt.plot(sorted(alphas), np.linspace(0,1,NN), 'k', label = 'Alphas')

## Plot the cumulative distribution of the U(0,1) uniform for comparison
plt.plot([0,1], [0,1], 'k:', label = 'U(0,1)')

plt.legend()
plt.show()

# Fourth Example, global singh plot for estimating probabilities with a fixed sample size. A good result would be for the minimum rate of coverage to dominate the uniform.

## Fixed Distribution parameters
N = 10

## Singh Plot Parameters
GN = 100 # Number of parameters to try
NN = 1000

## Scan over varying parameters
ps = np.linspace(0,1,GN)

# Calculate minimum confidence levels for coverage for each set of data sets.
alphas = [sorted([1-BalchStruct(d, N).possibility(p) for d in sps.binom.rvs(p=p, n=N, size=NN)]) for p in ps]

[plt.plot(A, np.linspace(0,1,NN), alpha = 0.3) for A in alphas]
plt.plot(np.max(alphas, axis=0), np.linspace(0,1,NN), 'k')

## Plot the cumulative distribution of the U(0,1) uniform for comparison
plt.plot([0,1], [0,1], 'k:', label = 'U(0,1)')
plt.show()

## Note that a low sample count for the Singh plot may obscure good performance. Try upping the sample count if unsure.

## Fixed Distribution parameters
N = 10

## Singh Plot Parameters
GN = 100 # Number of parameters to try
NN = 10000

## Scan over varying parameters
ps = np.linspace(0,1,GN)

# Calculate minimum confidence levels for coverage for each set of data sets.
alphas = [sorted([1-BalchStruct(d, N).possibility(p) for d in sps.binom.rvs(p=p, n=N, size=NN)]) for p in ps]

## Plot coverages
[plt.plot(A, np.linspace(0,1,NN), alpha = 0.3) for A in alphas]

## Plot minimum observed coverage.
plt.plot(np.max(alphas, axis=0), np.linspace(0,1,NN), 'k', lavbel = 'Min Coverage')

## Plot the cumulative distribution of the U(0,1) uniform for comparison
plt.plot([0,1], [0,1], 'k:', label = 'U(0,1)')
plt.legend()
plt.show()

# Example Five, ECDF
# Distribution parameters
dof = 3
XN = 20

# data
data = sps.chi2.rvs(dof, size=[NN, XN+1])

# alphas
alphas = [1-ECDFStruct(d[:-1]).possibility(d[-1]) for d in data]

## Plot the ECDF of the alpha levels for each bound
### Lower Bound
plt.plot(sorted(alphas), np.linspace(0,1,NN), 'k', label = 'Alphas')

## Plot the cumulative distribution of the U(0,1) uniform for comparison
plt.plot([0,1], [0,1], 'k:', label = 'U(0,1)')

plt.legend()
plt.show()


## Final example, Singh plot for methods that don't have a direct inverse. Estimate minimum confidence by binary search. Output will be a function of two inputs, one a probability and the other a random deviate drawn from a Chi Squared distribution.
from Useful import FuncPossibility, ECDFStruct

# Distribution 1 parameters
p = 0.3
N = 100

# Distribution 2 parameters
dof = 3
XN = 200

# Singh plot parameters
NN = 1000

# Distribution 1 data
data1 = sps.binom.rvs(p=p, n=N, size=NN)
data2 = sps.chi2.rvs(dof, size=[NN, XN+1])

# Calculate confidence levels using a binary search algorithm
alphas = [1-FuncPossibility([BalchStruct(data1[i], N), ECDFStruct(data2[i][:-1], origin = 0.5)], p*data2[i][-1], lambda a, b: a*b).left for i in range(NN)]

## Plot the ECDF of the alpha levels for each bound
### Lower Bound
plt.plot(sorted(alphas), np.linspace(0,1,NN), 'k', label = 'Alphas')

## Plot the cumulative distribution of the U(0,1) uniform for comparison
plt.plot([0,1], [0,1], 'k:', label = 'U(0,1)')

plt.legend()
plt.show()
