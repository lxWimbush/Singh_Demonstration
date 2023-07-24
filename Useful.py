import scipy.stats as sps
import numpy as np
from scipy.special import digamma
from scipy.special import betaln
from scipy.special import beta as betafun
from pba import Interval

def _spsconvert(theta, func):
    if isinstance(theta, Interval):
        C = Interval(func(theta.left), func(theta.right))
    elif isinstance(theta, (list, tuple, np.ndarray)):
        C = [func(t) for t in theta]
    else:
        C = func(theta)
    return C

def possibilitytool(theta, func, core = None):
    target = lambda t: _spsconvert(t, lambda theta: func(theta))
    if not(isinstance(theta, (list, tuple, np.ndarray))):
        theta = [theta]
    C = []
    for c in theta:
        # If theta intersects the core of the positive structure, then max possibiltiy is 1.
        if not core is None:
            if core.straddles(c): C.append(1)
        else:
            # Otherwise either evaluate the point or take the max of the endpoints
            if isinstance(c, Interval):
                C.append(max(target(c)))
            else:
                C.append(target(c))
    # if theta wasn't iterable, return it as a non-iterable.
    if len(C) == 1: C = max(C)
    return C

def FuncPossibility(Structs, Output, fun, precis = 0.001):
    Poss = Interval(0, 1)
    if fun(*[S.cut(Poss.right) for S in Structs]).intersection(Output) is None:
        divisions = int(np.log(precis)/np.log(0.5))+1
        for _ in range(divisions):
            P = Poss.midpoint()
            if fun(*[S.cut(P) for S in Structs]).intersection(Output) is None:
                Poss.right = P
            else:
                Poss.left = P
    else:
        Poss = Interval(1)
    return Poss

def quickclop(p, k, n):
    if k == 0:
        A = 0
    else:
        A = sps.beta.cdf(p, k, n-k+1)
    if k == n:
        B = 1
    else:
        B = sps.beta.cdf(p, k+1, n-k)
    return Interval(A, B)

# Clopper-Pearson implementation. Perhaps redundant due to pba implementation.
class ClopperPearson():
    # Define a structure with left and right bounds based on the observed binary data.
    def __init__(self, k, n):
        # if k or n are themselves intervals, give the envelope of these possibilities.
        if any([isinstance(x, Interval) for x in [k, n]]):
            self.n, self.k = [Interval(x) for x in [k, n]]
            L = [self.n.left - self.k.right, self.k.right + 1]
            R = [self.n.right - self.k.left + 1, self.k.left]
        # Otherwise just sort out the cbox as usual.
        else:
            self.n, self.k = n, k
            L = [self.k, self.n - self.k + 1]
            R = [self.k + 1, self.n - self.k]
        # Define the cbox as the envelope of two beta distributions.
        self.box = beta(
                [L[0], R[0]],
                [L[1], R[1]]
                )
        self.core = self.ppf(0.5)

    # Get the cumulative density of a given theta, just convert to allow for interval outputs.
    def cdf(self, theta):
        return _spsconvert(
            theta, 
            lambda t: Interval(np.nan_to_num(self.box.cdf(t), nan=[1,0]))
            )

    # Get the theta interval of a given percentile, just convert to allow for interval outputs.
    def ppf(self, theta):
        return _spsconvert(
            theta, 
            lambda t: Interval(np.nan_to_num(self.box.ppf(t), nan=[0,1]))
            )
    
    # Assess the possiiblity of a given theta, whether precise or an interval. 
    def possibility(self, theta):
        # Convert the cbox into a possibility structure
        target = lambda c: _spsconvert(
            c, 
            lambda t: max(1-abs(1-2*np.nan_to_num(self.box.cdf(t), nan=[0,1])))
            )
        # Make sure theta is iterable.
        if not(isinstance(theta, (list, tuple, np.ndarray))):
            theta = [theta]
        C = []
        # Iterate through theta and provide possibiltiy for each.
        for c in theta:
            # If theta intersects the core of the positive structure, then max possibiltiy is 1.
            if self.core.straddles(c):
                C+=[1]
            else:
                # Otherwise either evaluate the point or take the max of the endpoints
                if isinstance(c, Interval):
                    C+=[max(target(c))]
                else:
                    C+=[target(c)]
        # if theta wasn't iterable, return it as a non-iterable.
        if len(C) == 1: C = max(C)
        return C
    
    # Provide an alpha cut of the possibility transform of the cbox.
    def cut(self, alpha):
        if isinstance(alpha, (tuple, list, np.ndarray)):
            L = [self.ppf(a/2).left for a in alpha]
            R = [self.ppf(1-a/2).right for a in alpha]
            I = [Interval(L[i], R[i]) for i in range(len(alpha))]
        else:
            L = self.ppf(alpha/2).left
            R = self.ppf(1-alpha/2).right
            I = Interval(L, R)
        return I

class NormTPivot():
    def __init__(self, data, origin = 0.5):
        self.data = data
        self.sample_mean = np.mean(data)
        self.sample_std = np.std(data, ddof = 1)
        self.sample_n = len(data)
        self.origin = origin
        self.Dist = sps.t(self.sample_n - 1, loc = self.sample_mean, scale = self.sample_std/(self.sample_n)**0.5)
    
    def possibility(self, theta):
        return possibilitytool(theta, lambda t: 1-abs(1-self.Dist.cdf(t)*2))
    
    def cut(self, alpha):
        aU = self.origin+(1-alpha)*(1-self.origin)
        aL = self.origin-(1-alpha)*self.origin
        return Interval(self.Dist.ppf(aL), self.Dist.ppf(aU))


class BalchStruct():
    def __init__(self, k, n, precision = 0.0001):
        self.k = k
        self.n = n
        self.precision = precision
        self.Walley_Ints()
        self.steps = np.zeros(len(self.WallInts))

    def Walley_Ints(self):
        self.WallInts = np.zeros(self.n + 3)
        for i in range(0, self.n + 1):
            if i < (self.k - 1):
                gki = (1/(self.k - i - 1))*(betaln(self.k, self.n - self.k + 1)-betaln(i + 1, self.n - i))
            elif i == (self.k - 1):
                gki = digamma(self.k) - digamma(self.n - self.k + 1)
            elif i == self.k:
                gki = -np.inf
            elif i == (self.k + 1):
                gki = digamma(self.k + 1) - digamma(self.n - self.k)
            elif i > (self.k + 1):
                gki = (1/(i - self.k - 1))*(betaln(i, self.n - i + 1)-betaln(self.k + 1, self.n - self.k))
            self.WallInts[i] = 1/(1+np.exp(-gki))
        self.WallInts[-2] = 1
        self.WallInts[-1] = 0

    def possibility(self, theta):
        if not isinstance(theta, Interval):
            r = 0
            if self.k > 0:
                for i in range(0, int(self.k)):
                    if self.WallInts[i - 1] <= theta and theta < self.WallInts[i]:
                        r = sps.binom.cdf(i - 1, self.n, theta) + 1 - sps.binom.cdf(self.k - 1, self.n, theta)
            if self.WallInts[self.k - 1] <= theta and theta <= self.WallInts[self.k + 1]:
                r = 1
            if self.k < self.n:
                for i in range(int(self.k + 1), int(self.n+1)):
                    if self.WallInts[i] < theta and theta <= self.WallInts[i + 1]:
                        r = 1 - sps.binom.cdf(i, self.n, theta) + sps.binom.cdf(self.k, self.n, theta)
        else:
            if theta.straddles(Interval(self.WallInts[self.k - 1], self.WallInts[self.k + 1])):
                r = 1
            else:
                r = max(self.possibility(theta.left), self.possibility(theta.right))
        return r
    
    def cut(self, alpha, precision = None):
        if precision is None:
            precision = self.precision
        LI, RI = np.arange(0, self.k), np.arange(self.k+1, self.n+1)
        # get left bound
        if self.k == 0:
            LBound = 0
        else:
            for i in LI[::-1]:
                if not self.steps[i-1] == 0:
                    LPoss = self.steps[i-1]
                else:
                    self.steps[i-1] = self.possibility(self.WallInts[i-1])
                    LPoss = self.steps[i-1]
                if LPoss<=alpha:
                    if not self.steps[i] == 0:
                        LPoss = self.steps[i]
                    else:
                        self.steps[i] = self.possibility(self.WallInts[i])
                        LPoss = self.steps[i]
                    L = i
                    break
            if LPoss-sps.binom.pmf(L, self.n, self.WallInts[L])<=alpha<=LPoss:
                LBound = self.WallInts[L]
            else:
                LBound = Interval(self.WallInts[L-1], self.WallInts[L])
                if LBound.width() > 0:
                    divisions = int(np.log(precision/LBound.width())/np.log(0.5))+1
                    for _ in range(divisions):
                        if self.possibility(LBound.midpoint())>=alpha:
                            LBound = Interval(LBound.left, LBound.midpoint())
                        else:
                            LBound = Interval(LBound.midpoint(), LBound.right)
                LBound = LBound.left
        if self.k == self.n:
            RBound = 1
        else:
            for i in RI:
                if not self.steps[i+1] == 0:
                    RPoss = self.steps[i+1]
                else:
                    self.steps[i+1] = self.possibility(self.WallInts[i+1])
                    RPoss = self.steps[i+1]
                if RPoss<=alpha:
                    if not self.steps[i] == 0:
                        RPoss = self.steps[i]
                    else:
                        self.steps[i] = self.possibility(self.WallInts[i])
                        RPoss = self.steps[i]
                    R = i
                    break
            if RPoss-sps.binom.pmf(R, self.n, self.WallInts[R])<=alpha<=RPoss:
                RBound = self.WallInts[R]
            else:
                RBound = Interval(self.WallInts[R], self.WallInts[R+1])
                if RBound.width() > 0:
                    divisions = int(np.log(precision/RBound.width())/np.log(0.5))+1
                    for _ in range(divisions):
                        if self.possibility(RBound.midpoint())>=alpha:
                            RBound = Interval(RBound.midpoint(), RBound.right)
                        else:
                            RBound = Interval(RBound.left, RBound.midpoint())
                RBound = RBound.right
        return Interval(LBound, RBound)

def betaintersect(a, b1, b2):
    return 1-(betafun(a, b1)/betafun(a, b2))**(1/(b1-b2))

def rside(b, k):
    return 1-np.exp(digamma(b-k+1))/np.exp(digamma(b+1))

class KNBalchStruct(BalchStruct):
    def __init__(self, k, n, maxn = None, precision = 0.0001):
        if maxn is None:
            self.maxn = n*100
        else:
            self.maxn = maxn
        self.k = k
        self.n = n
        self.precision = precision
        self.Walley_Ints()
        self.steps = np.zeros(len(self.WallInts))
        
    def Walley_Ints(self):
        self.WallInts = [
            betaintersect(self.k, i-self.k, self.n-self.k) for i in range(self.k, self.n)
            ] + [rside(self.n-1,self.k), rside(self.n,self.k)
            ] + [betaintersect(self.k, i-self.k+1, self.n-self.k+1) for i in range(self.n+1, self.maxn)
            ]

    def possibility(self, theta):
        i = len([I for I in self.WallInts if I>=theta])
        r = 0
        # r = sps.nbinom.pmf(self.n-self.k, p=theta, n=self.n)
        if i <= 1:
            r = sps.beta.sf(theta, self.k, self.n-self.k) if self.n>self.k else 1
        elif i<=(self.n-self.k):
            r = sps.beta.sf(theta, self.k, self.n-self.k) + sps.beta.cdf(theta, self.k, i-1) if self.n>self.k else 1
        elif i<=(self.n-self.k+1):
            r = 1
        elif i<len(self.WallInts):
            r = sps.beta.sf(theta, self.k, i) + sps.beta.cdf(theta, self.k, self.n-self.k+1)
        elif i==len(self.WallInts):
            r = (
                sps.beta.cdf(self.WallInts[-1], self.k, self.n-self.k+1) + 
                sps.beta.sf(self.WallInts[-1], self.k, i)
            )
        return r
        
class ECDFStruct():
    def __init__(self, data, origin = 0, lbound = None, rbound = None):
        self.data = np.sort(data) # Maybe unnecessary? Useful to get it into a numpy array
        self.n = len(self.data)
        assert not Interval(0,1).intersection(origin) is None, 'Invalid Origin, please provide a value between 0 and 1.'
        self.origin = origin
        if lbound == None:
            self.lbound = -np.inf
        else:
            self.lbound = lbound
        if rbound == None:
            self.rbound = np.inf
        else:
            self.rbound = rbound

    def possibility(self, theta):
        k = sum(self.data<=theta)
        if (k+1)<=(self.n+1)*self.origin:
            a = (k+1)/((self.n+1)*self.origin)
        elif (k)>=(self.n+1)*self.origin:
            a = (self.n+1-k)/((self.n+1)*(1-self.origin))
        else:
            a = 1
        return a
    
    def cut(self, alpha):
        L = int(np.floor(alpha*self.origin*(self.n+1))-1)
        if L == -1:
            L = self.lbound
        else:
            L = self.data[L]
        if self.origin <=1:
            R = int(np.ceil((1-alpha*(1-self.origin))*(self.n+1)))
            if R == self.n+1:
                R = self.rbound
            else:
                R = self.data[R-1]
        else:
            R = self.rbound
        return Interval(L, R)

def scoreinterval(conf, k, n):
    phat = k/n
    z = sps.norm.ppf(1-(1-conf)/2)
    denom = 1+(z**2)/n
    numer = phat+((z**2)/(2*n))+(
        Interval(-1,1)*(z*((phat*(1-phat)+(z**2)/(4*n))/n)**0.5))
    return numer/denom