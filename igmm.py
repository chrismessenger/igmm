import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, gamma
import scipy.special as spec
import argparse
import time
import copy
import time
import corner
import sys 
from ARS import ARS

# the maximum positive integer for use in setting the ARS seed
MAXINT = sys.maxint

class Sample:
    """Class for defining a single sample"""
    def __init__(self,mu,s,pi,lam,r,beta,w,alpha,k):
        self.mu = np.reshape(mu,(1,-1))
        self.s = np.reshape(s,(1,-1))
        self.pi = np.reshape(pi,(1,-1))
        self.lam = lam
        self.r = r
        self.beta = beta
        self.w = w
        self.k = k
        self.alpha = alpha

class Samples:
    """Class for generating a collection of samples"""
    def __init__(self):
        self.sample = []

    def __getitem__(self, key): 
        return self.sample[key]

    def addsample(self,S):
        return self.sample.append(S)

def IntegralApprox(y,lam,r,beta,w,size=100):
    """estimates the integral in Eq.17 of Rasmussen (2000)"""
    mu = norm.rvs(loc=lam,scale=1.0/np.sqrt(r),size=size)
    s = drawGammaRas(beta,1.0/w,size=size)
    return np.mean(norm.pdf(y,loc=mu,scale=1.0/np.sqrt(s)))

def logpalpha(alpha,k=1,N=1):
    """The log of Eq.15 in Rasmussen (2000)"""
    return (k-1.5)*np.log(alpha) - 0.5/alpha + spec.gammaln(alpha) - spec.gammaln(N+alpha)

def logpalphaprime(alpha,k=1,N=1):
    """The derivative (wrt alpha) of the log of Eq.15 in Rasmussen (2000)"""
    return (k-1.5)/alpha + 0.5/(alpha*alpha) + spec.psi(alpha) - spec.psi(alpha+N)

def logpbeta(beta,k=1,s=1,w=1):
    """The log of the second part of Eq.9 in Rasmussen (2000)"""
    return -k*spec.gammaln(0.5*beta) - 0.5/beta + 0.5*(k*beta-3.0)*np.log(0.5*beta) + 0.5*beta*np.sum(np.log(w) + np.log(s)) - beta*np.sum(0.5*s*w)

def logpbetaprime(beta,k=1,s=1,w=1):
    """The derivative (wrt beta) of the log of Eq.9 in Rasmussen (2000)"""
    return -0.5*k*spec.psi(0.5*beta) + 0.5/(beta**2) + 0.5*k*np.log(0.5*beta) + 0.5*(k*beta-3)/beta + 0.5*np.sum(np.log(s) + np.log(w) - s*w)

def drawGammaRas(a,theta,size=1):
    """Returns Gamma distributed variables according to 
    the Rasmussen (2000) definition"""
    return gamma.rvs(0.5*a,loc=0,scale=2.0*theta/a,size=size)

def drawNormal(mean=0,var=1,size=1):
    """Returns normally distributed variables"""
    return norm.rvs(loc=mean,scale=np.sqrt(var),size=size)

def drawIndicator(nvec,pvec):
    """Draws stochastic indicator values from multinomial distributions """
    res = np.zeros(pvec.shape[1])
    
    # loop over each data point
    for j in xrange(pvec.shape[1]):
        c = np.cumsum(pvec[:,j])        # the cumulative un-scaled probabilities
        R = np.random.uniform(0,c[-1],1)        # a random number  
        r = (c-R)>0                     # truth table (less or greater than R)
        y = (i for i,v in enumerate(r) if v)    # find first instant of truth
        try:
            res[j] = y.next()           # record component index
        except:                 # if no solution (must have been all zeros)
            res[j] = np.random.randint(0,pvec.shape[0]) # pick uniformly
    return res

def drawAlpha(k,N,size=1):
    """Draw alpha from its distribution (Eq.15 Rasmussen 2000) using ARS
    Make it robust with an expanding range in case of failure"""
    flag = True
    cnt = 0
    while flag:
        xi = np.logspace(-2-cnt,3+cnt,50)       # update range if needed
        try:
            ars = ARS(logpalpha,logpalphaprime,xi=xi,lb=0, ub=np.inf, k=k, N=N)
            flag = False
        except:
            cnt += 1
    
    # draw alpha but also pass random seed to ARS code
    return ars.draw(size,np.random.randint(MAXINT))

def drawBeta(k,s,w,size=1):
    """Draw beta from its distribution (Eq.9 Rasmussen 2000) using ARS
    Make it robust with an expanding range in case of failure"""
    flag = True
    cnt = 0
    while flag:
        xi = np.logspace(-2-cnt,3+cnt,50)       # update range if needed
        try:
            ars = ARS(logpbeta,logpbetaprime,xi=xi,lb=0, ub=np.inf, k=k, s=s, w=w)
            flag = False
        except:
            cnt += 1

    # draw beta but also pass random seed to ARS code
    return ars.draw(size,np.random.randint(MAXINT))

def plotResult(Samp,Y,Nburn,Nstep,mu,s,pi,outfile,N=250):
    """Plots the original data, the true distribution, and the
    90% confidence regions estimated from the analysis"""
    
    # initialis the plot and the x-axis ranges
    plt.figure()
    miny = np.min(mu-5.0/np.sqrt(s))
    maxy = np.max(mu+5.0/np.sqrt(s))
    yvec = np.linspace(miny,maxy,N)
    truth = np.zeros(N)

    # histogram the data
    plt.hist(Y,np.sqrt(len(Y)),histtype='stepfilled',align='mid',color='r',alpha=0.5,normed=True)

    # make the true curve
    for i,m in enumerate(mu):
        truth = truth + pi[i]*norm.pdf(yvec,loc=mu[i],scale=1.0/np.sqrt(s[i]))
    
    # if we have passed samples then generate confidence bounds 
    # as a function of y
    if Samp!=None:
        fun = np.zeros((3,N))
        for idx,yy in enumerate(yvec):
            temp = []
            bb = 0
            for aa,S in enumerate(Samp):
                if (aa > Nburn) and ((aa - Nburn) % Nstep == 0):
                    temp.append(0)
                    for newpi,newmu,news in zip(S.pi,S.mu,S.s):
                        for newnewpi,newnewmu,newnews in zip(newpi,newmu,news):
                            temp[bb] += newnewpi*norm.pdf(yy,loc=newnewmu,scale=1.0/np.sqrt(newnews))
                    bb += 1
            temp = np.reshape(np.squeeze(temp),(1,-1))
            fun[0,idx] = np.percentile(temp,50)
            fun[1,idx] = np.percentile(temp,5)
            fun[2,idx] = np.percentile(temp,95)

        # plot shaded 90% region (between 5% and 95%) and median curve
        plt.fill_between(yvec, fun[1,:], fun[2,:], edgecolor='b', facecolor='b', alpha=0.5, interpolate=True)
        plt.plot(yvec,fun[0,:],'b',linewidth=2)
    
    # plot true distribution and label plot
    plt.plot(yvec,truth,'--k',linewidth=2)
    plt.xlim([miny, maxy])
    plt.xlabel(r'$y$',fontsize=16)
    plt.ylabel(r'$p(y)$',fontsize=16)
    plt.savefig(outfile)

def plotchains(Samp,truths,label,outfile):
    """Plot the chains of the mean, precision, or pi variables"""
    plt.figure()
    i = 0
    # choose the quantity and plot it
    if label=='$\mu$':
        for sample in Samp:
            plt.plot(i,sample.mu,'.k')
            i += 1
    elif label=='$s$':
        for sample in Samp:
            plt.plot(i,sample.s,'.k')
            i += 1
    elif label=='$\pi$':
        for sample in Samp:
            plt.plot(i,sample.pi,'.k')
            i += 1
    else:
        print 'ERROR : no known parameter {}'.format(label)
        exit(1)

    # plot the true values and label the plot
    for t in truths:
        plt.plot((0, i-1), (t, t), 'r--')
    plt.xlabel(r'$i$',fontsize=16)
    plt.ylabel(label,fontsize=16)
    if np.min(truths) == np.max(truths):
        plt.ylim([np.min(truths)-1, np.max(truths)+1])
    else:
        R = np.max(truths)-np.min(truths)
        plt.ylim([np.min(truths)-0.5*R, np.max(truths)+0.5*R])
    plt.savefig(outfile)

# command line parser
def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='igmm.py',description='Applies an infinite Gaussian mixture model to data')
    parser.add_argument('-s', '--prec', type=float, nargs='+', default=[0.5,1,2], help='the precision of each Gaussian component')
    parser.add_argument('-m', '--mu', type=float, nargs='+', default=[-3,1,15], help='the means of the simulated Gaussians')
    parser.add_argument('-p', '--pi', type=float, nargs='+', default=[0.5,0.25,0.25], help='the simulated Gaussian weights')
    parser.add_argument('-N', '--Ndata', type=int, default=100, help='the number of input data samples to use')
    parser.add_argument('-n', '--Nsamples', type=int, default=100, help='the number of samples to produce')
    parser.add_argument('-a', '--Nint', type=int, default=100, help='the number of samples used in approximating the tricky integral')
    parser.add_argument('-z', '--seed', type=int, default=1, help='the random seed')
    parser.add_argument('-v', '--verb', action='count', default=0)
    return parser.parse_args()

# the main part of the code
def main():
    """Takes command line args and computes samples from the joint posterior
    using Gibbs sampling"""

    # record the start time
    t = time.time()

    # get the command line args
    args = parser()
    if args.seed>0:
        np.random.seed(args.seed)
    N = args.Ndata

    # generate some data to start with
    args.pi = args.pi/np.sum(args.pi)
    M = np.transpose(np.random.multinomial(N, args.pi, size=1))
    Y = []
    for i,m in enumerate(M):
        Y.append(norm.rvs(loc=args.mu[i],scale=1.0/np.sqrt(args.prec[i]),size=m))
    Y = np.random.permutation(np.hstack(Y))

    # compute some data derived quantities
    muy = np.mean(Y)
    sig2y = np.var(Y)
    inv_sig2y = 1.0/sig2y

    # initialise a single sample
    Samp = Samples()
    c = np.zeros(N)                             # initialise the stochastic indicators
    pi = np.zeros(1)                           # initialise the weights
    mu = np.zeros(1)                           # initialise the means
    s = np.zeros(1)                            # initialise the precisions
    n = np.zeros(1)                             # initialise the occupation numbers
    mu[0] = muy                                # set first mu to the mean of all data    
    pi[0] = 1.0                                # only one component so pi=1
    beta = 1.0/drawGammaRas(1.0,1.0)           # draw beta from prior
    w = drawGammaRas(1.0,sig2y)                # draw w from prior
    s[0] = drawGammaRas(beta,1.0/w)          # draw s from prior
    n[0] = N                                    # all samples are in the only component
    lam = drawNormal(mean=muy,var=sig2y)       # draw lambda from prior
    r = drawGammaRas(1.0,inv_sig2y)            # draw r from prior
    alpha = 1.0/drawGammaRas(1.0,1.0)          # draw alpha from prior
    k = 1                                      # set only 1 component
    S = Sample(mu,s,pi,lam,r,beta,w,alpha,k)       # define the sample
    Samp.addsample(S)                           # add the sample
    
    # loop over samples
    z = 1
    while z<args.Nsamples:
    
        # for each represented muj value
        ybarj = [np.sum(Y[np.argwhere(c==j)])/nj for j,nj in enumerate(n)]
        temp_sigSq = np.hstack([1.0/(nj*sj + r) for nj,sj in zip(n,s)])
        mu = drawNormal(mean=(n*ybarj*s + lam*r)*temp_sigSq,var=temp_sigSq,size=k)

        # for lambda (depends on mu vector, k, and r)
        temp_sigSq = 1.0/(inv_sig2y + k*r)
        temp_mu = (muy*inv_sig2y + r*np.sum(mu))*temp_sigSq
        lam = drawNormal(mean=temp_mu,var=temp_sigSq)

        # for r (depnds on k, mu, and lambda)
        r = drawGammaRas(k+1,(k+1)/(sig2y + np.sum((mu-lam)**2)))

        # from alpha (depends on k)
        alpha = drawAlpha(k,N)

        # for each represented sj value (depends on mu, c, beta, w)
        temp = [np.sum((Y[np.argwhere(c==j)]-mu[j])**2) for j in xrange(k)]
        s = drawGammaRas(beta + n,(beta + n)/(w*beta + temp),size=k)

        # compute the unrepresented probability 
        p_unrep = [alpha/(N-1.0+alpha)*IntegralApprox(Yi,lam,r,beta,w,size=args.Nint) for Yi in Y]
        p_temp = np.outer(np.ones(k+1),p_unrep)
        
        # for the represented components
        for j in xrange(k):
            nij = n[j] - (c==j).astype(int)
            idx = nij>0         # only apply to indices where we have multi occupancy
            p_temp[j,idx] = nij[idx]/(N-1.0+alpha)*np.sqrt(s[j])*np.exp(-0.5*s[j]*(Y[idx]-mu[j])**2)

        # stochastic indicator (we could have a new component)
        jvec = np.arange(k+1)
        c = np.hstack(drawIndicator(jvec,p_temp))

        # for w
        w = drawGammaRas(k*beta + 1.0,(k*beta + 1.0)/(inv_sig2y + beta*np.sum(s)))

        # from beta
        beta = drawBeta(k,s,w)
        
        # sort out based on new stochastic indicators
        nij = np.sum(c==k)        # see if the *new* component has occupancy
        if nij>0:
            # draw from priors and increment k
            newmu = drawNormal(mean=lam,var=1.0/r)
            news = drawGammaRas(beta,1.0/w)
            mu = np.concatenate((mu,newmu))
            s = np.concatenate((s,news)) 
            k = k + 1
    
        # find unrepresented components
        n = np.array([np.sum(c==j) for j in xrange(k)])
        badidx = np.argwhere(n==0)
        Nbad = len(badidx)

        # remove unrepresented components
        if Nbad>0:
            mu = np.delete(mu,badidx)
            s = np.delete(s,badidx)
            for cnt,i in enumerate(badidx):
                idx = np.argwhere(c>=(i-cnt))
                c[idx] = c[idx]-1
            k -= Nbad        # update component number

        # recompute n
        n = np.array([np.sum(c==j) for j in xrange(k)])

        # from pi
        pi = n.astype(float)/np.sum(n)

        # add sample
        S = Sample(mu,s,pi,lam,r,beta,w,alpha,k)
        newS = copy.deepcopy(S)
        Samp.addsample(newS)
        print '{}: generated {}/{} samples'.format(time.asctime(),z,args.Nsamples)
        z += 1
    
    # print computation time
    print "time to complete = %.3f sec" % (time.time()-t)    

    # make corner plot for prior params (lambda,r,beta,w,alpha,k)
    x = [[sample.lam,sample.r,np.log(sample.beta),sample.w,sample.alpha] for sample in Samp] 
    x = np.transpose(np.hstack(x))
    figure = corner.corner(x, labels=[r"$\lambda$", r"$r$", r"$\log \beta$",
                                     r"$w$", r"$\alpha$"],
                         quantiles=[0.16, 0.5, 0.84],
                         show_titles=True, title_args={"fontsize": 16})
    figure.savefig("./priorparams.png")

    # plot chains for mu,s, and pi
    plotchains(Samp,args.mu,r'$\mu$','./muchains.png')
    plotchains(Samp,args.prec,r'$s$','./schains.png')
    plotchains(Samp,args.pi,r'$\pi$','./pichains.png')
    
    # plot the represented components
    plt.figure()
    kvec = [samples.k for samples in Samp]
    kvec = kvec[int(0.75*args.Nsamples)::5]     # take only the last 25% and every 5th
    kmax = 3 + np.max(kvec)
    plt.hist(kvec,bins=np.arange(kmax),histtype='stepfilled',align='left',color='b',normed=True)
    plt.xlim([0,kmax-1])
    plt.xlabel(r'$k$',fontsize=16)
    plt.ylabel(r'$P(k)$',fontsize=16)
    plt.savefig('./components.png')

    # plot the distribution
    plotResult(Samp,Y,int(0.75*args.Nsamples),5,args.mu,args.prec,args.pi,'./distributions.png')
    print 'success.'    

if __name__ == "__main__":
    exit(main())


#gibbs()
