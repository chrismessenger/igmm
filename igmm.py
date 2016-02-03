#########################################################################
#    igmm.py - An implementation of an infinite Gaussian mixture model
#    Copyright (C) 2016  C.Messenger
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.stats import norm, uniform, gamma, chi2 # wishart
from scipy.stats import multivariate_normal as mv_norm
from numpy.linalg import inv,eig,det,solve,cond,cholesky,slogdet
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
    def __init__(self,N,nd):
        self.sample = []
        self.N = N
        self.nd = nd

    def __getitem__(self, key): 
        return self.sample[key]

    def addsample(self,S):
        return self.sample.append(S)

# the sampler
def igmm_sampler(Y,Nsamples,missmat=None,Nint=1,anneal=False,verb=False):
    """Takes command line args and computes samples from the joint posterior
    using Gibbs sampling

    input:
        Y - the input dataset
        Nsamples - the number of Gibbs samples
        missmat - the matrix indicating missing data
        Nint - the samples used for evaluating the tricky integral
        anneal - perform simple siumulated annealing
        verb - show verbose output

    output:
        Samp - the output samples

    """

    # first rescale the data to be more manageable
    Y,scale = scaledata(Y,missmat)

    # compute some data derived quantities
    N,nd = Y.shape
    muy = np.zeros(nd)
    covy = np.zeros((nd,nd))
    for i in xrange(nd):
        idx_i = np.argwhere(np.squeeze(missmat[:,i])==0)
        muy[i] = np.mean(Y[idx_i,i])
        covy[i,i] = np.var(Y[idx_i,i])
    inv_covy = inv(covy) if nd>1 else np.reshape(1.0/covy,(1,1))
    if verb:
        print '{}: mean(Y) = {}'.format(time.asctime(),np.reshape(muy,(1,-1)))
        print '{}: cov(Y) = {}'.format(time.asctime(),np.reshape(covy,(1,-1)))
        print '{}: min(Y) = {}'.format(time.asctime(),np.min(Y,0))
        print '{}: max(Y) = {}'.format(time.asctime(),np.max(Y,0))

    # compute indices of data with any missing elements
    missidx = []
    for i,m in enumerate(missmat):
        idx = np.argwhere(m==1)
        if idx.size:
            missidx.append(i)
            for j in idx:
                Y[i,j] = 0.0

    # initialise a single sample
    Samp = Samples(Nsamples,nd)

    c = np.zeros(N)            # initialise the stochastic indicators
    pi = np.zeros(1)           # initialise the weights
    mu = np.zeros((1,nd))      # initialise the means
    s = np.zeros((1,nd*nd))    # initialise the precisions
    n = np.zeros(1)            # initialise the occupation numbers

    mu[0,:] = muy              # set first mu to the mean of all data
    pi[0] = 1.0                # only one component so pi=1
    temp = drawGamma(0.5,2.0/float(nd))
    beta = np.squeeze(float(nd) - 1.0 + 1.0/temp)     # draw beta from prior
    w = drawWishart(nd,covy/float(nd))                # draw w from prior
  
    # draw s from prior
    s[0,:] = np.squeeze(np.reshape(drawWishart(float(beta),inv(beta*w)),(nd*nd,-1)))

    n[0] = N                   # all samples are in the only component
    lam = drawMVNormal(mean=muy,cov=covy)       # draw lambda from prior
    r = drawWishart(nd,inv(nd*covy))            # draw r from prior
    alpha = 1.0/drawGamma(0.5,2.0)              # draw alpha from prior
    k = 1                                       # set only 1 component
    S = Sample(mu,s,pi,lam,r,beta,w,alpha,k)    # define the sample
    Samp.addsample(S)                           # add the sample
    print '{}: initialised parameters'.format(time.asctime())

    # loop over samples
    z = 1
    oldpcnt = 0
    while z<Nsamples:

        # define simulated annealing temperature
        G = max(1.0,float(0.5*Nsamples)/float(z+1)) if anneal else 1.0

        # sample missing data
        for m in missidx:

            # compute component probabilities
            idx = np.argwhere(missmat[m,:]==0).reshape(-1)
            nidx = np.argwhere(missmat[m,:]==1).reshape(-1)

            # conditionally draw from that component
            Y[m,nidx] = drawmissing(mu[c[m],:],s[c[m],:].reshape(nd,nd),nd,idx,nidx,Y[m,:])

        # recompute muy and covy
        muy = np.mean(Y,axis=0)
        covy = np.cov(Y,rowvar=0)
        inv_covy = inv(covy) if nd>1 else np.reshape(1.0/covy,(1,1))

        # for each represented muj value
        ybarj = [np.sum(Y[np.argwhere(c==j),:],0)/nj for j,nj in enumerate(n)]
        mu = np.zeros((k,nd))
        j = 0
        for yb,nj,sj in zip(ybarj,n,s):
            sj = np.reshape(sj,(nd,nd))
            muj_cov = inv(nj*sj + r)
            muj_mean = np.dot(muj_cov,nj*np.dot(sj,np.squeeze(yb)) + np.dot(r,lam))
            mu[j,:] = drawMVNormal(mean=muj_mean,cov=muj_cov,size=1)
            j += 1

        # for lambda (depends on mu vector, k, and r)
        lam_cov = inv(inv_covy + k*r)
        lam_mean = np.dot(lam_cov,np.dot(inv_covy,muy) + np.dot(r,np.sum(mu,0)))
        lam = drawMVNormal(mean=lam_mean,cov=lam_cov)

        # for r (depnds on k, mu, and lambda)
        temp = np.zeros((nd,nd))
        for muj in mu:
            temp += np.outer((muj-lam),np.transpose(muj-lam))
        r = drawWishart(k+nd,inv(nd*covy + temp))

        # from alpha (depends on k)
        alpha = drawAlpha(k,N)

        # for each represented sj value (depends on mu, c, beta, w)
        for j,nj in enumerate(n):
            temp = np.zeros((nd,nd))
            temptemp = np.zeros((nd,nd))
            idx = np.argwhere(c==j)
            yj = np.reshape(Y[idx,:],(idx.shape[0],nd))
            for yi in yj:
                temp += np.outer((mu[j,:]-yi),np.transpose(mu[j,:]-yi))
            temp_s = drawWishart(beta + nj,inv(beta*w + temp))
            s[j,:] = np.reshape(temp_s,(1,nd*nd))

        # compute the unrepresented probability - apply simulated annealing
        # here
        p_unrep = (alpha/(N-1.0+alpha))*IntegralApprox(Y,lam,r,beta,w,G,size=Nint)
        p_temp = np.outer(np.ones(k+1),p_unrep)

        # for the represented components
        for j in xrange(k):
            nij = n[j] - (c==j).astype(int)
            idx = np.argwhere(nij>0)         # only apply to indices where we have multi occupancy
            temp_s = G*np.reshape(s[j,:],(nd,nd))     # apply simulated annealing to this parameter
            Q = np.array([np.dot(np.squeeze(Y[i,:]-mu[j,:]),np.dot(np.squeeze(Y[i,:]-mu[j,:]),temp_s)) for i in idx])
            p_temp[j,idx] = nij[idx]/(N-1.0+alpha)*np.reshape(np.exp(-0.5*Q),idx.shape)*np.sqrt(det(temp_s))

        # stochastic indicator (we could have a new component)
        jvec = np.arange(k+1)
        c = np.hstack(drawIndicator(jvec,p_temp))

        # for w
        w = drawWishart(k*beta + nd,inv(nd*inv_covy + beta*np.reshape(np.sum(s,0),(nd,nd))))

        # from beta
        beta = drawBeta(k,s,w)

        # sort out based on new stochastic indicators
        nij = np.sum(c==k)        # see if the *new* component has occupancy
        if nij>0:
            # draw from priors and increment k
            newmu = drawMVNormal(mean=lam,cov=inv(r))
            news = drawWishart(float(beta),inv(beta*w))
            mu = np.concatenate((mu,np.reshape(newmu,(1,nd))))
            s = np.concatenate((s,np.reshape(news,(1,nd*nd))))
            k = k + 1

        # find unrepresented components
        n = np.array([np.sum(c==j) for j in xrange(k)])
        badidx = np.argwhere(n==0)
        Nbad = len(badidx)

        # remove unrepresented components
        if Nbad>0:
            mu = np.delete(mu,badidx,axis=0)
            s = np.delete(s,badidx,axis=0)
            for cnt,i in enumerate(badidx):
                idx = np.argwhere(c>=(i-cnt))
                c[idx] = c[idx]-1
            k -= Nbad        # update component number

        # recompute n
        n = np.array([np.sum(c==j) for j in xrange(k)])

        # from pi
        pi = n.astype(float)/np.sum(n)

        pcnt = int(100.0*z/float(Nsamples))
        if pcnt>oldpcnt:
            print '{}: %--- {}% complete ----------------------%'.format(time.asctime(),pcnt)
            if verb:
                print '{}: ybarj = {}'.format(time.asctime(),np.reshape(ybarj,(1,-1)))
                print '{}: mu = {}'.format(time.asctime(),np.reshape(mu,(1,-1)))
                print '{}: lam = {}'.format(time.asctime(),np.reshape(lam,(1,-1)))
                print '{}: r = {}'.format(time.asctime(),np.reshape(r,(1,-1)))
                print '{}: alpha = {}'.format(time.asctime(),np.reshape(alpha,(1,-1)))
                print '{}: s = {}'.format(time.asctime(),np.reshape(s,(1,-1)))
                print '{}: w = {}'.format(time.asctime(),np.reshape(w,(1,-1)))
                print '{}: beta = {}'.format(time.asctime(),np.reshape(beta,(1,-1)))
                print '{}: k = {}'.format(time.asctime(),k)
                print '{}: n = {}'.format(time.asctime(),np.reshape(n,(1,-1)))
                print '{}: pi = {}'.format(time.asctime(),np.reshape(pi,(1,-1)))
            oldpcnt = pcnt

        # add sample
        S = Sample(mu,s,pi,lam,r,beta,w,alpha,k)
        newS = copy.deepcopy(S)
        Samp.addsample(newS)
        z += 1

    # rescale the samples
    Samp,Y = rescaleResults(Samp,Y,scale)

    return Samp,Y

def rescaleResults(Samp,Y,scale):
    """Rescales the samples back to the original data scale
    input:
        Samp - the samples
        scale - the menas and stdevs of the orginal data
    output:
        Samp - the rescaled samples
    """
    
    # rescale the samples
    nd = Samp.nd
    N = Samp.N
    temp = np.outer(scale[:,1],scale[:,1])
    for i in xrange(N):

        # rescale only the dimensionful parameters
        # leave pi, beta, alpha, k untouched 
        samples = Samp[i]
        m = np.reshape(samples.mu,(samples.k,nd))
        s = np.reshape(samples.s,(samples.k,nd,nd))
        for j in xrange(samples.k):
            m[j] = m[j]*scale[:,1] + scale[:,0]
            s[j] /= temp
        Samp[i].lam = Samp[i].lam*scale[:,1] + scale[:,0]
        Samp[i].r /= temp
        Samp[i].w /= temp    

    # rescale data
    for i in xrange(Y.shape[0]):
        Y[i,:] = Y[i,:]*scale[:,1] + scale[:,0]

    return Samp,Y

def scaledata(Y,missmat=None):
    """Scales the data to have unit variance and zero mean
    inputs:
        Y - input data
        missmat - missing data matrix
    outputs:
        Y - rescaled data
        scale - the scale parameters
    """

    # if no miss matrix set then make and empty one
    if missmat is None:
        missmat = np.zeros(Y.shape)

    j = 0
    _,nd = Y.shape
    scale = np.zeros((nd,2))  # stores the means and variances of each dimension
    for i,y in zip(missmat.transpose(),Y.transpose()):
        z = y[np.argwhere(i==0)]
        scale[j,0] = np.mean(z)
        scale[j,1] = np.std(z)
        Y[:,j] = (Y[:,j] - scale[j,0])/scale[j,1]
        Y[np.argwhere(i==1),j] = 0
        j += 1

    return Y,scale

def IntegralApprox(y,lam,r,beta,w,G=1,size=100):
    """estimates the integral in Eq.17 of Rasmussen (2000)"""
    temp = np.zeros(len(y))
    inv_betaw = inv(beta*w)
    inv_r = inv(r)
    i = 0
    bad = 0
    while i<size:
        mu = mv_norm.rvs(mean=lam,cov=inv_r,size=1)
        s = drawWishart(float(beta),inv_betaw)
        try:
            temp += mv_norm.pdf(y,mean=np.squeeze(mu),cov=G*inv(s))
        except:
            bad += 1
            pass
        i += 1
    return temp/float(size)

def logpalpha(alpha,k=1,N=1):
    """The log of Eq.15 in Rasmussen (2000)"""
    return (k-1.5)*np.log(alpha) - 0.5/alpha + spec.gammaln(alpha) - spec.gammaln(N+alpha)

def logpalphaprime(alpha,k=1,N=1):
    """The derivative (wrt alpha) of the log of Eq.15 in Rasmussen (2000)"""
    return (k-1.5)/alpha + 0.5/(alpha*alpha) + spec.psi(alpha) - spec.psi(alpha+N)

def logpbeta(beta,k=1,s=1,w=1,nd=1,logdetw=1,temp=1):
    """The log of the second part of Eq.9 in Rasmussen (2000)"""
    return -1.5*np.log(beta - nd + 1.0) \
        - 0.5*nd/(beta - nd + 1.0) \
        + 0.5*beta*k*nd*np.log(0.5*beta) \
        + 0.5*beta*k*logdetw \
        + 0.5*beta*temp \
        - k*spec.multigammaln(0.5*beta,nd)

def logpbetaprime(beta,k=1,s=1,w=1,nd=1,logdetw=1,temp=1):
    """The derivative (wrt beta) of the log of Eq.9 in Rasmussen (2000)"""
    psi = 0.0
    for j in xrange(1,nd+1):
        psi += spec.psi(0.5*beta + 0.5*(1.0 - j)) 
    return -1.5/(beta - nd + 1.0) \
        + 0.5*nd/(beta - nd + 1.0)**2 \
        + 0.5*k*nd*(1.0 + np.log(0.5*beta)) \
        + 0.5*k*logdetw \
        + 0.5*temp \
        - 0.5*k*psi  

def drawGammaRas(a,theta,size=1):
    """Returns Gamma distributed variables according to
    the Rasmussen (2000) definition"""
    return gamma.rvs(0.5*a,loc=0,scale=2.0*theta/a,size=size)

def drawGamma(a,theta,size=1):
    """Returns Gamma distributed variables"""
    return gamma.rvs(a,loc=0,scale=theta,size=size)

def drawWishart(df,scale):
    """Returns Wishart distributed variables"""
    """Currently broken in scipy so using alternative"""
    #return wishart.rvs(df=df,scale=scale,size=size)
    return wishartrand(df,scale)

def wishartrand(nu, phi):
    """Returns wishart distributed variables (modified from
    https://gist.github.com/jfrelinger/2638485)"""
    dim = phi.shape[0]
    chol = cholesky(phi)
    foo = np.tril(norm.rvs(loc=0,scale=1,size=(dim,dim)))
    temp = [np.sqrt(chi2.rvs(nu-(i+1)+1)) for i in np.arange(dim)]
    foo[np.diag_indices(dim)] = temp
    
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

def drawMVNormal(mean=0,cov=1,size=1):
    """Returns multivariate normally distributed variables"""
    return mv_norm.rvs(mean=mean,cov=cov,size=size)

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
        xi = np.logspace(-2-cnt,3+cnt,200)       # update range if needed
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
    nd = w.shape[0]
    
    # precompute some things for speed
    logdetw = slogdet(w)[1]
    temp = 0
    for sj in s:
        sj = np.reshape(sj,(nd,nd))
        temp += slogdet(sj)[1]
        temp -= np.trace(np.dot(w,sj))
    
    lb = nd - 1.0
    flag = True
    cnt = 0
    while flag:
        xi = lb + np.logspace(-3-cnt,1+cnt,200)       # update range if needed
        flag = False
        try:
            ars = ARS(logpbeta,logpbetaprime,xi=xi,lb=lb,ub=np.inf, \
                k=k, s=s, w=w, nd=nd, logdetw=logdetw, temp=temp)
        except:
            cnt += 1
            flag = True

    # draw beta but also pass random seed to ARS code
    return ars.draw(size,np.random.randint(MAXINT))

def greedy(x):
    """computes the enclosed probability
    """
    s = x.shape
    x = np.reshape(x,(1,-1))
    x = np.squeeze(x/np.sum(x))
    idx = np.squeeze(np.argsort(x))
    test = x[idx]
    z = np.cumsum(x[idx])
    d = np.zeros(len(z))
    d[idx] = z
    return 1.0 - np.reshape(d,s)

def computemargp(Samp,xvec,randidx,nel=1e6):
    """computes the 2 and 1 marginalised posteriors
    """ 

    nd = Samp.nd
    Ngrid = xvec.shape[1]
    if nd>1:
        n = min(int(np.log10(float(nel))/np.log10(float(Ngrid))),nd)
        nel = Ngrid**n
        nchunk = (Ngrid**(nd-n))
    else:
        nel = Ngrid
        nchunk = 1
    

    # loop over manageable chunks of the grid and compute the result
    res2 = np.zeros((nd,nd,Ngrid,Ngrid)) if nd>1 else np.zeros(Ngrid)
    Np = np.array(Ngrid**np.arange(nd))
    idx = np.zeros((nel,nd)).astype('int')
    for cnt in xrange(nchunk):
        
        # make grid locations for this chunk
        temp = np.array(cnt*nel + np.arange(nel))
        idx[:,0] = np.array(temp/Np[-1]).astype('int')
        for i in xrange(1,nd):
            temp = temp - idx[:,i-1]*Np[nd-i]
            idx[:,i] = np.array(temp/Np[nd-i-1]).astype('int')
        grid = np.array([xvec[i,idx[:,i]] for i in xrange(nd)]).transpose()

        # loop over the samples
        prob = np.zeros(nel)
        for k in randidx:
            samples = Samp[k]
            s = np.reshape(samples.s,(samples.k,nd*nd))
            m = np.reshape(samples.mu,(samples.k,nd))
            p = np.reshape(np.array(np.squeeze(samples.pi)),(-1,1))
            
            # loop over the components
            for b in xrange(samples.k):
                prob += p[b]*mv_norm.pdf(grid,mean=m[b,:],cov=inv(np.reshape(s[b,:],(nd,nd))))

        # for each 2D pair of dimensions
        if nd>1:
            for i in xrange(nd):
                for j in xrange(nd):
                    for k,gc in enumerate(idx):
                        res2[i,j,gc[j],gc[i]] += prob[k]
        else:
            res2 += prob

    # make 1D results
    temp = np.arange(1,nd)
    res1 = np.zeros((nd,Ngrid))
    if nd>1:
        for i in xrange(nd):
            res1[i,:] = np.squeeze(np.sum(res2[0,i,:,:],axis=1))
    else:
        res1 = np.reshape(res2,(1,-1))

    return res2,res1


def plotresult(Samp,Y,outfile,missmat=None,Ngrid=100,M=4,plottype='ellipse'):
    """Plots samples of ellipses drawn from the posterior"""
    
    nd = Samp.nd
    N = Samp.N
    lower = np.min(Y,axis=0)
    upper = np.max(Y,axis=0)
    lower = lower - 0.5*(upper-lower)
    upper = upper + 0.5*(upper-lower)
    xvec = np.zeros((nd,Ngrid))
    for i in xrange(nd):
        xvec[i,:] = np.linspace(lower[i],upper[i],Ngrid)
    label = ['$x_{}$'.format(i) for i in xrange(nd)]
    levels = [0.68, 0.95,0.999]
    alpha = [1.0, 0.5, 0.2]

    plt.figure(figsize = (nd,nd))
    gs1 = gridspec.GridSpec(nd, nd)
    gs1.update(left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0, hspace=0)

    # fill in miss matrix if not set
    if missmat is None:
        missmat = np.zeros(Y.shape)

    # pick random samples to use
    randidx = np.random.randint(N/2,N,M)

    # compute 2 and 1D marginalised probabilities
    if plottype=='map':
        res2,res1 = computemargp(Samp,xvec,randidx)
    
    cnt = 0
    for i in xrange(nd):
        for j in xrange(nd):
            
            ij = np.unravel_index(cnt,[nd,nd])
            ax1 = plt.subplot(gs1[ij])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])    

            # scatter plot the data in lower triangle plots
            if i>j:
                ax1.plot(Y[:,j],Y[:,i],'r.',alpha=0.5,markersize=0.5)
                ax1.set_xlim([lower[j],upper[j]])
                ax1.set_ylim([lower[i],upper[i]])
            elif i==j:  # otherwise on the diagonal plot histograms
                if nd>1:
                    newY = Y[np.argwhere(missmat[:,i]==0),i]
                    ax1.hist(newY,25,histtype='stepfilled',normed=True,alpha=0.5,edgecolor='None',facecolor='red')            
                    ax1.set_xlim([lower[j],upper[j]])
                else:
                    newY = Y[np.argwhere(missmat[:,i]==0),i]
                    plt.hist(newY,25,histtype='stepfilled',normed=True,alpha=0.5,edgecolor='None',facecolor='red')
                    plt.xlim([lower[j],upper[j]])
                    plt.ylim([lower[i],upper[i]])

            if plottype=='ellipse':
                
                # if off the diagonal
                if i>=j:

                    # loop over randomly selected samples
                    for k in randidx:
                        samples = Samp[k]
                        s = np.reshape(samples.s,(samples.k,nd*nd))
                        m = np.reshape(samples.mu,(samples.k,nd))
                        p = np.reshape(np.array(np.squeeze(samples.pi)),(-1,1))
                    
                        # loop over components in this sample
                        for b in xrange(samples.k):
                            tempC = inv(np.reshape(s[b,:],(nd,nd)))
                            ps = tempC[np.ix_([i,j],[i,j])] if i!=j else tempC[i,i]                        

                            # if we have a 2D covariance after projecting
                            if ps.size==4:
                                w,v = eig(ps)
                                e = Ellipse(xy=m[b,[j,i]],width=2.0*np.sqrt(6.0*w[1]), \
                                    height=2*np.sqrt(6.0*w[0]), \
                                    angle=(180.0/np.pi)*np.arctan2(v[0,1],v[0,0]), \
                                    alpha=np.squeeze(p[b]))
                                e.set_facecolor('none')
                                e.set_edgecolor('b')
                                ax1.add_artist(e)
                            elif ps.size==1:
                                if nd>1:
                                    ax1.plot(xvec[i,:],p[b]*norm.pdf(xvec[i,:],loc=m[b,i],scale=np.sqrt(np.squeeze(ps))),'b',alpha=p[b]) 
                                else:
                                    plt.plot(xvec[i,:],p[b]*norm.pdf(xvec[i,:],loc=m[b,i],scale=np.sqrt(np.squeeze(ps))),'b',alpha=p[b])
                            else:
                                print '{}: ERROR strange number of elements in projected matrix'.format(time.asctime())
                                exit(0)
                
            elif plottype=='map':

                if i>j:

                    proj = np.squeeze(res2[i,j,:,:])
                    z = greedy(proj)
                    xtemp = xvec[j,:].flatten()
                    ytemp = xvec[i,:].flatten()
                    for lev,a in zip(levels,alpha):
                        plt.contour(xvec[j,:].flatten(), xvec[i,:].flatten(), np.transpose(z), [lev], \
                            colors='blue',linestyles=['solid'], alpha=a, \
                            linewidth=0.5)
                    newY = Y[np.squeeze(np.argwhere(np.all(missmat==0,1))),:]
                    xY = Y[np.argwhere(missmat[:,i]==1),j]
                    yY = Y[np.argwhere(missmat[:,j]==1),i]
                    ax1.plot(Y[np.argwhere(missmat[:,j]==1),j],Y[np.argwhere(missmat[:,j]==1),i],'k+',alpha=1,markersize=3)
                    ax1.plot(Y[np.argwhere(missmat[:,i]==1),j],Y[np.argwhere(missmat[:,i]==1),i],'ko',alpha=1,markersize=1)
                    ax1.plot(newY[:,j],newY[:,i],'r.',alpha=0.5,markersize=0.5)
                    ax1.plot(np.ones(len(yY))*upper[j],yY,'g+')
                    ax1.plot(xY,np.ones(len(xY))*upper[i],'g+')

                elif i==j:  # for diagonal elements plot 1D marginalised posteriors
                    proj = np.squeeze(res1[i,:])
                    z = proj/(np.sum(proj)*(xvec[i,1]-xvec[i,0]))
                    ax1.plot(xvec[i],z,'b')

            else:
                print '{} : ERROR unknown plottype {}. Exiting.'.format(time.asctime(),plottype)
                exit(1)

            if j>i:
                ax1.axis('off') if nd>1 else plt.axis('off')
            if cnt>=nd*(nd-1):
                plt.xlabel(label[j],fontsize=12)
                ax1.xaxis.labelpad = -5
            if (cnt % nd == 0) and cnt>0:
                plt.ylabel(label[i],fontsize=12)
                ax1.yaxis.labelpad = -3
            cnt += 1

    plt.savefig(outfile,dpi=300)

def extractchain(Samp,label):
    """Extract the chains of the mean, precision, pi etc, variables"""
    
    i = 0
    nd = Samp.nd
    xdata = []
    ydata = []
    # choose the quantity and plot it
    if label=='$\\mu$':
        for sample in Samp:
            m = np.reshape(sample.mu,(sample.k,nd))
            for d in xrange(sample.k):
                xdata.append(i*np.ones(nd))
                ydata.append(np.squeeze(m[d,:]))
            i += 1
    elif label=='$[s]^{-1}$':
        for sample in Samp:
            s = np.reshape(sample.s,(sample.k,nd,nd))
            for d in xrange(sample.k):
                xdata.append(i*np.ones(nd*nd))
                ydata.append(np.squeeze(np.reshape(inv(s[d,:,:]),(1,nd*nd))))
            i += 1
    elif label=='$\\pi$':
        for sample in Samp:
            newp = np.reshape(np.array(np.squeeze(sample.pi)),(-1,1))
            for p in newp:
                xdata.append(i)
                ydata.append(np.squeeze(p))
            i += 1
    elif label=='$\\lambda$':
        for sample in Samp:
            xdata.append(i*np.ones(nd))
            ydata.append(np.squeeze(sample.lam))
            i += 1
    elif label=='$r$':
        for sample in Samp:
            xdata.append(i*np.ones(nd*nd))
            ydata.append(np.squeeze(np.reshape(sample.r,(1,nd*nd))))
            i += 1
    elif label=='$\\log\\beta$':
        for sample in Samp:
            xdata.append(i)
            ydata.append(np.log(np.squeeze(sample.beta)))
            i += 1
    elif label=='$\\beta$':
        for sample in Samp:
            xdata.append(i)
            ydata.append(np.squeeze(sample.beta))
            i += 1
    elif label=='$w$':
        for sample in Samp:
            xdata.append(i*np.ones(nd*nd))
            ydata.append(np.squeeze(np.reshape(sample.w,(1,nd*nd))))
            i += 1
    elif label=='$\\alpha$':
        for sample in Samp:
            xdata.append(i)
            ydata.append(np.squeeze(sample.alpha))
            i += 1
    elif label=='$k$':
        for sample in Samp:
            xdata.append(i)
            ydata.append(np.squeeze(sample.k))
            i += 1
    else:
        print 'ERROR : no known parameter {}'.format(label)
        exit(1)
    
    return np.array(xdata),np.array(ydata)

def plotsamples(Samp,args,chainfile,histfile):
    """Generates plots of samples as a function of index
       and also plots histograms of samples"""

    nd = Samp.nd
    f1, ax1 = plt.subplots(3,3)
    f2, ax2 = plt.subplots(3,3)

    if args.inputfile:
        truths = [None] * 9
    else:
        truths = [args.mu,args.cov,args.pi, \
            None,None,None,None,None,len(args.pi)]

    label = [r'$\mu$',r'$[s]^{-1}$',r'$\pi$', \
        r'$\lambda$',r'$r$',r'$\log\beta$', \
        r'$w$',r'$\alpha$',r'$k$']
    idx = 0
    for l,t in zip(label,truths):
        
        ij = np.unravel_index(idx,[3,3])
        xx,yy = extractchain(Samp,l)

        # plot the chains
        x = np.squeeze(np.reshape(xx,(1,-1)))
        y = np.squeeze(np.reshape(yy,(1,-1)))
        ax1[ij].plot(x,y,'.k',markersize=3)
        for s in np.reshape(t,(1,-1)):
            ax1[ij].plot((x[0], x[-1]+1), (s, s), 'r-',alpha=0.5)
        ax1[ij].set_xlabel(r'$i$',fontsize=10)
        ax1[ij].set_ylabel(l,fontsize=12)
        
        # define lower and upper ranges to plot (use the data)
        lower = np.min(y[len(y)/2:])
        upper = np.max(y[len(y)/2:])
        if l=='$\pi$':
            lower = 0
            upper = 1
        if l=='$k$' or l=='$\\alpha$':
            lower = 0
        if l=='$[s]^{-1}$':
            lower = np.percentile(y,5)
            upper = np.percentile(y,95)

        ax1[ij].set_ylim([lower, upper])
        ax1[ij].set_xlim([x[0],x[-1]+1])
        ax1[ij].tick_params(axis='both', which='major', labelsize=10)

        # plot the hist
        if yy.ndim==1:
            yy = np.expand_dims(yy, axis=1)
        for newy in np.transpose(yy):
            n = newy.shape[0]
            if l=='$k$':
                ax2[ij].hist(newy[n/2:],np.arange(0,np.max(newy[n/2:])+1)+0.5, \
                    normed=True,histtype='stepfilled',alpha=0.5,edgecolor='None')    
            elif l=='$[s]^{-1}$':
                ax2[ij].hist(newy[n/2:], \
                    np.linspace(np.percentile(newy,5),np.percentile(newy,95),25), \
                    normed=True,histtype='stepfilled', \
                    alpha=0.5,edgecolor='None')
            else:
                ax2[ij].hist(newy[n/2:],25,normed=True,histtype='stepfilled', \
                alpha=0.5,edgecolor='None')
        ax2[ij].axes.get_yaxis().set_ticks([])
        ax2[ij].set_xlabel(l,fontsize=12)
        hupper = ax2[ij].get_ylim()[1]
        for s in np.reshape(t,(1,-1)):
            ax2[ij].plot((s,s), (0, hupper), 'r-',alpha=0.5)
        hlower = 0
        if l=='$k$':
            ax2[ij].set_xlim([0, np.max(newy[n/2:])+1])
        ax2[ij].set_ylim([hlower, hupper])
        ax2[ij].tick_params(axis='both', which='major', labelsize=10)    

        idx += 1
    f1.subplots_adjust(hspace=.5)
    f1.subplots_adjust(wspace=.5)
    f2.subplots_adjust(hspace=.5)
    f2.subplots_adjust(wspace=.5)
    f1.savefig(chainfile)
    f2.savefig(histfile)

def drawmissing(mu,s,nd,idx,nidx,x):

    j = idx.shape[0]
    k = nd - j
    y = np.array(x[idx] - mu[idx])        # the known distance form the mean
    A = s[np.ix_(nidx,nidx)].reshape(k,k)       # the precision matrix on the missing data
    B = np.reshape(s,(nd,nd))[np.ix_(idx,nidx)].reshape(j,k)        # the precision matrix off-diagonal terms
    newmu = np.squeeze(mu[nidx] - np.transpose(np.dot(solve(A,np.transpose(B)),y.reshape(j,1))))
    newcov = inv(A)
    return drawMVNormal(mean=newmu,cov=newcov,size=1) 






