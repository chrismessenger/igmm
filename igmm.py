import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self,N):
        self.sample = []
        self.N = N

    def __getitem__(self, key): 
        return self.sample[key]

    def addsample(self,S):
        return self.sample.append(S)

def IntegralApprox(y,lam,r,beta,w,size=100):
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
            temp += mv_norm.pdf(y,mean=np.squeeze(mu),cov=inv(s))
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
        xi = lb + np.logspace(-3-cnt,1+cnt,50)       # update range if needed
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

def plotresult(Samp,Y,nd,outfile,Ngrid,nstep):
    """Plots the original data, with a averaged result in 
       2D projections"""

    # extract data from samples
    n = nd-1
    plt.figure()
    i,mu = extractchain(Samp,nd,'$\mu$')
    _,prec = extractchain(Samp,nd,'$s$')
    _,pi = extractchain(Samp,nd,'$\pi$')
    if nd==1:
        mu = np.expand_dims(np.array(mu), axis=1)
        prec = np.expand_dims(np.array(prec), axis=1)
    redmu = mu[mu.shape[0]/2::nstep,:]
    redprec = prec[prec.shape[0]/2::nstep,:]
    redpi = pi[pi.shape[0]/2::nstep]
    label = []
    for i in xrange(nd):
        temp = '$x_{}$'.format(i)
        label.append(temp)    

    # make N-D grid
    xvec = np.linspace(-10,10,Ngrid)
    dx = xvec[1]-xvec[0]
    grid = np.array(np.meshgrid(*[xvec for i in xrange(nd)],indexing='ij'))
    grid = np.reshape(np.transpose(grid),(-1,nd),order='F')
    
    # loop over each sample and compute the result in N-D grid
    prob = np.zeros(Ngrid**nd)
    for m,s,p in zip(redmu,redprec,redpi):
        print m,s,p
        prob += p*mv_norm.pdf(grid,mean=m,cov=np.reshape(s,(nd,nd)))
    prob = np.reshape(prob,([Ngrid for q in xrange(nd)]),order='C')        

    # make the 2D grid
    xy = np.array(np.meshgrid(xvec,xvec))
    xy = np.transpose(np.reshape(grid,(nd,-1)))

    levels = [0.68, 0.95,0.999]
    k = 0
    for j in xrange(nd):
        for i in xrange(nd):
            if i>=j:
                ij = np.unravel_index(k,[nd,nd])
                plt.subplot(nd,nd,k+1)
                ores = np.zeros(Ngrid)
                nres = np.zeros((Ngrid,Ngrid))
                sumidx = np.reshape(np.squeeze(np.argwhere((np.arange(nd)!=i)*(np.arange(nd)!=j))),(1,-1)).astype('int')
                idx = tuple(map(tuple, sumidx))[0]
                proj = np.squeeze(np.sum(prob,axis=idx,keepdims=True))
                if proj.ndim==2:
                    z = greedy(proj)
                    plt.contour(xvec, xvec, z, levels, \
                        colors='blue',linestyles=['solid','dashed','dotted'])
                    plt.plot(Y[:,i],Y[:,j],'r.',alpha=0.5,markersize=3)
                    plt.xlim([-10,10])
                    plt.ylim([-10,10])
                    plt.ylabel(label[j],fontsize=16)
                    plt.xlabel(label[i],fontsize=16)
                else:
                    plt.hist(Y[:,i],25,histtype='stepfilled',normed=True,alpha=0.5,edgecolor='None',facecolor='red')
                    z = proj/(np.sum(proj)*dx)
                    plt.plot(xvec,z,'b')
                    plt.yticks([])
                    plt.xlabel(label[i],fontsize=16)
                plt.subplots_adjust(hspace=.5) 
                plt.subplots_adjust(wspace=.5)
            k += 1
    
    plt.savefig(outfile)   
             

def project(x,idx):
    """projects out dimension of matrix"""
    n = x.shape[0]
    y = np.zeros((n,n))
    for k in np.reshape(idx,(-1,1)):
        for i in xrange(n):
            for j in xrange(n):
                y[i,j] = x[i,j] - x[i,k]*x[k,j]/x[k,k]
        
    return np.squeeze(np.delete(np.delete(y,idx,0),idx,1))        

def plotellipses(Samp,Y,nd,outfile,Ngrid,Nellipse):
    """Plots samples of ellipses drawn frok the posterior"""
    
    N = Samp.N
    xvec = np.linspace(-10,10,Ngrid)
    label = []
    for i in xrange(nd):
        temp = '$x_{}$'.format(i)
        label.append(temp)
    fig = plt.figure()
    f, ax = plt.subplots(nd,nd)
    randidx = np.random.randint(N/2,N,Nellipse)
    cnt = 0
    for i in xrange(nd):
        for j in xrange(nd):
            
            # plot the data
            if i>j:
                ax[j,i].plot(Y[:,i],Y[:,j],'r.',alpha=0.5,markersize=1)
            elif i==j:
                print Y.shape
                if nd>1:
                    ax[j,i].hist(Y[:,i],25,histtype='stepfilled',normed=True,alpha=0.5,edgecolor='None',facecolor='red')            
                else:
                    plt.hist(Y[:,i],25,histtype='stepfilled',normed=True,alpha=0.5,edgecolor='None',facecolor='red')


            # if off the diagonal
            if i>=j:
                ij = np.unravel_index(cnt,[nd,nd])

                # select 
                for k in randidx:
                    samples = Samp[k]
                    s = np.reshape(samples.s,(samples.k,nd*nd))
                    m = np.reshape(samples.mu,(samples.k,nd))
                    p = np.reshape(np.array(np.squeeze(samples.pi)),(-1,1))
                    
                    for b in xrange(samples.k):
                        tempC = np.reshape(s[b,:],(nd,nd))
                        idx = np.reshape(np.squeeze(np.argwhere((np.arange(nd)!=i)*(np.arange(nd)!=j))),(1,-1)).astype('int')
                        ps = tempC if np.any(np.array(idx.shape)==0) else project(tempC,idx)
                        if ps.size==4:
                            w,v = eig(inv(ps))
                            e = Ellipse(xy=m[b,[i,j]], width=2.0*np.sqrt(6.0*w[1]), \
                                height=2*np.sqrt(6.0*w[0]), \
                                angle=(180.0/np.pi)*np.arctan2(v[0,1],v[0,0]), \
                                alpha=np.squeeze(p[b]))
                            e.set_facecolor('none')
                            e.set_edgecolor('b')
                            ax[j,i].add_artist(e)
                            ax[j,i].set_xlim([-10,10])
                            ax[j,i].set_ylim([-10,10])
                            ax[j,i].set_ylabel(label[j],fontsize=16)
                            ax[j,i].set_xlabel(label[i],fontsize=16)    
                        elif ps.size==1:
                            if nd>1:
                                ax[j,i].plot(xvec,p[b]*norm.pdf(xvec,loc=m[b,i],scale=1.0/np.sqrt(np.squeeze(ps))),'b',alpha=p[b]) 
                                ax[j,i].set_yticks([])
                                ax[j,i].set_xlabel(label[i],fontsize=16)
                            else:
                                plt.plot(xvec,p[b]*norm.pdf(xvec,loc=m[b,i],scale=1.0/np.sqrt(np.squeeze(ps))),'b',alpha=p[b])
                                plt.yticks([])
                                plt.xlabel(label[i],fontsize=16)
                        else:
                            print '{}: ERROR strange number of elements in projected matrix'.format(time.asctime())
                            exit(0)

            else:
                ax[j,i].axis('off') if nd>1 else plt.axis('off')
            cnt += 1

    f.subplots_adjust(hspace=.5)
    f.subplots_adjust(wspace=.5)        
    plt.savefig(outfile)

def extractchain(Samp,nd,label):
    """Extract the chains of the mean, precision, pi etc, variables"""
    i = 0
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
    elif label=='$s$':
        for sample in Samp:
            s = np.reshape(sample.s,(sample.k,nd*nd))
            for d in xrange(sample.k):
                xdata.append(i*np.ones(nd*nd))
                ydata.append(np.squeeze(np.reshape(inv(np.reshape(s[d,:],(nd,nd))),(1,nd*nd))))
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

def plotsamples(Samp,nd,args,chainfile,histfile):
    """Generates plots of samples as a function of index
       and also plots histograms of samples"""

    f1, ax1 = plt.subplots(3,3)
    f2, ax2 = plt.subplots(3,3)

    truths = [args.mu,np.reshape(args.cov,(1,-1)),np.reshape(args.pi,(1,-1)), \
        None,None,None,None,None,len(args.pi)]
    label = [r'$\mu$',r'$s$',r'$\pi$', \
        r'$\lambda$',r'$r$',r'$\log\beta$', \
        r'$w$',r'$\alpha$',r'$k$']
    idx = 0
    for l,t in zip(label,truths):
        
        ij = np.unravel_index(idx,[3,3])
        xx,yy = extractchain(Samp,nd,l)

        # plot the chains
        x = np.squeeze(np.reshape(xx,(1,-1)))
        y = np.squeeze(np.reshape(yy,(1,-1)))
        ax1[ij].plot(x,y,'.k',markersize=3)
        for s in np.reshape(t,(1,-1)):
            ax1[ij].plot((x[0], x[-1]+1), (s, s), 'r-',alpha=0.5)
        ax1[ij].set_xlabel(r'$i$',fontsize=10)
        ax1[ij].set_ylabel(l,fontsize=12)
        
        dum = t if np.any(t) else y
        rng = np.max(dum)-np.min(dum)
        lower = np.min(dum) - 0.5*rng
        upper = np.max(dum) + 0.5*rng
        if l=='$\pi$':
            lower = 0
            upper = 1
        if l=='$k$' or l=='$\\alpha$':
            lower = 0
        if l=='$k$':
            upper = np.max(y) + 1

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
            elif l=='$s$':
                ax2[ij].hist(newy[n/2:], \
                    np.linspace(np.percentile(newy,5),np.percentile(newy,95),25), \
                    normed=True,histtype='stepfilled', \
                    alpha=0.5,edgecolor='None')
            else:
                ax2[ij].hist(newy[n/2:],25,normed=True,histtype='stepfilled', \
                alpha=0.5,edgecolor='None')
        ax2[ij].axes.get_yaxis().set_ticks([])
        ax2[ij].set_xlabel(l,fontsize=12)
        upper = ax2[ij].get_ylim()[1]
        for s in np.reshape(t,(1,-1)):
            ax2[ij].plot((s,s), (0, upper), 'r-',alpha=0.5)
        lower = 0
        if l=='$k$':
            ax2[ij].set_xlim([0, np.max(yy)+1])
        ax2[ij].set_ylim([lower, upper])
        ax2[ij].tick_params(axis='both', which='major', labelsize=10)    

        idx += 1
    f1.subplots_adjust(hspace=.5)
    f1.subplots_adjust(wspace=.5)
    f2.subplots_adjust(hspace=.5)
    f2.subplots_adjust(wspace=.5)
    f1.savefig(chainfile)
    f2.savefig(histfile)

# command line parser
def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='igmm.py',description='Applies an N-Dimensional infinite Gaussian mixture model to data')
    parser.add_argument('-d', '--Ndim', type=int, default=2, help='the dimension of the data')
    parser.add_argument('-c', '--cov', type=float, nargs='+', default=[3.0,1.0,1.0,5.0,5.0,0.0,0.0,2.0,2.0,-1.5,-1.5,2.0], help='the precision of each Gaussian component')
    parser.add_argument('-m', '--mu', type=float, nargs='+', default=[-3,1,0,7,5,-5], help='the means of the simulated Gaussians')
    parser.add_argument('-p', '--pi', type=float, nargs='+', default=[0.4,0.4,0.2], help='the simulated Gaussian weights')
    parser.add_argument('-N', '--Ndata', type=int, default=100, help='the number of input data samples to use')
    parser.add_argument('-n', '--Nsamples', type=int, default=1000, help='the number of samples to produce')
    parser.add_argument('-a', '--Nint', type=int, default=10, help='the number of samples used in approximating the tricky integral')
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
    nd = args.Ndim
    args.mu = np.reshape(args.mu,(-1,nd))
    args.cov = np.reshape(args.cov,(-1,nd,nd))

    # generate some data to start with
    args.pi = args.pi/np.sum(args.pi)
    M = np.transpose(np.random.multinomial(N, args.pi, size=1))
    Y = []
    for i,m in enumerate(M):
        temp = mv_norm.rvs(mean=args.mu[i],cov=args.cov[i],size=m)
        if nd==1:
            temp = np.expand_dims(temp,axis=1)
        Y.append(temp)
    Y = np.vstack(Y)

    # compute some data derived quantities
    muy = np.mean(Y,0)
    print muy.shape
    covy = np.reshape(np.cov(Y,rowvar=0),(nd,nd))
    inv_covy = inv(covy) if nd>1 else np.reshape(1.0/covy,(1,1))
    if args.verb:
        print '{}: true mean(Y) = {}'.format(time.asctime(),np.reshape(args.mu,(1,-1)))
        print '{}: true cov(Y) = {}'.format(time.asctime(),np.reshape(args.cov,(1,-1)))
        print '{}: mean(Y) = {}'.format(time.asctime(),np.reshape(muy,(1,-1)))
        print '{}: cov(Y) = {}'.format(time.asctime(),np.reshape(covy,(1,-1)))

    # initialise a single sample
    Samp = Samples(args.Nsamples)
    c = np.zeros(N)                             # initialise the stochastic indicators
    pi = np.zeros(1)                            # initialise the weights
    mu = np.zeros((1,nd))                       # initialise the means
    s = np.zeros((1,nd*nd))                     # initialise the precisions
    n = np.zeros(1)                             # initialise the occupation numbers
    
    mu[0,:] = muy                               # set first mu to the mean of all data    
    pi[0] = 1.0                                 # only one component so pi=1
    temp = drawGamma(0.5,2.0/float(nd))
    beta = np.squeeze(float(nd) - 1.0 + 1.0/temp)     # draw beta from prior
    print covy.shape
    w = drawWishart(nd,covy/float(nd))                # draw w from prior
    s[0,:] = np.squeeze(np.reshape(drawWishart(float(beta),inv(beta*w)),(nd*nd,-1))) # draw s from prior

    n[0] = N                                    # all samples are in the only component
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
    while z<args.Nsamples:

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

        # compute the unrepresented probability
        p_unrep = (alpha/(N-1.0+alpha))*IntegralApprox(Y,lam,r,beta,w,size=args.Nint)
        p_temp = np.outer(np.ones(k+1),p_unrep)

        # for the represented components
        for j in xrange(k):
            nij = n[j] - (c==j).astype(int)
            idx = np.argwhere(nij>0)         # only apply to indices where we have multi occupancy
            temp_s = np.reshape(s[j,:],(nd,nd))
            p_temp[j,idx] = nij[idx]/(N-1.0+alpha)*((2.0*np.pi)**(0.5*nd))*np.reshape(mv_norm.pdf(Y[idx,:],mean=mu[j,:],cov=inv(temp_s)),idx.shape)

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

        pcnt = int(100.0*z/float(args.Nsamples))
        if pcnt>oldpcnt:
            print '{}: %--- {}% complete ----------------------%'.format(time.asctime(),pcnt)
            if args.verb:
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

    # print computation time
    print "{}: time to complete main analysis = {} sec".format(time.asctime(),time.time()-t)    

    # plot chains, histograms, average maps, and overlayed ellipses
    print '{}: making output plots'.format(time.asctime()) 
    Ngrid = min(100,int(1e6**(1.0/nd)))
    plotsamples(Samp,nd,args,'./chains.png','./hist.png')
    plotresult(Samp,Y,nd,'./maps.png',Ngrid,10)
    plotellipses(Samp,Y,nd,'./ellipses.png',Ngrid,10)

    print '{}: success'.format(time.asctime())    

if __name__ == "__main__":
    exit(main())


#gibbs()
