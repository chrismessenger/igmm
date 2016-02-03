#########################################################################
#    igmm_test.py - Driver script for an infinite Gaussian mixture model
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
import argparse
import time
import time
import sys
import cPickle as pickle
from scipy.stats import multivariate_normal as mv_norm
from igmm import igmm_sampler, plotsamples, plotresult
import matplotlib.pylab as plt

# the maximum positive integer for use in setting the ARS seed
MAXINT = sys.maxint

def parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='igmm_test.py',description='Applies an N-Dimensional infinite Gaussian mixture model to data')

    # arguments for reading in a data file
    parser.add_argument('-i', '--inputfile', type=str, default=None, help='the input file name')
    parser.add_argument('-C', '--cols', type=int, nargs='+',default=None, help='the data columns to read from the file')
    parser.add_argument('-s', '--sep', type=str, default=',', help='the field seperator')
    parser.add_argument('-L', '--logdata', type=str, nargs='+',default=None, help='should the input data be logged before analysis')

    # arguments for generating random data
    parser.add_argument('-k', '--krand', type=int, default=None, help='how many random components to generate')    
    parser.add_argument('-d', '--Ndim', type=int, default=None, help='the dimension of the data')
    parser.add_argument('-N', '--Ndata', type=int, default=None, help='the number of input data samples to use')
    parser.add_argument('-n', '--Nsamples', type=int, default=2000, help='the number of samples to produce')

    # arguments for generating random data from a specific model
    parser.add_argument('-m', '--mu', type=float, nargs='+', default=None, help='the means of the simulated Gaussians')
    parser.add_argument('-c', '--cov', type=float, nargs='+', default=None, help='the covariance of each Gaussian component')
    parser.add_argument('-p', '--pi', type=float, nargs='+', default=None, help='the simulated Gaussian weights')
    parser.add_argument('-f', '--missfrac', type=float, default=0, help='the fraction of missing data')
    
    # general analysis parameters
    parser.add_argument('-o', '--path', type=str, default='./test', help='the output file path')
    parser.add_argument('-I', '--Nint', type=int, default=1, help='the number of samples used in approximating the tricky integral')
    parser.add_argument('-z', '--seed', type=int, default=1, help='the random seed') 
    parser.add_argument('-a', '--anneal', action='count', default=0, help='perform simulated annealing')
    parser.add_argument('-P', '--image', action='count', default=0, help='simulate only 2D image data')
    parser.add_argument('-v', '--verb', action='count', default=0)
 
    # catch any input errors   
    args = parser.parse_args()
    if not args.inputfile and not args.krand and not args.mu:
        print '{} : ERROR - must specify either an input file, a number of random components or a set of specific components. Exiting.'.format(time.asctime())
        exit(1)
    if args.krand and not (args.Ndim and args.Ndata):
        print '{} : ERROR - must specify dimensions and number of input data points. Exiting.'.format(time.asctime())
        exit(1)
    if (args.mu and not (args.cov and args.pi and args.Ndata)) \
    or (args.cov and not (args.mu and args.pi and args.Ndata)) \
    or (args.pi and not (args.cov and args.mu and args.Ndata)):
        print '{} : ERROR - must specify covariance, occupation fractions and number of data points. Exiting.'.format(time.asctime())
        exit(1)
    if args.missfrac<0 or args.missfrac>=1:
        print '{} : ERROR - the missing data fraction must be between 0 and 1. Exiting.'.format(time.asctime())
        exit(1)
    if args.Nint<1:
        print '{} : ERROR - the integration samples must be > 0. Exiting.'.format(time.asctime())
        exit(1)
    if args.Nsamples<1:
        print '{} : ERROR - the number of igmm samples must be > 0. Exiting.'.format(time.asctime())
        exit(1)
    if args.krand:
        if args.krand<1:
            print '{} : ERROR - the number of igmm components must be > 0. Exiting.'.format(time.asctime())
            exit(1)
    if args.cols:
        if np.any(args.cols<0):
            print '{} : ERROR - column indices must be > 0. Exiting.'.format(time.asctime())
            exit(1) 
        if len(np.unique(args.cols))!=len(args.cols):
            print '{} : ERROR - column indices must be unique. Exiting.'.format(time.asctime())
            exit(1)
    if args.Ndata:
        if args.Ndata<1:
            print '{} : ERROR - the number of input data samples must be > 0. Exiting.'.format(time.asctime())
            exit(1) 
    
    return parser.parse_args()

def readdata(inputfile,Ndata,cols=None,sep=',',logdata=None):
    """reads in data from an input text file
    
        inputfile - the name of the input file
        Ndata - the number of rows to read (None=all rows)
        cols - the columns to read (starting at 0)
        sep - the field seperator
        logdata - a boolean array indicating which fields to log
    
    """

    temp = []
    missmat = []
    k = 0
    if Ndata==None:
        Ndata = MAXINT 
    with open(inputfile, 'r') as f:
        for line in f:
            if line[0]!='#':
                newline = line.split() if sep==' ' else line.split(sep)
                for i,s in enumerate(newline):
                    if s=='' or s=='\n':
                        missmat.append(1)
                        newline[i] = '0'
                    else:
                        missmat.append(0)
                temp.append(newline)
                k += 1
                if k>=Ndata:
                    break

    # extract desired columns (use all cols if not specified)
    temp = np.array(temp).astype('str')
    if not cols:
        cols = np.arange(temp.shape[1]-1)
    Y = np.array(temp).astype('str')[:,np.array(cols).astype('int')]
    Y = np.array(Y).astype('float')
    missmat = np.array(missmat).astype('int').reshape(k,-1)
    missmat = missmat[:,cols]    

    # log data in specific columns if requested
    if logdata:
        logdata = np.array(logdata).astype('int')
        if (logdata.size==1) and (logdata == 1):
            Y = np.log(Y)
        elif (logdata.size>1):
            for l in np.argwhere(logdata==1):
                Y[:,l] = np.log(Y[:,l])
    N,nd = Y.shape

    return Y,N,nd,missmat

def gendata(args):
    """generates artificial data with gaps
    """

    N = args.Ndata
    nd = args.Ndim
    if args.krand:
        k = args.krand
        mu = mv_norm.rvs(mean=np.zeros(nd),cov=np.diag(np.ones(nd)),size=k)
        cov = np.zeros((k,nd*nd))
        for j in xrange(k):
            temp = np.random.rand(nd,nd)
            cov[j,:] = np.dot(temp,temp.transpose()).flatten()
        cov = np.reshape(cov,(k,nd,nd))
        mu = np.reshape(mu,(-1,nd))
        pi = np.random.rand(k)
        pi = pi/np.sum(pi)
        args.mu = mu
        args.cov = cov
        args.pi = pi
    else:
        mu = np.reshape(args.mu,(-1,nd))
        cov = np.reshape(args.cov,(-1,nd,nd))
        pi = args.pi/np.sum(args.pi)

    # generate some data to start with
    M = np.transpose(np.random.multinomial(N, pi, size=1))
    Y = []
    for i,m in enumerate(M):
        temp = mv_norm.rvs(mean=mu[i],cov=cov[i],size=m)
        if nd==1:
            temp = np.expand_dims(temp,axis=1)
        Y.append(temp)
    Y = np.vstack(Y)

    # remove some data at random
    if not args.image:
        missmat = np.random.choice(2, args.Ndata*nd, p=[1.0-args.missfrac,args.missfrac])
        missmat = np.reshape(missmat,(args.Ndata,nd))
        bidx = []
        for i,m in enumerate(missmat):
            if np.sum(m)==nd:
                bidx.append(i)
        bidx = np.array(bidx)
        if bidx.size:
            Y = np.delete(Y,bidx,axis=0)
            missmat = np.delete(missmat,bidx,axis=0)
    else:
        # only have nd-2 data points in each row
        missmat = np.zeros((args.Ndata,nd))
        if nd>2:
            ndrem = nd - 2
        elif nd==2:
            ndrem = 1
        else:
            print '{}: ERROR cannot simulate image data for 1D. Exiting.'.format(time.asctime())
            exit(1)
        for i in xrange(args.Ndata):
            rem = np.random.choice(nd, ndrem, replace=False, p=np.ones(nd)/nd)
            missmat[i,rem] = np.ones(ndrem)

    # save data to file (taking care to not include missing data)
    if np.all(missmat==0):
        np.savetxt(args.path + "_data.csv", Y, delimiter=",")
    else:
        fp = open(args.path + "_data.csv", "w")
        for m,y in zip(missmat,Y):
            y = y.astype('str')
            z = []
            k = 0
            for a,s in zip(m,y):
                if a==0:
                    z.append(s)
                else:
                    z.append('')
                k += 1            

            fp.write(','.join(z))
            fp.write(',\n')
        fp.close()
    
    return Y,N,nd,missmat,args

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
    
    # read in data if required
    if args.inputfile:
        Y,N,nd,missmat = readdata(args.inputfile,args.Ndata,args.cols,args.sep,args.logdata)
    else:
        Y,N,nd,missmat,args = gendata(args)

    # call igmm Gibbs sampler
    Samp,Y = igmm_sampler(Y,args.Nsamples,missmat,args.Nint,anneal=args.anneal,verb=args.verb) 

    # print computation time
    print "{}: time to complete main analysis = {} sec".format(time.asctime(),time.time()-t)    

    # save data to file
    pickle.dump(zip(Samp,Y,missmat), open( args.path + "_output.p", "wb" ) )

    # plot chains, histograms, average maps, and overlayed ellipses
    print '{}: making output plots'.format(time.asctime()) 
    plotsamples(Samp,args,args.path + '_chains.png',args.path + '_hist.png')
    plotresult(Samp,Y,args.path + '_ellipses.png',M=4,Ngrid=100,plottype='ellipse')
    plotresult(Samp,Y,args.path + '_maps.png',missmat=missmat,M=100,Ngrid=100,plottype='map')

    print '{}: success'.format(time.asctime())    

if __name__ == "__main__":
    exit(main())


#gibbs()
