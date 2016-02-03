# igmm
A python implementation of the infinite Gaussian mixture model (Rasmussen 2000)

This is currently the N-dimensional version alluded to in the paper. It has the
added functionality that it can account for missing data i.e. where there are
multiple missing fields in any of the data entries. 

An example driver script igmm_test.py is provided to give an idea of how to use
the igmm.py library. To see the required/optional command line arguments simply
run

python igmm_test.py -h

This will allow you to read in existing data, or generate simulated data in
N-dimensions. To this the user must either specify a number of Gaussian
components that will be selected randomly or specify specific component
properties.  The user can also specify the desired fraction of missing data.

In the testdata directory there are some example input files for 2 and 3D cases
with varying ammounts of data and varying levels of missing data.

To run on the testdata the command line arguments should be

python igmm_test.py -i ./testdata/<inputfile>

More thorough documentation will be provided soon.
 
