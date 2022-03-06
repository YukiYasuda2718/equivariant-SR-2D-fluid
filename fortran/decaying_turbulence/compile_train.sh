#!/bin/sh

OMP_NUM_THREADS=1
export OMP_NUM_THREADS
ulimit -s unlimited
ulimit -m unlimited

OPT="-O3 -fno-automatic -fbounds-check -fbacktrace -ffpe-trap=invalid,zero -fcheck=array-temps,bounds,do,mem,pointer,recursion -Waliasing -Wampersand -Wconversion -Wsurprising -Wc-binding-type -Wintrinsics-std -Wtabs -Wintrinsic-shadow -Wline-truncation -Wtarget-lifetime -Winteger-division -Wreal-q-constant -Wunused -Wundefined-do-loop -Werror"

OPT_L="-L/usr/local/lib/ -lisp" 

cd `dirname $0`
rm -rf *.mod *.o *.out
cp -f /workspace/fortran/mt19937-64.f95 .
gfortran ${OPT} mt19937-64.f95 decaying_turbulence_train.f90 ${OPT_L}
rm -f ./mt19937-64.f95
