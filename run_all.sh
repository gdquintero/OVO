export ALGENCAN=/home/gustavo/algencan-3.1.1

rm -f measles mumps rubella

gfortran -O3 -w -fcheck=all -g measles.f90 -L$ALGENCAN/lib -lalgencan -lhsl sort.o subset.o -o measles 
gfortran -O3 -w -fcheck=all -g mumps.f90 -L$ALGENCAN/lib -lalgencan -lhsl sort.o subset.o -o mumps     
gfortran -O3 -w -fcheck=all -g rubella.f90 -L$ALGENCAN/lib -lalgencan -lhsl sort.o subset.o -o rubella 

./measles & ./mumps & ./rubella