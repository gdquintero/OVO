export ALGENCAN=/home/gustavo/algencan-3.1.1

rm -f measles

gfortran -O3 -w -fcheck=all -g measles.f90 -L$ALGENCAN/lib -lalgencan -lhsl sort.o subset.o -o measles

./measles
