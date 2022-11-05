export ALGENCAN=/home/gustavo/algencan-3.1.1

rm -f meyer

gfortran -O3 -w -fcheck=all -g meyer.f90 -L$ALGENCAN/lib -lalgencan -lhsl sort.o subset.o -o meyer

./meyer
