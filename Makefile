NVFLAGS= -g -lcurand -arch=compute_20 -code=sm_20 -L/usr/local/cuda/include

# list .c and .cu source files here
# use -02 for optimization during timed runs


SRCFILES = main.cu pokerlib.cu
RANDFILES = rand.cu
TARGET = ./poker

all:	poker

poker: $(SRCFILES)
	nvcc $(NVFLAGS) -o poker $^
	
rand: $(RANDFILES)
	nvcc $(NVFLAGS) -o rand $^


run: poker
	./poker --c1 Kh --c2 9d --c3 8d --c4 7c --c5 As

clean: 
	rm -f *.o poker
	
