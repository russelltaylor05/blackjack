NVFLAGS= -g -lcurand -arch=compute_20 -code=sm_20 -L/usr/local/cuda/include

# list .c and .cu source files here
# use -02 for optimization during timed runs


SRCFILES = main.cu pokerlib.cu
RANDFILES = rand.cu
TARGET = ./gpu_poker

all:	poker

poker: $(SRCFILES)
	nvcc $(NVFLAGS) -o gpu_poker $^
	
rand: $(RANDFILES)
	nvcc $(NVFLAGS) -o rand $^


run: poker
	./gpu_poker --c1 Kh --c2 2d --c3 8d --c4 7c --c5 As

clean: 
	rm -f *.o poker gpu_poker
	
