NVFLAGS= -g -O2 -arch=compute_20 -code=sm_20 

# list .c and .cu source files here
# use -02 for optimization during timed runs


SRCFILES = main.cu pokerlib.cu
TARGET = ./poker

all:	poker

poker: $(SRCFILES) pokerlib.o
	nvcc $(NVFLAGS) -o poker $^

poker2: $(SRCFILES)
	nvcc $(NVFLAGS) -o poker $^


pokerlib.o: pokerlib.cu lookuptable.h
	nvcc ${NVFLAGS} pokerlib.cu -o pokerlib.o

run: poker
	@./poker

clean: 
	rm -f *.o poker
	
