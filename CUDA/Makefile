NVFLAGS = -g -O2 -arch=compute_20 -code=sm_20 -DSINGLE

# list .c and .cu source files here
SRCFILES = sample_cuda_code.cu 

all:	sample_cuda_code	

sample_cuda_code: $(SRCFILES) 
	nvcc $(NVFLAGS) -o sample_cuda_code $^

clean: 
	rm -f *.o sample_cuda_code
