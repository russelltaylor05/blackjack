#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include "poker.h"

static void HandleError( cudaError_t err, const char * file, int line)
{
  if(err !=cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



int main(int argc, char *argv[])
{

  int *dev_a, *a;
  
  a = (int *) malloc(sizeof(int));
  
  HANDLE_ERROR(cudaMalloc(&dev_a, sizeof(int)));

  analyze<<<1,10>>>(1);
  
  HANDLE_ERROR(cudaMemcpy(a, dev_a, sizeof(int), cudaMemcpyDeviceToHost));

  return 0;
}