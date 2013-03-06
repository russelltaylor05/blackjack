#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "poker.h"

static void HandleError( cudaError_t err, const char * file, int line)
{
  if(err !=cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)



int main(int argc, char *argv[])
{
  int deck[52], staticHand[HAND_SIZE];
  float analyzeResults[ANALYZE_RESOLUTION];
  int size;
  //int throwAwayCards[10];
  
  int *devHand;
  float *devAnalyzeResults;
  
  curandState *devStates;
  
  CUDA_CALL(cudaMalloc((void **)&devStates, 10 * sizeof(curandState)));
  
  /* initialize the deck */
  init_deck(deck);
  
  /* Set Hands */
  setStaticHand(deck, staticHand);  
  //setRandomHand(deck, randomHand, throwAwayCards, 0);   
  
  size = HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devHand, HAND_SIZE * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(devHand, staticHand, size, cudaMemcpyHostToDevice));

  size = ANALYZE_RESOLUTION * sizeof(float);
  HANDLE_ERROR(cudaMalloc(&devAnalyzeResults, size));  

  analyzeHand<<<1,2>>>(devHand, devHand, HAND_SIZE, devAnalyzeResults, devStates);
  
  size = ANALYZE_RESOLUTION * sizeof(float);
  HANDLE_ERROR(cudaMemcpy(analyzeResults, devAnalyzeResults, size, cudaMemcpyDeviceToHost));


  return 0;
}