#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "cpu_poker.h"
#include "poker.h"

void    srand48();
double  drand48();

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
  int deck[52];
  int staticHand[HAND_SIZE];
  int blockCnt;
  int analyzeResults[ANALYZE_RESOLUTION];
  int size, sum =0;
  int i, score, rank;
  ARGSP *argsp;
  
  clock_t start, stop;
  
  int *devHand;
  int *devAnalyzeResults;  
  curandState *devStates;

  argsp = (ARGSP *)malloc(sizeof(ARGSP));  
  if(getArgs(argsp, argc, argv) < 0) {
    printf("Card arguments broken\n");
    return EXIT_FAILURE;
  };

  srand48((int) time(NULL));  
    
  init_deck_cpu(deck);
  setHandFromArgs(deck, staticHand, argsp);

  score = eval_5hand_cpu(staticHand);
  rank = hand_rank_cpu(score);
  printf("Hand: \t\t");
  print_hand_cpu(staticHand, HAND_SIZE);
  printf("\nScore: \t\t%d\n", score);
  printf("Rank: \t\t%s\n", value_str_cpu[rank]);  

  start = clock();

  // Cuda Memeory Setup
  HANDLE_ERROR(cudaMalloc((void **)&devStates, ANALYZE_RESOLUTION * sizeof(curandState)));
  
  size = HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devHand, HAND_SIZE * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(devHand, staticHand, size, cudaMemcpyHostToDevice));

  size = ANALYZE_RESOLUTION * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devAnalyzeResults, size));  

  blockCnt = (ANALYZE_RESOLUTION + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  analyzeHand<<<blockCnt,THREADS_PER_BLOCK>>>(devHand, devHand, HAND_SIZE, devAnalyzeResults, devStates);
  
  size = ANALYZE_RESOLUTION * sizeof(int);
  HANDLE_ERROR(cudaMemcpy(analyzeResults, devAnalyzeResults, size, cudaMemcpyDeviceToHost));

  for(i = 0; i < ANALYZE_RESOLUTION; i++) {
    sum +=  analyzeResults[i];
  }
  stop = clock();
  
  printf("Score: \t\t%.2f%%\n", (float)sum / (float)ANALYZE_RESOLUTION * 100.0);
  printf("Time: \t\t%f seconds\n", (double)(stop - start) / CLOCKS_PER_SEC);  

  // Free Cleanup
  cudaFree(devAnalyzeResults);
  cudaFree(devHand);
  free(argsp);

  return EXIT_SUCCESS;
}

