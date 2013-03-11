#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "cpu_poker.h"
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
  int deck[52];
  int staticHand[HAND_SIZE];
  //int tempHand[HAND_SIZE];
  int throwAway[HAND_SIZE];
  int throwCombosResults[THROWAWAY_RESOLUTION * HAND_SIZE];
  int *throwResults;
  int score, rank, throwCnt, size, blockCnt, i;
  int sum = 0;
  ARGSP *argsp;
  
  clock_t start, stop;
  
  int *devHand;
  int *devThrowCards;
  int *devThrowCombosResults; 
  int *devThrowResults; 
  curandState *devStates;

  argsp = (ARGSP *)malloc(sizeof(ARGSP));  
  if(getArgs(argsp, argc, argv) < 0) {
    printf("Card arguments broken\n");
    return EXIT_FAILURE;
  };
  
  throwResults = (int *)malloc(THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION * sizeof(int)); 
  if (throwResults == NULL) {
    fprintf(stderr, "failed to allocate memory.\n");
    return -1;
  }
  

  srand48((int) time(NULL));  
    
  init_deck_cpu(deck);
  setHandFromArgs(deck, staticHand, argsp);
  setThrowFromArgs(deck, throwAway, &throwCnt, argsp);

  score = eval_5hand_cpu(staticHand);
  rank = hand_rank_cpu(score);
  printf("Hand: \t\t");    print_hand_cpu(staticHand, HAND_SIZE);
  printf("\nThrow: \t\t"); print_hand_cpu(throwAway, throwCnt);
  printf("\nScore: \t\t%d\n", score);
  printf("Rank: \t\t%s\n", value_str_cpu[rank]);  
  printf("ThrowAway Res: \t%d\n", THROWAWAY_RESOLUTION);  

  start = clock();

  // Cuda Memeory Setup
  HANDLE_ERROR(cudaMalloc((void **)&devStates, THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION * sizeof(curandState)));  

  size = HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devHand, HAND_SIZE * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(devHand, staticHand, size, cudaMemcpyHostToDevice));  

  size = throwCnt * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devThrowCards, throwCnt * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(devThrowCards, throwAway, size, cudaMemcpyHostToDevice));

  size = THROWAWAY_RESOLUTION * HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devThrowCombosResults, size));

  size = THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devThrowResults, size));  


  // Kernel Calls
  __global__ void curandSetup(curandState *state);
  blockCnt = (THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  curandSetup<<<blockCnt,THREADS_PER_BLOCK>>>(devStates);     
  
  blockCnt = (THROWAWAY_RESOLUTION + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  printf("K1 blockcnt: \t%d\n",blockCnt);
  printf("K1 threadcnt: \t%d\n\n", blockCnt * THREADS_PER_BLOCK);
  //createThrowCombos<<<blockCnt,THREADS_PER_BLOCK>>>(devHand, devThrowCards, throwCnt, devThrowCombosResults, devStates);    

  blockCnt = (THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  printf("K2 blockcnt: \t%d\n", blockCnt);
  printf("K2 threadcnt: \t%d\n", blockCnt * THREADS_PER_BLOCK);
  //analyzeThrowCombos<<<blockCnt,THREADS_PER_BLOCK>>>(devHand, devThrowCombosResults, devThrowResults, devStates);


  // Return Results 
  size = THROWAWAY_RESOLUTION * HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMemcpy(throwCombosResults, devThrowCombosResults, size, cudaMemcpyDeviceToHost));

  size = THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION * sizeof(int);
  HANDLE_ERROR(cudaMemcpy(throwResults, devThrowResults, size, cudaMemcpyDeviceToHost));

  for(i = 0; i < THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION; i++) {
    sum += throwResults[i];
    //printf("%d, ",throwResults[i]);
  }


  /*
  printf("CPU Combos: \n");
  for (i = 0; i < THROWAWAY_RESOLUTION * HAND_SIZE; i++) {
    for(j = 0; j < 5; j++) {
      tempHand[j] = throwCombosResults[i];
      i++;      
    }
    i--;
    score = eval_5hand_cpu(tempHand);
    rank = hand_rank_cpu(score);
    print_hand_cpu(tempHand, HAND_SIZE);
    printf("\t%d", score);
    printf("\t%s", value_str_cpu[rank]);
    printf("\n");
  }
  */
  

  stop = clock();
  printf("Sum: \t\t%d\n",sum);
  //printf("Result Size: \t%d\n", THROWAWAY_RESOLUTION);
  printf("throwScore: \t%.2f%%\n", (float)sum / (float)(THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION) * 100.0);
  printf("Time: \t\t%f seconds\n", (double)(stop - start) / CLOCKS_PER_SEC);  

  HANDLE_ERROR(cudaFree(devStates));
  HANDLE_ERROR(cudaFree(devHand));
  HANDLE_ERROR(cudaFree(devThrowCards));
  HANDLE_ERROR(cudaFree(devThrowCombosResults));
  HANDLE_ERROR(cudaFree(devThrowResults));
  free(argsp);

  return EXIT_SUCCESS;
}

