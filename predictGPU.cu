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
  float predictScores[31];
  float predictWinTempScore;
  int predictWinIndex;
  short *mask;
  int card;
  
  int score, rank, throwCnt, size, i, j;
  int comboBlockCnt, analyzeBlockCnt;
  int sum = 0;
  ARGSP *argsp;
  
  cudaEvent_t start, stop;
  float   elapsedTime;

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

  srand48((int) time(NULL));  
    
  init_deck_cpu(deck);
  setHandFromArgs(deck, staticHand, argsp);

  score = eval_5hand_cpu(staticHand);
  rank = hand_rank_cpu(score);

  printf("Hand: \t\t");    print_hand_cpu(staticHand, HAND_SIZE);
  printf("\n");
  /*
  printf("\nThrow: \t\t"); print_hand_cpu(throwAway, throwCnt);
  printf("\nScore: \t\t%d\n", score);
  printf("Rank: \t\t%s\n", value_str_cpu[rank]);  
  printf("ThrowAway Res: \t%d\n", THROWAWAY_RESOLUTION);  
  printf("Analyze Res: \t%d\n\n", ANALYZE_RESOLUTION);  
  */

  // Cuda Memeory Setup
  analyzeBlockCnt = (THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  comboBlockCnt = (THROWAWAY_RESOLUTION + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  
  throwResults = (int *)malloc(analyzeBlockCnt * sizeof(int)); 
  if (throwResults == NULL) {
    fprintf(stderr, "failed to allocate memory.\n");
    return -1;
  }
  
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
  
  size = THROWAWAY_RESOLUTION * ANALYZE_RESOLUTION * sizeof(curandState);
  //printf("curandSize: \t%d\n", sizeof(curandState));
  //printf("curandSize: \t%d\n", size);
  HANDLE_ERROR(cudaMalloc((void **)&devStates, size));  

  size = HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devHand, HAND_SIZE * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(devHand, staticHand, size, cudaMemcpyHostToDevice));  

  size = THROWAWAY_RESOLUTION * HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devThrowCombosResults, size));

  size = analyzeBlockCnt * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devThrowResults, size));
  HANDLE_ERROR(cudaMemset(devThrowResults, 0, size));

  size = throwCnt * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devThrowCards, size));
  HANDLE_ERROR(cudaMemcpy(devThrowCards, throwAway, size, cudaMemcpyHostToDevice));
  
  for(i = 0; i < 31; i++) {
  
    setThrowCombo(throwAway, &throwCnt, staticHand, i);
    
    size = throwCnt * sizeof(int);
    HANDLE_ERROR(cudaMalloc(&devThrowCards, size));
    HANDLE_ERROR(cudaMemcpy(devThrowCards, throwAway, size, cudaMemcpyHostToDevice));

    // Kernel Calls
    curandSetup<<<analyzeBlockCnt,THREADS_PER_BLOCK>>>(devStates);         
    createThrowCombos<<<comboBlockCnt,THREADS_PER_BLOCK>>>(devHand, devThrowCards, throwCnt, devThrowCombosResults, devStates);    
    analyzeThrowCombos<<<analyzeBlockCnt,THREADS_PER_BLOCK>>>(devHand, devThrowCombosResults, devThrowResults, devStates);
  
    size = analyzeBlockCnt * sizeof(int);
    HANDLE_ERROR(cudaMemcpy(throwResults, devThrowResults, size, cudaMemcpyDeviceToHost));
  
    sum = 0;
    for(j = 0; j < analyzeBlockCnt; j++) {
      sum += throwResults[j];
    }
    predictScores[i] = (float)sum / (float)(analyzeBlockCnt * THREADS_PER_BLOCK) * 100.0; 
    printf("win[%d]: \t %.2f \t", i, predictScores[i]);
    print_hand_cpu(throwAway, throwCnt);
    printf("\n");    
  }
  
  predictWinTempScore = 0;
  for(i = 0; i < 31; i++) {
    if(predictScores[i] > predictWinTempScore) {
      predictWinTempScore  = predictScores[i];
      predictWinIndex = i;
    }  
  }

  mask = throwMask[predictWinIndex];
  for(j = 0; j < 5; j++) {
    if(mask[j] == 1) {
      card = staticHand[j];
      print_card_cpu(card);
    }
  }
  

  /*
  printf("\"Title\": \"GPU Throw\",\n",score);
  printf("\"Hand\":\"");
  print_hand_cpu(staticHand, HAND_SIZE);
  printf("\",\n");
  printf("\"Throw\":\"");
  print_hand_cpu(throwAway, throwCnt);
  printf("\",\n");
  printf("\"Score\":%d,\n",score);
  printf("\"Rank\":\"%s\",\n", value_str_cpu[rank]);
  printf("\"Win\":%.2f,\n", (float)sum / (float)(analyzeBlockCnt * THREADS_PER_BLOCK) * 100.0);
  printf("\"Sum\":%d,\n", sum);
  printf("\"Kernel_Time\":%.2f,\n", elapsedTime);
  printf("\"Analyze_Res\":%d,\n", ANALYZE_RESOLUTION);
  printf("\"Throw_Res\":%d,\n", THROWAWAY_RESOLUTION);
  printf("\"Block_Cnt\":%d,\n", analyzeBlockCnt);
  printf("\"Thread_cnt\":%d\n", analyzeBlockCnt * THREADS_PER_BLOCK);
  */
  

  HANDLE_ERROR(cudaEventDestroy( start ));
  HANDLE_ERROR(cudaEventDestroy( stop ));  

  HANDLE_ERROR(cudaFree(devStates));
  HANDLE_ERROR(cudaFree(devHand));
  HANDLE_ERROR(cudaFree(devThrowCards));
  HANDLE_ERROR(cudaFree(devThrowCombosResults));
  HANDLE_ERROR(cudaFree(devThrowResults));

  free(argsp);

  return EXIT_SUCCESS;
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

