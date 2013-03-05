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
  int deck[52], randomHand[HAND_SIZE], staticHand[HAND_SIZE];
  float analyzeResults[ANALYZE_RESOLUTION];
  int size;
  int throwAwayCards[10];
  
  int *devDeck;
  int *devHand;
  float *devAnalyzeResults;
  
  /* initialize the deck */
  init_deck(deck);
  
  /* Set Hands */
  setStaticHand(deck, staticHand);  
  //setRandomHand(deck, randomHand, throwAwayCards, 0);   


  print_hand(staticHand, HAND_SIZE);
  printf("\n");
  
  size = HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devHand, HAND_SIZE * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(devHand, staticHand, size, cudaMemcpyHostToDevice));

  size = 52 * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devDeck, 52 * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(devDeck, deck, size, cudaMemcpyHostToDevice));

  size = ANALYZE_RESOLUTION * sizeof(float);
  HANDLE_ERROR(cudaMalloc(&devAnalyzeResults, size));  

  analyzeHand<<<1,10>>>(devHand, devDeck, devHand, HAND_SIZE, devAnalyzeResults);
  
  size = ANALYZE_RESOLUTION * sizeof(float);
  HANDLE_ERROR(cudaMemcpy(analyzeResults, devAnalyzeResults, size, cudaMemcpyDeviceToHost));


  return 0;
}