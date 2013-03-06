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

void print_bits(int number ){
   unsigned long mask = 1 << 30;
   int cnt = 1;
   while(mask){
      (mask & number) ? printf("X") : printf(".");
      mask = mask >> 1 ; 
      if(!(cnt % 8)){
         putchar('|');
      }
      cnt++;
   }
   putchar('\n');
}


int main(int argc, char *argv[])
{
  int deck[52], staticHand[HAND_SIZE];
  int analyzeResults[ANALYZE_RESOLUTION];
  int size;
  
  int *devHand;
  int *devAnalyzeResults;
  
  curandState *devStates;
  
  CUDA_CALL(cudaMalloc((void **)&devStates, ANALYZE_RESOLUTION * sizeof(curandState)));
  
  /* initialize the deck */
  init_deck(deck);
  
  /* Set Hands */
  setStaticHand(deck, staticHand);
  //setRandomHand(deck, randomHand, throwAwayCards, 0);   

  /*
  printf("CPU Hand: ");
  print_hand(staticHand, 5);
  print_bits(staticHand[0]);
  print_bits(staticHand[1]);
  print_bits(staticHand[2]);
  print_bits(staticHand[3]);
  print_bits(staticHand[4]);  
  */
  
  size = HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devHand, HAND_SIZE * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(devHand, staticHand, size, cudaMemcpyHostToDevice));

  size = ANALYZE_RESOLUTION * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devAnalyzeResults, size));  

  int blockCnt = (ANALYZE_RESOLUTION + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  printf("block cnt: %d\n", blockCnt);
  
  
  analyzeHand<<<blockCnt,THREADS_PER_BLOCK>>>(devHand, devHand, HAND_SIZE, devAnalyzeResults, devStates);
  
  size = ANALYZE_RESOLUTION * sizeof(int);
  HANDLE_ERROR(cudaMemcpy(analyzeResults, devAnalyzeResults, size, cudaMemcpyDeviceToHost));

  printf("Score: %.2f%%\n", (float)analyzeResults[0] / (float)ANALYZE_RESOLUTION * 100.0);

  return 0;
}


/* Manually set a hand of cards
 * Check out poker.h for the #define variables
 * Assumes deck is initialized
 */
void setStaticHand(int *deck, int *hand) 
{
  int cardIndex = 0;
  
  cardIndex = find_card(Nine, DIAMOND, deck);
  hand[0] = deck[cardIndex];  
  cardIndex = find_card(Ace, HEART, deck);
  hand[1] = deck[cardIndex];
  cardIndex = find_card(Queen, SPADE, deck);
  hand[2] = deck[cardIndex];
  cardIndex = find_card(Queen, HEART, deck);
  hand[3] = deck[cardIndex];
  cardIndex = find_card(King, HEART, deck);
  hand[4] = deck[cardIndex];
}