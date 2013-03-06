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
  int size, rank;
  int handScore;
  //int throwAwayCards[10];
  
  int *devHand;
  int *devAnalyzeResults;
  
  curandState *devStates;
  
  CUDA_CALL(cudaMalloc((void **)&devStates, ANALYZE_RESOLUTION * sizeof(curandState)));
  
  /* initialize the deck */
  init_deck(deck);
  
  /* Set Hands */
  setStaticHand(deck, staticHand);
  //setRandomHand(deck, randomHand, throwAwayCards, 0);   

  handScore = eval_5hand(staticHand);
  rank = hand_rank(handScore);
  printf("CPU Hand: ");
  print_hand(staticHand, 5);
  printf("CPU Score: ");
  printf("%d\n", handScore);  
  printf("CPU Rank: %s\n\n", value_str[rank]);
  
  
  size = HAND_SIZE * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devHand, HAND_SIZE * sizeof(int)));
  HANDLE_ERROR(cudaMemcpy(devHand, staticHand, size, cudaMemcpyHostToDevice));

  size = ANALYZE_RESOLUTION * sizeof(int);
  HANDLE_ERROR(cudaMalloc(&devAnalyzeResults, size));  

  analyzeHand<<<1,ANALYZE_RESOLUTION>>>(devHand, devHand, HAND_SIZE, devAnalyzeResults, devStates);
  
  size = ANALYZE_RESOLUTION * sizeof(float);
  HANDLE_ERROR(cudaMemcpy(analyzeResults, devAnalyzeResults, size, cudaMemcpyDeviceToHost));


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


/*
int findit2( int key )
{
    int low = 0, high = 4887, mid;

    while ( low <= high )
    {
        mid = (high+low) >> 1;      // divide by two
        if ( key < products[mid] )
            high = mid - 1;
        else if ( key > products[mid] )
            low = mid + 1;
        else
            return( mid );
    }
    printf("ERROR:  no match found; key = %d\n", key );
    return( -1 );
}

short eval_5cards2( int c1, int c2, int c3, int c4, int c5 )
{
    int q;
    short s;
    q = (c1|c2|c3|c4|c5) >> 16;
    if ( c1 & c2 & c3 & c4 & c5 & 0xF000 )
  	 return( flushes[q] );
    s = unique5[q];
    if ( s )  return ( s );
    q = (c1&0xFF) * (c2&0xFF) * (c3&0xFF) * (c4&0xFF) * (c5&0xFF);
    q = findit2( q );

    return( values[q] );
}


short eval_5hand2( int *hand )
{
    int c1, c2, c3, c4, c5;
    c1 = *hand++;
    c2 = *hand++;
    c3 = *hand++;
    c4 = *hand++;
    c5 = *hand;
    return( eval_5cards2(c1,c2,c3,c4,c5) );
}
*/
