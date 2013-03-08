#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include "cpu_poker.h"

void    srand48();
double  drand48(); 


int main(int argc, char *argv[])
{
  int deck[52];
  int staticHand[HAND_SIZE_CPU];
  int blockCnt;
  int analyzeResults[ANALYZE_RESOLUTION_CPU];
  int size, sum =0;
  int i, score, rank;
  ARGSP *argsp;
  float originalResults;
  
  clock_t start, stop;
  
  argsp = (ARGSP *)malloc(sizeof(ARGSP));  
  if(getArgs(argsp, argc, argv) < 0) {
    printf("Card arguments broken\n");
    return EXIT_FAILURE;
  };

  srand48((int) time(NULL));  
    
  start = clock();
  
  init_deck_cpu(deck);
  setHandFromArgs(deck, staticHand, argsp);

  score = eval_5hand_cpu(staticHand);
  rank = hand_rank_cpu(score);
  printf("Hand: \t\t");
  print_hand_cpu(staticHand, HAND_SIZE_CPU);
  printf("\nScore: \t\t%d\n", score);
  printf("Rank: \t\t%s\n", value_str_cpu[rank]);  

  originalResults = analyzeHand(staticHand, deck, staticHand, HAND_SIZE_CPU);
  printf("Win Ration: %.2f%% \n\n", originalResults);

  stop = clock();

  free(argsp);

  return EXIT_SUCCESS;
}

