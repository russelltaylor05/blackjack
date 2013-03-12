#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <getopt.h>
#include "poker.h"


int main(int argc, char *argv[])
{
  int deck[52], staticHand[HAND_SIZE];
  int score, rank;
  float results;  
  clock_t start, stop;
  ARGSP *argsp;
  
  argsp = (ARGSP *)malloc(sizeof(ARGSP));  
  if(getArgs(argsp, argc, argv) < 0) {
    printf("Card arguments broken\n");
    return EXIT_FAILURE;
  }; 

  /* seed the random number generator */
  srand48((int) time(NULL));
  
  /* initialize the deck */
  init_deck(deck);
  
  /* Set Hand from Args */
  setHandFromArgs(deck, staticHand, argsp);
    
  
  start = clock();
  
  results = analyzeHand(staticHand, deck, staticHand, HAND_SIZE);
    
  stop = clock();

  /* Original Hand */
  printf("Hand: \t\t");
  print_hand(staticHand, HAND_SIZE);
  score = eval_5hand(staticHand);
  rank = hand_rank(score);      
  printf("\nScore: \t\t%s (%d)\n", value_str[rank], score);
  printf("Win Ration: \t%.2f%% \n", results);
  printf("Time: \t\t%.4f \n", (float)(stop - start) / CLOCKS_PER_SEC); 
  printf("Analyze Res: \t%d \n", ANALYZE_RESOLUTION);   
  printf("Throw Res: \t%d \n", THROWAWAY_RESOLUTION);

  free(argsp);
    
  return 0;
}
