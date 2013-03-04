#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "poker.h"


int main(int argc, char *argv[])
{
  int deck[52], randomHand[HAND_SIZE], staticHand[HAND_SIZE];
  int score, rank;
  int throwAwayCnt;
  int throwAwayCards[HAND_SIZE * 2];
  int bestThrowAway[5];
  int bestThrowAwaySize = 0;
  float results;
  
  /* seed the random number generator */
  srand48((int) time(NULL));
  
  /* initialize the deck */
  init_deck(deck);
  
  /* Set Hands */
  setStaticHand(deck, staticHand);  
  setRandomHand(deck, randomHand, throwAwayCards, 0);   

  /* Random Hand */
  printf("\nRandom Hand: ");
  print_hand(randomHand, HAND_SIZE);
  score = eval_5hand(randomHand);
  rank = hand_rank(score);      
  printf("\nScore: %s (%d)\n", value_str[rank], score);

  results = analyzeHand(randomHand, deck, randomHand, HAND_SIZE);
  printf("Win Ration: %.2f%% \n\n", results);
  

  /* Throw Away logic - Should be moved to a function */
  /*printf("Throw Away First Two Cards: \n");
  throwAwayCards[0] = randomHand[0];
  throwAwayCards[1] = randomHand[1];
  throwAwayCnt = 0;
  results = analyzeThrowAway(randomHand, deck, throwAwayCards, throwAwayCnt);
  printf("\nNew Win Ratio: %.2f%% \n\n", results);  */

  analyzePrediction(randomHand, deck, bestThrowAway, bestThrowAwaySize);
    
  return 0;
}
