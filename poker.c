#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "poker.h"


int main(int argc, char *argv[])
{
  int deck[52], randomHand[HAND_SIZE], staticHand[HAND_SIZE];
  int score, rank;
  int throwAwayCards[HAND_SIZE * 2];
  int *bestThrowAway;
  int bestThrowAwaySize = 0;
  float results;

  if (NULL == (bestThrowAway = malloc(sizeof(int) * 5)))
  {
     printf("Malloc Error\n");
     exit(EXIT_FAILURE);
  }
  
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
  
  /*Exhaustively analyze all possible throw away combinations*/
  results = analyzePrediction(randomHand, deck, bestThrowAway, &bestThrowAwaySize);
  printf("\n\nDiscard the follwing %i cards: ", bestThrowAwaySize);
  print_hand(bestThrowAway, bestThrowAwaySize);
  printf("\nTo acheieve %.2f%% odds of winning\n",results);
    
  return 0;
}
