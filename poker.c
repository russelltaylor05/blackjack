#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "poker.h"


int main(int argc, char *argv[])
{
  int deck[52], randomHand[HAND_SIZE], staticHand[HAND_SIZE];
  int score, rank;
  
  /* seed the random number generator */
  srand48((int) time(NULL));
  
  /* initialize the deck */
  init_deck(deck);
    
  setStaticHand(deck, staticHand);  
  setRandomHand(deck, randomHand);  
  
  /* Static Hand */
  printf("\nStatic Hand\n");
  print_hand(staticHand, HAND_SIZE);
  score = eval_5hand(staticHand);
  rank = hand_rank(score);      
  printf("\nScore: %d\n", score);
  printf("Rank: %s\n\n\n",  value_str[rank]);
  
  /* Random Hand */
  printf("Random Hand\n");
  print_hand(randomHand, HAND_SIZE);
  score = eval_5hand(randomHand);
  rank = hand_rank(score);      
  printf("\nScore: %d\n", score);
  printf("Rank: %s\n\n",  value_str[rank]);


  /* Randomize hand until finding a Full House */
  printf("Looking for a FULL_HOUSE:\n");
  while(rank != FULL_HOUSE) {
    shuffle_deck(deck);
    setRandomHand(deck, randomHand);  
    score = eval_5hand(randomHand);
    rank = hand_rank(score);      
    printf(".");
  }
  printf("\nFull House Score: %d\n", score);

  printf("\n");
  return 0;
}



/* Manually set a hand of cards
 * Check out poker.h for the #define variables
 * Assumes deck is initialed with all cards
 */
void setStaticHand(int *deck, int *hand) 
{
  int cardIndex = 0;
  
  cardIndex = find_card(Nine, DIAMOND, deck);
  hand[0] = deck[cardIndex];  
  cardIndex = find_card(Nine, HEART, deck);
  hand[1] = deck[cardIndex];
  cardIndex = find_card(Nine, SPADE, deck);
  hand[2] = deck[cardIndex];
  cardIndex = find_card(Queen, HEART, deck);
  hand[3] = deck[cardIndex];
  cardIndex = find_card(King, HEART, deck);
  hand[4] = deck[cardIndex];
}


/* Picks 5 random cards and sets them in *hand
 * inArray() makes sure we don't have duplicates
 */
void setRandomHand(int *deck, int *hand) 
{
  int randNum;  
  int history[HAND_SIZE];
  int historyCnt = 0;
  
  srand((int) time(NULL));
  
  while (historyCnt < HAND_SIZE) {
    randNum = rand() % (52);
    if (!inArray(randNum, history, historyCnt)) {
      history[historyCnt] = randNum;
      hand[historyCnt] = deck[randNum];
      historyCnt++;      
    }      
    
  } 

}

/* Return 1 if value is in array
 * Return 0 if value is not in array
 */
int inArray(int value, int *array, int size) 
{ 
  int i;
  for(i = 0; i < size; i++) {
    if (array[i] == value) {
      return 1;
    }    
  }
  return 0;
}