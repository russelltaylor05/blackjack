#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "poker.h"


int main(int argc, char *argv[])
{
  int deck[52], randomHand[HAND_SIZE], staticHand[HAND_SIZE], tempHand[HAND_SIZE];
  int score, rank, staticScore, i= 0, tally = 0;
  int numHands;

  /* seed the random number generator */
  srand48((int) time(NULL));
  
  /* initialize the deck */
  init_deck(deck);
  
  //printRankTable(deck);
  
  setStaticHand(deck, staticHand);  
  setRandomHand(deck, randomHand, NULL);  
  
  /* Static Hand */
  printf("\nStatic Hand: ");
  print_hand(staticHand, HAND_SIZE);
  score = eval_5hand(staticHand);
  rank = hand_rank(score);      
  printf("\nScore: %s (%d)\n\n", value_str[rank], score);

  /* Random Hand */
  printf("Random Hand: ");
  print_hand(randomHand, HAND_SIZE);
  score = eval_5hand(randomHand);
  rank = hand_rank(score);      
  printf("\nScore: %s (%d)\n\n", value_str[rank], score);
  staticScore = score;

  /* Compare Random hand to static Hand */
  numHands = 10000;
  printf("Running %d random hands:\n", numHands);
  while(i < numHands) {
    shuffle_deck(deck);
    setRandomHand(deck, tempHand, randomHand);  
    score = eval_5hand(tempHand);
    rank = hand_rank(score);          
    if(score < staticScore) {
      tally++;
    }
    i++;
  }
  printf("%.2f%% of Hands will beat: ", (float)tally / (float)numHands * 100.00);
  print_hand(randomHand, HAND_SIZE);
  printf("\n");
    
  printf("\n");
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
  cardIndex = find_card(Ten, SPADE, deck);
  hand[2] = deck[cardIndex];
  cardIndex = find_card(Queen, HEART, deck);
  hand[3] = deck[cardIndex];
  cardIndex = find_card(King, HEART, deck);
  hand[4] = deck[cardIndex];
}


/* Picks 5 random cards and sets them in *hand
 * excludedCards will be excluded from random hand
 * Pass NULL to excludedCards if there are none 
 */
void setRandomHand(int *deck, int *hand, int *excludedCards) 
{
  int randNum;  
  int history[HAND_SIZE];
  int historyCnt = 0;  
  int excludeCheckSize;
  srand((int) time(NULL));
  
  /* If excludedCards is NULL we pass a 0 to the size parameter of inArray() */
  excludeCheckSize = (!excludedCards) ? 0 : HAND_SIZE;
  
  while (historyCnt < HAND_SIZE) {
    randNum = rand() % (52);
    if (!inArray(randNum, history, historyCnt)) { // make sure we don't have duplicates 
      if(!inArray(deck[randNum], excludedCards, excludeCheckSize)) { // check excluded hand
        history[historyCnt] = randNum;
        hand[historyCnt] = deck[randNum];
        historyCnt++;      
      }
    }       
  } 
}


/* Print a table for frequency of each Hand by Rank */
void printRankTable(int *deck) 
{
  int hand[5], freq[10];
  int a, b, c, d, e, i, j;

  for ( i = 0; i < 10; i++ )
    freq[i] = 0;
  
  for (a=0;a<48;a++) {
    hand[0] = deck[a];
    for (b=a+1;b<49;b++) {
      hand[1] = deck[b];
      for (c=b+1;c<50;c++) {
        hand[2] = deck[c];
        for (d=c+1;d<51;d++) {
          hand[3] = deck[d];
          for (e=d+1;e<52;e++) {
            hand[4] = deck[e];  
            i = eval_5hand( hand );
            j = hand_rank(i);
            freq[j]++;
          }
        }
      }
    }
  }
  printf("Frequency of Hands\n");
  for(i=1;i<=9;i++) {
    printf( "%15s: %8d\n", value_str[i], freq[i] );
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
