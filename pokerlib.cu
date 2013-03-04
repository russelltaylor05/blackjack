#include <stdio.h>
#include "lookuptable.h"
#include "poker.h"


/* Picks 5 random cards and sets them in *hand
 * excludedCards will be excluded from random hand
 */
__device__ void setRandomHand(int *deck, int *hand, int *excludedCards, int excludeCnt) 
{
  int i;  
  int excludedCardsTemp[HAND_SIZE * 2];

  /* Copy exclude cards to new temp array */
  copyHand(excludedCardsTemp, excludedCards, excludeCnt);
  
  /* Every time we get a new random card, add
   * it to the excludedCardsTemp array so that it won't
   * get choosen again.
   */
  for(i = 0; i < HAND_SIZE; i++) {
    hand[i] = getRandomCard(deck, excludedCardsTemp, excludeCnt);
    excludedCardsTemp[excludeCnt] = hand[i];
    excludeCnt++;    
  }
}


/* Updates a hand's cards specified by throwAwayCards[]
 */
__device__ void updateHand(int *deck, int *hand, int *throwAwayCards, int throwAwayCnt)
{
  int index, i = 0;
  int excludeCnt = HAND_SIZE;
  int excludedCards[HAND_SIZE * 2]; // large enough for a 5 card hand plus 5 cards to throw away
      
  /* Copy hand into excludeCards array */
  for(i = 0; i < HAND_SIZE; i++)
    excludedCards[i] = hand[i];
  
  /* For each throw away card, choose a new random card
   * that is not in our excluded cards array.
   * Once a new card is choosen, add it the end of the 
   * excluded cards array
   */
  for(i = 0; i < throwAwayCnt; i++) {    
    index = findCardIndex(hand, throwAwayCards[i], HAND_SIZE);
    hand[index] = getRandomCard(deck, excludedCards, excludeCnt);
    excludedCards[excludeCnt] = hand[index];
    excludeCnt++;
  }
}

/* Returns random card VALUE that is not in exclude array */
__device__ int getRandomCard(int *deck, int *exclude, int excludeSize) 
{
  int i = 0;  
  shuffle_deck(deck);
  while(inArray(deck[i], exclude, excludeSize )) {
    i++;
  }
         
  return deck[i];
}


/* Manually set a hand of cards
 * Check out poker.h for the #define variables
 * Assumes deck is initialized
 */
__device__ void setStaticHand(int *deck, int *hand) 
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


/* copies source hand to destination hand */
__device__ void copyHand (int *dest, int *source, int handSize) {
  int i;    
  for(i = 0; i < handSize; i++) {
    dest[i] = source[i];
  }  
}


/* Return 1 if value is in array
 * Return 0 if value is not in array
 */
__device__ int inArray(int value, int *array, int size) 
{ 
  int i;
  for(i = 0; i < size; i++) {
    if (array[i] == value) {
      return 1;
    }    
  }
  return 0;
}

/* returns index of cardValue in *hand */
__device__ int findCardIndex(int *hand, int cardValue, int handSize) 
{ 
  int i;
  for(i = 0; i < handSize; i++) {
    if (hand[i] == cardValue) {
      return i;
    }    
  }
  return -1;
}

__device__ void printHandStats(int *hand, float results) 
{
    print_hand(hand, HAND_SIZE);
    int score = eval_5hand(hand);
    int rank = hand_rank(score);
    //printf("\t %.2f%%\t %s\n", results,  value_str[rank]);

}




/* Print a table for frequency of each Hand by Rank */
__device__ void printRankTable(int *deck) 
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
    //printf( "%15s: %8d\n", value_str[i], freq[i] );
  }
}
