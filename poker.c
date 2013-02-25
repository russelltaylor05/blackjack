#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "poker.h"

int main(int argc, char *argv[])
{
  int deck[52], randomHand[HAND_SIZE], staticHand[HAND_SIZE];
  int score, rank, i= 0;
  int throwAwayCnt;
  int throwAwayCards[] = {0,0,0,0,0,0,0,0,0,0,0,0};
  int excludeCards[] = {0,0,0,0,0,0,0,0,0,0,0,0};
  float results, resultTotal = 0;

  /* seed the random number generator */
  srand48((int) time(NULL));

/* UNCOMMENT TO RUN ONLY FAKE ANALYSIS
  if (analyzePrediction(NULL,NULL,0,0))
  {
     return 0;
  }*/
  
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
  printf("Throw Away First Two Cards: \n");
  
  for(i = 0; i < THROWAWAY_RESOLUTION; i++) {
  
    throwAwayCards[0] = randomHand[0];
    throwAwayCards[1] = randomHand[1];
    throwAwayCnt = 2;

    copyHand(excludeCards, randomHand, HAND_SIZE);
    updateHand(deck, randomHand, throwAwayCards, throwAwayCnt);   

    excludeCards[HAND_SIZE] = randomHand[0];
    excludeCards[HAND_SIZE+1] = randomHand[1];

    results = analyzeHand(randomHand, deck, excludeCards, HAND_SIZE + throwAwayCnt);
    
//    print_hand(randomHand, HAND_SIZE);
    score = eval_5hand(randomHand);
    rank = hand_rank(score);
//    printf("\t %.2f%%\t %s\n", results,  value_str[rank]);
    
    resultTotal += results;    
  }
  printf("\n");  
  printf("New Win Ratio: %.2f%% \n\n", resultTotal / THROWAWAY_RESOLUTION);  
    
  return 0;
}

/* This function uses many loops to ensure that all possible card combinations
 * are analyzed.  It then compares each return of analyzeThrowAway and compares
 * the win percentage returned to bestWinPercent and tracks the best win 
 * percentage calculated.  Each time bestWinPercent is updated, it also
 * updates bestThrowAwaySize depending on which loop block it is in, and also
 * stores the card positions thrown away in a,b,c,d,e.  These are used at the
 * end to determine what the best cards to throw away are (using inArray) and
 * these best throw away cards are returned via *bestThrowAway*/

float analyzePrediction(int *hand, int *deck, int *bestThrowAway, int bestThrowAwaySize)
{
   /*Loop variables*/
   int i=0,j=0,k=0,l=0;
   /*Used to store which cards and return value give the best hand*/
   float bestWinPercent = 0.0;
   int a=-1,b=-1,c=-1,d=-1,e=-1;

   /*Variables just used for testing*/
   int m=0;
   float dummyReturnValue;
   srand(time(NULL));

   /*Replace one card*/
   for (i=0;i<5;i++)
   {
      printf("Replacing one card with (%i)\n",i);

      /*This will be changed from a += to simply an = when function works*/
      dummyReturnValue = analyzeThrowAway(NULL,NULL,NULL,0);

      /*Check if we found a new best win percentage!*/
      if (bestWinPercent < dummyReturnValue)
      {
         a=i;
         bestWinPercent = dummyReturnValue;
         bestThrowAwaySize = 1;
      }
   }

   /*Replace two cards*/
   for (j=0; j<5; j++)
   {
      for (i=j+1; i<5; i++)
      {
         printf("Replacing two cards (%i,%i)\n",i,j);

         /*This will be changed from a += to simply an = when function works*/
         dummyReturnValue = analyzeThrowAway(NULL,NULL,NULL,0);

         /*Check if we found a new best win percentage!*/
         if (bestWinPercent < dummyReturnValue)
         {
            a=i;b=j;
            bestWinPercent = dummyReturnValue;
            bestThrowAwaySize = 2;
         }
      }
   }

   /*Replace three cards*/
   for (k=0; k<5; k++)
   {
      for (j=k+1; j<5; j++)
      {
         for (i=j+1; i<5; i++)
         {
            printf("Replacing three cards (%i,%i,%i)\n",i,j,k);

            /*This will be changed from a += to simply an = when function works*/
            dummyReturnValue = analyzeThrowAway(NULL,NULL,NULL,0);

            /*Check if we found a new best win percentage!*/
            if (bestWinPercent < dummyReturnValue)
            {
               a=i;b=j;c=k;
               bestWinPercent = dummyReturnValue;
               bestThrowAwaySize = 3;
            }
         }
      }
   }

   /*Replace four cards*/
   for (l=0; l<5; l++)
   {
      for (k=l+1; k<5; k++)
      {
         for (j=k+1; j<5; j++)
         {
            for (i=j+1; i<5; i++)
            {
               printf("Replacing four cards (%i,%i,%i,%i)\n",i,j,k,l);
               
               /*This will be changed from a += to simply an = when function works*/
               dummyReturnValue = analyzeThrowAway(NULL,NULL,NULL,0);

               /*Check if we found a new best win percentage!*/
               if (bestWinPercent < dummyReturnValue)
               {
                  a=i;b=j;c=k;d=l;
                  bestWinPercent = dummyReturnValue;
                  bestThrowAwaySize = 4;
               }
            }
         }
      }
   }

   /*Replace five cards--just for fun, this will be hard coded*/
   for (m=0; m<5; m++)
   {
      for (l=m+1; l<5; l++)
      {
         for (k=l+1; k<5; k++)
         {
            for (j=k+1; j<5; j++)
            {
               for (i=j+1; i<5; i++)
               {
                  printf("Replacing five cards (%i,%i,%i,%i,%i)\n",i,j,k,l,m);
                  
                  /*This will be changed from a += to simply an = when function works*/
                  dummyReturnValue = analyzeThrowAway(NULL,NULL,NULL,0);

                  /*Check if we found a new best win percentage!*/
                  if (bestWinPercent < dummyReturnValue)
                  {
                     a=i;b=j;c=k;d=l;e=m;
                     bestWinPercent = dummyReturnValue;
                     bestThrowAwaySize = 5;
                  }
               }
            }
         }
      }
   }

   printf("Best Win Percentage is %.2f%%, by replacing %i cards\n",
         bestWinPercent, bestThrowAwaySize);

   /*Insert loop here using inArray to determine which of a,b,c,d,e are
    * cards that were actually in the user's hand.  The cards that are
    * NOT found by inArray will be passed back via bestThrowAway, as
    * these are the cards that the user should throw away*/

   return bestWinPercent;
}


float analyzeThrowAway(int *hand, int *deck, int *throwAway, int throwAwaySize)
{
   /*Return random number between 0 and 75*/
   return ((rand()%75));  
}


/* Returns: %chance that hand will win */ 
float analyzeHand(int *hand, int *deck, int *exclude, int excludeSize)
{
  int resolution = ANALYZE_RESOLUTION;
  int tempHand[HAND_SIZE];
  int handScore, tempScore, i;
  int wins = 0;

  handScore = eval_5hand(hand);  
  for(i = 0; i < resolution; i++) {
    setRandomHand(deck, tempHand, hand, HAND_SIZE);
    tempScore = eval_5hand(tempHand);
    if(handScore < tempScore) {
      wins++;
    }
  }  
  return (float)wins / (float)resolution * 100.00;
}



/* Picks 5 random cards and sets them in *hand
 * excludedCards will be excluded from random hand
 */
void setRandomHand(int *deck, int *hand, int *excludedCards, int excludeCnt) 
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
void updateHand(int *deck, int *hand, int *throwAwayCards, int throwAwayCnt)
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
int getRandomCard(int *deck, int *exclude, int excludeSize) 
{
  int i = 0;  
  shuffle_deck(deck);
  while(inArray(deck[i], exclude, excludeSize ))
    i++;
  
  if (inArray(deck[i], exclude, excludeSize))
    printf("we got duplicate\n");
     
  return deck[i];
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


/* copies source hand to destination hand */
void copyHand (int *dest, int *source, int handSize) {
  int i;    
  for(i = 0; i < handSize; i++) {
    dest[i] = source[i];
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

/* returns index of cardValue in *hand */
int findCardIndex(int *hand, int cardValue, int handSize) 
{ 
  int i;
  for(i = 0; i < handSize; i++) {
    if (hand[i] == cardValue) {
      return i;
    }    
  }
  return -1;
}

void printHandStats(int *hand, float results) 
{
    print_hand(hand, HAND_SIZE);
    int score = eval_5hand(hand);
    int rank = hand_rank(score);
    printf("\t %.2f%%\t %s\n", results,  value_str[rank]);

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


