#include <stdio.h>
#include "lookuptable.h"
#include "poker.h"

void    srand48();
double  drand48();

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
   float bestWinPercent = 0.0, newWinPercent = 0.0;
   int newThrowAway[5];
   
   /*Variables just used for testing*/
   int m=0;
   
   /*Replace one card*/
   for (i=0;i<5;i++)
   {
      printf("Replacing one card with (%i)\n",i);
      newThrowAway[0] = hand[i];
      
      /*This will be changed from a += to simply an = when function works*/
      newWinPercent = analyzeThrowAway(hand,deck,newThrowAway,1);
      
      /*Check if we found a new best win percentage!*/
      if (bestWinPercent < newWinPercent)
      {
         bestThrowAway[0] = newThrowAway[0];
         bestWinPercent = newWinPercent;
         bestThrowAwaySize = 1;
      }
   }
   
   /*Replace two cards*/
   for (j=0; j<5; j++)
   {
      for (i=j+1; i<5; i++)
      {
         printf("Replacing two cards (%i,%i)\n",i,j);
         newThrowAway[0] = hand[i];
         newThrowAway[1] = hand[j];
         
         /*This will be changed from a += to simply an = when function works*/
         newWinPercent = analyzeThrowAway(hand,deck,newThrowAway,2);
         
         /*Check if we found a new best win percentage!*/
         if (bestWinPercent < newWinPercent)
         {
            bestThrowAway[0] = newThrowAway[0];
            bestThrowAway[1] = newThrowAway[1];
            bestWinPercent = newWinPercent;
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
            newThrowAway[0] = hand[i];
            newThrowAway[1] = hand[j];
            newThrowAway[2] = hand[k];
            
            /*This will be changed from a += to simply an = when function works*/
            newWinPercent = analyzeThrowAway(hand,deck,newThrowAway,3);
            
            /*Check if we found a new best win percentage!*/
            if (bestWinPercent < newWinPercent)
            {
               bestThrowAway[0] = newThrowAway[0];
               bestThrowAway[1] = newThrowAway[1];
               bestThrowAway[2] = newThrowAway[2];
               bestWinPercent = newWinPercent;
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
               newThrowAway[0] = hand[i];
               newThrowAway[1] = hand[j];
               newThrowAway[2] = hand[k];
               newThrowAway[3] = hand[l];
               
               /*This will be changed from a += to simply an = when function works*/
               newWinPercent = analyzeThrowAway(hand,deck,newThrowAway,4);
               
               /*Check if we found a new best win percentage!*/
               if (bestWinPercent < newWinPercent)
               {
                  bestThrowAway[0] = newThrowAway[0];
                  bestThrowAway[1] = newThrowAway[1];
                  bestThrowAway[2] = newThrowAway[2];
                  bestThrowAway[3] = newThrowAway[3];
                  bestWinPercent = newWinPercent;
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
                  newThrowAway[0] = hand[i];
                  newThrowAway[1] = hand[j];
                  newThrowAway[2] = hand[k];
                  newThrowAway[3] = hand[l];
                  newThrowAway[4] = hand[m];
                  
                  /*This will be changed from a += to simply an = when function works*/
                  newWinPercent = analyzeThrowAway(hand,deck,newThrowAway,5);
                  
                  /*Check if we found a new best win percentage!*/
                  if (bestWinPercent < newWinPercent)
                  {
                     bestThrowAway[0] = newThrowAway[0];
                     bestThrowAway[1] = newThrowAway[1];
                     bestThrowAway[2] = newThrowAway[2];
                     bestThrowAway[3] = newThrowAway[3];
                     bestThrowAway[4] = newThrowAway[4];
                     bestWinPercent = newWinPercent;
                     bestThrowAwaySize = 5;
                  }
               }
            }
         }
      }
   }
   
   printf("Best Win Percentage is %.2f%%, by replacing %i cards\n",
         bestWinPercent, bestThrowAwaySize);
   
   return bestWinPercent;
}

/* Generates new random cards specified in throwAwayCards
 * returns avergae win % of new hands compared against random dealer hands
 */ 
float analyzeThrowAway(int *hand, int *deck, int *throwAwayCards, int throwAwayCnt)
{
  float results, resultsTotal = 0;
  int excludeCards[HAND_SIZE*2], originalHand[HAND_SIZE];
  int score, rank, i;
  int excludeCnt = HAND_SIZE + throwAwayCnt;
  
  copyHand(excludeCards, hand, HAND_SIZE);
  copyHand(originalHand, hand, HAND_SIZE);

  /* Add throw away cards to exclude list */
  for (i = 0; i < throwAwayCnt; i++) {
    excludeCards[HAND_SIZE + i] = throwAwayCards[i];
  }  
  
  for(i = 0; i < THROWAWAY_RESOLUTION; i++) {

    copyHand(hand, originalHand, HAND_SIZE);
    updateHand(deck, hand, throwAwayCards, throwAwayCnt);   

    results = analyzeHand(hand, deck, excludeCards, excludeCnt);
    
    print_hand(hand, HAND_SIZE);
    score = eval_5hand(hand);
    rank = hand_rank(score);
    printf("\t %.2f%%\t %s\n", results,  value_str[rank]);
    
    resultsTotal += results;    
  }
  
  return resultsTotal / THROWAWAY_RESOLUTION;
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
  while(inArray(deck[i], exclude, excludeSize )) {
    i++;
  }
         
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





// perform a binary search on a pre-sorted array
//
int findit( int key )
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
    fprintf( stderr, "ERROR:  no match found; key = %d\n", key );
    return( -1 );
}

//
//   This routine initializes the deck.  A deck of cards is
//   simply an integer array of length 52 (no jokers).  This
//   array is populated with each card, using the following
//   scheme:
//
//   An integer is made up of four bytes.  The high-order
//   bytes are used to hold the rank bit pattern, whereas
//   the low-order bytes hold the suit/rank/prime value
//   of the card.
//
//   +--------+--------+--------+--------+
//   |xxxbbbbb|bbbbbbbb|cdhsrrrr|xxpppppp|
//   +--------+--------+--------+--------+
//
//   p = prime number of rank (deuce=2,trey=3,four=5,five=7,...,ace=41)
//   r = rank of card (deuce=0,trey=1,four=2,five=3,...,ace=12)
//   cdhs = suit of card
//   b = bit turned on depending on rank of card
//
void init_deck( int *deck )
{
    int i, j, n = 0, suit = 0x8000;

    for ( i = 0; i < 4; i++, suit >>= 1 )
        for ( j = 0; j < 13; j++, n++ )
            deck[n] = primes[j] | (j << 8) | suit | (1 << (16+j));
}


//  This routine will search a deck for a specific card
//  (specified by rank/suit), and return the INDEX giving
//  the position of the found card.  If it is not found,
//  then it returns -1
//
int
find_card( int rank, int suit, int *deck )
{
	int i, c;

	for ( i = 0; i < 52; i++ )
	{
		c = deck[i];
		if ( (c & suit)  &&  (RANK(c) == rank) )
			return( i );
	}
	return( -1 );
}


//
//  This routine takes a deck and randomly mixes up
//  the order of the cards.
//
void shuffle_deck( int *deck )
{
    int i, n, temp[52];

    for ( i = 0; i < 52; i++ )
        temp[i] = deck[i];

    for ( i = 0; i < 52; i++ )
    {
        do {
            n = (int)(51.9999999 * drand48());
        } while ( temp[n] == 0 );
        deck[i] = temp[n];
        temp[n] = 0;
    }
}


void print_hand( int *hand, int n )
{
    int i, r;
    char suit;
    static char *rank = "23456789TJQKA";

    for ( i = 0; i < n; i++ ) 
    {
        r = (*hand >> 8) & 0xF;
        if ( *hand & 0x8000 )
            suit = 'c';
        else if ( *hand & 0x4000 )
            suit = 'd';
        else if ( *hand & 0x2000 )
            suit = 'h';
        else
            suit = 's';

        printf( "%c%c ", rank[r], suit );
        hand++;
    }
}

void print_card(int card) 
{
  int r;
  char suit;
  static char *rank = "23456789TJQKA";

  r = (card >> 8) & 0xF;
  if ( card & 0x8000 )
      suit = 'c';
  else if ( card & 0x4000 )
      suit = 'd';
  else if ( card & 0x2000 )
      suit = 'h';
  else
      suit = 's';

  printf( "%c%c ", rank[r], suit );
}


int hand_rank( short val )
{
    if (val > 6185) return(HIGH_CARD);        // 1277 high card
    if (val > 3325) return(ONE_PAIR);         // 2860 one pair
    if (val > 2467) return(TWO_PAIR);         //  858 two pair
    if (val > 1609) return(THREE_OF_A_KIND);  //  858 three-kind
    if (val > 1599) return(STRAIGHT);         //   10 straights
    if (val > 322)  return(FLUSH);            // 1277 flushes
    if (val > 166)  return(FULL_HOUSE);       //  156 full house
    if (val > 10)   return(FOUR_OF_A_KIND);   //  156 four-kind
    return(STRAIGHT_FLUSH);                   //   10 straight-flushes
}


short eval_5cards( int c1, int c2, int c3, int c4, int c5 )
{
    int q;
    short s;

    q = (c1|c2|c3|c4|c5) >> 16;

    /* check for Flushes and StraightFlushes
    */
    if ( c1 & c2 & c3 & c4 & c5 & 0xF000 )
	return( flushes[q] );

    /* check for Straights and HighCard hands
    */
    s = unique5[q];
    if ( s )  return ( s );

    /* let's do it the hard way
    */
    q = (c1&0xFF) * (c2&0xFF) * (c3&0xFF) * (c4&0xFF) * (c5&0xFF);
    q = findit( q );

    return( values[q] );
}


short eval_5hand( int *hand )
{
    int c1, c2, c3, c4, c5;

    c1 = *hand++;
    c2 = *hand++;
    c3 = *hand++;
    c4 = *hand++;
    c5 = *hand;

    return( eval_5cards(c1,c2,c3,c4,c5) );
}


// This is a non-optimized method of determining the
// best five-card hand possible out of seven cards.
// I am working on a faster algorithm.
//
short eval_7hand( int *hand )
{
    int i, j, q, best = 9999, subhand[5];

	for ( i = 0; i < 21; i++ )
	{
		for ( j = 0; j < 5; j++ )
			subhand[j] = hand[ perm7[i][j] ];
		q = eval_5hand( subhand );
		if ( q < best )
			best = q;
	}
	return( best );
}
