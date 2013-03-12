#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "lookuptable.h"
#include "poker.h"


__global__ void curandSetup(curandState *state) 
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //curand_init(clock(), index, 0, &state[index]);
  curand_init((clock() << 20) + index, 0 , 0, &state[index]);
}


__global__ void analyzeThrowCombos(int *hand, int *devthrowCombosResults, int *devThrowResults, curandState *state)
{
  int deck[52];
  int randomHand[HAND_SIZE];
  int compareHand[HAND_SIZE];
  int excludeCards[HAND_SIZE * 2];
  int excludeCnt = 10;
  int randomScore, handScore;
  int i;
  __shared__ int tempResults[THREADS_PER_BLOCK];
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int compareIndex = (index / ANALYZE_RESOLUTION) * 5;
      
  init_deck(deck);

  
  curandState localState = state[index];
  // If THROWAWAY_RESOLUTION < 20 the curand stuff destroys cuda memory ????
  

  for(i = 0; i < 5; i++) {
    compareHand[i] = devthrowCombosResults[compareIndex + i];
    excludeCards[i] = compareHand[i];
  }
  for(i = 5; i < 10; i++) {
    excludeCards[i] = hand[i - HAND_SIZE];
  }
  
  //setRandomHand(deck, randomHand, excludeCards, excludeCnt, localState);
  setRandomHand(deck, randomHand, hand, HAND_SIZE, localState);
  /* error with exclude Cards */
  
  handScore = eval_5hand(compareHand);
  randomScore = eval_5hand(randomHand);
  // if(randomScore == 666) we got a bad hand. do not include
  
  tempResults[threadIdx.x] =  (handScore < randomScore);

  for (i = THREADS_PER_BLOCK / 2;  i > 0; i >>= 1) {
    __syncthreads();
    if(threadIdx.x < i) {
      tempResults[threadIdx.x] = tempResults[threadIdx.x] + tempResults[threadIdx.x + i];
    }
  }
  
  __syncthreads();
  if(threadIdx.x == 0) {
    //printf("Resutls: %d\t blockid: %d\n", tempResults[0], blockIdx.x);
    devThrowResults[blockIdx.x] = tempResults[0];
  }
  

  /*
  if(index == 0) {
    randomScore = eval_5hand(hand);
    rank = hand_rank(randomScore);
    printf("K2 result: \t%d\n", devThrowResults[index]);
    printf("K2 Hand: \t");   print_hand(hand, HAND_SIZE);
    printf("K2 Comp Hand: \t");   print_hand(compareHand, HAND_SIZE);
    printf("K2 Rand: \t");   print_hand(randomHand, HAND_SIZE);
    printf("K2 Score: \t%d\n", randomScore);
    printf("K2 rank: \t%s\n", value_str[rank]);    
  }
  */
}

__global__ void createThrowCombos(int *hand, int *throwCards, int throwCnt, int *devthrowCombosResults, curandState *state)
{
  int deck[52];
  int tempHand[HAND_SIZE];
  int i;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int resultIndex = index * 5;
  
  init_deck(deck);
  curandState localState = state[index];
  /* If THROWAWAY_RESOLUTION < 20 the curand stuff destroys cuda memory ???? */
  
  /*
  if(index == 0) {
    tempScore = eval_5hand(hand);
    rank = hand_rank(tempScore);
    printf("GPU Hand: \t");   print_hand(hand, HAND_SIZE);
    printf("GPU Throw: \t");  print_hand(throwCards, throwCnt);
    printf("GPU Score: \t%d\n", tempScore);
    printf("GPU rank: \t%s\n", value_str[rank]);    
  }
  */

  copyHand (tempHand, hand, HAND_SIZE);
  updateHand(deck, tempHand, throwCards, throwCnt, localState);
  
  for(i = 0; i < HAND_SIZE; i++) {
    devthrowCombosResults[resultIndex++] = tempHand[i];      
  }
  
}


__global__ void analyzeHand(int *hand, int *exclude, int excludeSize, int *devAnalyzeResults, curandState *state)
{
  int deck[52];
  int tempHand[HAND_SIZE];
  int tempScore;
  int handScore;
  int rank;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  curandState localState = state[index];
  
  init_deck(deck);  
  handScore = eval_5hand(hand);

  /*  
  if(index == 0) {
    rank = hand_rank(handScore);
    printf("GPU Hand: \t");   print_hand(hand, HAND_SIZE);
    printf("GPU Score: \t%d\n", handScore);
    printf("GPU rank: \t%s\n", value_str[rank]);    
  }
  */
  
  setRandomHand(deck, tempHand, hand, HAND_SIZE, localState);
  tempScore = eval_5hand(tempHand); 
  devAnalyzeResults[index] =  (handScore < tempScore);
}


__device__ void setStaticHandDev(int *deck, int *hand) 
{
  int cardIndex = 0;
  
  cardIndex = find_card(Nine, DIAMOND, deck);
  hand[0] = deck[cardIndex];  
  cardIndex = find_card(Ten, HEART, deck);
  hand[1] = deck[cardIndex];
  cardIndex = find_card(Queen, SPADE, deck);
  hand[2] = deck[cardIndex];
  cardIndex = find_card(Queen, HEART, deck);
  hand[3] = deck[cardIndex];
  cardIndex = find_card(King, HEART, deck);
  hand[4] = deck[cardIndex];
}


/* Picks 5 random cards and sets them in *hand
 * excludedCards will be excluded from random hand
 */
__device__ void setRandomHand(int *deck, int *hand, int *excludedCards, int excludeCnt, curandState localState) 
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
    hand[i] = getRandomCard(deck, excludedCardsTemp, excludeCnt, localState);
    excludedCardsTemp[excludeCnt] = hand[i];
    excludeCnt++;    
  }
}


/* Updates a hand's cards specified by throwAwayCards[]
 */
__device__ void updateHand(int *deck, int *hand, int *throwAwayCards, int throwAwayCnt, curandState localState)
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
    hand[index] = getRandomCard(deck, excludedCards, excludeCnt, localState);
    excludedCards[excludeCnt] = hand[index];
    excludeCnt++;
  }
}

/* Returns random card VALUE that is not in exclude array */
__device__ int getRandomCard2(int *deck, int *exclude, int excludeSize, curandState localState) 
{
  int i = 0;  
  shuffle_deck(deck, localState);
  while(inArray(deck[i], exclude, excludeSize )) {
    i++;
  }
         
  return deck[i];
}

/* Returns random card VALUE that is not in exclude array */
__device__ int getRandomCard(int *deck, int *exclude, int excludeSize, curandState localState) 
{
  int n;
  
  n = (int)(51.9999999 * curand_uniform(&localState));
  while(inArray(deck[n], exclude, excludeSize )) {
    n = (int)(51.9999999 * curand_uniform(&localState));
  }
         
  return deck[n];
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

__device__ void printHandStats(int *hand) 
{
    print_hand(hand, HAND_SIZE);
    int score = eval_5hand(hand);
    int rank = hand_rank(score);
    printf("%d\t %s\n", score, value_str[rank]);
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


// perform a binary search on a pre-sorted array
//
__device__ int findit( int key )
{
    int low = 0, high = 4887, mid;

    //printf("key: %d", key);
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
__device__ void init_deck( int *deck )
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
__device__  int find_card( int rank, int suit, int *deck )
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
__device__ void shuffle_deck(int *deck, curandState localState)
{
    int i, n, temp[52];

  //printf("%d) %f\n", threadIdx.x, curand_uniform(&localState));    
  //printf("%d) %f\n", threadIdx.x, curand_uniform(&localState));    

    for ( i = 0; i < 52; i++ )
        temp[i] = deck[i];

    for ( i = 0; i < 52; i++ )
    {
        do {
            n = (int)(51.9999999 * curand_uniform(&localState));            
        } while ( temp[n] == 0 );        
        deck[i] = temp[n];
        temp[n] = 0;
    }
    //printf("%d,", deck[0]);
}


__device__ void print_hand( int *hand, int n )
{
  int i, r;
  char suit;
  char *rank = "23456789TJQKA";
  char handString[17];
  int pnt = 0;

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

      handString[pnt++] = rank[r];
      handString[pnt++] = suit;
      handString[pnt++] = ' ';
      
      hand++;
  }
  handString[pnt++] = '\n';
  handString[pnt] = '\0';
  printf("%s", handString);
}

__device__ void print_card(int card) 
{
  int r;
  char suit;
  char *rank = "23456789TJQKA";

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


__device__ int hand_rank( short val )
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


__device__ short eval_5cards( int c1, int c2, int c3, int c4, int c5 )
{
    int q;
    short s;

    q = (c1|c2|c3|c4|c5) >> 16;

    /* check for Flushes and StraightFlushes */
    if ( c1 & c2 & c3 & c4 & c5 & 0xF000 )
      return( flushes[q] );

    /* check for Straights and HighCard hands */
    s = unique5[q];
    if ( s )  return ( s );

    /* let's do it the hard way */
    q = (c1&0xFF) * (c2&0xFF) * (c3&0xFF) * (c4&0xFF) * (c5&0xFF);
    q = findit(q);
    
    if(q < 0)
      return 666;

    return( values[q] );
}


__device__ short eval_5hand( int *hand )
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
 __device__ short eval_7hand( int *hand )
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


__device__ void print_bits_dev(int number ){
   unsigned long mask = 1 << 30;
   int cnt = 1;
   while(mask){
      (mask & number) ? printf("X") : printf(".");
      mask = mask >> 1 ; 
      if(!(cnt % 8)){
         printf("|");
      }
      cnt++;
   }
   printf("\n");
}



/* Generates new random cards specified in throwAwayCards
 * returns avergae win % of new hands compared against random dealer hands
 */ 
 /*
__global__ void analyzeThrowAway(int *hand, int *deck, int *throwAwayCards, int throwAwayCnt)
{
  float results, resultsTotal = 0;
  int excludeCards[HAND_SIZE*2], originalHand[HAND_SIZE];
  int score, rank, i;
  int excludeCnt = HAND_SIZE + throwAwayCnt;
  
  copyHand(excludeCards, hand, HAND_SIZE);
  copyHand(originalHand, hand, HAND_SIZE);

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
*/
