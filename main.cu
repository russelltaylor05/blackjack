#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include "lookuptable.h"
#include "poker.h"

static void HandleError( cudaError_t err, const char * file, int line)
{
  if(err !=cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



int main(int argc, char *argv[])
{

  int *dev_a, *a;
  
  a = (int *) malloc(sizeof(int));
  
  HANDLE_ERROR(cudaMalloc(&dev_a, sizeof(int)));

  analyze<<<1,10>>>(1);
  
  HANDLE_ERROR(cudaMemcpy(a, dev_a, sizeof(int), cudaMemcpyDeviceToHost));

  return 0;
}


__global__ void analyze(int i)
{
  int deck[52], hand[HAND_SIZE], staticHand[HAND_SIZE];
  int score, rank;
  int throwAwayCnt;
  int throwAwayCards[HAND_SIZE * 2];
  float results;
  int cardIndex = 0;

  init_deck(deck);

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

  print_hand(hand, HAND_SIZE);
  printf("\n");
}




// perform a binary search on a pre-sorted array
//
__device__ int findit( int key )
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
    printf( "ERROR:  no match found; key = %d\n", key );
    
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
__device__ int find_card( int rank, int suit, int *deck )
{
	int i, c;

	for ( i = 0; i < 52; i++ ) {
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
__device__ void shuffle_deck( int *deck )
{
    int i, n, temp[52];

    for ( i = 0; i < 52; i++ ) {
      temp[i] = deck[i];
    }

    for ( i = 0; i < 52; i++ ) {
        do {
            n = (int)(51.9999999 * 1); //drand48());
        } while ( temp[n] == 0 );
        deck[i] = temp[n];
        temp[n] = 0;
    }
}


__device__ void print_hand( int *hand, int n )
{
  int i, r;
  char suit;
  char *rank = "23456789TJQKA";

  for ( i = 0; i < n; i++ )  {
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

  /* let's do it the hard way  */
  q = (c1&0xFF) * (c2&0xFF) * (c3&0xFF) * (c4&0xFF) * (c5&0xFF);
  q = findit( q );

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

	for ( i = 0; i < 21; i++ ) {
		for ( j = 0; j < 5; j++ ) {
			subhand[j] = hand[ perm7[i][j] ];
		}
		q = eval_5hand( subhand );
		if ( q < best )
			best = q;
	}
	return( best );
}
