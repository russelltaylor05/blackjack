#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <time.h>
#include <ctype.h>
#include <unistd.h>
#include "cpu_lookuptable.h"
#include "cpu_poker.h"




/* Returns: %chance that hand will win */ 
float analyzeHand(int *hand, int *deck, int *exclude, int excludeSize)
{
  int resolution = ANALYZE_RESOLUTION_CPU;
  int tempHand[HAND_SIZE_CPU];
  int handScore, tempScore, i;
  int wins = 0;

  handScore = eval_5hand_cpu(hand);  
  for(i = 0; i < resolution; i++) {
    setRandomHand_cpu(deck, tempHand, hand, HAND_SIZE_CPU);
    tempScore = eval_5hand_cpu(tempHand);
    if(handScore < tempScore) {
      wins++;
    }
  }  
  return (float)wins / (float)resolution * 100.00;
}

int getArgs(ARGSP *argsp, int argc, char *argv[])
{
  
  int c, option_index = 0;

  static struct option long_options[] =
  {
    /* These options set a flag. */
    {"c1",  required_argument, 0, 'a'},
    {"c2",  required_argument, 0, 'b'},
    {"c3",  required_argument, 0, 'c'},
    {"c4",  required_argument, 0, 'd'},
    {"c5",  required_argument, 0, 'e'},
    {"t1",  required_argument, 0, 'f'},
    {"t2",  required_argument, 0, 'g'},
    {"t3",  required_argument, 0, 'h'},
    {"t4",  required_argument, 0, 'i'},
    {"t5",  required_argument, 0, 'j'},

    {0, 0, 0, 0}
  };  

  
  while ((c = getopt_long(argc, argv, "a:b:c", long_options, &option_index)) != -1) {
    switch (c)
    {
      case 'a':
        argsp->c1Flag = 1;
        argsp->c1 = optarg;
        break;
      case 'b':
        argsp->c2Flag = 1;
        argsp->c2 = optarg;
        break;
      case 'c':
        argsp->c3Flag = 1;
        argsp->c3 = optarg;
        break;
      case 'd':
        argsp->c4Flag = 1;
        argsp->c4 = optarg;
        break;
      case 'e':
        argsp->c5Flag = 1;
        argsp->c5 = optarg;
        break;
      case 'f':
        argsp->t1Flag = 1;
        argsp->t1 = optarg;
        break;
      case 'g':
        argsp->t2Flag = 1;
        argsp->t2 = optarg;
        break;
      case 'h':
        argsp->t3Flag = 1;
        argsp->t3 = optarg;
        break;
      case 'i':
        argsp->t4Flag = 1;
        argsp->t4 = optarg;
        break;
      case 'j':
        argsp->t5Flag = 1;
        argsp->t5 = optarg;
        break;

    }
  }

  if(argsp->c1Flag 
      && argsp->c2Flag 
      && argsp->c3Flag
      && argsp->c4Flag
      && argsp->c5Flag){
    return 1;    
  } else {
    return -1;  
  }
      
}  



/* Picks 5 random cards and sets them in *hand
 * excludedCards will be excluded from random hand
 */
void setRandomHand_cpu(int *deck, int *hand, int *excludedCards, int excludeCnt) 
{
  int i;  
  int excludedCardsTemp[HAND_SIZE_CPU * 2];

  /* Copy exclude cards to new temp array */
  copyHand_cpu(excludedCardsTemp, excludedCards, excludeCnt);
  
  /* Every time we get a new random card, add
   * it to the excludedCardsTemp array so that it won't
   * get choosen again.
   */
  for(i = 0; i < HAND_SIZE_CPU; i++) {
    hand[i] = getRandomCard_cpu(deck, excludedCardsTemp, excludeCnt);
    excludedCardsTemp[excludeCnt] = hand[i];
    excludeCnt++;    
  }
}


/* Updates a hand's cards specified by throwAwayCards[]
 */
void updateHand_cpu(int *deck, int *hand, int *throwAwayCards, int throwAwayCnt)
{
  int index, i = 0;
  int excludeCnt = HAND_SIZE_CPU;
  int excludedCards[HAND_SIZE_CPU * 2]; // large enough for a 5 card hand plus 5 cards to throw away
      
  /* Copy hand into excludeCards array */
  for(i = 0; i < HAND_SIZE_CPU; i++)
    excludedCards[i] = hand[i];
  
  /* For each throw away card, choose a new random card
   * that is not in our excluded cards array.
   * Once a new card is choosen, add it the end of the 
   * excluded cards array
   */
  for(i = 0; i < throwAwayCnt; i++) {    
    index = findCardIndex_cpu(hand, throwAwayCards[i], HAND_SIZE_CPU);
    hand[index] = getRandomCard_cpu(deck, excludedCards, excludeCnt);
    excludedCards[excludeCnt] = hand[index];
    excludeCnt++;
  }
}


/* Returns random card VALUE that is not in exclude array */
int getRandomCard_cpu(int *deck, int *exclude, int excludeSize) 
{
  int i = 0;  
  shuffle_deck_cpu(deck);
  while(inArray_cpu(deck[i], exclude, excludeSize )) {
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
  
}


/* copies source hand to destination hand */
void copyHand_cpu (int *dest, int *source, int handSize) {
  int i;    
  for(i = 0; i < handSize; i++) {
    dest[i] = source[i];
  }  
}


/* Return 1 if value is in array
 * Return 0 if value is not in array
 */
int inArray_cpu(int value, int *array, int size) 
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
int findCardIndex_cpu(int *hand, int cardValue, int handSize) 
{ 
  int i;
  for(i = 0; i < handSize; i++) {
    if (hand[i] == cardValue) {
      return i;
    }    
  }
  return -1;
}



// perform a binary search on a pre-sorted array
//
int findit_cpu( int key )
{
    int low = 0, high = 4887, mid;

    while ( low <= high )
    {
        mid = (high+low) >> 1;      // divide by two
        if ( key < products_cpu[mid] )
            high = mid - 1;
        else if ( key > products_cpu[mid] )
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
void init_deck_cpu( int *deck )
{
    int i, j, n = 0, suit = 0x8000;

    for ( i = 0; i < 4; i++, suit >>= 1 )
        for ( j = 0; j < 13; j++, n++ )
            deck[n] = primes_cpu[j] | (j << 8) | suit | (1 << (16+j));
}




//  This routine will search a deck for a specific card
//  (specified by rank/suit), and return the INDEX giving
//  the position of the found card.  If it is not found,
//  then it returns -1
//
int
find_card_cpu( int rank, int suit, int *deck )
{
	int i, c;

	for ( i = 0; i < 52; i++ )
	{
		c = deck[i];
		if ( (c & suit)  &&  (RANK_CPU(c) == rank) )
			return( i );
	}
	return( -1 );
}


//
//  This routine takes a deck and randomly mixes up
//  the order of the cards.
//
void shuffle_deck_cpu( int *deck )
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


void print_hand_cpu( int *hand, int n )
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

void print_card_cpu(int card) 
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


int hand_rank_cpu( short val )
{
    if (val > 6185) return(HIGH_CARD_CPU);        // 1277 high card
    if (val > 3325) return(ONE_PAIR_CPU);         // 2860 one pair
    if (val > 2467) return(TWO_PAIR_CPU);         //  858 two pair
    if (val > 1609) return(THREE_OF_A_KIND_CPU);  //  858 three-kind
    if (val > 1599) return(STRAIGHT_CPU);         //   10 straights
    if (val > 322)  return(FLUSH_CPU);            // 1277 flushes
    if (val > 166)  return(FULL_HOUSE_CPU);       //  156 full house
    if (val > 10)   return(FOUR_OF_A_KIND_CPU);   //  156 four-kind
    return(STRAIGHT_FLUSH_CPU);                   //   10 straight-flushes
}


short eval_5cards_cpu( int c1, int c2, int c3, int c4, int c5 )
{
    int q;
    short s;

    q = (c1|c2|c3|c4|c5) >> 16;

    /* check for Flushes and StraightFlushes
    */
    if ( c1 & c2 & c3 & c4 & c5 & 0xF000 )
	return( flushes_cpu[q] );

    /* check for Straights and HighCard hands
    */
    s = unique5_cpu[q];
    if ( s )  return ( s );

    /* let's do it the hard way
    */
    q = (c1&0xFF) * (c2&0xFF) * (c3&0xFF) * (c4&0xFF) * (c5&0xFF);
    q = findit_cpu( q );

    return( values_cpu[q] );
}


short eval_5hand_cpu( int *hand )
{
    int c1, c2, c3, c4, c5;

    c1 = *hand++;
    c2 = *hand++;
    c3 = *hand++;
    c4 = *hand++;
    c5 = *hand;

    return( eval_5cards_cpu(c1,c2,c3,c4,c5) );
}

/**********************/
/**  DFEBUG          **/
/**********************/



void print_bits(int number ){
   unsigned long mask = 1 << 30;
   int cnt = 1;
   while(mask){
      (mask & number) ? printf("X") : printf(".");
      mask = mask >> 1 ; 
      if(!(cnt % 8)){
         putchar('|');
      }
      cnt++;
   }
   putchar('\n');
}

void setThrowFromArgs(int *deck, int *throwAway, int *throwCnt, ARGSP *argsp)
{
  int index = 0;
  int cnt = 0;

  if(argsp->t1Flag) {
    index = parseCard(argsp->t1, deck);
    throwAway[cnt++] = deck[index];
  }
  if(argsp->t2Flag) {
    index = parseCard(argsp->t2, deck);
    throwAway[cnt++] = deck[index];
  }
  if(argsp->t3Flag) {
    index = parseCard(argsp->t3, deck);
    throwAway[cnt++] = deck[index];
  }
  if(argsp->t4Flag) {
    index = parseCard(argsp->t4, deck);
    throwAway[cnt++] = deck[index];
  }
  if(argsp->t5Flag) {
    index = parseCard(argsp->t5, deck);
    throwAway[cnt++] = deck[index];
  }
  
  *throwCnt = cnt;
}

void setHandFromArgs(int *deck, int *hand, ARGSP *argsp) 
{
  int index = -1;

  if(index = parseCard(argsp->c1, deck)) {
    hand[0] = deck[index];
  } else { printf("Set Hand Error\n");}

  if(index = parseCard(argsp->c2, deck)) {
    hand[1] = deck[index];
  } else { printf("Set Hand Error\n");}
  
  index = parseCard(argsp->c3, deck);
  hand[2] = deck[index];
  index = parseCard(argsp->c4, deck);
  hand[3] = deck[index];
  index = parseCard(argsp->c5, deck);
  hand[4] = deck[index];  
  
}

/* Returns index of the Argument card string */
int parseCard(char *str, int *deck) 
{
  int rank, suit;

  //printf("%c%c, ", str[0], str[1]);
  switch (str[0])
  {
    case '2':
      rank = Deuce_CPU;
      break;
    case '3':
      rank = Trey_CPU;
      break;
    case '4':
      rank = Four_CPU;
      break;
    case '5':
      rank = Five_CPU;
      break;
    case '6':
      rank = Six_CPU;
      break;
    case '7':
      rank = Seven_CPU;
      break;
    case '8':
      rank = Eight_CPU;
      break;
    case '9':
      rank = Nine_CPU;
      break;
    case 'T':
      rank = Ten_CPU;
      break;
    case 'J':
      rank = Jack_CPU;
      break;
    case 'Q':
      rank = Queen_CPU;
      break;
    case 'K':
      rank = King_CPU;
      break;
    case 'A':
      rank = Ace_CPU;
      break;
    default:
      rank = -1;
      break;
  }
  switch (str[1])
  {
    case 'd':
      suit = DIAMOND_CPU;
      break;
    case 'h':
      suit = HEART_CPU;
      break;
    case 's':
      suit = SPADE_CPU;
      break;
    case 'c':
      suit = CLUB_CPU;
      break;
    default:
      suit = -1;
      break;
  }
  
  if(suit >= 0 && rank >= 0) {
    return find_card_cpu(rank, suit, deck);
  } else  {
    return 0;
  }
}
