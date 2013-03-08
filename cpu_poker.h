#ifndef CPU_POKER
#define CPU_POKER

#define	STRAIGHT_FLUSH_CPU	1
#define	FOUR_OF_A_KIND_CPU	2
#define	FULL_HOUSE_CPU	3
#define	FLUSH_CPU		4
#define	STRAIGHT_CPU	5
#define	THREE_OF_A_KIND_CPU	6
#define	TWO_PAIR_CPU	7
#define	ONE_PAIR_CPU	8
#define	HIGH_CARD_CPU	9

#define HAND_SIZE_CPU 5

#define	RANK_CPU(x)		((x >> 8) & 0xF)

static char *value_str_cpu[] = {
	"",
	"Straight Flush",
	"Four of a Kind",
	"Full House",
	"Flush",
	"Straight",
	"Three of a Kind",
	"Two Pair",
	"One Pair",
	"High Card"
};

#define CLUB_CPU	0x8000
#define DIAMOND_CPU 0x4000
#define HEART_CPU   0x2000
#define SPADE_CPU   0x1000

#define Deuce_CPU	0
#define Trey_CPU	1
#define Four_CPU	2
#define Five_CPU	3
#define Six_CPU	  4
#define Seven_CPU	5
#define Eight_CPU	6
#define Nine_CPU	7
#define Ten_CPU	  8
#define Jack_CPU	9
#define Queen_CPU	10
#define King_CPU	11
#define Ace_CPU	  12


typedef struct argsp
{
  int c1Flag;
  int c2Flag;  
  int c3Flag;
  int c4Flag;
  int c5Flag;  
  char *c1;
  char *c2;
  char *c3;
  char *c4;
  char *c5; 
  int t1Flag;
  int t2Flag;  
  int t3Flag;
  int t4Flag;
  int t5Flag;  
  char *t1;
  char *t2;
  char *t3;
  char *t4;
  char *t5;  

} ARGSP;

int getArgs(ARGSP *argsp, int argc, char *argv[]);
void freeArgs(ARGSP *argsp);

void print_bits(int number);

void setHandFromArgs(int *deck, int *hand, ARGSP *argsp);
int parseCard(char *str, int *deck);

void setStaticHand_cpu (int *deck, int *hand);
void setRandomHand_cpu (int *deck, int *hand, int *excludedCards, int excludeCnt); 
void updateHand_cpu (int *deck, int *hand, int *throwAwayCards, int throwAwayCnt);
int inArray_cpu (int value, int *array, int size);
void printRankTable_cpu (int *deck);
int findCardIndex_cpu (int *hand, int cardValue, int handSize);
int getRandomCard_cpu(int *deck, int *exclude, int excludeSize);
void copyHand_cpu (int *hand1, int *hand2, int handSize);
void printHandStats_cpu(int *hand);

int findit_cpu( int key );
void init_deck_cpu( int *deck );
int find_card_cpu( int rank, int suit, int *deck );
void shuffle_deck_cpu( int *deck);
void print_hand_cpu( int *hand, int n );
void print_card_cpu( int card );
int hand_rank( short val );
short eval_5cards_cpu( int c1, int c2, int c3, int c4, int c5 );
short eval_5hand_cpu( int *hand );

int hand_rank_cpu( short val );


#endif
