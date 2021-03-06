#ifndef CPU_POKER
#define CPU_POKER

#define	STRAIGHT_FLUSH	1
#define	FOUR_OF_A_KIND	2
#define	FULL_HOUSE	3
#define	FLUSH		4
#define	STRAIGHT	5
#define	THREE_OF_A_KIND	6
#define	TWO_PAIR	7
#define	ONE_PAIR	8
#define	HIGH_CARD	9

#define HAND_SIZE 5
#define ANALYZE_RESOLUTION 4096
#define THROWAWAY_RESOLUTION 4096

#define	RANK(x)		((x >> 8) & 0xF)

#define CLUB	0x8000
#define DIAMOND 0x4000
#define HEART   0x2000
#define SPADE   0x1000

#define Deuce	0
#define Trey	1
#define Four	2
#define Five	3
#define Six	4
#define Seven	5
#define Eight	6
#define Nine	7
#define Ten	8
#define Jack	9
#define Queen	10
#define King	11
#define Ace	12

static char *value_str[] = {
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



void    srand48();
double  drand48();

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

int findit( int key );
void init_deck( int *deck );
int find_card( int rank, int suit, int *deck );
void shuffle_deck( int *deck );
void print_hand( int *hand, int n );
void print_card( int card );
int hand_rank( short val );
short eval_5cards( int c1, int c2, int c3, int c4, int c5 );
short eval_5hand( int *hand );
short eval_7hand( int *hand );

int getArgs(ARGSP *argsp, int argc, char *argv[]);
void freeArgs(ARGSP *argsp);
void setHandFromArgs(int *deck, int *hand, ARGSP *argsp);
void setThrowFromArgs(int *deck, int *throwAway, int *throwCnt, ARGSP *argsp);
int parseCard(char *str, int *deck);

void setStaticHand (int *deck, int *hand);
void setRandomHand (int *deck, int *hand, int *excludedCards, int excludeCnt) ; 
void updateHand (int *deck, int *hand, int *throwAwayCards, int throwAwayCnt);
int inArray (int value, int *array, int size);
void printRankTable (int *deck);
int findCardIndex (int *hand, int cardValue, int handSize);
int getRandomCard(int *deck, int *exclude, int excludeSize);
void copyHand (int *hand1, int *hand2, int handSize);
float analyzeHand(int *hand, int *deck, int *exclude, int excludeSize);
float analyzeThrowAway(int *hand, int *deck, int *throwAwayCards, int throwAwayCnt);
float analyzePrediction(int *hand, int *deck, int *bestThrowAway, int *bestThrowAwaySize);


#endif