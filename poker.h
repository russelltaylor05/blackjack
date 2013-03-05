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
#define ANALYZE_RESOLUTION 10000
#define THROWAWAY_RESOLUTION 10


#define	RANK(x)		((x >> 8) & 0xF)

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

__global__ void analyzeHand(int *hand, int *deck, int *exclude, int excludeSize, float *devAnalyzeResults);
__global__ void analyzeThrowAway(int *hand, int *deck, int *throwAwayCards, int throwAwayCnt);


void setStaticHand (int *deck, int *hand);
__device__ void setRandomHand (int *deck, int *hand, int *excludedCards, int excludeCnt) ; 
__device__ void updateHand (int *deck, int *hand, int *throwAwayCards, int throwAwayCnt);
__device__ int inArray (int value, int *array, int size);
__device__ void printRankTable (int *deck);
__device__ int findCardIndex (int *hand, int cardValue, int handSize);
__device__ int getRandomCard(int *deck, int *exclude, int excludeSize);
__device__ void copyHand (int *hand1, int *hand2, int handSize);

__device__ int findit( int key );
__host__ __device__ void init_deck( int *deck );
__host__ __device__ int find_card( int rank, int suit, int *deck );
__device__ void shuffle_deck( int *deck );
__host__ __device__ void print_hand( int *hand, int n );
__host__ __device__ void print_card( int card );
__device__ int hand_rank( short val );
__device__ short eval_5cards( int c1, int c2, int c3, int c4, int c5 );
__device__ short eval_5hand( int *hand );
__device__ short eval_7hand( int *hand );


