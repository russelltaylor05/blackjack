#include "header.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*Modify to change the number of Monte Carlo Simulations*/
unsigned int MONTE_CARLO = 10000;

void generateCards(int *rand1, int *rand2, int *rand3, int *rand4, int *rand5);
int readSuit(int rand);
int readRank(int rand);
void initializeHand(int hand[2][5], int rand1, int rand2, int rand3,
     int rand4, int rand5);
int scoreHand(int hand[2][5]);
int checkRoyalFlush(int hand[2][5]);
int checkStraightFlush(int hand[2][5]);
int checkFourKind(int hand[2][5]);
int checkFullHouse(int hand[2][5]);
int checkFlush(int hand[2][5]);
int checkStraight(int hand[2][5]);
int checkThreeKind(int hand[2][5]);
int checkTwoPair(int hand[2][5]);
int checkOnePair(int hand[2][5]);
int checkHighCard(int hand[2][5]);

int main(int argc, char *argv[])
{
   /*Cards for the player's hand, randomly generated*/
   int rand1, rand2, rand3, rand4, rand5;

   /*Stores suit first, then rank of each card in hand*/
   int hand[2][5];

   /*Generate and store the player's cards*/
   generateCards(&rand1, &rand2, &rand3, &rand4, &rand5);
   initializeHand(hand, rand1, rand2, rand3, rand4, rand5);

   /*TODO:Score the player's hand*/
   scoreHand(hand);

   return 0;
}

int checkRoyalFlush(int hand[2][5])
{
   /*Determine if hand contains a Straight Flush of A,K,Q,J,10*/
   if (checkStraightFlush(hand))
   {
      /*TODO:Check that cards are exactly A,K,Q,J,10*/
      /*If they are, return 1 here*/
      /*return 1;*/
   }
   return 0;
}

int checkStraightFlush(int hand[2][5])
{
   /*Determine if hand contains a straight AND a flush*/
   if (checkFlush(hand) && checkStraight(hand))
      return 1;

   return 0;
}

int checkFourKind(int hand[2][5])
{
   /*TODO:Determine if hand contains four cards of the same rank*/
   return 0;
}

int checkFullHouse(int hand[2][5])
{
   /*TODO:Determine if hand contains three of a kind AND one pair*/
   return 0;
}

int checkFlush(int hand[2][5])
{
   /*TODO:Determine if hand contains five cards with the same suit*/
   return 0;
}

int checkStraight(int hand[2][5])
{
   /*TODO:Determine if hand contains three sequentially ranked cards*/
   return 0;
}

int checkThreeKind(int hand[2][5])
{
   /*TODO:Determine if hand contains three cards of the same rank*/
   return 0;
}

int checkTwoPair(int hand[2][5])
{
   /*TODO:Determine if hand contains two sets of two cards of the same rank*/
   return 0;
}

int checkOnePair(int hand[2][5])
{
   /*Determine if hand contains two cards of the same rank*/
   int i=0, j=0;

   for (j=0; j<5; j++)
   {
      for (i=0;i<5;i++)
      {
         if (i == j)
         {
            /*Don't check the card against itself*/
            continue;
         }

         /*See if the two cards have the same rank*/
         if (hand[1][i] == hand[1][j])
         {
            printf("FOUND A PAIR!\n");
            return 1;
         }
      }
   }

   return 0;
}

int checkHighCard(int hand[2][5])
{
   /*This function has no purpose right now...*/
   return 1;
}

int scoreHand(int hand[2][5])
{
   if (1 == checkRoyalFlush(hand))
      return 10;
   if (1 == checkStraightFlush(hand))
      return 9;
   if (1 == checkFourKind(hand))
      return 8;
   if (1 == checkFullHouse(hand))
      return 7;
   if (1 == checkFlush(hand))
      return 6;
   if (1 == checkStraight(hand))
      return 5;
   if (1 == checkThreeKind(hand))
      return 4;
   if (1 == checkTwoPair(hand))
      return 3;
   if (1 == checkOnePair(hand))
      return 2;
   if (1 == checkHighCard(hand))
      return 1;

   printf("Unable to score hand\n");
   {
      exit(EXIT_FAILURE);
   }
   return -1;
}

void initializeHand(int hand[2][5], int rand1, int rand2, int rand3,
     int rand4, int rand5)
{
   hand[1][0] = readRank(rand1);
   hand[0][0] = readSuit(rand1);
   hand[1][1] = readRank(rand2);
   hand[0][1] = readSuit(rand2);
   hand[1][2] = readRank(rand3);
   hand[0][2] = readSuit(rand3);
   hand[1][3] = readRank(rand4);
   hand[0][3] = readSuit(rand4);
   hand[1][4] = readRank(rand5);
   hand[0][4] = readSuit(rand5);
}

int readRank(int rand)
{
   int rank = 1 + (rand % 13);

   switch (rank)
   {
   case 11:
      printf("Jack ");
      break;
   case 12:
      printf("Queen ");
      break;
   case 13:
      printf("King ");
      break;
   case 1:
      printf("Ace ");
      break;
   default:
      printf("%i ", rank);
      break;
   }
   return rank;
}

int readSuit(int rand)
{
   switch (rand/13)
   {
   case 0:
      {
         printf("of Spades\n");
         return 0;
      }
   case 1:
      {
         printf("of Diamonds\n");
         return 1;
      }
   case 2:
      {
         printf("of Clubs\n");
         return 2;
      }
   case 3:
      {
         printf("of Hearts\n");
         return 3;
      }
   default:
      {
         printf("Suit Unknown\n");
         exit(EXIT_FAILURE);
      }
   }
}

void generateCards(int *rand1, int *rand2, int *rand3, int *rand4, int *rand5)
{
   srand((int) time(NULL));
   /*TODO:Need logic to ensure that same card cannot be dealt twice...*/
   *rand1 = rand() % (52);
   *rand2 = rand() % (52);
   *rand3 = rand() % (52);   
   *rand4 = rand() % (52);   
   *rand5 = rand() % (52);   
}
