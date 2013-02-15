#include "header.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*Modify to change the number of Monte Carlo Simulations*/
unsigned int MONTE_CARLO = 10000;

void generateCards(int *rand1, int *rand2, int *rand3, int *rand4, int *rand5);
int readSuit(int rand);
int readRank(int rand);
int determineHand(int hand[2][5]);

int main(int argc, char *argv[])
{
   int rand1, rand2, rand3, rand4, rand5;
   int hand[2][5]; /*Store suite and rank of each card*/

   /*Initialize player's cards*/
   generateCards(&rand1, &rand2, &rand3, &rand4, &rand5);
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

   determineHand(hand);
   return 0;
}

int determineHand(int hand[2][5])
{
   int i, j;

   printf("\nDetermining what you hold\n");

   /*Check for pair(s)*/
   for (j = 0; j < 5; j++)
   {      
      for (i = 0; i < 5; i++)
      {
         if (i == j)
         {
            printf("Skipping\n");
            continue;
         }

         if (hand[1][j] == hand[1][i])      
         {
            printf("FOUND A PAIR!\n\nCard1:\n");
            readRank(hand[1][j]);
            readSuit(hand[1][j]);
            printf("Card2:\n");
            readRank(hand[1][i]);
            readSuit(hand[1][i]);
         }
      }
   }
   return 0;
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
   *rand1 = rand() % (52);
   *rand2 = rand() % (52);
   *rand3 = rand() % (52);   
   *rand4 = rand() % (52);   
   *rand5 = rand() % (52);   
}
