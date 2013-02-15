#include "header.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*Modify to change the number of decks being played with*/
unsigned int NUM_DECKS = 1;

void generateCards(int *rand1, int *rand2, int *rand3, int *rand4, int *rand5);
int readSuit(int rand);
int readRank(int rand);
int determine_strategy(unsigned int *all_cards, int pCard1, int pCard2,
      int dCard);

int main(int argc, char *argv[])
{
   int rand1, rand2, rand3, rand4, rand5;

   /*Initialize player's cards*/
   generateCards(&rand1, &rand2, &rand3, &rand4, &rand5);
   readSuit(rand1);
   readRank(rand1);
   readSuit(rand2);
   readRank(rand2);
   readSuit(rand3);
   readRank(rand3);
   readSuit(rand4);
   readRank(rand4);
   readSuit(rand5);
   readRank(rand5);
   return 0;
}

int readRank(int rand)
{
   int rank = rand % 13;

   switch (rank)
   {
   case 11:
      printf("Jack\n");
      break;
   case 12:
      printf("Queen\n");
      break;
   case 13:
      printf("King\n");
      break;
   case 1:
      printf("Ace\n");
      break;
   default:
      printf("%i\n", rank);
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
         printf("Suit is Spade\n");
         return 0;
      }

   case 1:
      {
         printf("Suit is Diamonds\n");
         return 1;
      }

   case 2:
      {
         printf("Suit is Clubs\n");
         return 2;
      }

   case 3:
      {
         printf("Suit is Hearts\n");
         return 3;
      }

   default:
      {
         printf("Suit Unknown\n");
         exit(EXIT_FAILURE);
      }
   }
}

/*THIS FUNCTION IS WHERE THE CUDA MAGIC WILL HAPPEN!*/
/*Function returns 1 for "hit" and 0 for "stand"*/
int determine_strategy(unsigned int *all_cards, int pCard1, int pCard2,
      int dCard)
{
   int i = 0;
   int hit = -2; /*Start at -3 to offset for known cards (0s in deck)*/
   int stand = 0;
   int pSum = all_cards[pCard1] + all_cards[pCard2];

   /*Always hit below 12*/
/*   if (pSum < 12)
   {
      return 1;
   }*/

   /*Always stand above 16*/
/*   if (pSum > 16)
   {
      return 0;
   }*/

   /*TODO: Incorporate better dealer logic, this code won't execute...EVER*/
   if (all_cards[dCard] + 10 > 21)
   {
      printf("Dealer is showing %i and is likely to bust...you should Stand\n",
            all_cards[dCard] + 10);
      return 0;
   }

   /*Zero all known cards so they won't be used*/
   all_cards[pCard1] = 0;
   all_cards[pCard2] = 0;
   all_cards[dCard] = 0;

   for (i=0; i<52*NUM_DECKS; i++)
   {
      if (pSum + all_cards[i] > 21)
      {
         /*Check for Ace*/
         if (all_cards[i] == 11)
         {
            /*If Ace, treat value as 1 instead*/
            if (pSum + all_cards[i] > 31)
            {
               /*Player was already at 21*/
               printf("THIS SHOULD NEVER HAPPEN!\n");
               stand++;
            }
            else
            {
               /*Should occur frequently, means 12<pSum<20*/
               hit++;
            }
         }
         else
         {
            /*Player will bust with this card!*/
            stand++;
         }
      }
      else
      {
         /*Player won't bust with this extra card*/
         hit++;
      }
   }

   printf("Final counts, hit: %i, stand: %i\n", hit, stand);

   /*If there is a good chance the player won't bust, tell them to hit!*/
   if (hit > stand)
   {
      return 1;
   }

   /*Otherwise tell them to stand*/
   return 0;
}

void generateCards(int *rand1, int *rand2, int *rand3, int *rand4, int *rand5)
{
   srand((int) time(NULL));
   *rand1 = rand() % (52*NUM_DECKS);
   *rand2 = rand() % (52*NUM_DECKS);
   *rand3 = rand() % (52*NUM_DECKS);   
   *rand4 = rand() % (52*NUM_DECKS);   
   *rand5 = rand() % (52*NUM_DECKS);   
}
