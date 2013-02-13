#include "header.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*Modify to change the number of decks being played with*/
unsigned int NUM_DECKS = 1;

void generateCards(int *rand1, int *rand2, int *rand3);
int determine_strategy(unsigned int *all_cards, int pCard1, int pCard2,
      int dCard);

int main(int argc, char *argv[])
{
   unsigned int *all_cards;
   unsigned int i = 0;
   int rand1, rand2, rand3;

   /*Initialize player's cards*/
   generateCards(&rand1, &rand2, &rand3);
   printf("Rands are %i, %i and %i\n", rand1, rand2, rand3);

   /*Malloc all of the cards to deal from*/
   if (NULL == (all_cards = malloc(NUM_DECKS*ONE_DECK*sizeof(cards))))
   {
      handle_error("malloc");
   }

   /*Initialize all of the cards to play with*/
   for (i=0; i<52*NUM_DECKS; i++)
   {
      all_cards[i] = cards[i%13];
   }

   printf("Dealing two cards...\n");

   printf("CARDS: %i %i, sum: %i\n", all_cards[rand1], all_cards[rand2],
         all_cards[rand1] + all_cards[rand2]);

   printf("DEALER: %i (%i)\n", all_cards[rand3], all_cards[rand3]+10);

   if (determine_strategy(all_cards, rand1, rand2, rand3))
   {
      printf("You should hit\n");
   }
   else
   {
      printf("You should stand\n");
   }

   return 0;
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

void generateCards(int *rand1, int *rand2, int *rand3)
{
   srand((int) time(NULL));
   *rand1 = rand() % (52*NUM_DECKS);
   *rand2 = rand() % (52*NUM_DECKS);
   *rand3 = rand() % (52*NUM_DECKS);   
}
