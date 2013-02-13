/*Prevents multiple, redundant includes*/
#ifndef HEADER_H
#define HEADER_H

/*Macro to define how many cards[] make up a deck*/
#define ONE_DECK 4

/*Global to define how many cards are in a deck*/
unsigned const int DECK_SIZE = 52;

/*Unique cards in a deck*/
unsigned int cards[13] = {2,3,4,5,6,7,8,9,10,10,10,10,11};

/*CPU Error Handling Macro*/
#define handle_error(msg) \
   do { perror(msg); exit(EXIT_FAILURE); } while (0)

/*Leave this at the end of the file!*/
#endif
