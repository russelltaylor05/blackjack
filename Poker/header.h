/*Prevents multiple, redundant includes*/
#ifndef HEADER_H
#define HEADER_H

/*Macro to define how many cards[] make up a deck*/
#define ONE_DECK 4

/*Global to define how many cards are in a deck*/
unsigned const int DECK_SIZE = 52;

/*Unique cards in a deck*/
unsigned int cards[52] = {0,1,2,3,4,5,6,7,8,9,10,11,12,
                       13,14,15,16,17,18,19,20,21,22,23,24,25,
                       26,27,28,29,30,31,32,33,34,35,36,37,38,
                       39,40,41,42,43,44,45,46,47,48,49,50,51};

/*CPU Error Handling Macro*/
#define handle_error(msg) \
   do { perror(msg); exit(EXIT_FAILURE); } while (0)

/*Leave this at the end of the file!*/
#endif
