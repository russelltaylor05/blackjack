CC = gcc

SOURCES = poker.c

CFLAGS = -Wall -g

all: poker
	
poker: poker.c pokerlib.o
	${CC} ${CFLAGS} poker.c pokerlib.o -s -o poker	

pokerlib.o: pokerlib.c lookuptable.h
	${CC} -c ${CFLAGS} pokerlib.c -o pokerlib.o

run: poker
	@./poker

clean:
	rm -rf *.o poker main strategy
