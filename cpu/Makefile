CC = gcc

SOURCES = poker_analyze.c

CFLAGS = -Wall -g

all: analyze throw
	
analyze: poker_analyze.c pokerlib.o
	${CC} ${CFLAGS} poker_analyze.c pokerlib.o -o cpu_analyze	

throw: poker_throw.c pokerlib.o
	${CC} ${CFLAGS} poker_throw.c pokerlib.o -o cpu_throw	

pokerlib.o: pokerlib.c lookuptable.h
	${CC} -c ${CFLAGS} pokerlib.c -o pokerlib.o

run: poker
	@./poker

clean:
	rm -rf *.o poker main strategy cpu_analyze cpu_throw
