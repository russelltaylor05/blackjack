CC = gcc -O2 

SOURCES = strategy.c

CFLAGS = -Wall -g -pedantic -ansi

all: strategy

strategy: $(SOURCES)
	$(CC) $(CFLAGS) -o strategy $(SOURCES)
	./strategy

clean:
	rm strategy
