CC      := gcc
CFLAGS  := -O2 -Wall -Wextra -std=c99 -I include -I tests
LDFLAGS := -lm

SRC     := src/mfcc.c src/mfcc_tables.c tests/test_mfcc_host.c
BIN     := tests/test_mfcc_host

.PHONY: all test clean

all: $(BIN)

$(BIN): $(SRC) include/mfcc.h tests/test_vectors.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

tests/test_vectors.h tests/test_vectors.npz src/mfcc_tables.c: generate_mfcc_tables.py
	python3 generate_mfcc_tables.py

test: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)
