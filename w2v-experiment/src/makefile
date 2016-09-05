
SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

all: word2vec word2phrase

word2vec: word2vec.c
	$(CC) word2vec.c -o ${BIN_DIR}/word2vec $(CFLAGS)

word2phrase : word2phrase.c
	$(CC) word2phrase.c -o ${BIN_DIR}/word2phrase $(CFLAGS)


clean: SHELL := /bin/bash
clean:
	pushd ${BIN_DIR} && rm -rf word2vec word2phrase; popd
