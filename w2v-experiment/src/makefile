
SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

all: word2vec

word2vec: word2vec.c
	$(CC) word2vec.c -o ${BIN_DIR}/word2vec $(CFLAGS)


clean: SHELL := /bin/bash
clean:
	pushd ${BIN_DIR} && rm -rf word2vec; popd
