#!/bin/bash

DATA_DIR=../../data
BIN_DIR=../bin
SRC_DIR=../src
OUT_DIR=../model

VOCAB_FILE=$DATA_DIR/vocab.jsonl
TEXT_DATA=$DATA_DIR/jb-zh-combine.txt
VECTOR_DATA=$OUT_DIR/syn0.bin

pushd ${SRC_DIR} && make; popd

# if [ ! -e $VECTOR_DATA ]; then
  
  if [ ! -e $TEXT_DATA ]; then
    echo 'Training file does not exist, aborting...'
    exit
  fi
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/word2vec -save-vocab $VOCAB_FILE -train $TEXT_DATA -output $VECTOR_DATA -cbow 1 -size 100 -iteration 5 -window 10 -negative 50 -hs 0 -sample 1e-3 -threads 12 -binary 1
  
# fi

echo -----------------------------------------------------------------------------------------------------
echo -- Training finished...
