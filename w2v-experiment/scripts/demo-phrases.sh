#!/bin/bash

DATA_DIR=../../data/content-category 
BIN_DIR=../bin
SRC_DIR=../src
# OUT_DIR=../model


TEXT_DATA=$DATA_DIR/makeup.txt
PHRASES_DATA=$DATA_DIR/makeup-phrase.txt

pushd ${SRC_DIR} && make; popd

if [ ! -e $TEXT_DATA ]; then
  echo 'Training file does not exist, aborting...'
  exit
fi

echo -----------------------------------------------------------------------------------------------------
echo -- Creating phrases...
time $BIN_DIR/word2phrase -train $TEXT_DATA -output $PHRASES_DATA -threshold 200 -debug 2
