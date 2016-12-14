DATA_DIR=../../data
BIN_DIR=../kenlm/build/bin
OUT_DIR=../model

ORDER=5
MEM_LIMIT=80%

TEXT_DATA=$DATA_DIR/gf-zh.txt
OUT_DATA=$OUT_DIR/gf-lm$ORDER.arpa

if [ ! -e $TEXT_DATA ]; then
    echo 'Training file does not exist, aborting...'
    exit
fi

echo -----------------------------------------------------------------------------------------------------
echo -- Training vectors...
time $BIN_DIR/lmplz -o $ORDER -S $MEM_LIMIT <$TEXT_DATA >$OUT_DATA
echo -----------------------------------------------------------------------------------------------------
echo -- Training finished...
