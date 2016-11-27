DATA_DIR=../../data
BIN_DIR=../irstlm-5.80.08/trunk/build/bin
SCR_DIR=../irstlm-5.80.08/trunk/scripts
OUT_DIR=../model

NGRAM=3

TEXT_DATA=$DATA_DIR/uni-mini.txt
INTERMIDIATE=$OUT_DIR/uni-mini-lm$NGRAM.gz
OUT_DATA=$OUT_DIR/uni-mini-lm$NGRAM.arpa

if [ ! -e $TEXT_DATA ]; then
    echo 'Training file does not exist, aborting...'
    exit
fi

echo -----------------------------------------------------------------------------------------------------
echo -- Training language model ...
time $SCR_DIR/build-lm.sh -i $TEXT_DATA -o $INTERMIDIATE -n $NGRAM -k 5 -p -u
echo -----------------------------------------------------------------------------------------------------
echo -- Transform into ARPA format ...
time $BIN_DIR/quantize-lm $INTERMIDIATE $OUT_DATA
rm -f $INTERMIDIATE
echo -- Job finished...
