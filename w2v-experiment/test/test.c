#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char **argv) {
	FILE *fin;
	float *weight;
	char *word;
	char c, c1 = 0, c2 = 0;

	long long vocab_size, layer1_size, a, b, received_vocab = 0;
	
	// fin = fopen("../model/tokenized-zh-seg-mini-vector.bin", "rb");
	fin = fopen("../model/test.bin", "rb");
	if(fin == NULL) {
		printf("ERROR: debug file not found!\n");
		exit(1);
	}
	fscanf(fin, "%lld %lld\n", &vocab_size, &layer1_size);
	fprintf(stderr, "%lld %lld\n", vocab_size, layer1_size);
	weight = (float *)calloc(layer1_size, sizeof(float));
	word = (char *)calloc(100, sizeof(char));

	while (!feof(fin)) {
		fscanf(fin, "%s%c", word, &c);
		if(feof(fin)) break;
		fprintf(stderr, "%s, %d", word, (int)c);
		fread(weight, sizeof(float), layer1_size, fin);
		received_vocab++;
    	// for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    	// len = 0;
    	// for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    	// len = sqrt(len);
    	// for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
  	fclose(fin);

	// while (!feof(fin)) {
	// 	a = 0;
	// 	while (!feof(fin)) {
	// 		word[a] = fgetc(fin);
	// 		if (word[a] == 32 || a >= 100 - 1) break;
	// 		a++;
	// 	}
	// 	if(feof(fin)) break;
	// 	word[a] = 0;
	// 	fprintf(stderr, "%s", word);
	// 	fread(weight, sizeof(float), layer1_size, fin);
	// 	received_vocab++;
	// }

	fprintf(stderr, "\nTotal %lld vocab received\n", received_vocab);
	free(weight);
	return 0;
}