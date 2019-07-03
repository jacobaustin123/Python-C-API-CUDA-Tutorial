#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

extern double * myVectorAdd(double * h_A, double * h_B, int numElements);

int main() {
    double * array1 = (double *) malloc(sizeof(double) * 200);
    double * array2 = (double *) malloc(sizeof(double) * 200);

    for (int i = 0; i < 200; i++) {
        array1[i] = 2;
        array2[i] = 3;
    }

    double * out = myVectorAdd(array1, array2, 200);
    for (int i = 0; i < 200; i++) {
        assert(out[i] == 5);
    }

    return 0;
}


