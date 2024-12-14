#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void CovarianceMatrix(double** p, double sigma_kl, double l_kl, int npts) {

    // Variáveis para armazenar o tempo inicial e final
    clock_t start, end;
    double cpu_time_used;

    // Captura o tempo inicial
    start = clock();

    double** Q = (double**) malloc(npts * sizeof(double*));
    if (Q == NULL) {
    fprintf(stderr, "Erro ao alocar memória para as linhas da matriz Q.\n");
    exit(1);
    }
    for (int i = 0; i < npts; i++) {
        Q[i] = (double*) malloc(npts * sizeof(double));
        if (Q[i] == NULL) {
        fprintf(stderr, "Erro ao alocar memória para a linha %d da matriz Q.\n", i);
        exit(1);
    }
    }

    double sigmakl2 = sigma_kl * sigma_kl;
    double lkl2 = l_kl * l_kl;

    for (int i = 0; i < npts; i++) {
        for (int j = 0; j <= i; j++) {
            double dist = sqrt(pow(p[0][i] - p[0][j], 2) + pow(p[1][i] - p[1][j], 2));
            Q[i][j] = sigmakl2 * exp(-dist / (2 * lkl2));
            Q[j][i] = Q[i][j];
        }
    }

    // Salvando a matriz de covariância em um arquivo
    char filename[50];
    sprintf(filename, "covmat_%d_c.txt", npts);
    FILE* f = fopen(filename, "w");
    for (int i = 0; i < npts; i++) {
        for (int j = 0; j < npts; j++) {
            fprintf(f, "%.18e ", Q[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    // Liberando memória
    for (int i = 0; i < npts; i++) {
        free(Q[i]);
    }
    free(Q);

    end = clock();
    // Calcula o tempo gasto em segundos
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Ellapsed time: %f s\n", cpu_time_used);
}

int main() {
    FILE* f = fopen("my_points_test.txt", "r");

    int npts = 0;
    double temp;
    while (fscanf(f, "%lf", &temp) != EOF) {
        npts++;
    }
    npts /= 2;  // Dividindo por 2 pois há duas linhas de coordenadas

    rewind(f);

    double** p = (double**) malloc(2 * sizeof(double*));
    p[0] = (double*) malloc(npts * sizeof(double));
    p[1] = (double*) malloc(npts * sizeof(double));

    for (int i = 0; i < npts; i++) {
        fscanf(f, "%lf", &p[0][i]);
    }
    for (int i = 0; i < npts; i++) {
        fscanf(f, "%lf", &p[1][i]);
    }
    fclose(f);

    double sigma_kl = 0.5;
    double l_kl = 0.5;

    CovarianceMatrix(p, sigma_kl, l_kl, npts);

    free(p[0]);
    free(p[1]);
    free(p);

    return 0;
}
