#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void CovarianceMatrix(double** p, double sigma_kl, double l_kl, int npts, int num_ths) {

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

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < npts; i++) {
        for (int j = 0; j <= i; j++) {
            double dist = sqrt(pow(p[0][i] - p[0][j], 2) + pow(p[1][i] - p[1][j], 2));
            Q[i][j] = sigmakl2 * exp(-dist / (2 * lkl2));
            Q[j][i] = Q[i][j];
        }
    }

    // Salvando a matriz de covariância em um arquivo
    char filename[50];
    sprintf(filename, "cov_mat_omp.txt", num_ths);
    FILE* f = fopen(filename, "w");
    for (int i = 0; i < npts; i++) {
        for (int j = 0; j < npts; j++) {
            fprintf(f, "%lf ", Q[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

    // Liberando memória
    for (int i = 0; i < npts; i++) {
        free(Q[i]);
    }
    free(Q);
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

    int thread_counts[] = {1, 2, 4, 8};

    printf("======================================================================================================\n");
    printf("\tCovariance Matrix with %d points (Parallel with OpenMP)\n", npts);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Threads\t\t\tRuntime (s)\n");
    printf("------------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < sizeof(thread_counts)/sizeof(thread_counts[0]); i++) {
        int num_ths = thread_counts[i];
        omp_set_num_threads(num_ths);
        double start_time = omp_get_wtime();
        CovarianceMatrix(p, sigma_kl, l_kl, npts, num_ths);
        double runtime = omp_get_wtime() - start_time;
        printf("%d\t\t\t%f\n", num_ths, runtime);
    }

    printf("------------------------------------------------------------------------------------------------------\n");

    free(p[0]);
    free(p[1]);
    free(p);

    return 0;
}
