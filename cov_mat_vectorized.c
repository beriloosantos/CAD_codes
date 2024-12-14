#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void CovarianceMatrix(double** p, double sigma_kl, double l_kl, int npts) {

    // Variáveis para armazenar o tempo inicial e final
    clock_t start, end;
    double cpu_time_used;

    int num_elements = ((npts-1)*npts)/2;  // Tamanho do vetor para a parte triangular superior
    double* Qvec = (double*)malloc(num_elements * sizeof(double));
    if (Qvec == NULL) {
        fprintf(stderr, "Erro ao alocar memória para o vetor Qvec.\n");
        exit(1);
    }

    // Vetores auxiliares para armazenar os pares de coordenadas
    double* coordp1x = (double*)malloc(num_elements * sizeof(double));
    double* coordp2x = (double*)malloc(num_elements * sizeof(double));
    double* coordp1y = (double*)malloc(num_elements * sizeof(double));
    double* coordp2y = (double*)malloc(num_elements * sizeof(double));
    if (coordp1x == NULL || coordp2x == NULL || coordp1y == NULL || coordp2y == NULL) {
        fprintf(stderr, "Erro ao alocar memória para os vetores de coordenadas.\n");
        exit(1);
    }

    double sigmakl2 = sigma_kl * sigma_kl;
    double lkl2 = l_kl * l_kl;

    // Calcular os pares de índices do triângulo inferior
    int countcoord = 0;
    for (int p1 = 1; p1 < npts; p1++) {
        for (int p2 = 0; p2 < p1; p2++) {
            coordp1x[countcoord] = p[0][p1];
            coordp2x[countcoord] = p[0][p2];
            coordp1y[countcoord] = p[1][p1];
            coordp2y[countcoord] = p[1][p2];
            countcoord++;
        }
    }

    // Assegurando que todos os pares foram calculados corretamente
    if (countcoord != num_elements) {
        fprintf(stderr, "Erro no cálculo dos pares de índices.\n");
        free(Qvec);
        free(coordp1x);
        free(coordp2x);
        free(coordp1y);
        free(coordp2y);
        exit(1);
    }

    // Captura o tempo inicial
    start = clock();

    // Calcular os valores para o vetor da parte triangular inferior
    for (int i = 0; i < num_elements; i++) {
        double p1x = coordp1x[i];
        double p2x = coordp2x[i];
        double p1y = coordp1y[i];
        double p2y = coordp2y[i];
        double dist = sqrt((p1x-p2x)*(p1x-p2x) + (p1y-p2y)*(p1y-p2y));
        //double dist = sqrt(pow(p[0][p1] - p[0][p2], 2) + pow(p[1][p1] - p[1][p2], 2));
        Qvec[i] = sigmakl2 * exp(-dist / (2 * lkl2));
    }

    // Salvando o vetor em um arquivo
    char filename[50];
    sprintf(filename, "covmat_vectorized_%d.txt", npts);
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo para escrita.\n");
        free(Qvec);
        free(coordp1x);
        free(coordp2x);
        free(coordp1y);
        free(coordp2y);
        exit(1);
    }

    for (int i = 0; i < num_elements; i++) {
        fprintf(f, "%.18e", Qvec[i]);
    }
    fclose(f);
    free(Qvec);
    free(coordp1x);
    free(coordp2x);
    free(coordp1y);
    free(coordp2y);

    end = clock();
    // Calcula o tempo gasto em segundos
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Ellapsed time: %f s\n", cpu_time_used);
}

int main() {

    FILE* f = fopen("/home/berilo/scripts_CAD/my_points_fat.txt", "r");

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
