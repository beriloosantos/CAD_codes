#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void CovarianceMatrix(double** p, double sigma_kl, double l_kl, int npts, int num_ths) {

    omp_set_num_threads(num_ths);

    int num_elements = ((npts-1)*npts)/2;  // Tamanho do vetor para a parte triangular
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

    double start_time = omp_get_wtime();

    // Calcular os valores para o vetor da parte triangular inferior
    #pragma omp parallel for num_threads(num_ths) shared(coordp1x, coordp2x, coordp1y, coordp2y, Qvec, sigmakl2, lkl2)
    for (int i = 0; i < num_elements; i++) {
        double p1x = coordp1x[i];
        double p2x = coordp2x[i];
        double p1y = coordp1y[i];
        double p2y = coordp2y[i];
        double dist = sqrt((p1x-p2x)*(p1x-p2x) + (p1y-p2y)*(p1y-p2y));
        Qvec[i] = sigmakl2 * exp(-dist / (2 * lkl2));
    }

// Salvando o vetor em um arquivo
    char filename[50];
    sprintf(filename, "covmat_vectorized_%d_omp_%d.txt", npts, num_ths);
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
        fprintf(f, "%.18e\n", Qvec[i]);
    }
    fclose(f);
    free(Qvec);
    free(coordp1x);
    free(coordp2x);
    free(coordp1y);
    free(coordp2y);

    double runtime = omp_get_wtime() - start_time;
    printf("%d\t\t\t%f\n", num_ths, runtime);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s <num_threads>\n", argv[0]);
        return 1;
    }

    int num_ths = atoi(argv[1]);  // Lê o número de threads a partir do argumento
    if (num_ths <= 0) {
        fprintf(stderr, "O número de threads deve ser maior que zero.\n");
        return 1;
    }

    FILE* f = fopen("/home/berilo/scripts_CAD/my_points_fat.txt", "r");
    if (f == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo de pontos.\n");
        return 1;
    }

    int npts = 0;
    double temp;
    while (fscanf(f, "%lf", &temp) != EOF) {
        npts++;
    }
    npts /= 2;  // Dividindo por 2 pois há duas linhas de coordenadas

    rewind(f);

    double** p = (double**)malloc(2 * sizeof(double*));
    p[0] = (double*)malloc(npts * sizeof(double));
    p[1] = (double*)malloc(npts * sizeof(double));

    if (p[0] == NULL || p[1] == NULL) {
        fprintf(stderr, "Erro ao alocar memória para os pontos.\n");
        free(p[0]);
        free(p[1]);
        free(p);
        fclose(f);
        return 1;
    }

    for (int i = 0; i < npts; i++) {
        fscanf(f, "%lf", &p[0][i]);
    }
    for (int i = 0; i < npts; i++) {
        fscanf(f, "%lf", &p[1][i]);
    }
    fclose(f);

    double sigma_kl = 0.5;
    double l_kl = 0.5;

    CovarianceMatrix(p, sigma_kl, l_kl, npts, num_ths);

    free(p[0]);
    free(p[1]);
    free(p);

    return 0;
}