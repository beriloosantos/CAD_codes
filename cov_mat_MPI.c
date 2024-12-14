#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Função para calcular a matriz de covariância
void CovarianceMatrix(double sigma_kl, double l_kl, int npts, int local_n, double* coordp1x, double* coordp2x, double* coordp1y, double* coordp2y, int rank, int size) {

    double sigmakl2 = sigma_kl * sigma_kl;
    double lkl2 = l_kl * l_kl;

    // Alocação da matriz vetorizada local
    double* Qvec_loc = (double*)malloc(local_n * sizeof(double));
    if (Qvec_loc == NULL) {
        fprintf(stderr, "Erro ao alocar memória para o vetor Qvec.\n");
        exit(1);
    }

    // Medir o tempo de início
    double start_time = MPI_Wtime();

    // Calcular os valores para o vetor da parte triangular inferior
    for (int local_i = 0; local_i < local_n; local_i++) {
        double p1x = coordp1x[local_i];
        double p2x = coordp2x[local_i];
        double p1y = coordp1y[local_i];
        double p2y = coordp2y[local_i];
        double dist = sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y));
        Qvec_loc[local_i] = sigmakl2 * exp(-dist / (2 * lkl2));
    }

    // Coletar as partes do vetor de cada processo
    double* Qvec = NULL;
    if (rank == 0) {
        Qvec = (double*)malloc(local_n * size * sizeof(double));
    }

    // Coletar os resultados em Qvec no processo root
    MPI_Gather(Qvec_loc, local_n, MPI_DOUBLE, Qvec, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Processo root salva o vetor em um arquivo
    if (rank == 0) {
        char filename[50];
        sprintf(filename, "covmat_vectorized_%d.txt", npts);
        FILE* f = fopen(filename, "w");
        if (f == NULL) {
            fprintf(stderr, "Erro ao abrir o arquivo para escrita.\n");
            free(Qvec);
            exit(1);
        }

        for (int i = 0; i < local_n * size; i++) {
            fprintf(f, "%.18e\n", Qvec[i]);
        }
        fclose(f);
        free(Qvec);
    }

    // Liberando a memória
    free(Qvec_loc);

    // Medir o tempo de término e imprimir o tempo de execução
    if (rank == 0) {
        double runtime = MPI_Wtime() - start_time;
        printf("%d \t %f \n", size, runtime);
    }
}

int main(int argc, char** argv) {
    // Inicializar MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int npts = 0;
    double** p = NULL;
    int num_elements;

    double* coordp1x = NULL, * coordp2x = NULL, * coordp1y = NULL, * coordp2y = NULL;
    double* local_coordp1x = NULL, * local_coordp2x = NULL, * local_coordp1y = NULL, * local_coordp2y = NULL;

    if (rank == 0) {
        FILE* f = fopen("/home/berilo/scripts_CAD/my_points_fat.txt", "r");

        double temp;
        while (fscanf(f, "%lf", &temp) != EOF) {
            npts++;
        }
        npts /= 2;
        rewind(f);

        p = (double**)malloc(2 * sizeof(double*));
        p[0] = (double*)malloc(npts * sizeof(double));
        p[1] = (double*)malloc(npts * sizeof(double));

        for (int i = 0; i < npts; i++) {
            fscanf(f, "%lf", &p[0][i]);
        }
        for (int i = 0; i < npts; i++) {
            fscanf(f, "%lf", &p[1][i]);
        }
        fclose(f);

        num_elements = ((npts - 1) * npts) / 2;

        coordp1x = (double*)malloc(num_elements * sizeof(double));
        coordp2x = (double*)malloc(num_elements * sizeof(double));
        coordp1y = (double*)malloc(num_elements * sizeof(double));
        coordp2y = (double*)malloc(num_elements * sizeof(double));

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
    }

    // Broadcast do número de pontos e número de elementos
    MPI_Bcast(&num_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = num_elements / size + (rank < num_elements % size ? 1 : 0);

    local_coordp1x = (double*)malloc(local_n * sizeof(double));
    local_coordp2x = (double*)malloc(local_n * sizeof(double));
    local_coordp1y = (double*)malloc(local_n * sizeof(double));
    local_coordp2y = (double*)malloc(local_n * sizeof(double));

    int* recvcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            recvcounts[i] = num_elements / size + (i < num_elements % size ? 1 : 0);
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    // Scatterv para distribuir partes dos vetores
    MPI_Scatterv(coordp1x, recvcounts, displs, MPI_DOUBLE, local_coordp1x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(coordp2x, recvcounts, displs, MPI_DOUBLE, local_coordp2x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(coordp1y, recvcounts, displs, MPI_DOUBLE, local_coordp1y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(coordp2y, recvcounts, displs, MPI_DOUBLE, local_coordp2y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double sigma_kl = 0.5;
    double l_kl = 0.5;

    // Chamada da função para calcular a matriz de covariância
    CovarianceMatrix(sigma_kl, l_kl, npts, local_n, local_coordp1x, local_coordp2x, local_coordp1y, local_coordp2y, rank, size);

    // Liberar a memória
    free(local_coordp1x);
    free(local_coordp2x);
    free(local_coordp1y);
    free(local_coordp2y);

    if (rank == 0) {
        free(coordp1x);
        free(coordp2x);
        free(coordp1y);
        free(coordp2y);
        free(recvcounts);
        free(displs);
    }

    // Finalizar MPI
    MPI_Finalize();
    return 0;
}

