#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Função para calcular a matriz de covariância

void CovarianceMatrix(double sigma_kl, double l_kl, int npts, int num_elements, double* coordp1x, double* coordp2x, double* coordp1y, double* coordp2y, int rank, int size) {

    double sigmakl2 = sigma_kl * sigma_kl;
    double lkl2 = l_kl * l_kl;

    int local_n = num_elements / size;
    int remainder = num_elements % size;
    int start = rank * local_n + (rank < remainder ? rank : remainder);
    local_n += (rank < remainder);

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
        int i = start + local_i;
        double p1x = coordp1x[i];
        double p2x = coordp2x[i];
        double p1y = coordp1y[i];
        double p2y = coordp2y[i];
        double dist = sqrt((p1x-p2x)*(p1x-p2x) + (p1y-p2y)*(p1y-p2y));
        Qvec_loc[local_i] = sigmakl2 * exp(-dist / (2 * lkl2));
    }

    // Coletar as partes do vetor de cada processo
    double* Qvec = NULL;
    int* recvcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        Qvec = (double*)malloc(num_elements * sizeof(double));
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        for (int i = 0; i < size; i++) {
            recvcounts[i] = num_elements / size + (i < remainder ? 1 : 0);
            displs[i] = i * (num_elements / size) + (i < remainder ? i : remainder);
        }
    }

    // Coletar os resultados em Qvec no processo root
    MPI_Gatherv(Qvec_loc, local_n, MPI_DOUBLE, Qvec, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Processo root salva o vetor em um arquivo
    if (rank == 0) {
        char filename[50];
        sprintf(filename, "covmat_vectorized_%d.txt", npts);
        FILE* f = fopen(filename, "w");
        if (f == NULL) {
            fprintf(stderr, "Erro ao abrir o arquivo para escrita.\n");
            free(Qvec);
            free(recvcounts);
            free(displs);
            exit(1);
        }

        for (int i = 0; i < num_elements; i++) {
            fprintf(f, "%.18e\n", Qvec[i]);
        }
        fclose(f);

        free(Qvec);
        free(recvcounts);
        free(displs);
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

    // Ler os dados de pontos (apenas no processo root)
    int npts = 0;
    double** p = NULL;
    int num_elements;

    double* coordp1x, * coordp2x, * coordp1y, * coordp2y;
    
    if (rank == 0) {
        FILE* f = fopen("my_points_test.txt", "r");

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
    }

    // Broadcast do número de pontos e alocação dos dados nos outros processos
    
    MPI_Bcast(&num_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&npts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    coordp1x = (double*)malloc(num_elements * sizeof(double));
    coordp2x = (double*)malloc(num_elements * sizeof(double));
    coordp1y = (double*)malloc(num_elements * sizeof(double));
    coordp2y = (double*)malloc(num_elements * sizeof(double));

    if (rank == 0) {
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

    double sigma_kl = 0.5;
    double l_kl = 0.5;

    // Broadcast dos vetores de coordenadas dos pontos
    MPI_Bcast(coordp1x, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(coordp2x, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(coordp1y, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(coordp2y, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Chamada da função para calcular a matriz de covariância
    CovarianceMatrix(sigma_kl, l_kl, npts, num_elements, coordp1x, coordp2x, coordp1y, coordp2y, rank, size);

    // Liberar a memória
    if (rank == 0) {
    free(p[0]);
    free(p[1]);
    free(p);
    }
    free(coordp1x);
    free(coordp2x);
    free(coordp1y);
    free(coordp2y);

    // Finalizar MPI
    MPI_Finalize();

    return 0;
}
