#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Kernel CUDA para calcular os valores da matriz de covariância
__global__ void CovarianceMatrixKernel(double* Qvec, double* coordp1x, double* coordp2x, double* coordp1y, double* coordp2y, double sigmakl2, double lkl2, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        double p1x = coordp1x[idx];
        double p2x = coordp2x[idx];
        double p1y = coordp1y[idx];
        double p2y = coordp2y[idx];
        double dist = sqrt((p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y));
        Qvec[idx] = sigmakl2 * exp(-dist / (2 * lkl2));
    }
}

// Função host para configurar e chamar o kernel CUDA
void CovarianceMatrix(double** p, double sigma_kl, double l_kl, int npts, int blockSize) {
    clock_t start, end;
    double cpu_time_used;

    int num_elements = ((npts - 1) * npts) / 2;

    // Alocar memória unificada para as variáveis
    double* Qvec;
    double* coordp1x;
    double* coordp2x;
    double* coordp1y;
    double* coordp2y;

    cudaMallocManaged(&Qvec, num_elements * sizeof(double));
    cudaMallocManaged(&coordp1x, num_elements * sizeof(double));
    cudaMallocManaged(&coordp2x, num_elements * sizeof(double));
    cudaMallocManaged(&coordp1y, num_elements * sizeof(double));
    cudaMallocManaged(&coordp2y, num_elements * sizeof(double));

    // Calcular os pares de índices para a parte triangular
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

    if (countcoord != num_elements) {
        fprintf(stderr, "Erro no cálculo dos pares de índices.\n");
        cudaFree(Qvec);
        cudaFree(coordp1x);
        cudaFree(coordp2x);
        cudaFree(coordp1y);
        cudaFree(coordp2y);
        exit(1);
    }

    start = clock();

    // Definir o número de blocos
    int numBlocks = (num_elements + blockSize - 1) / blockSize;

    double sigmakl2 = sigma_kl * sigma_kl;
    double lkl2 = l_kl * l_kl;

    // Chamar o kernel CUDA
    CovarianceMatrixKernel<<<numBlocks, blockSize>>>(Qvec, coordp1x, coordp2x, coordp1y, coordp2y, sigmakl2, lkl2, num_elements);

    // Sincronizar antes de acessar os resultados
    cudaDeviceSynchronize();

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Salvar o vetor em um arquivo
    char filename[50];
    sprintf(filename, "covmat_vectorized_%d_cuda_%d_%d.txt", npts, numBlocks, blockSize);
    FILE* f = fopen(filename, "w");
    if (f == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo para escrita.\n");
        cudaFree(Qvec);
        cudaFree(coordp1x);
        cudaFree(coordp2x);
        cudaFree(coordp1y);
        cudaFree(coordp2y);
        exit(1);
    }

    for (int i = 0; i < num_elements; i++) {
        fprintf(f, "%.18e\n", Qvec[i]);
    }
    fclose(f);

    // Liberar a memória unificada
    cudaFree(Qvec);
    cudaFree(coordp1x);
    cudaFree(coordp2x);
    cudaFree(coordp1y);
    cudaFree(coordp2y);

    printf("%d\t%d\t\t\t%f\n", numBlocks, blockSize, cpu_time_used);
}

int main() {
    FILE* f = fopen("/home/berilo/scripts_CAD/my_points_fat.txt", "r");
    if (f == NULL) {
        fprintf(stderr, "Error opening input file\n");
        return EXIT_FAILURE;
    }

    int npts = 0;
    double temp;
    while (fscanf(f, "%lf", &temp) != EOF) {
        npts++;
    }
    npts /= 2;

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

    int block_opts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    printf("======================================================================================================\n");
    printf("\tCovariance Matrix with %d points (Parallel with CUDA)\n", npts);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Num Blocks\tBlock size\t\t\tRuntime (s)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < sizeof(block_opts) / sizeof(block_opts[0]); i++) {
        int blockSize = block_opts[i];
        CovarianceMatrix(p, sigma_kl, l_kl, npts, blockSize);
    }
    printf("------------------------------------------------------------------------------------------------------\n");

    free(p[0]);
    free(p[1]);
    free(p);

    return 0;
}
