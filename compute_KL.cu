// compilação >>> nvcc -o compute_kl_cuda compute_kl.cu -lcublas -lcusolver -lm
// /usr/local/cuda-12.6/bin/nvcc -O3 -Xcompiler -O3 -o compute_kl_cuda compute_kl.cu -lcublas -lcusolver -lm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSOLVER(call) \
    do { \
        cusolverStatus_t err = call; \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSOLVER error: %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Sorting helper functions
typedef struct {
    double value;
    int index;
} EigenPair;

int compare_eigenpairs(const void* a, const void* b) {
    EigenPair* pa = (EigenPair*)a;
    EigenPair* pb = (EigenPair*)b;
    return (pb->value > pa->value) ? 1 : -1;
}

// Kernels
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

__global__ void BuildQFlatKernel(double* Qflat, double* Qvec, double sigmakl2, int npts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npts * npts) {
        int i = idx / npts;
        int j = idx % npts;
        if (i == j) {
            Qflat[idx] = sigmakl2;
        } else if (i > j) {
            int qvec_idx = i * (i - 1) / 2 + j;
            Qflat[idx] = Qvec[qvec_idx];
            Qflat[j * npts + i] = Qvec[qvec_idx]; // Simetria
        }
    }
}

// -----------------------------------------------------------------------------------------------------
void save_matrix_to_file(const char* filename, double* matrix, int npts) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo para salvar a matriz: %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < npts; i++) {
        for (int j = 0; j < npts; j++) {
            fprintf(file, "%.18e ", matrix[i * npts + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
    printf("Matriz salva no arquivo: %s\n", filename);
}

void print_matrix_sample(const char* label, double* matrix, int npts, int sample_size) {
    printf("\n%s (amostra de %dx%d):\n", label, sample_size, sample_size);
    for (int i = 0; i < sample_size && i < npts; i++) {
        for (int j = 0; j < sample_size && j < npts; j++) {
            printf("%.6e ", matrix[i * npts + j]);
        }
        printf("\n");
    }
}

void normalizeEigenvectors(double* eigenvectors, const double* M, int n) {
    for (int col = 0; col < n; col++) {
        // B-norm: v^T * M * v
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                norm += eigenvectors[i + col * n] * M[i * n + j] * eigenvectors[j + col * n];
            }
        }
        norm = sqrt(norm);

        // Normalize
        for (int i = 0; i < n; i++) {
            eigenvectors[i + col * n] /= norm;
        }

        // Ensure sign consistency (optional)
        double max_val = 0.0;
        int max_idx = 0;
        for (int i = 0; i < n; i++) {
            if (fabs(eigenvectors[i + col * n]) > fabs(max_val)) {
                max_val = eigenvectors[i + col * n];
                max_idx = i;
            }
        }
        if (max_val < 0) {
            for (int i = 0; i < n; i++) {
                eigenvectors[i + col * n] *= -1;
            }
        }
    }
}

// Funcao host Matriz de Covariancia
double* CovarianceMatrix(double** p, double sigma_kl, double l_kl, int npts, int num_elements, int blockSize, int numBlocks) {

    printf("Assembling Covariance Matrix\n");

    // Alocar memoria unificada para as variaveis
    double* Qvec;
    double* Qflat;
    double* coordp1x;
    double* coordp2x;
    double* coordp1y;
    double* coordp2y;
    cudaMallocManaged(&Qvec, num_elements * sizeof(double));
    cudaMallocManaged(&Qflat, npts * npts * sizeof(double));
    cudaMallocManaged(&coordp1x, num_elements * sizeof(double));
    cudaMallocManaged(&coordp2x, num_elements * sizeof(double));
    cudaMallocManaged(&coordp1y, num_elements * sizeof(double));
    cudaMallocManaged(&coordp2y, num_elements * sizeof(double));
    if (Qvec == NULL || Qflat == NULL || coordp1x == NULL || coordp2x == NULL || coordp1y == NULL || coordp2y == NULL) {
        fprintf(stderr, "Erro ao alocar memoria para os vetores de coordenadas.\n");
        exit(1);
    }

    // Calcular os pares de indices para a parte triangular
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
        fprintf(stderr, "Erro no calculo dos pares de indices.\n");
        cudaFree(Qvec);
        cudaFree(Qflat);
        cudaFree(coordp1x);
        cudaFree(coordp2x);
        cudaFree(coordp1y);
        cudaFree(coordp2y);
        exit(1);
    }

    double sigmakl2 = sigma_kl * sigma_kl;
    double lkl2 = l_kl * l_kl;

    // Chamar o kernel CUDA para a Matriz de Covariancia
    CovarianceMatrixKernel<<<numBlocks, blockSize>>>(Qvec, coordp1x, coordp2x, coordp1y, coordp2y, sigmakl2, lkl2, num_elements);

    // Sincronizar antes de acessar os resultados
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    numBlocks = (npts * npts + blockSize - 1) / blockSize;

    // Converter a Matriz Q para uma matriz cheia linearizada
    BuildQFlatKernel<<<numBlocks, blockSize>>>(Qflat, Qvec, sigmakl2, npts);

    // Liberar a memoria unificada
    cudaFree(Qvec);
    cudaFree(coordp1x);
    cudaFree(coordp2x);
    cudaFree(coordp1y);
    cudaFree(coordp2y);

    // Sincronizar antes de acessar os resultados
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    print_matrix_sample("Matriz Q", Qflat, npts, 5);

    return Qflat;
}

void Compute_KL(double* Qflat, double* h_Mflat, int npts, const char* eigenvalues_file, const char* eigenvectors_file, int blockSize, int numBlocks) {

    // Alocar memoria unificada para as variaveis
    double *d_Mflat, *d_auxm, *d_Tmat, *d_eigenvalues;
    CHECK_CUDA_ERROR(cudaMalloc(&d_Mflat, npts*npts*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_auxm, npts*npts*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Tmat, npts*npts*sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_eigenvalues, npts*sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemset(d_auxm, 0, npts*npts*sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_Mflat, h_Mflat, npts*npts*sizeof(double), cudaMemcpyHostToDevice));

    // Monitoramento das matrizes
    print_matrix_sample("Matriz M host", h_Mflat, npts, 5);    
    double* h_Mflat_device_copy = (double*)malloc(npts*npts*sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(h_Mflat_device_copy, d_Mflat, npts*npts*sizeof(double), cudaMemcpyDeviceToHost));
    print_matrix_sample("Matriz M device", h_Mflat_device_copy, npts, 5);
    free(h_Mflat_device_copy);

    printf("\nAssembling T Matrix\n");

    // Calcular auxm = Q * M
    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0; // Escalar multiplicador
    const double beta = 1.0;  // Escalar para a matriz de saída

    // Chamada para cublasDgemm
    cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N, // Não transpor as matrizes
        npts, npts, npts,         // Dimensões m, n, k
        &alpha,                   // Ponteiro para alpha
        Qflat, npts,             // Matriz A e seu lda
        d_Mflat, npts,             // Matriz B e seu ldb
        &beta,                    // Ponteiro para beta
        d_auxm, npts                // Matriz C (resultado) e seu ldc
    );

    // Sincronizar antes de acessar os resultados
    cudaDeviceSynchronize();

    double* h_auxm = (double*)malloc(npts*npts*sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(h_auxm, d_auxm, npts*npts*sizeof(double), cudaMemcpyDeviceToHost));
    print_matrix_sample("Matriz Aux", h_auxm, npts, 5);
    free(h_auxm);

    // Calcular Tmat = M^T * auxm
    // Chamada para cublasDgemm
    cublasDgemm(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N, // Transpor a matriz M
        npts, npts, npts,         // Dimensões m, n, k
        &alpha,                   // Ponteiro para alpha
        d_Mflat, npts,             // Matriz A e seu lda
        d_auxm, npts,             // Matriz B e seu ldb
        &beta,                    // Ponteiro para beta
        d_Tmat, npts                // Matriz C (resultado) e seu ldc
    );
    cublasDestroy(handle); // Liberar recursos do handle

    // Sincronizar antes de acessar os resultados
    cudaDeviceSynchronize();

    // Monitoramento das matrizes
    double* h_Tmat = (double*)malloc(npts*npts*sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpy(h_Tmat, d_Tmat, npts*npts*sizeof(double), cudaMemcpyDeviceToHost));
    print_matrix_sample("Matriz T", h_Tmat, npts, 5);

    save_matrix_to_file("matrix_Q_cu.txt", Qflat, npts);
    save_matrix_to_file("matrix_T_cu.txt", h_Tmat, npts);

    printf("\nSolving Eigeinvalue Problem\n");

    // Resolver o problema de autovalores generalizado para Tmat
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    int lwork = 0;
    int* dev_info;
    cudaMalloc((void**)&dev_info, sizeof(int));
    double* work;
    cudaMalloc((void**)&work, lwork * sizeof(double));

    // Consultar o tamanho do workspace
    CHECK_CUSOLVER(cusolverDnDsygvd_bufferSize(
        cusolverH, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        npts, d_Tmat, npts, d_Mflat, npts, d_eigenvalues, &lwork));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&work, lwork * sizeof(double)));

    // Resolver o problema generalizado: Tmat * v = lambda * Mflat * v
    CHECK_CUSOLVER(cusolverDnDsygvd(
        cusolverH, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        npts, d_Tmat, npts, d_Mflat, npts, d_eigenvalues, work, lwork, dev_info));

    // Verificar erros
    int info;
    CHECK_CUDA_ERROR(cudaMemcpy(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        fprintf(stderr, "Erro ao calcular autovalores e autofuncoes com cuSolver (info = %d).\n", info);
        cudaFree(dev_info);
        cudaFree(work);
        cusolverDnDestroy(cusolverH);
        exit(EXIT_FAILURE);
    }

    double* h_eigenvalues = (double*)malloc(npts*sizeof(double));

    CHECK_CUDA_ERROR(cudaMemcpy(h_eigenvalues, d_eigenvalues, npts*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_Tmat, d_Tmat, npts*npts*sizeof(double), cudaMemcpyDeviceToHost));

    // Normalize and adjust sign of eigenvectors
    normalizeEigenvectors(h_Tmat, h_Mflat, npts);  // h_Mflat = your B matrix

    // 3. Efficient sorting using qsort
    EigenPair* pairs = (EigenPair*)malloc(npts*sizeof(EigenPair));
    for(int i=0; i<npts; i++) {
        pairs[i].value = h_eigenvalues[i];
        pairs[i].index = i;
    }
    qsort(pairs, npts, sizeof(EigenPair), compare_eigenpairs);

    // Reorder eigenvalues and eigenvectors
    double* sorted_eigenvalues = (double*)malloc(npts*sizeof(double));
    double* sorted_Tmat = (double*)malloc(npts*npts*sizeof(double));

    for (int i = 0; i < npts; i++) {
        sorted_eigenvalues[i] = pairs[i].value;
        int src_col = pairs[i].index;

        // Copy entire eigenvector column from h_Tmat to sorted_Tmat
        for (int j = 0; j < npts; j++) {
            // Column-major: [j + src_col * npts] -> [j + i * npts]
            sorted_Tmat[j + i * npts] = h_Tmat[j + src_col * npts];
        }
    }

    //printf("First eigenvector (sorted): ");
    //for (int i = 0; i < npts; i++) {
    //    printf("%.3e ", sorted_Tmat[i + 0 * npts]); // First column
    //}
    //printf("\n");

    // Salvar autovalores em arquivo
    FILE* f_eigenvalues = fopen(eigenvalues_file, "w");
    if (f_eigenvalues == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo para salvar autovalores.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < npts; i++) {
        fprintf(f_eigenvalues, "%.18e\n", sorted_eigenvalues[i]);
    }
    fclose(f_eigenvalues);

    // Salvar autovetores em arquivo
    FILE* f_eigenvectors = fopen(eigenvectors_file, "w");
    if (f_eigenvectors == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo para salvar autofuncoes.\n");
        exit(EXIT_FAILURE);
    }

    // Transpose and save (row-major output)
    for (int i = 0; i < npts; i++) {       // Rows
        for (int j = 0; j < npts; j++) {   // Columns
            fprintf(f_eigenvectors, "%.18e ", sorted_Tmat[j * npts + i]); // Transposed access
        }
        fprintf(f_eigenvectors, "\n");
    }
    fclose(f_eigenvectors);

    // Liberar recursos
    free(pairs);
    free(sorted_eigenvalues);
    free(sorted_Tmat);
    free(h_Tmat);
    free(h_eigenvalues);
    cudaFree(dev_info);
    cudaFree(work);
    cudaFree(d_Mflat);
    cudaFree(d_auxm);
    cudaFree(d_Tmat);
    cudaFree(d_eigenvalues);
    cusolverDnDestroy(cusolverH);
}

// -----------------------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {

    const char* base_path = "/home/berilo/scripts_CAD";  // Default
    if (argc >= 2) {
        base_path = argv[1];
    }

    // Carregamento do arquivo de coordenadas
    char coordinates_file[512];
    snprintf(coordinates_file, sizeof(coordinates_file), "%s/my_points_test.txt", base_path);
    FILE* f = fopen(coordinates_file, "r");
    // FILE* f = fopen("/home/berilo/scripts_CAD/my_points.txt", "r");
    if (f == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo de coordenadas");
        return EXIT_FAILURE;
    }

    int npts = 0;
    double temp;
    while (fscanf(f, "%lf", &temp) != EOF) {
        npts++;
    }
    npts /= 2;  // Dividindo por 2 pois ha duas linhas de coordenadas

    rewind(f);

    double** p = (double**) malloc(2 * sizeof(double*));
    if (p == NULL) {
        fprintf(stderr, "Erro ao alocar memoria para p.\n");
        fclose(f);
        return EXIT_FAILURE;
    }
    p[0] = (double*) malloc(npts * sizeof(double));
    p[1] = (double*) malloc(npts * sizeof(double));
    if (p[0] == NULL || p[1] == NULL) {
        fprintf(stderr, "Erro ao alocar memoria para as coordenadas.\n");
        free(p);
        fclose(f);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < npts; i++) {
        fscanf(f, "%lf", &p[0][i]);
    }
    for (int i = 0; i < npts; i++) {
        fscanf(f, "%lf", &p[1][i]);
    }
    fclose(f);

    // Carregamento da matriz de massa
    char mass_file[512];
    snprintf(mass_file, sizeof(mass_file), "%s/matrix_M.txt", base_path);

    FILE* ff = fopen(mass_file, "r");
    //FILE* ff = fopen("/home/berilo/scripts_CAD/matrix_M.txt", "r");
    if (ff == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo da matriz de massa");
        free(p[0]);
        free(p[1]);
        free(p);
        return EXIT_FAILURE;
    }

    // Alocar memoria para a matriz no formato denso
    double* h_Mflat = (double*) malloc(npts * npts * sizeof(double));
    if (h_Mflat == NULL) {
        fprintf(stderr, "Erro ao alocar memoria para a matriz de massa.\n");
        fclose(ff);
        free(p[0]);
        free(p[1]);
        free(p);
        return EXIT_FAILURE;
    }
    // Ler os dados do arquivo e preencher a matriz
    int row, col;
    double value;
    while (fscanf(ff, "%d %d %lf", &row, &col, &value) == 3) {
        if (row >= npts || col >= npts) {
            fprintf(stderr, "Indice fora dos limites: row=%d, col=%d\n", row, col);
            fclose(ff);
            free(p[0]);
            free(p[1]);
            free(p);
            free(h_Mflat);
            return EXIT_FAILURE;
        }
        h_Mflat[row * npts + col] = value; // Preenchendo o formato row-major
    }
    fclose(ff);

    double sigma_kl = 0.5;
    double l_kl = 0.5;

    int num_elements = ((npts - 1) * npts) / 2;

    // int block_opts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    double* Qflat;

    // printf("======================================================================================================\n");
    // printf("\tCovariance Matrix with %d points (Parallel with CUDA)\n", npts);
    // printf("------------------------------------------------------------------------------------------------------\n");
    // printf("Num Blocks\tBlock size\t\t\tRuntime (s)\n");
    printf("------------------------------------------------------------------------------------------------------\n");

    //for (int i = 0; i < sizeof(block_opts) / sizeof(block_opts[0]); i++) {

        int blockSize = 8; //block_opts[i];
        // Definir o numero de blocos
        int numBlocks = (num_elements + blockSize - 1) / blockSize;

        clock_t start, end;
        double cpu_time_used;
        start = clock();

        Qflat = CovarianceMatrix(p, sigma_kl, l_kl, npts, num_elements, blockSize, numBlocks);
        if (Qflat == NULL) {
            fprintf(stderr, "Erro ao calcular a matriz de covariancia.\n");
            free(p[0]);
            free(p[1]);
            free(p);
            free(h_Mflat);
            return EXIT_FAILURE;
        }

        Compute_KL(Qflat, h_Mflat, npts, "eigenvalues_cu.txt", "eigenfunctions_cu.txt", blockSize, numBlocks);

        printf("Eigenproblem solved successfully\n");

        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("%d\t%d\t\t\t%f\n", numBlocks, blockSize, cpu_time_used);

    //}
    printf("------------------------------------------------------------------------------------------------------\n");

    free(p[0]);
    free(p[1]);
    free(p);
    cudaFree(Qflat);
    free(h_Mflat);

    return 0;
}