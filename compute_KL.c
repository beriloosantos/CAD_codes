// compilação >>> gcc -o compute_kl compute_KL.c -llapacke -lblas -lm -O3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <lapacke.h>
#include <cblas.h>

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

double* CovarianceMatrix(double** p, double sigma_kl, double l_kl, int npts) {

    printf("Assembling Covariance Matrix\n");
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
    printf("\n");
    free(coordp1x);
    free(coordp2x);
    free(coordp1y);
    free(coordp2y);

    // Alocar memória para Q_flat
    double* Q_flat = (double*)malloc(npts * npts * sizeof(double));

    // Construir Q_flat a partir de Qvec
    for (int i = 0; i < npts; i++) {
        Q_flat[i * npts + i] = sigmakl2; // Autocorrelação padrão
    }

    // Preencher parte triangular inferior e garantir simetria
    int idx = 0;
    for (int i = 1; i < npts; i++) {
        for (int j = 0; j < i; j++) {
            Q_flat[i * npts + j] = Qvec[idx];  // Parte inferior
            Q_flat[j * npts + i] = Qvec[idx];  // Simetria
            idx++;
        }
    }

    free(Qvec);
    return Q_flat;
}

void compute_KL(double* Q_flat, double* M_flat, int npts, const char* eigenvalues_file, const char* eigenvectors_file) {
    
    // Alocar memória para matrizes intermediárias e resultados
    double* auxm = (double*)malloc(npts * npts * sizeof(double));
    double* Tmat = (double*)malloc(npts * npts * sizeof(double));
    double* eigenvalues = (double*)malloc(npts * sizeof(double));
    double* work = (double*)malloc(4 * npts * sizeof(double));
    int lda = npts, info;

    printf("Assembling T Matrix\n");

    // Calcular a matriz auxiliar: auxm = Q * M
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                npts, npts, npts, 1.0, Q_flat, npts, M_flat, npts, 0.0, auxm, npts);

    // Calcular Tmat = M^T * auxm
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                npts, npts, npts, 1.0, M_flat, npts, auxm, npts, 0.0, Tmat, npts);

    // Monitoramento das matrizes
    save_matrix_to_file("matrix_Q_c.txt", Q_flat, npts);
    save_matrix_to_file("matrix_T_c.txt", Tmat, npts);
    print_matrix_sample("Matriz M", M_flat, npts, 5);
    print_matrix_sample("Matriz Q", Q_flat, npts, 5);
    print_matrix_sample("Matriz Aux", auxm, npts, 5);
    print_matrix_sample("Matriz T", Tmat, npts, 5);

    free(auxm); // Liberar memória intermediária

    printf("Solving Eigeinvalue Problem\n");

    // Resolver o problema de autovalores generalizado para Tmat
    //info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', npts, Tmat, lda, eigenvalues);
    info = LAPACKE_dsygv(LAPACK_ROW_MAJOR, 1, 'V', 'U', npts, Tmat, lda, M_flat, lda, eigenvalues);
    if (info > 0) {
        fprintf(stderr, "Erro ao calcular autovalores.\n");
        free(Tmat);
        free(eigenvalues);
        free(work);
        exit(1);
    }

    // Ordenar os autovalores em ordem decrescente e reorganizar as autofunções
    for (int i = 0; i < npts - 1; i++) {
        for (int j = i + 1; j < npts; j++) {
            if (eigenvalues[i] < eigenvalues[j]) {
                // Trocar autovalores
                double temp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = temp;

                // Trocar autofunções correspondentes
                for (int k = 0; k < npts; k++) {
                    double temp_vec = Tmat[k * npts + i];
                    Tmat[k * npts + i] = Tmat[k * npts + j];
                    Tmat[k * npts + j] = temp_vec;
                }
            }
        }
    }

    // Salvar autovalores
    FILE* f_eigenvalues = fopen(eigenvalues_file, "w");
    if (f_eigenvalues == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo para salvar autovalores.\n");
        exit(1);
    }

    for (int i = 0; i < npts; i++) {
        fprintf(f_eigenvalues, "%.18e\n", eigenvalues[i]);
    }
    fclose(f_eigenvalues);

    // Salvar autovetores (armazenados em Tmat após LAPACKE_dsyev)
    FILE* f_eigenvectors = fopen(eigenvectors_file, "w");
    if (f_eigenvectors == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo para salvar autovetores.\n");
        exit(1);
    }

    for (int i = 0; i < npts; i++) {
        for (int j = 0; j < npts; j++) {
            fprintf(f_eigenvectors, "%.18e ", Tmat[i * npts + j]);
        }
        fprintf(f_eigenvectors, "\n");
    }
    fclose(f_eigenvectors);

    free(Tmat);
    free(eigenvalues);
    free(work);
}


int main() {
    //int argc, char* argv[]
    //if (argc < 2) {
    //    printf("File path não especificado, considerando padrão\n");
    const char* base_path = "/home/lamap/Documents/CAD";
    //}

    //const char* base_path = argv[1];

    // Carregamento do arquivo de coordenadas
    char coordinates_file[512];
    snprintf(coordinates_file, sizeof(coordinates_file), "%s/my_points.txt", base_path);
    FILE* f = fopen(coordinates_file, "r");
    if (f == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo de coordenadas: %s\n", coordinates_file);
        return EXIT_FAILURE;
    }

    int npts = 0;
    double temp;
    while (fscanf(f, "%lf", &temp) != EOF) {
        npts++;
    }
    npts /= 2;  // Dividindo por 2 pois há duas linhas de coordenadas

    rewind(f);

    double** p = (double**) malloc(2 * sizeof(double*));
    if (p == NULL) {
        fprintf(stderr, "Erro ao alocar memória para p.\n");
        fclose(f);
        return EXIT_FAILURE;
    }
    p[0] = (double*) malloc(npts * sizeof(double));
    p[1] = (double*) malloc(npts * sizeof(double));
    if (p[0] == NULL || p[1] == NULL) {
        fprintf(stderr, "Erro ao alocar memória para as coordenadas.\n");
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
    if (ff == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo da matriz de massa: %s\n", mass_file);
        free(p[0]);
        free(p[1]);
        free(p);
        return EXIT_FAILURE;
    }

    // Alocar memória para a matriz no formato denso
    double* M_flat = (double*)calloc(npts * npts, sizeof(double));
    if (M_flat == NULL) {
        fprintf(stderr, "Erro ao alocar memória para a matriz de massa.\n");
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
            fprintf(stderr, "Índice fora dos limites: row=%d, col=%d\n", row, col);
            fclose(ff);
            free(p[0]);
            free(p[1]);
            free(p);
            free(M_flat);
            return EXIT_FAILURE;
        }
        M_flat[row * npts + col] = value; // Preenchendo o formato row-major
    }
    fclose(ff);

    // Medir tempo de execução
    clock_t start, end;
    start = clock();

    double sigma_kl = 0.5;
    double l_kl = 0.5;

    double* Q_flat = CovarianceMatrix(p, sigma_kl, l_kl, npts);
    if (Q_flat == NULL) {
        fprintf(stderr, "Erro ao calcular a matriz de covariância.\n");
        free(p[0]);
        free(p[1]);
        free(p);
        free(M_flat);
        return EXIT_FAILURE;
    }

    compute_KL(Q_flat, M_flat, npts, "eigenvalues_c.txt", "eigenfunctions_c.txt");

    printf("Eigenproblem solved succesfully\n");

    end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Ellapsed time: %.6f s\n", cpu_time_used);

    // Liberar memória
    free(p[0]);
    free(p[1]);
    free(p);
    free(Q_flat);
    free(M_flat);

    return 0;
}
