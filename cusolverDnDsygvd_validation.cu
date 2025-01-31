#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUSOLVER(call) \
    do { \
        cusolverStatus_t err = call; \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSOLVER error: %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Normalize eigenvectors and adjust sign
void normalizeEigenvectors(double* eigenvectors, int n) {
    for (int j = 0; j < n; j++) {
        // Find the largest component (in absolute value) for the current eigenvector
        double maxVal = 0.0;
        int maxIndex = 0;
        for (int i = 0; i < n; i++) {
            if (fabs(eigenvectors[i + j * n]) > fabs(maxVal)) {
                maxVal = eigenvectors[i + j * n];
                maxIndex = i;
            }
        }

        // Flip the sign of the eigenvector if the largest component is negative
        if (maxVal < 0) {
            for (int i = 0; i < n; i++) {
                eigenvectors[i + j * n] = -eigenvectors[i + j * n];
            }
        }

        // Normalize the eigenvector to have unit length
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += eigenvectors[i + j * n] * eigenvectors[i + j * n];
        }
        norm = sqrt(norm);

        for (int i = 0; i < n; i++) {
            eigenvectors[i + j * n] /= norm;
        }
    }
}

int main() {
    const int n = 3; // Size of the matrices (n x n)

    // Input matrices A and B (symmetric, column-major order)
    double A[] = {4.0, 1.0, 1.0,
                  1.0, 2.0, 0.0,
                  1.0, 0.0, 3.0};
    double B[] = {2.0, 0.0, 0.0,
                  0.0, 3.0, 0.0,
                  0.0, 0.0, 1.0};

    // Output eigenvalues and eigenvectors
    double* eigenvalues = (double*)malloc(n * sizeof(double));
    double* eigenvectors = (double*)malloc(n * n * sizeof(double));

    // Device memory pointers
    double *d_A, *d_B, *d_eigenvalues, *d_work;
    int *d_info;
    int lwork = 0;

    // Create cuSOLVER handle
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_A, n * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, n * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_eigenvalues, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));

    // Copy matrices A and B to device
    CHECK_CUDA(cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Query workspace size
    CHECK_CUSOLVER(cusolverDnDsygvd_bufferSize(
        handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n, d_A, n, d_B, n, d_eigenvalues, &lwork));

    // Allocate workspace
    CHECK_CUDA(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

    // Solve generalized eigenvalue problem
    CHECK_CUSOLVER(cusolverDnDsygvd(
        handle, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n, d_A, n, d_B, n, d_eigenvalues, d_work, lwork, d_info));

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(eigenvalues, d_eigenvalues, n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(eigenvectors, d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    // Normalize and adjust sign of eigenvectors
    normalizeEigenvectors(eigenvectors, n);

    // Print eigenvalues
    printf("Eigenvalues: ");
    for (int i = 0; i < n; i++) {
        printf("%f ", eigenvalues[i]);
    }
    printf("\n");

    // Print eigenvectors (column-major)
    printf("Eigenvectors (column-major):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", eigenvectors[i + j * n]);
        }
        printf("\n");
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_eigenvalues));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUSOLVER(cusolverDnDestroy(handle));

    free(eigenvalues);
    free(eigenvectors);

    return 0;
}