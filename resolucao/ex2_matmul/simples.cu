#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// Cada thread calcula um elemento C[i][j] (mesmo laço em k da versão CPU).
__global__ void multiplicacaoGPU(const double* A, const double* B, double* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // linha
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // coluna
    if (i >= N || j >= N)
        return;

    double soma = 0.0;
    for (int k = 0; k < N; k++) {
        double val = A[i * N + k] * B[k * N + j];
        soma += val * sin(val) + cos(val);
    }
    C[i * N + j] = soma;
}

int main() {

    const int N = 2048;

    vector<double> A(N * N);
    vector<double> B(N * N);
    vector<double> C(N * N, 0.0);

    // Inicialização dos dados
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            int idx = i * N + j;

            A[idx] = sin(i + j);
            B[idx] = cos(i - j);
        }
    }

    // Memória na GPU
    size_t bytes = N * N * sizeof(double);
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Blocos de 16 x 16 threads, um thread por elemento de C
    dim3 threads(16, 16);
    dim3 blocos((N + 15) / 16, (N + 15) / 16);

    // Marca o início da medição (cópias + kernel)
    auto t0 = chrono::high_resolution_clock::now();

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    multiplicacaoGPU<<<blocos, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // Marca o final da medição
    auto t1 = chrono::high_resolution_clock::now();

    // Calcula o tempo de execução
    double tempo =
        chrono::duration<double>(t1 - t0).count();

    cout << "N = " << N << endl;
    cout << "Tempo: " << tempo << " s" << endl;

    // Exibe alguns valores para validação
    cout << "Para validacao dos dados:" << endl;

    cout << "Primeiro elemento da matriz C[0][0] = "
        << C[0] << endl;

    cout << "Elemento do meio da matriz C[N/2][N/2] = "
    << C[(N / 2) * N + (N / 2)] << endl;

    cout << "Ultimo elemento da matriz C[N-1][N-1] = "
        << C[(N - 1) * N + (N - 1)] << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
