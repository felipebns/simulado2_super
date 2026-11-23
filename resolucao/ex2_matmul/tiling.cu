#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

const int TILE = 16;  // tile de 16 x 16 = bloco de 256 threads

// Cada bloco calcula um tile de C. A cada passo, as threads do bloco copiam juntas
// um tile de A e um tile de B para a memória compartilhada e reutilizam esses valores.
__global__ void multiplicacaoTiling(const double* A, const double* B, double* C, int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * TILE + ty;  // linha
    int j = blockIdx.x * TILE + tx;  // coluna

    double soma = 0.0;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        // Cada thread carrega um elemento de A e um de B
        int colA = t * TILE + tx;
        int linB = t * TILE + ty;
        As[ty][tx] = (i < N && colA < N) ? A[i * N + colA] : 0.0;
        Bs[ty][tx] = (linB < N && j < N) ? B[linB * N + j] : 0.0;
        __syncthreads();  // tile completo antes de usar

        int kmax = min(TILE, N - t * TILE);  // último tile pode ser menor
        for (int k = 0; k < kmax; k++) {
            double val = As[ty][k] * Bs[k][tx];
            soma += val * sin(val) + cos(val);
        }
        __syncthreads();  // todos terminaram antes de sobrescrever o tile
    }

    if (i < N && j < N)
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

    // Um bloco de TILE x TILE threads por tile de C
    dim3 threads(TILE, TILE);
    dim3 blocos((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Marca o início da medição (cópias + kernel)
    auto t0 = chrono::high_resolution_clock::now();

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    multiplicacaoTiling<<<blocos, threads>>>(d_A, d_B, d_C, N);
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
