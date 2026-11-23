#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

const int THREADS = 1024;  // threads por bloco: cada bloco faz o scan de 1024 elementos

// Etapa 1a: scan inclusivo dentro de cada bloco (scan simples da aula, em memória
// compartilhada). A última thread guarda a soma do bloco em somas_blocos.
__global__ void scan_blocos(const float* entrada, float* saida, float* somas_blocos,
                            int inicio, int fim) {
    __shared__ float XY[THREADS];
    int tid = threadIdx.x;
    int gid = inicio + blockIdx.x * blockDim.x + tid;

    XY[tid] = (gid < fim) ? entrada[gid] : 0.0f;
    __syncthreads();

    for (int passo = 1; passo < blockDim.x; passo *= 2) {
        float temp = 0.0f;
        if (tid >= passo)
            temp = XY[tid - passo];
        __syncthreads();
        XY[tid] += temp;
        __syncthreads();
    }

    if (gid < fim)
        saida[gid] = XY[tid];
    if (tid == blockDim.x - 1)
        somas_blocos[gid / THREADS] = XY[tid];
}

// Etapa 1b: soma em cada elemento o total dos blocos anteriores
__global__ void somar_offset(float* prefix, const float* somas_blocos, int inicio, int fim) {
    int gid = inicio + blockIdx.x * blockDim.x + threadIdx.x;
    int bloco = gid / THREADS;
    if (gid < fim && bloco > 0)
        prefix[gid] += somas_blocos[bloco - 1];
}

// Etapa 2: normalização (divide pela soma total)
__global__ void normalizar(float* prefix, const float* total, int inicio, int fim) {
    int gid = inicio + blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < fim)
        prefix[gid] /= *total;
}

int main() {
    const size_t N = 87654;
    std::vector<float> v(N);
    std::vector<float> prefix(N);

    const int nblocos = (N + THREADS - 1) / THREADS;  // 86 (precisa ser <= THREADS)
    const size_t bytes = N * sizeof(float);

    float *d_v, *d_prefix, *d_somas, *d_total;
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_prefix, bytes);
    cudaMalloc(&d_somas, nblocos * sizeof(float));
    cudaMalloc(&d_total, sizeof(float));

    auto inicio = std::chrono::high_resolution_clock::now();

    // Gera o vetor
    for (size_t i = 0; i < N; ++i)
        v[i] = static_cast<float>((i + 1) * 2);

    cudaMemcpy(d_v, v.data(), bytes, cudaMemcpyHostToDevice);

    // prefix sum: scan de cada bloco -> scan das somas dos blocos -> soma do offset
    scan_blocos<<<nblocos, THREADS>>>(d_v, d_prefix, d_somas, 0, N);
    scan_blocos<<<1, THREADS>>>(d_somas, d_somas, d_total, 0, nblocos);  // d_total = soma total
    somar_offset<<<nblocos, THREADS>>>(d_prefix, d_somas, 0, N);

    // Normalização
    normalizar<<<nblocos, THREADS>>>(d_prefix, d_total, 0, N);
    cudaDeviceSynchronize();

    float total;
    cudaMemcpy(prefix.data(), d_prefix, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost);

    auto fim = std::chrono::high_resolution_clock::now();
    double tempo = std::chrono::duration<double, std::milli>(fim - inicio).count();

    // RESULTADOS
    std::cout << "\nÚltimos 10 valores do prefix sum:\n[ ";
    for (size_t i = N - 10; i < N; ++i)
        std::cout << prefix[i] * total << " ";
    std::cout << "]\n";

    std::cout << "Últimos 10 valores normalizados:\n[ ";
    for (size_t i = N - 10; i < N; ++i)
        std::cout << prefix[i] << " ";
    std::cout << "]\n";

    std::cout << "\nTempo GPU (síncrono): " << tempo << " ms\n";

    cudaFree(d_v);
    cudaFree(d_prefix);
    cudaFree(d_somas);
    cudaFree(d_total);
}
