#include <iostream>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

const int THREADS = 1024;  // threads por bloco: cada bloco faz o scan de 1024 elementos
const int NSTREAMS = 2;    // número de streams (um pedaço do vetor por stream)

// Os kernels são os mesmos da versão síncrona.

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
    const int nblocos = (N + THREADS - 1) / THREADS;  // 86 (precisa ser <= THREADS)
    const size_t bytes = N * sizeof(float);

    // Tamanho do pedaço de cada stream (múltiplo de THREADS para não cortar um bloco)
    const size_t CHUNK = (nblocos + NSTREAMS - 1) / NSTREAMS * THREADS;

    // Memória pinned: sem ela o cudaMemcpyAsync não é assíncrono de verdade
    float *v, *prefix;
    cudaMallocHost(&v, bytes);
    cudaMallocHost(&prefix, bytes);

    float *d_v, *d_prefix, *d_somas, *d_total;
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_prefix, bytes);
    cudaMalloc(&d_somas, nblocos * sizeof(float));
    cudaMalloc(&d_total, sizeof(float));

    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; ++s)
        cudaStreamCreate(&streams[s]);

    auto inicio = std::chrono::high_resolution_clock::now();

    // Fase 1: a CPU gera o pedaço da stream s enquanto a GPU já copia e faz o scan
    // do pedaço da stream anterior
    for (int s = 0; s < NSTREAMS; ++s) {
        size_t ini = s * CHUNK;
        size_t fim = std::min(ini + CHUNK, N);
        int blocos = (fim - ini + THREADS - 1) / THREADS;

        for (size_t i = ini; i < fim; ++i)
            v[i] = static_cast<float>((i + 1) * 2);

        cudaMemcpyAsync(d_v + ini, v + ini, (fim - ini) * sizeof(float),
                        cudaMemcpyHostToDevice, streams[s]);
        scan_blocos<<<blocos, THREADS, 0, streams[s]>>>(d_v, d_prefix, d_somas, ini, fim);
    }

    // Dependência: o scan das somas dos blocos precisa de todos os pedaços prontos
    for (int s = 0; s < NSTREAMS; ++s)
        cudaStreamSynchronize(streams[s]);
    scan_blocos<<<1, THREADS, 0, streams[0]>>>(d_somas, d_somas, d_total, 0, nblocos);
    cudaStreamSynchronize(streams[0]);

    // Fase 2: offset + normalização + cópia de volta, um pedaço por stream
    // (a cópia de um pedaço sobrepõe o cálculo do pedaço da outra stream)
    for (int s = 0; s < NSTREAMS; ++s) {
        size_t ini = s * CHUNK;
        size_t fim = std::min(ini + CHUNK, N);
        int blocos = (fim - ini + THREADS - 1) / THREADS;

        somar_offset<<<blocos, THREADS, 0, streams[s]>>>(d_prefix, d_somas, ini, fim);
        normalizar<<<blocos, THREADS, 0, streams[s]>>>(d_prefix, d_total, ini, fim);
        cudaMemcpyAsync(prefix + ini, d_prefix + ini, (fim - ini) * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[s]);
    }

    // A CPU só lê o resultado depois que todas as streams terminam
    for (int s = 0; s < NSTREAMS; ++s)
        cudaStreamSynchronize(streams[s]);

    float total;
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

    std::cout << "\nTempo GPU (assíncrono): " << tempo << " ms\n";

    for (int s = 0; s < NSTREAMS; ++s)
        cudaStreamDestroy(streams[s]);
    cudaFree(d_v);
    cudaFree(d_prefix);
    cudaFree(d_somas);
    cudaFree(d_total);
    cudaFreeHost(v);
    cudaFreeHost(prefix);
}
