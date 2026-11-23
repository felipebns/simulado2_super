#include <iostream>
#include <vector>
#include <chrono>

int main() {
    const size_t N = 87654;
    std::vector<float> v(N);
    std::vector<float> prefix(N);

    auto inicio = std::chrono::high_resolution_clock::now();

    // Gera o vetor
    for (size_t i = 0; i < N; ++i)
        v[i] = static_cast<float>((i + 1) * 2);

    // prefix sum
    prefix[0] = v[0];
    for (size_t i = 1; i < N; ++i)
        prefix[i] = prefix[i - 1] + v[i];

    // Normalização
    float total = prefix.back();  // soma total
    for (size_t i = 0; i < N; ++i)
        prefix[i] /= total;

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

    std::cout << "\nTempo CPU: " << tempo << " ms\n";
}
