#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;

// Multiplicação sequencial em CPU.
void multiplicacaoCPU(
    const vector<double>& A,
    const vector<double>& B,
    vector<double>& C,
    int N
) {

    for (int i = 0; i < N; i++) {

        for (int j = 0; j < N; j++) {

            double soma = 0.0;

            for (int k = 0; k < N; k++) {

                // Acessa A[i][k]
                double a = A[i * N + k];

                // Acessa B[k][j]
                double b = B[k * N + j];

                // Produto intermediário
                double val = a * b;

                // Operação computacionalmente custosa
                soma += val * sin(val) + cos(val);
            }

            // Armazena o resultado em C[i][j]
            C[i * N + j] = soma;
        }
    }
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

    // Marca o início da medição
    auto t0 = chrono::high_resolution_clock::now();

    // Executa a multiplicação sequencial
    multiplicacaoCPU(A, B, C, N);

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

    return 0;
}
