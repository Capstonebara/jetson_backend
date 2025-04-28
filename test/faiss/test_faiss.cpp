#include <faiss/IndexFlat.h>
#include <vector>
#include <iostream>
#include <random>

int main() {
    int dim = 5;
    int nb = 4;
    int nq = 1;
    // int k = 1;

    std::mt19937 rng(std::random_device{}()); // seed RNG
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    std::vector<float> xb(dim * nb);
    std::vector<float> xq(dim * nq);

    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < dim; ++j)
            xb[dim * i + j] = distrib(rng);
        xb[dim * i] += static_cast<float>(i) / 1000.f;
    }

    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < dim; ++j)
            xq[dim * i + j] = distrib(rng);
        xq[dim * i] += static_cast<float>(i) / 1000.f;
    }

    faiss::IndexFlatL2 index(dim);
    std::cout << "is_trained = " << (index.is_trained ? "true" : "false") << std::endl;

    index.add(nb, xb.data());
    std::cout << "ntotal = " << index.ntotal << std::endl;
    return 0;
}
