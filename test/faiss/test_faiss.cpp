#include <faiss/IndexFlat.h>
#include <iostream>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <utils.h>        // For mapping users to embeddings
#include <cstdlib>  // For rand()
#include <algorithm>  // For std::generate
#include <vectordb.h>  // For std::generate

int main() {
    
    VectorDB db("/home/jetson/FaceRecognitionSystem/jetson/backend/embeddings/");

    std::vector<float> query_vector(128);  // Your query vector
    std::generate(query_vector.begin(), query_vector.end(), [](){ return static_cast<float>(rand()) / RAND_MAX; });

    auto pair = db.search(query_vector.data());
    int index = pair.first;
    faiss::idx_t distance = pair.second;

    std::cout << "Index: " << index << " , distance: " << distance << std::endl;
    std::cout << "Name: " << db.findName(index) << std::endl;

    return 0;
}
