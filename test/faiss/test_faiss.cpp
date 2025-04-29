#include <faiss/IndexFlat.h>
#include <iostream>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <utils.h>        // For mapping users to embeddings
#include <cstdlib>  // For rand()
#include <algorithm>  // For std::generate



int main() {
    std::string embeddings_dir = "/home/jetson/FaceRecognitionSystem/jetson/backend/embeddings/";
    
    // Load all user embeddings
    std::vector<utils::UserEmbedding> users = utils::load_all_embeddings(embeddings_dir);
    std::cout << "Loaded embeddings for " << users.size() << " users" << std::endl;
    
    if (users.empty()) {
        std::cerr << "No user embeddings found. Exiting." << std::endl;
        return 1;
    }
    
    // Create FAISS index using the embedding size from the first user
    // (assuming all embeddings have the same dimension)
    int embedding_size = users.front().embedding_size;
    std::cout << "Creating FAISS index with dimension: " << embedding_size << std::endl;
    
    faiss::IndexFlatL2 index(embedding_size);
    
    // Create a map to track which index entries belong to which user
    std::map<int, int> index_to_user_map;  // Maps FAISS index to user index in our vector
    
    // Add all embeddings to the FAISS index
    for (size_t user_idx = 0; user_idx < users.size(); ++user_idx) {
        const utils::UserEmbedding& user = users[user_idx];
        
        int num_vectors = user.embeddings.size() / embedding_size;
        std::cout << "Adding " << num_vectors << " embeddings for user: " 
                  << user.name << " (ID: " << user.id << ")" << std::endl;
        
        int start_idx = index.ntotal;
        index.add(num_vectors, user.embeddings.data());
        std::cout << "Added vectors. Index now contains " << index.ntotal << " vectors" << std::endl;
        
        // Map the added indices to this user
        for (int i = 0; i < num_vectors; ++i) {
            index_to_user_map[start_idx + i] = user_idx;
        }
    }
    
    std::cout << "FAISS index contains a total of " << index.ntotal << " vectors" << std::endl;
    
    // Example of how to perform a search (if needed)
    // You would need a query vector for this

    const int k = 3;  // Number of nearest neighbors to find
    std::vector<float> query_vector(embedding_size);  // Your query vector
    std::generate(query_vector.begin(), query_vector.end(), [](){ return static_cast<float>(rand()) / RAND_MAX; });
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> indices(k);

    index.search(1, query_vector.data(), k, distances.data(), indices.data());

    for (int i = 0; i < k; ++i) {
        if (indices[i] >= 0 && indices[i] < index.ntotal) {
            int user_idx = index_to_user_map[indices[i]];
            std::cout << "Match " << i << ": User " << users[user_idx].name 
                      << " (dist: " << distances[i] << ")" << std::endl;
        }
    }
    
    return 0;
}
