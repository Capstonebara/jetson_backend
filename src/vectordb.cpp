#include <vectordb.h>
#include <embedding_loader.h>
#include <iostream>
#include <cmath>
#include <iomanip>

VectorDB::VectorDB(const std::string &db_path, const int embedding_size): m_users(embedding_loader::load_all_embeddings(db_path)),
    m_embedding_size(embedding_size), 
    m_index(m_embedding_size)
{
    std::cout << "Loaded embeddings for " << m_users.size() << " users" << std::endl;

    if (m_users.empty()) {
        std::cerr << "No user embeddings found. exiting." << std::endl;
    }

    for (size_t user_idx = 0; user_idx < m_users.size(); ++user_idx) {
        User& user = m_users[user_idx];
        
        int num_vectors = user.embeddings.size() / embedding_size;
        std::cout << "adding " << num_vectors << " embeddings for user: " 
                  << user.name << " (id: " << user.id << ")" << std::endl;
        
        int start_idx = m_index.ntotal;
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Before" << std::endl;
        for (unsigned int i =  0; i < 5; ++i) {
            std::cout << user.embeddings[i] << std::endl;
        }
        // embedding_loader::l2_normalize(user.embeddings);
        std::cout << "After" << std::endl;
        for (unsigned int i = 0; i < 5; ++i) {
            std::cout << user.embeddings[i] << std::endl;
        }
        m_index.add(num_vectors, user.embeddings.data());
        std::cout << "added vectors. index now contains " << m_index.ntotal << " vectors" << std::endl;
        
        // map the added indices to this user
        for (int i = 0; i < num_vectors; ++i) {
            m_index_to_user_map[start_idx + i] = user_idx;
        }
    }
    
    std::cout << "faiss index contains a total of " << m_index.ntotal << " vectors" << std::endl;

}

std::pair<faiss::idx_t, float> VectorDB::search(float *query_ptr) const {
    const int k = 1;  // number of nearest neighbors to find
    std::vector<float> similarities(k);
    std::vector<faiss::idx_t> indices(k);
    m_index.search(1, query_ptr, k, similarities.data(), indices.data());
    return std::make_pair(indices[0], similarities[0]);
}

int VectorDB::getRow() const {
    return m_index.ntotal;
}

int VectorDB::getColumn() const {
    return m_embedding_size;
}
std::string VectorDB::findName(faiss::idx_t idx) {
    const int user_idx = m_index_to_user_map[idx];
    std::cout << "User idx " << user_idx << std::endl;
    std::cout << "Number of User: " << m_users.size() << std::endl;
    const User& user = m_users[user_idx];
    std::cout << "User: " << user.name << std::endl;
    return user.name;
}
