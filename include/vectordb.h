#ifndef VECTORDB_H
#define VECTORDB_H

#include <unordered_map>
#include <faiss/IndexFlat.h>
#include <user.h>
#include <utility>

class VectorDB {
    std::vector<User> m_users;
    int m_embedding_size;
    faiss::IndexFlatIP m_index;
    std::unordered_map<int, int> m_index_to_user_map;
public:
    VectorDB(const std::string &db_path, const int embedding_size=128);
    std::pair<faiss::idx_t, float> search(float *query_ptr) const;
    std::string findName(faiss::idx_t idx);
    int getUserID(faiss::idx_t idx);

    int getRow() const;
    int getColumn() const;
};

#endif // VECTORDB_H
