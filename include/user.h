#ifndef USER_H
#define USER_H

struct User {
    int id;
    std::string name;
    int embedding_size;
    std::vector<float> embeddings;
};

#endif // USER_H
