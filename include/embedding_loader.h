#ifndef EMBEDDING_LOADER_H
#define EMBEDDING_LOADER_H

#include <string>
#include <vector>
#include <user.h>

namespace embedding_loader {

void load_embeddings_from_bin(const std::string& file_path, User& user);

void find_bin_files_recursive(const std::string& dir_path, std::vector<std::string>& bin_files);

std::vector<User> load_all_embeddings(const std::string& base_directory);

void l2_normalize(std::vector<float>& vec);

}

#endif // EMBEDDING_LOADER_H
