#include <embedding_loader.h>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <dirent.h>  // POSIX library for directory traversal
#include <sys/types.h> // For dirent.h (POSIX)

namespace embedding_loader {

void load_embeddings_from_bin(const std::string& file_path, User& user) {
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("Failed to open file");

    struct stat sb;
    if (fstat(fd, &sb) == -1) throw std::runtime_error("Failed to stat file");

    size_t filesize = sb.st_size;
    char* data = static_cast<char*>(mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fd, 0));
    close(fd);

    if (data == MAP_FAILED) throw std::runtime_error("mmap failed");

    size_t offset = 0;

    // Read id
    std::memcpy(&user.id, data + offset, sizeof(int));
    offset += sizeof(int);

    // Read name
    int name_length;
    std::memcpy(&name_length, data + offset, sizeof(int));
    offset += sizeof(int);

    user.name.assign(data + offset, name_length);
    offset += name_length;

    // Read num_embeddings and embedding_size
    int num_embeddings;
    std::memcpy(&num_embeddings, data + offset, sizeof(int));
    offset += sizeof(int);

    std::memcpy(&user.embedding_size, data + offset, sizeof(int));
    offset += sizeof(int);

    // Read float embeddings (zero-copy)
    float* float_data = reinterpret_cast<float*>(data + offset);
    size_t total_floats = static_cast<size_t>(num_embeddings) * user.embedding_size;

    // Copy into vector (you can also avoid this if just passing pointer to FAISS)
    user.embeddings.assign(float_data, float_data + total_floats);

    // Unmap
    munmap(data, filesize);
}

// Find all .bin files recursively in a directory
void find_bin_files_recursive(const std::string& dir_path, std::vector<std::string>& bin_files) {
    DIR* dir = opendir(dir_path.c_str());
    if (dir == nullptr) {
        std::cerr << "Failed to open directory: " << dir_path << std::endl;
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        
        // Skip . and .. directories
        if (name == "." || name == "..") {
            continue;
        }
        
        std::string full_path = dir_path + "/" + name;
        
        // Check if it's a directory
        if (entry->d_type == DT_DIR) {
            // Recursive call for subdirectories
            find_bin_files_recursive(full_path, bin_files);
        } 
        // Check if it's a .bin file
        else if (entry->d_type == DT_REG && name.size() > 4 && 
                 name.substr(name.size() - 4) == ".bin") {
            bin_files.push_back(full_path);
        }
    }
    
    closedir(dir);
}

std::vector<User> load_all_embeddings(const std::string& base_directory) {
    std::vector<User> all_users;
    
    // Find all .bin files recursively
    std::vector<std::string> bin_files;
    find_bin_files_recursive(base_directory, bin_files);
    
    std::cout << "Found " << bin_files.size() << " .bin files" << std::endl;
    
    // Load each .bin file
    for (const auto& file_path : bin_files) {
        try {
            User user;
            load_embeddings_from_bin(file_path, user);
            std::cout << "Loaded embedding from: " << file_path 
                      << " (ID: " << user.id 
                      << ", Name: " << user.name << ")" << std::endl;
            all_users.push_back(std::move(user));
        } catch (const std::exception& e) {
            std::cerr << "Failed to load embedding file " << file_path << ": " << e.what() << std::endl;
        }
    }
    
    return all_users;
}
void l2_normalize(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float v: vec) norm += v * v;
    norm = std::sqrt(norm);
    for (float &v: vec) v /= norm;
}
} // namespace embedding_loader
