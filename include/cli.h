#ifndef CLI_H
#define CLI_H
#include <iostream>
#include <string>
#include <unordered_map>

namespace cli {

class Args {
    private:
    std::unordered_map<std::string, std::string> m_args; 

    public:
    Args(int argc, const char* argv[]);

    std::string get(std::string key, std::string default_value="") const; 

    size_t size() const;

    // auto begin() const -> decltype(m_args.begin());
    // auto end() const -> decltype(m_args.end());

};

void show_help(std::string binary_name);
void list(const Args &args);

}

#endif // CLI_H
