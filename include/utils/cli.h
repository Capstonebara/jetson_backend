#ifndef CLI_H
#define CLI_H
#include <iostream>
#include <string>

namespace cli {
    void list(int argc, const char* argv[]);
    void show_help(std::string_view binary_name);
}

#endif // CLI_H
