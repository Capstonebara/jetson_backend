#include <utils/cli.h>

void cli::show_help(std::string_view binary_name) {
    std::cout << "Usage: " << binary_name << " [--model <path of your model>] [--output <where to save your output>]"<< std::endl;
}

void cli::list(int argc, const char* argv[]) {
    for (int i = 0; i < argc; ++i) {
        std::string key{argv[i]};
        std::cout << "argv[" << i << "] = " << argv[i] << std::endl;
    }
}
