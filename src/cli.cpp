#include <cli.h>


namespace cli {


Args::Args(int argc, const char* argv[]) {
    m_args["binary_name"] = argv[0];
    for (int i = 1; i + 1 <= argc; i += 2) {
        std::string key{argv[i]};
        std::string value{argv[i + 1]};

        m_args[key] = value;
    }
}


size_t Args::size() const {
    return m_args.size();
}

std::string Args::get(std::string key, std::string default_value) const {
    auto it = m_args.find(std::string(key));
    return (it != m_args.end()) ? it->second : std::string(default_value);
}

void show_help(std::string binary_name) {
    std::cout << "Usage: " << binary_name << " [--model <path of your model>] [--output <where to save your output>]"<< std::endl;
}

void list(const Args &args) {
    if (args.size() <= 2 ) {
        cli::show_help(args.get("binary_name"));
    }

}
// auto Args::begin() const -> decltype(m_args.begin()) {
//     return m_args.begin();
// };
//
// auto Args::end() const -> decltype(m_args.end()) {
//     return m_args.end();
// };

} // namespace cli
