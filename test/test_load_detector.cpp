#include <torch/torch.h>
#include <torchvision/vision.h>
#include <utils/cli.h>

int main(int argc, const char* argv[]) {
    cli::Args args(argc, argv);
    cli::list(args);
    std::string binary_name = args.get("binary_name");
    std::string wrong_key = args.get("???", "khong co");
    std::cout << binary_name << std::endl;
    std::cout << wrong_key << std::endl;
    return 0;
}
