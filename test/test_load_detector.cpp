#include <torch/torch.h>
#include <torchvision/vision.h>
#include <utils/cli.h>

int main(int argc, const char* argv[]) {
    if (argc <= 2 ) {
        cli::show_help(argv[0]);
        return -1; // Fail state
    }
    cli::parse(argc, argv);
    return 0;
}
