#include "models/lbrc_nvcomp_zstd_batched.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void usage()
{
    std::cout
        << "CAESAR LBRC batched nvCOMP Zstd residual evaluator\n\n"
        << "Usage:\n"
        << "  caesar_lbrc --original original.bin --recons base_recon.bin --shape B,C,T,H,W [options]\n\n"
        << "Options:\n"
        << "  --nrmse <val>       Target block NRMSE (default 1e-5)\n"
        << "  --latent-bit <n>    Base CAESAR latent bits for total CR (default 0)\n"
        << "  --block-t <n>       Temporal block size (default 60)\n"
        << "  --block-h <n>       Block height (default 120)\n"
        << "  --block-w <n>       Block width (default 120)\n"
        << "  --level <n>         Zstd compression level (default 21)\n"
        << "  --quant-iter <n>    Binary search iterations (default 16)\n"
        << "  --workers <n>       CPU workers, <=0 uses hardware concurrency (default 16)\n"
        << "  -h, --help          Show this message\n";
}

std::string requireValue(int& i, int argc, char** argv)
{
    if (i + 1 >= argc)
        throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string originalPath;
        std::string reconsPath;
        std::string shapeText;
        caesar::lbrc_nvcomp_zstd_batched::Options options;
        options.workers = 16;

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                usage();
                return 0;
            }
            if (arg == "--original")
                originalPath = requireValue(i, argc, argv);
            else if (arg == "--recons")
                reconsPath = requireValue(i, argc, argv);
            else if (arg == "--shape")
                shapeText = requireValue(i, argc, argv);
            else if (arg == "--nrmse")
                options.targetNrmse = std::stod(requireValue(i, argc, argv));
            else if (arg == "--latent-bit")
                options.latentBit = std::stoll(requireValue(i, argc, argv));
            else if (arg == "--block-t")
                options.blockT = std::stoll(requireValue(i, argc, argv));
            else if (arg == "--block-h")
                options.blockH = std::stoll(requireValue(i, argc, argv));
            else if (arg == "--block-w")
                options.blockW = std::stoll(requireValue(i, argc, argv));
            else if (arg == "--level")
                options.zstdLevel = std::stoi(requireValue(i, argc, argv));
            else if (arg == "--quant-iter")
                options.quantIterations = std::stoi(requireValue(i, argc, argv));
            else if (arg == "--workers")
                options.workers = std::stoi(requireValue(i, argc, argv));
            else
                throw std::runtime_error("unknown argument: " + arg);
        }

        if (originalPath.empty() || reconsPath.empty() || shapeText.empty())
        {
            usage();
            return 2;
        }

        const caesar::lbrc_nvcomp_zstd_batched::Shape5D shape = caesar::lbrc_nvcomp_zstd_batched::parseShape5D(shapeText);
        const int64_t elements = caesar::lbrc_nvcomp_zstd_batched::numElements(shape);
        std::vector<float> original = caesar::lbrc_nvcomp_zstd_batched::loadFloat32File(originalPath, elements);
        std::vector<float> recons = caesar::lbrc_nvcomp_zstd_batched::loadFloat32File(reconsPath, elements);

        const auto start = std::chrono::steady_clock::now();
        const caesar::lbrc_nvcomp_zstd_batched::EvalResult result =
            caesar::lbrc_nvcomp_zstd_batched::evaluate(original, recons, shape, options);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = end - start;

        std::cout << "Target NRMSE: " << options.targetNrmse << "\n";
        std::cout << "Final NRMSE: " << result.finalNrmse << "\n";
        std::cout << "Encoded NRMSE: " << result.encodedNrmse << "\n";
        std::cout << "CR: " << result.cr << "\n";
        std::cout << "Correction bytes: " << result.correctionBytes << "\n";
        std::cout << "Latent bytes: " << result.latentBytes << "\n";
        std::cout << "Blocks: " << result.blocks << "\n";
        std::cout << "LBRC time: " << elapsed.count() << " s\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
