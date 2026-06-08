#include "models/runGaeCuda.h"

#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string need_value(int& i, int argc, char** argv)
{
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

static std::vector<int64_t> parse_shape(const std::string& s)
{
    std::vector<int64_t> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        std::string part = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        out.push_back(std::stoll(part));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (out.size() != 5) throw std::runtime_error("--shape must be B,C,T,H,W");
    return out;
}

static int64_t numel(const std::vector<int64_t>& shape)
{
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

static torch::Tensor load_f32(const std::string& path, const std::vector<int64_t>& shape)
{
    int64_t n = numel(shape);
    std::vector<float> buf((size_t)n);

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("cannot open file: " + path);

    f.read(reinterpret_cast<char*>(buf.data()), n * sizeof(float));
    if (!f) throw std::runtime_error("failed to read expected float32 data: " + path);

    return torch::from_blob(buf.data(), torch::IntArrayRef(shape), torch::kFloat32).clone();
}

static double nrmse(const torch::Tensor& a, const torch::Tensor& b)
{
    torch::Tensor diff = (a - b).to(torch::kFloat64);
    return std::sqrt(diff.pow(2).mean().item<double>());
}

int main(int argc, char** argv)
{
    try {
        std::string original_path;
        std::string recons_path;
        std::string shape_text;
        double target_nrmse = 1e-5;
        int64_t latent_bit = 0;
        std::string codec = "Zstd";
        std::string device = torch::cuda::is_available() ? "cuda" : "cpu";

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--original") original_path = need_value(i, argc, argv);
            else if (arg == "--recons") recons_path = need_value(i, argc, argv);
            else if (arg == "--shape") shape_text = need_value(i, argc, argv);
            else if (arg == "--nrmse") target_nrmse = std::stod(need_value(i, argc, argv));
            else if (arg == "--latent-bit") latent_bit = std::stoll(need_value(i, argc, argv));
            else if (arg == "--codec") codec = need_value(i, argc, argv);
            else if (arg == "--device") device = need_value(i, argc, argv);
            else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: runGAE --original original.bin --recons recons.bin --shape B,C,T,H,W --nrmse 1e-5 --latent-bit N\n";
                return 0;
            }
            else throw std::runtime_error("unknown argument: " + arg);
        }

        if (original_path.empty() || recons_path.empty() || shape_text.empty())
            throw std::runtime_error("--original, --recons, and --shape are required");

        std::vector<int64_t> shape = parse_shape(shape_text);
        torch::Tensor original = load_f32(original_path, shape);
        torch::Tensor recons = load_f32(recons_path, shape);

        torch::Tensor original_cpu = original.to(torch::kCPU);
        float mean = original_cpu.mean().item<float>();
        float xmin = original_cpu.min().item<float>();
        float xmax = original_cpu.max().item<float>();
        float range = xmax - xmin;
        if (range == 0.0f) range = 1.0f;

        torch::Tensor original_norm = (original - mean) / range;
        torch::Tensor recons_norm = (recons - mean) / range;

        double init_nrmse = nrmse(original_norm, recons_norm);
        double original_gib = (double)original.numel() * original.element_size() / (1024.0 * 1024.0 * 1024.0);

        PCACompressor compressor(target_nrmse, 2.0, device, codec, {8, 8});

        auto t0 = std::chrono::high_resolution_clock::now();
        GAECompressionResult result = compressor.compress(original_norm, recons_norm);
        auto t1 = std::chrono::high_resolution_clock::now();

        torch::Tensor recons_gae;
        if (result.dataBytes > 0) {
            recons_gae = compressor.decompress(recons_norm, result.metaData, *result.compressedData);
        } else {
            recons_gae = recons_norm;
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        double enc_s = std::chrono::duration<double>(t1 - t0).count();
        double dec_s = std::chrono::duration<double>(t2 - t1).count();
        double total_s = std::chrono::duration<double>(t2 - t0).count();

        double final_nrmse = nrmse(original_norm, recons_gae);
        double compressed_bytes = (double)result.dataBytes + (double)latent_bit / 8.0;
        double cr = compressed_bytes > 0.0 ? ((double)original.numel() * original.element_size()) / compressed_bytes : 0.0;

        std::cout << "Device: " << device << "\n";
        std::cout << "Codec: " << codec << "\n";
        std::cout << "Target NRMSE: " << target_nrmse << "\n";
        std::cout << "Initial NRMSE: " << init_nrmse << "\n";
        std::cout << "Final NRMSE: " << final_nrmse << "\n";
        std::cout << "Correction bytes: " << result.dataBytes << "\n";
        std::cout << "Latent bits: " << latent_bit << "\n";
        std::cout << "CR: " << cr << "\n";
        std::cout << "Encoding time: " << enc_s << " s\n";
        std::cout << "Decoding time: " << dec_s << " s\n";
        std::cout << "Total time: " << total_s << " s\n";
        std::cout << "Encoding GB/s: " << original_gib / enc_s << "\n";
        std::cout << "Decoding GB/s: " << original_gib / dec_s << "\n";
        std::cout << "Overall GB/s: " << original_gib / total_s << "\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
