#include "models/caesar_compress.h"
#include "data_utils.h"

#include <torch/torch.h>

#include <chrono>
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
    const int64_t n = numel(shape);
    std::vector<float> buf((size_t)n);

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("cannot open input: " + path);

    f.read(reinterpret_cast<char*>(buf.data()), n * sizeof(float));
    if (!f) throw std::runtime_error("failed to read expected float32 data from: " + path);

    return torch::from_blob(buf.data(), torch::IntArrayRef(shape), torch::kFloat32).clone();
}

int main(int argc, char** argv)
{
    try {
        std::string input;
        std::string shape_text;
        float nrmse = 1e-5f;
        int batch_size = 32;
        int n_frame = 8;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--input") input = need_value(i, argc, argv);
            else if (arg == "--shape") shape_text = need_value(i, argc, argv);
            else if (arg == "--nrmse") nrmse = std::stof(need_value(i, argc, argv));
            else if (arg == "--batch-size") batch_size = std::stoi(need_value(i, argc, argv));
            else if (arg == "--n-frame") n_frame = std::stoi(need_value(i, argc, argv));
            else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: caesar_gae_bench --input original.bin --shape B,C,T,H,W --nrmse 1e-5\n";
                return 0;
            }
            else throw std::runtime_error("unknown argument: " + arg);
        }

        if (input.empty() || shape_text.empty()) throw std::runtime_error("--input and --shape are required");

        std::vector<int64_t> shape = parse_shape(shape_text);
        torch::Tensor raw = load_f32(input, shape);

        PaddingInfo padding_info;
        torch::Tensor raw_5d;
        bool force_padding = false;
        torch::Tensor squeezed = raw.squeeze();
        std::tie(raw_5d, padding_info) = to_5d_and_pad(squeezed, shape[3], shape[4], force_padding);

        torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

        DatasetConfig config;
        config.memory_data = raw_5d;
        config.variable_idx = 0;
        config.n_frame = n_frame;
        config.dataset_name = "GAE benchmark";
        config.section_range = std::nullopt;
        config.frame_range = std::nullopt;
        config.train_size = 256;
        config.inst_norm = true;
        config.norm_type = "mean_range";
        config.train_mode = false;
        config.n_overlap = 0;
        config.test_size = {256, 256};
        config.augment_type = {};

        Compressor compressor(device);

        auto t0 = std::chrono::high_resolution_clock::now();
        CompressionResult result = compressor.compress(config, batch_size, nrmse);
        auto t1 = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(t1 - t0).count();

        size_t latent_bytes = 0;
        for (const auto& s : result.encoded_latents) latent_bytes += s.size();
        for (const auto& s : result.encoded_hyper_latents) latent_bytes += s.size();

        double original_bytes = (double)numel(shape) * sizeof(float);
        double cr = latent_bytes ? original_bytes / (double)latent_bytes : 0.0;

        std::cout << "Device: " << (device.type() == torch::kCUDA ? "CUDA" : "CPU") << "\n";
        std::cout << "Target NRMSE: " << nrmse << "\n";
        std::cout << "GAE time: " << seconds << " s\n";
        std::cout << "Latent bytes: " << latent_bytes << "\n";
        std::cout << "GAE CR latent_only: " << cr << "\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
