#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace caesar
{
namespace lbrc_nvcomp_batched
{

struct Shape5D
{
    int64_t b = 1;
    int64_t c = 1;
    int64_t t = 1;
    int64_t h = 1;
    int64_t w = 1;
};

struct Options
{
    double targetNrmse = 1e-5;
    int64_t blockT = 60;
    int64_t blockH = 120;
    int64_t blockW = 120;
    int zstdLevel = 21;
    int quantIterations = 16;
    int workers = 1;
    int64_t latentBit = 0;
};

struct Block
{
    int64_t b = 0;
    int64_t c = 0;
    int64_t t0 = 0;
    int64_t t1 = 0;
    int64_t h0 = 0;
    int64_t h1 = 0;
    int64_t w0 = 0;
    int64_t w1 = 0;
    double step = 1.0;
    uint32_t bitCount = 1;
    double sse = 0.0;
    int64_t num = 0;
    std::vector<std::vector<uint8_t>> streams;
};

struct Package
{
    Shape5D shape;
    double targetNrmse = 1e-5;
    float mean = 0.0f;
    float scale = 1.0f;
    int64_t latentBit = 0;
    int64_t originalBytes = 0;
    int64_t blockT = 0;
    int64_t blockH = 0;
    int64_t blockW = 0;
    double encodedNrmse = 0.0;
    std::vector<Block> blocks;
};

struct EvalResult
{
    double finalNrmse = 0.0;
    double encodedNrmse = 0.0;
    double cr = 0.0;
    size_t correctionBytes = 0;
    size_t latentBytes = 0;
    size_t blocks = 0;
};

std::vector<float> loadFloat32File(const std::string& path, int64_t elements);
void saveFloat32File(const std::string& path, const std::vector<float>& data);
Shape5D parseShape5D(const std::string& text);
int64_t numElements(const Shape5D& shape);

Package encode(const std::vector<float>& original,
               const std::vector<float>& baseRecon,
               const Shape5D& shape,
               const Options& options);

std::vector<float> decode(const Package& package,
                          const std::vector<float>& baseRecon,
                          const Options& options);

EvalResult evaluate(const std::vector<float>& original,
                    const std::vector<float>& baseRecon,
                    const Shape5D& shape,
                    const Options& options);

size_t serializedSize(const Package& package);
double nrmse(const std::vector<float>& a,
             const std::vector<float>& b,
             double scale);

} // namespace lbrc_nvcomp_batched_nvcomp_batched
} // namespace caesar
