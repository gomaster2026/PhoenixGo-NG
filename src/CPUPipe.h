#ifndef CPUPipe_H_INCLUDED
#define CPUPipe_H_INCLUDED
#include "config.h"
#include <cassert>
#include <vector>
#include "ForwardPipe.h"
class CPUPipe : public ForwardPipe {
public:
    virtual void initialize(int channels);
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val);
    virtual void push_weights(unsigned int filter_size, unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights);
private:
    void winograd_transform_in(const std::vector<float>& in, std::vector<float>& V, int C);
    void winograd_sgemm(const std::vector<float>& U, const std::vector<float>& V,
                        std::vector<float>& M, int C, int K);
    void winograd_transform_out(const std::vector<float>& M, std::vector<float>& Y, int K);
    void winograd_convolve3(int outputs, const std::vector<float>& input,
                            const std::vector<float>& U, std::vector<float>& V,
                            std::vector<float>& M, std::vector<float>& output);
    int m_input_channels;
    std::shared_ptr<const ForwardPipeWeights> m_weights;
    std::vector<float> m_conv_pol_w;
    std::vector<float> m_conv_val_w;
    std::vector<float> m_conv_pol_b;
    std::vector<float> m_conv_val_b;
};
#endif
