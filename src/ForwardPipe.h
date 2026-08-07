#ifndef FORWARDPIPE_H_INCLUDED
#define FORWARDPIPE_H_INCLUDED
#include "config.h"
#include <memory>
#include <vector>
class ForwardPipe {
public:
    class ForwardPipeWeights {
    public:
        std::vector<std::vector<float>> m_conv_weights;
        std::vector<std::vector<float>> m_conv_biases;
        std::vector<std::vector<float>> m_batchnorm_means;
        std::vector<std::vector<float>> m_batchnorm_stddevs;
        std::vector<std::vector<float>> m_batchnorm_gammas;
        std::vector<std::vector<float>> m_batchnorm_betas;
        std::vector<float> m_bn_trunk_means;
        std::vector<float> m_bn_trunk_stddevs;
        std::vector<float> m_conv_pol_w;
        std::vector<float> m_conv_pol_b;
        std::vector<float> m_conv_val_w;
        std::vector<float> m_conv_val_b;
    };
    virtual ~ForwardPipe() = default;
    virtual void initialize(int channels) = 0;
    virtual bool needs_autodetect() { return false; };
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val) = 0;
    virtual void push_weights(unsigned int filter_size, unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights) = 0;
    virtual void drain() {}
    virtual void resume() {}
};
#endif
