#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "Iir.h"
#include <DSPFilters/Dsp.h>

int main() {
    const double sampleRate = 48000.0;
    const double cutoffHz   = 1000.0;
    const int order         = 4;

    // Create filter object
    Dsp::FilterDesign<Dsp::Butterworth::LowPass<4>, 1> design;

    // Setup filter
    Dsp::Params params;
    params[0] = sampleRate;  // sample rate
    params[1] = cutoffHz;    // cutoff frequency
    design.setParams(params);

    auto& filter = design.getFilter();

    // Process one sample
    float x = 1.0f;
    float y = filter.process(x);

    // Process buffer
    std::vector<float> data = { /* your audio */ };
    filter.process(data.size(), data.data());

    return 0;
}

// Gaussian blur function
cv::Mat applyGaussianBlur(const cv::Mat& inputImage, int kernelSize = 5, double sigma = 0) {
    cv::Mat outputImage;
    cv::GaussianBlur(inputImage, outputImage, cv::Size(kernelSize, kernelSize), sigma);
    return outputImage;
}
