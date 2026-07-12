#pragma once

#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

struct Spot
{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int cls;
};

class SpotDetector
{
public:
    SpotDetector(const std::wstring& modelPath);
    std::vector<Spot> detect(const cv::Mat& image, float confThreshold = 0.0009f, float iouThreshold = 0.45f);

private:
    Ort::Env env;
    Ort::Session session;
};