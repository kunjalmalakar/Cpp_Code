#pragma once

#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

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
    std::vector<Spot> detect(const std::string& imagePath);

private:
    Ort::Env env;
    Ort::Session session;
};