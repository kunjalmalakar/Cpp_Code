#include "../include/SpotDetector.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <iostream>

SpotDetector::SpotDetector(const std::wstring& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "SpotDetector"),
      session(env, modelPath.c_str(), Ort::SessionOptions{})
{
}

std::vector<Spot> SpotDetector::detect(const cv::Mat& image, float confThreshold, float iouThreshold)
{
    std::vector<Spot> detections;

    if (image.empty())
    {
        std::cerr << "Error: Input image is empty" << std::endl;
        return detections;
    }

    int orig_w = image.cols;
    int orig_h = image.rows;

    // 1. YOLOv8 Letterbox Preprocessing
    float r = std::min(640.0f / orig_w, 640.0f / orig_h);
    int unpad_w = (int)std::round(orig_w * r);
    int unpad_h = (int)std::round(orig_h * r);

    int pad_w = 640 - unpad_w;
    int pad_h = 640 - unpad_h;

    float dw = (float)pad_w / 2.0f;
    float dh = (float)pad_h / 2.0f;

    int top = (int)std::round(dh - 0.1f);
    int bottom = (int)std::round(dh + 0.1f);
    int left = (int)std::round(dw - 0.1f);
    int right = (int)std::round(dw + 0.1f);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(unpad_w, unpad_h));
    cv::copyMakeBorder(resized, resized, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> inputTensorValues(3 * 640 * 640);
    for (int c = 0; c < 3; c++)
    {
        for (int y = 0; y < 640; y++)
        {
            for (int x = 0; x < 640; x++)
            {
                inputTensorValues[c * 640 * 640 + y * 640 + x] = resized.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    std::vector<int64_t> inputShape = { 1, 3, 640, 640 };

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorValues.size(),
        inputShape.data(),
        inputShape.size()
    );

    const char* inputNames[] = { "images" };
    const char* outputNames[] = { "output0" };

    auto outputTensors = session.Run(
        Ort::RunOptions{ nullptr },
        inputNames,
        &inputTensor,
        1,
        outputNames,
        1
    );

    float* output = outputTensors[0].GetTensorMutableData<float>();

    // Dynamically retrieve output tensor shape to handle arbitrary classes (nc)
    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputShape = outputTensorInfo.GetShape();

    // For YOLOv8, output shape is typically {1, 4 + nc, 8400}
    int64_t num_channels = outputShape[1];
    int64_t num_anchors = outputShape[2];

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int i = 0; i < num_anchors; ++i)
    {
        float cx = output[0 * num_anchors + i];
        float cy = output[1 * num_anchors + i];
        float w  = output[2 * num_anchors + i];
        float h  = output[3 * num_anchors + i];

        float max_score = 0.0f;
        int class_id = 0;
        for (int c = 4; c < num_channels; ++c)
        {
            float score = output[c * num_anchors + i];
            if (score > max_score)
            {
                max_score = score;
                class_id = c - 4;
            }
        }

        if (max_score >= confThreshold)
        {
            // Scale back coordinates: subtract padding and divide by ratio
            float cx_orig = (cx - left) / r;
            float cy_orig = (cy - top) / r;
            float w_orig  = w / r;
            float h_orig  = h / r;

            float x1 = cx_orig - w_orig / 2.0f;
            float y1 = cy_orig - h_orig / 2.0f;

            int ix1 = std::max(0, std::min((int)x1, orig_w - 1));
            int iy1 = std::max(0, std::min((int)y1, orig_h - 1));
            int iw  = std::max(1, std::min((int)w_orig, orig_w - ix1));
            int ih  = std::max(1, std::min((int)h_orig, orig_h - iy1));

            bboxes.push_back(cv::Rect(ix1, iy1, iw, ih));
            confidences.push_back(max_score);
            classIds.push_back(class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confidences, confThreshold, iouThreshold, indices);

    for (int idx : indices)
    {
        Spot s;
        s.x1 = (float)bboxes[idx].x;
        s.y1 = (float)bboxes[idx].y;
        s.x2 = (float)(bboxes[idx].x + bboxes[idx].width);
        s.y2 = (float)(bboxes[idx].y + bboxes[idx].height);
        s.confidence = confidences[idx];
        s.cls = classIds[idx];
        detections.push_back(s);
    }

    return detections;
}