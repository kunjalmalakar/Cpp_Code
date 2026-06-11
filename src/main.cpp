#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

#include "../include/SpotDetector.h"
#include "../include/RFCalculator.h"
#include "../include/AUCCalculator.h"

struct FinalSpot {
    int id;
    double x1, y1, x2, y2;
    double confidence;
    double rf;
    double intensity;
    double auc;
};

struct CallbackData {
    cv::Mat image_display;
    std::vector<cv::Point>* points;
    std::string window_name;
};

void mouse_click(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        CallbackData* data = static_cast<CallbackData*>(userdata);
        data->points->push_back(cv::Point(x, y));
        
        // Draw a visual marker for the clicked spot
        cv::circle(data->image_display, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(data->image_display, cv::Point(x, y), 20, cv::Scalar(0, 0, 255), 1);
        cv::imshow(data->window_name, data->image_display);
    }
}

// Convert a string to a wide string on Windows
std::wstring to_wstring(const std::string& str) {
    std::wstring wstr(str.begin(), str.end());
    return wstr;
}

int main(int argc, char* argv[]) {
    try {
        std::string img_path = "band3(1).png";
        std::string model_path = "best_spot5.onnx";
        bool headless = false;
        bool verbose = false;
        std::string manual_spots_str = "";
        std::string plot_output_path = "densitogram.png";

        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--headless") {
                headless = true;
            } else if (arg == "--verbose") {
                verbose = true;
            } else if (arg == "--manual" && i + 1 < argc) {
                manual_spots_str = argv[++i];
            } else if (arg == "--output-plot" && i + 1 < argc) {
                plot_output_path = argv[++i];
            } else {
                if (arg.rfind("--", 0) != 0) {
                    static int positional_count = 0;
                    if (positional_count == 0) {
                        img_path = arg;
                        positional_count++;
                    } else if (positional_count == 1) {
                        model_path = arg;
                        positional_count++;
                    }
                }
            }
        }

        if (verbose) {
            std::cout << "Loading image: " << img_path << std::endl;
            std::cout << "Loading model: " << model_path << std::endl;
            std::cout << "Headless mode: " << (headless ? "ENABLED" : "DISABLED") << std::endl;
        }

        cv::Mat image = cv::imread(img_path);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image: " << img_path << std::endl;
            return -1;
        }

        // 1. Run Spot Detection using ONNX Runtime
        std::wstring w_model_path = to_wstring(model_path);
        SpotDetector detector(w_model_path);
        std::vector<Spot> initial_spots = detector.detect(img_path);
        if (verbose) {
            std::cout << "Initial spots detected: " << initial_spots.size() << std::endl;
        }

        // 2. Interactive Manual Spot Selection or CLI injection
        std::vector<cv::Point> clicked_points;

        // Parse CLI manual spots: formatted as "x1,y1;x2,y2;..."
        if (!manual_spots_str.empty()) {
            std::stringstream ss(manual_spots_str);
            std::string item;
            while (std::getline(ss, item, ';')) {
                if (item.empty()) continue;
                std::stringstream pts(item);
                std::string sx, sy;
                if (std::getline(pts, sx, ',') && std::getline(pts, sy, ',')) {
                    try {
                        int x = std::stoi(sx);
                        int y = std::stoi(sy);
                        clicked_points.push_back(cv::Point(x, y));
                        if (verbose) {
                            std::cout << "Injected manual spot at: " << x << ", " << y << std::endl;
                        }
                    } catch (...) {
                        // Ignore invalid coordinates
                    }
                }
            }
        }

        if (!headless) {
            std::string win_name = "Select Missing Spots";
            CallbackData cb_data;
            cb_data.image_display = image.clone();
            cb_data.points = &clicked_points;
            cb_data.window_name = win_name;

            // Draw existing bounding boxes on the selection display image
            for (const auto& s : initial_spots) {
                cv::rectangle(cb_data.image_display, 
                             cv::Point((int)s.x1, (int)s.y1), 
                             cv::Point((int)s.x2, (int)s.y2), 
                             cv::Scalar(0, 255, 0), 2);
            }

            cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
            cv::setMouseCallback(win_name, mouse_click, &cb_data);
            cv::imshow(win_name, cb_data.image_display);
            
            std::cout << "Click on missing spots in the display window, then press ANY key on the keyboard to finish..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        // Combine auto-detected spots and manually selected spots
        std::vector<Spot> combined_spots = initial_spots;
        int box_size = 20;
        for (const auto& pt : clicked_points) {
            Spot manual_spot;
            manual_spot.x1 = (float)(pt.x - box_size);
            manual_spot.y1 = (float)(pt.y - box_size);
            manual_spot.x2 = (float)(pt.x + box_size);
            manual_spot.y2 = (float)(pt.y + box_size);
            manual_spot.confidence = 1.0f;
            manual_spot.cls = 0;
            combined_spots.push_back(manual_spot);
        }

        // 3. Filtration system to remove ghost spots
        std::vector<Spot> filtered_spots;
        double lane_width = image.cols;
        double lane_height = image.rows;
        double lane_center = lane_width / 2.0;

        for (const auto& spot : combined_spots) {
            double box_width = spot.x2 - spot.x1;
            double box_height = spot.y2 - spot.y1;
            double area = box_width * box_height;
            double center_x = (spot.x1 + spot.x2) / 2.0;

            if (verbose) {
                std::cout << "Candidate Spot: x1=" << spot.x1 << ", y1=" << spot.y1 
                          << ", x2=" << spot.x2 << ", y2=" << spot.y2 
                          << ", area=" << area << ", center_x=" << center_x 
                          << ", lane_center=" << lane_center << ", lane_width=" << lane_width 
                          << ", conf=" << spot.confidence << std::endl;
            }

            // Area filter
            if (area < 20 || area > 10000) {
                if (verbose) {
                    std::cout << "  -> Filtered out by Area: " << area << std::endl;
                }
                continue;
            }

            // Positioning filter to remove surrounding spots
            if (std::abs(center_x - lane_center) > lane_width * 0.45) {
                if (verbose) {
                    std::cout << "  -> Filtered out by Positioning: center_x diff=" << std::abs(center_x - lane_center) 
                              << " max allowed=" << (lane_width * 0.45) << std::endl;
                }
                continue;
            }

            // Confidence straining
            if (spot.confidence < 0.0009) {
                if (verbose) {
                    std::cout << "  -> Filtered out by Confidence: " << spot.confidence << std::endl;
                }
                continue;
            }

            if (verbose) {
                std::cout << "  -> Keep!" << std::endl;
            }
            filtered_spots.push_back(spot);
        }
        if (verbose) {
            std::cout << "Spots after filtration: " << filtered_spots.size() << std::endl;
        }

        // 4. Calculate intensities and Rf values
        std::vector<FinalSpot> final_spots;
        for (const auto& spot : filtered_spots) {
            int ix1 = std::max(0, std::min((int)spot.x1, image.cols - 1));
            int iy1 = std::max(0, std::min((int)spot.y1, image.rows - 1));
            int ix2 = std::max(0, std::min((int)spot.x2, image.cols));
            int iy2 = std::max(0, std::min((int)spot.y2, image.rows));
            
            int w = ix2 - ix1;
            int h = iy2 - iy1;
            
            double intensity = 0.0;
            if (w > 0 && h > 0) {
                cv::Mat spot_crop = image(cv::Rect(ix1, iy1, w, h));
                cv::Mat gray;
                cv::cvtColor(spot_crop, gray, cv::COLOR_BGR2GRAY);
                cv::Scalar mean_val = cv::mean(255 - gray);
                intensity = mean_val[0];
            }

            double spot_center_y = (spot.y1 + spot.y2) / 2.0;
            double rf = spot_center_y / lane_height;

            FinalSpot fs;
            fs.id = 0; // Will assign after sorting
            fs.x1 = spot.x1;
            fs.y1 = spot.y1;
            fs.x2 = spot.x2;
            fs.y2 = spot.y2;
            fs.confidence = spot.confidence;
            fs.rf = rf;
            fs.intensity = intensity;
            fs.auc = 0.0;

            final_spots.push_back(fs);
        }

        // Sort by Rf ascending
        std::sort(final_spots.begin(), final_spots.end(), [](const FinalSpot& a, const FinalSpot& b) {
            return a.rf < b.rf;
        });

        // 5. Generate chromatogram peaks and calculate AUC
        double max_intensity = 0.0;
        for (size_t i = 0; i < final_spots.size(); ++i) {
            final_spots[i].id = (int)(i + 1);
            
            std::vector<double> peak_x;
            std::vector<double> peak_y;
            AUCCalculator::generate_peak_data(final_spots[i].rf, final_spots[i].intensity, peak_x, peak_y);
            final_spots[i].auc = AUCCalculator::calculate_auc(peak_x, peak_y);

            if (final_spots[i].intensity > max_intensity) {
                max_intensity = final_spots[i].intensity;
            }
        }
        if (max_intensity == 0.0) {
            max_intensity = 255.0;
        }

        // 6. Draw Densitogram Plot using OpenCV
        int plot_w = 800;
        int plot_h = 500;
        cv::Mat plot(plot_h, plot_w, CV_8UC3, cv::Scalar(255, 255, 255));

        int margin_l = 80;
        int margin_r = 40;
        int margin_t = 50;
        int margin_b = 60;

        int graph_w = plot_w - margin_l - margin_r;
        int graph_h = plot_h - margin_t - margin_b;

        // Draw background grid lines (horizontal and vertical)
        // Horizontal grid
        for (int i = 0; i <= 4; ++i) {
            double ratio = i / 4.0;
            int y = plot_h - margin_b - (int)(ratio * graph_h);
            cv::line(plot, cv::Point(margin_l, y), cv::Point(plot_w - margin_r, y), cv::Scalar(230, 230, 230), 1);
            
            // Y Label
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << (ratio * max_intensity);
            cv::putText(plot, ss.str(), cv::Point(10, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(50, 50, 50), 1);
        }

        // Vertical grid
        for (int i = 0; i <= 5; ++i) {
            double ratio = i / 5.0;
            int x = margin_l + (int)(ratio * graph_w);
            cv::line(plot, cv::Point(x, margin_t), cv::Point(x, plot_h - margin_b), cv::Scalar(230, 230, 230), 1);

            // X Label
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << ratio;
            cv::putText(plot, ss.str(), cv::Point(x - 10, plot_h - margin_b + 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(50, 50, 50), 1);
        }

        // Draw axes lines
        cv::line(plot, cv::Point(margin_l, plot_h - margin_b), cv::Point(plot_w - margin_r, plot_h - margin_b), cv::Scalar(0, 0, 0), 2);
        cv::line(plot, cv::Point(margin_l, margin_t), cv::Point(margin_l, plot_h - margin_b), cv::Scalar(0, 0, 0), 2);

        // Titles and axis labels
        cv::putText(plot, "TLC Densitogram", cv::Point(plot_w / 2 - 80, margin_t - 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(plot, "Rf Value", cv::Point(plot_w / 2 - 30, plot_h - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
        cv::putText(plot, "Intensity", cv::Point(margin_l - 15, margin_t - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // Draw peaks and fill areas
        for (const auto& spot : final_spots) {
            std::vector<double> peak_x;
            std::vector<double> peak_y;
            AUCCalculator::generate_peak_data(spot.rf, spot.intensity, peak_x, peak_y);

            std::vector<cv::Point> pts;
            pts.reserve(peak_x.size() + 2);

            // First point on the baseline
            double first_x = peak_x.front();
            int first_px = margin_l + (int)((first_x - 0.0) / 1.0 * graph_w);
            pts.push_back(cv::Point(first_px, plot_h - margin_b));

            for (size_t j = 0; j < peak_x.size(); ++j) {
                int px = margin_l + (int)((peak_x[j] - 0.0) / 1.0 * graph_w);
                int py = plot_h - margin_b - (int)((peak_y[j] - 0.0) / max_intensity * graph_h);
                px = std::max(margin_l, std::min(px, plot_w - margin_r));
                py = std::max(margin_t, std::min(py, plot_h - margin_b));
                pts.push_back(cv::Point(px, py));
            }

            // Last point on the baseline
            double last_x = peak_x.back();
            int last_px = margin_l + (int)((last_x - 0.0) / 1.0 * graph_w);
            pts.push_back(cv::Point(last_px, plot_h - margin_b));

            // Fill area (semi-transparent pinkish-red)
            std::vector<std::vector<cv::Point>> fill_pts = { pts };
            cv::fillPoly(plot, fill_pts, cv::Scalar(220, 220, 255));

            // Draw peak line (red)
            std::vector<cv::Point> line_pts(pts.begin() + 1, pts.end() - 1);
            cv::polylines(plot, line_pts, false, cv::Scalar(0, 0, 255), 2);
        }

        // Save plot image
        cv::imwrite(plot_output_path, plot);
        if (verbose) {
            std::cout << "Saved densitogram plot to: " << plot_output_path << std::endl;
        }

        // 7. Output Results in tabular and JSON format
        std::cout << "\n================= TLC DENSITOMETRY REPORT =================\n";
        std::printf("%-8s %-10s %-12s %-10s\n", "Spot #", "Rf Value", "Intensity", "AUC");
        for (const auto& spot : final_spots) {
            std::printf("%-8d %-10.3f %-12.3f %-10.4f\n", spot.id, spot.rf, spot.intensity, spot.auc);
        }

        // Print structured JSON for frontend integration
        std::cout << "\nJSON_OUTPUT_START\n";
        std::cout << "{\n";
        std::cout << "  \"spots\": [\n";
        for (size_t i = 0; i < final_spots.size(); ++i) {
            const auto& spot = final_spots[i];
            std::cout << "    {\n";
            std::cout << "      \"id\": " << spot.id << ",\n";
            std::cout << "      \"rf\": " << spot.rf << ",\n";
            std::cout << "      \"intensity\": " << spot.intensity << ",\n";
            std::cout << "      \"auc\": " << spot.auc << ",\n";
            std::cout << "      \"confidence\": " << spot.confidence << ",\n";
            std::cout << "      \"box\": [" << spot.x1 << ", " << spot.y1 << ", " << spot.x2 << ", " << spot.y2 << "]\n";
            std::cout << "    }" << (i == final_spots.size() - 1 ? "" : ",") << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
        std::cout << "JSON_OUTPUT_END\n";

        // Display the densitogram plot if NOT headless
        if (!headless) {
            cv::imshow("TLC Densitogram", plot);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
}