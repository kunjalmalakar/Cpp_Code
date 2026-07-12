#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <map>

#include "../include/SpotDetector.h"
#include "../include/RFCalculator.h"
#include "../include/AUCCalculator.h"

struct FinalSpot {
    int id; // 1-indexed within the lane
    double x1, y1, x2, y2; // relative coordinates (in lane crop)
    double abs_x1, abs_y1, abs_x2, abs_y2; // absolute coordinates (in plate image)
    double confidence;
    double rf;
    double intensity;
    double auc;
};

struct Lane {
    int id; // 1-indexed from left to right
    double x1, y1, x2, y2; // coordinates in the plate image
    double confidence;
    cv::Mat crop;
    std::vector<FinalSpot> spots;
};

struct ClickCallbackData {
    cv::Mat image_display;
    std::vector<cv::Point>* points;
    std::string window_name;
};

void mouse_click(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        ClickCallbackData* data = static_cast<ClickCallbackData*>(userdata);
        data->points->push_back(cv::Point(x, y));
        
        // Draw a visual marker for the clicked spot
        cv::circle(data->image_display, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(data->image_display, cv::Point(x, y), 20, cv::Scalar(0, 0, 255), 1);
        cv::imshow(data->window_name, data->image_display);
        std::cout << "Clicked manual spot at: " << x << ", " << y << std::endl;
    }
}

// Convert an axis-aligned bounding box into a 4-point quadrilateral (quad)
std::vector<cv::Point> box_to_quad(double x1, double y1, double x2, double y2) {
    return {
        cv::Point((int)x1, (int)y1),
        cv::Point((int)x2, (int)y1),
        cv::Point((int)x2, (int)y2),
        cv::Point((int)x1, (int)y2)
    };
}

// Rasterizes both quads onto a local mask to compute their overlap percentage
// relative to the area of the smaller polygon
double polygon_overlap_percent(const std::vector<cv::Point>& poly_a, const std::vector<cv::Point>& poly_b) {
    std::vector<cv::Point> all_pts = poly_a;
    all_pts.insert(all_pts.end(), poly_b.begin(), poly_b.end());
    if (all_pts.empty()) return 0.0;
    
    cv::Rect bounding_rect = cv::boundingRect(all_pts);
    int min_x = bounding_rect.x;
    int min_y = bounding_rect.y;
    int w = std::max(bounding_rect.width + 2, 1);
    int h = std::max(bounding_rect.height + 2, 1);
    
    cv::Mat mask_a = cv::Mat::zeros(h, w, CV_8UC1);
    cv::Mat mask_b = cv::Mat::zeros(h, w, CV_8UC1);
    
    std::vector<cv::Point> local_a(poly_a.size());
    for (size_t i = 0; i < poly_a.size(); ++i) {
        local_a[i] = cv::Point(poly_a[i].x - min_x, poly_a[i].y - min_y);
    }
    
    std::vector<cv::Point> local_b(poly_b.size());
    for (size_t i = 0; i < poly_b.size(); ++i) {
        local_b[i] = cv::Point(poly_b[i].x - min_x, poly_b[i].y - min_y);
    }
    
    std::vector<std::vector<cv::Point>> pts_a = { local_a };
    std::vector<std::vector<cv::Point>> pts_b = { local_b };
    
    cv::fillPoly(mask_a, pts_a, cv::Scalar(1));
    cv::fillPoly(mask_b, pts_b, cv::Scalar(1));
    
    double area_a = cv::countNonZero(mask_a);
    double area_b = cv::countNonZero(mask_b);
    double min_area = std::min(area_a, area_b);
    if (min_area == 0.0) return 0.0;
    
    cv::Mat intersection_mask;
    cv::bitwise_and(mask_a, mask_b, intersection_mask);
    double intersection = cv::countNonZero(intersection_mask);
    
    return (intersection / min_area) * 100.0;
}

// Computes the lane-adaptive overlap threshold: mean + 0.5 * std of non-zero overlaps
double calculate_dynamic_threshold(const std::vector<double>& overlap_values, double min_thresh = 30.0, double max_thresh = 85.0) {
    std::vector<double> nonzero;
    for (double v : overlap_values) {
        if (v > 0.0) nonzero.push_back(v);
    }
    if (nonzero.empty()) return min_thresh;
    
    double sum = 0.0;
    for (double v : nonzero) sum += v;
    double mean = sum / nonzero.size();
    
    double sq_sum = 0.0;
    for (double v : nonzero) {
        sq_sum += (v - mean) * (v - mean);
    }
    double std_dev = std::sqrt(sq_sum / nonzero.size());
    
    double thresh = mean + 0.5 * std_dev;
    return std::max(min_thresh, std::min(thresh, max_thresh));
}

struct MergeResult {
    std::vector<Spot> spots;
    std::vector<bool> confirmed_flags;
    double dynamic_threshold;
};

// Compares manual quads against raw detections to link them or insert new manual spots
MergeResult merge_manual_and_detected_spots(
    const std::vector<std::vector<cv::Point>>& manual_quads,
    const std::vector<Spot>& detected_spots,
    double min_thresh = 30.0,
    double max_thresh = 85.0
) {
    std::vector<std::vector<cv::Point>> detected_quads;
    detected_quads.reserve(detected_spots.size());
    for (const auto& s : detected_spots) {
        detected_quads.push_back(box_to_quad(s.x1, s.y1, s.x2, s.y2));
    }
    
    std::vector<double> all_overlaps;
    all_overlaps.reserve(manual_quads.size() * detected_quads.size());
    std::vector<std::vector<double>> overlap_matrix(manual_quads.size(), std::vector<double>(detected_quads.size(), 0.0));
    
    for (size_t i = 0; i < manual_quads.size(); ++i) {
        for (size_t j = 0; j < detected_quads.size(); ++j) {
            double ov = polygon_overlap_percent(manual_quads[i], detected_quads[j]);
            overlap_matrix[i][j] = ov;
            all_overlaps.push_back(ov);
        }
    }
    
    double dynamic_threshold = calculate_dynamic_threshold(all_overlaps, min_thresh, max_thresh);
    
    std::vector<Spot> final_spots = detected_spots;
    std::vector<bool> confirmed_flags(final_spots.size(), false);
    
    std::cout << "\nManual overlap report (dynamic threshold = " << std::fixed << std::setprecision(2) << dynamic_threshold << "):" << std::endl;
    std::printf("%-15s %-18s %-16s %-30s\n", "Manual Spot #", "Best Overlap %", "Threshold Used", "Status");
    
    for (size_t i = 0; i < manual_quads.size(); ++i) {
        int best_j = -1;
        double best_overlap = 0.0;
        
        if (!detected_quads.empty()) {
            best_overlap = overlap_matrix[i][0];
            best_j = 0;
            for (size_t j = 1; j < detected_quads.size(); ++j) {
                if (overlap_matrix[i][j] > best_overlap) {
                    best_overlap = overlap_matrix[i][j];
                    best_j = (int)j;
                }
            }
        }
        
        std::string status;
        if (best_overlap >= dynamic_threshold && best_j != -1) {
            confirmed_flags[best_j] = true;
            std::stringstream ss;
            ss << "matched detection #" << best_j + 1 << " -> forced through filter";
            status = ss.str();
        } else {
            // Find bounding box for manual quad
            double min_x = manual_quads[i][0].x;
            double max_x = manual_quads[i][0].x;
            double min_y = manual_quads[i][0].y;
            double max_y = manual_quads[i][0].y;
            for (const auto& pt : manual_quads[i]) {
                if (pt.x < min_x) min_x = pt.x;
                if (pt.x > max_x) max_x = pt.x;
                if (pt.y < min_y) min_y = pt.y;
                if (pt.y > max_y) max_y = pt.y;
            }
            
            Spot manual_spot;
            manual_spot.x1 = (float)min_x;
            manual_spot.y1 = (float)min_y;
            manual_spot.x2 = (float)max_x;
            manual_spot.y2 = (float)max_y;
            manual_spot.confidence = 1.0f;
            manual_spot.cls = -1; // -1 flags manual
            
            final_spots.push_back(manual_spot);
            confirmed_flags.push_back(true);
            status = "manual-only spot added";
        }
        
        std::printf("%-15d %-18.2f %-16.2f %-30s\n", (int)(i + 1), best_overlap, dynamic_threshold, status.c_str());
    }
    
    MergeResult res;
    res.spots = final_spots;
    res.confirmed_flags = confirmed_flags;
    res.dynamic_threshold = dynamic_threshold;
    return res;
}

// Convert a string to a wide string on Windows
std::wstring to_wstring(const std::string& str) {
    std::wstring wstr(str.begin(), str.end());
    return wstr;
}

int main(int argc, char* argv[]) {
    try {
        std::string img_path = "band3(1).png";
        std::string strip_model_path = "strip.onnx";
        std::string spot_model_path = "best_spot8.onnx";
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
            } else if (arg == "--strip-model" && i + 1 < argc) {
                strip_model_path = argv[++i];
            } else if (arg == "--spot-model" && i + 1 < argc) {
                spot_model_path = argv[++i];
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
                    }
                }
            }
        }

        if (verbose) {
            std::cout << "Loading image: " << img_path << std::endl;
            std::cout << "Loading lane detector: " << strip_model_path << std::endl;
            std::cout << "Loading spot detector: " << spot_model_path << std::endl;
            std::cout << "Headless mode: " << (headless ? "ENABLED" : "DISABLED") << std::endl;
        }

        cv::Mat image = cv::imread(img_path);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image: " << img_path << std::endl;
            return -1;
        }

        // Parse CLI manual spots
        // Formats:
        // - Standard: "x1,y1;x2,y2;..." (auto-detects lane by coordinates)
        // - Explicit Lane: "lane_id:x1,y1;lane_id:x2,y2;..." (maps to specific lane)
        std::map<int, std::vector<cv::Point>> cli_manual_spots; // Maps lane_id -> list of points
        
        if (!manual_spots_str.empty()) {
            std::stringstream ss(manual_spots_str);
            std::string item;
            while (std::getline(ss, item, ';')) {
                if (item.empty()) continue;
                
                size_t colon_pos = item.find(':');
                if (colon_pos != std::string::npos) {
                    std::string slane = item.substr(0, colon_pos);
                    std::string coordinates = item.substr(colon_pos + 1);
                    std::stringstream pts(coordinates);
                    std::string sx, sy;
                    if (std::getline(pts, sx, ',') && std::getline(pts, sy, ',')) {
                        try {
                            int lane_id = std::stoi(slane);
                            int x = std::stoi(sx);
                            int y = std::stoi(sy);
                            cli_manual_spots[lane_id].push_back(cv::Point(x, y));
                        } catch (...) {}
                    }
                } else {
                    std::stringstream pts(item);
                    std::string sx, sy;
                    if (std::getline(pts, sx, ',') && std::getline(pts, sy, ',')) {
                        try {
                            int x = std::stoi(sx);
                            int y = std::stoi(sy);
                            cli_manual_spots[0].push_back(cv::Point(x, y));
                        } catch (...) {}
                    }
                }
            }
        }

        // 1. Run Lane/Strip Detection
        std::wstring w_strip_model_path = to_wstring(strip_model_path);
        SpotDetector strip_detector(w_strip_model_path);
        std::vector<Spot> lane_detections = strip_detector.detect(image, 0.25f, 0.45f);

        std::vector<Lane> lanes;
        for (const auto& det : lane_detections) {
            double width = det.x2 - det.x1;
            if (width < 20.0) {
                if (verbose) {
                    std::cout << "Filtering out lane candidate (width < 20): " << width << std::endl;
                }
                continue;
            }
            
            Lane lane;
            lane.id = 0; // Assigned after sorting
            lane.x1 = det.x1;
            lane.y1 = det.y1;
            lane.x2 = det.x2;
            lane.y2 = det.y2;
            lane.confidence = det.confidence;
            
            int ix1 = std::max(0, std::min((int)lane.x1, image.cols - 1));
            int iy1 = std::max(0, std::min((int)lane.y1, image.rows - 1));
            int ix2 = std::max(0, std::min((int)lane.x2, image.cols));
            int iy2 = std::max(0, std::min((int)lane.y2, image.rows));
            
            if (ix2 > ix1 && iy2 > iy1) {
                lane.crop = image(cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1)).clone();
                lanes.push_back(lane);
            }
        }

        // Sort lanes from left to right (based on x1)
        std::sort(lanes.begin(), lanes.end(), [](const Lane& a, const Lane& b) {
            return a.x1 < b.x1;
        });

        // Assign 1-indexed Lane IDs
        for (size_t i = 0; i < lanes.size(); ++i) {
            lanes[i].id = (int)(i + 1);
        }

        // Fallback: If no lanes are detected, treat the entire image as a single lane
        if (lanes.empty()) {
            if (verbose) {
                std::cout << "No lanes detected. Fallback: Treating full image as a single lane." << std::endl;
            }
            Lane lane;
            lane.id = 1;
            lane.x1 = 0;
            lane.y1 = 0;
            lane.x2 = image.cols;
            lane.y2 = image.rows;
            lane.confidence = 1.0;
            lane.crop = image.clone();
            lanes.push_back(lane);
        }

        if (verbose) {
            std::cout << "Active Lanes to analyze: " << lanes.size() << std::endl;
        }

        // 2. Spot Detection per Lane
        std::wstring w_spot_model_path = to_wstring(spot_model_path);
        SpotDetector spot_detector(w_spot_model_path);

        for (auto& lane : lanes) {
            std::vector<Spot> auto_spots = spot_detector.detect(lane.crop, 0.2f, 0.45f);
            
            std::vector<cv::Point> clicked_points;

            // Map CLI manual spots to this lane if they fall within its boundaries
            // Explicitly mapped:
            if (cli_manual_spots.count(lane.id) > 0) {
                for (const auto& pt : cli_manual_spots[lane.id]) {
                    if (pt.x >= lane.x1 && pt.x <= lane.x2 && pt.y >= lane.y1 && pt.y <= lane.y2) {
                        clicked_points.push_back(cv::Point((int)(pt.x - lane.x1), (int)(pt.y - lane.y1)));
                    } else {
                        clicked_points.push_back(pt); // Assume relative coordinates if not fitting absolute box
                    }
                }
            }
            // Auto-detected mapping for unassigned spots:
            if (cli_manual_spots.count(0) > 0) {
                for (const auto& pt : cli_manual_spots[0]) {
                    if (pt.x >= lane.x1 && pt.x <= lane.x2 && pt.y >= lane.y1 && pt.y <= lane.y2) {
                        clicked_points.push_back(cv::Point((int)(pt.x - lane.x1), (int)(pt.y - lane.y1)));
                    }
                }
            }

            // Interactive manual selection if UI is enabled
            if (!headless) {
                std::stringstream win_ss;
                win_ss << "Select Missing Spots - Lane " << lane.id;
                std::string win_name = win_ss.str();

                ClickCallbackData cb_data;
                cb_data.image_display = lane.crop.clone();
                cb_data.points = &clicked_points;
                cb_data.window_name = win_name;

                // Draw existing detected spots (Green boxes)
                for (const auto& s : auto_spots) {
                    cv::rectangle(cb_data.image_display, 
                                 cv::Point((int)s.x1, (int)s.y1), 
                                 cv::Point((int)s.x2, (int)s.y2), 
                                 cv::Scalar(0, 255, 0), 2);
                }

                // Draw existing manual spots (Red circles)
                for (const auto& pt : clicked_points) {
                    cv::circle(cb_data.image_display, pt, 5, cv::Scalar(0, 0, 255), -1);
                    cv::circle(cb_data.image_display, pt, 20, cv::Scalar(0, 0, 255), 1);
                }

                cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
                cv::setMouseCallback(win_name, mouse_click, &cb_data);
                cv::imshow(win_name, cb_data.image_display);

                std::cout << "Lane " << lane.id << ": Click missing spots in the window, then press ANY key to finish." << std::endl;
                cv::waitKey(0);
                cv::destroyWindow(win_name);
            }

            // Build manual quads from clicked points
            std::vector<std::vector<cv::Point>> manual_quads;
            int box_size = 20;
            for (const auto& pt : clicked_points) {
                manual_quads.push_back(box_to_quad(pt.x - box_size, pt.y - box_size, pt.x + box_size, pt.y + box_size));
            }

            std::vector<Spot> spots;
            std::vector<bool> confirmed_flags;

            if (!manual_quads.empty()) {
                MergeResult merge_res = merge_manual_and_detected_spots(manual_quads, auto_spots);
                spots = merge_res.spots;
                confirmed_flags = merge_res.confirmed_flags;
            } else {
                spots = auto_spots;
                confirmed_flags = std::vector<bool>(spots.size(), false);
            }

            // 3. Filtration system to filter out ghost spots
            std::vector<Spot> filtered_spots;
            double l_width = lane.crop.cols;
            double l_height = lane.crop.rows;
            double l_center = l_width / 2.0;

            for (size_t idx = 0; idx < spots.size(); ++idx) {
                const auto& spot = spots[idx];
                double box_width = spot.x2 - spot.x1;
                double box_height = spot.y2 - spot.y1;
                double area = box_width * box_height;
                double center_x = (spot.x1 + spot.x2) / 2.0;

                // Bypass filtration for manual spots
                if (confirmed_flags[idx]) {
                    filtered_spots.push_back(spot);
                    continue;
                }

                // Area filter
                if (area < 20.0 || area > 10000.0) {
                    continue;
                }

                // Positioning filter
                if (std::abs(center_x - l_center) > l_width * 0.45) {
                    continue;
                }

                // Confidence straining
                if (spot.confidence < 0.0009f) {
                    continue;
                }

                filtered_spots.push_back(spot);
            }

            // 4. Calculate intensities and Rf values
            for (const auto& spot : filtered_spots) {
                int ix1 = std::max(0, std::min((int)spot.x1, lane.crop.cols - 1));
                int iy1 = std::max(0, std::min((int)spot.y1, lane.crop.rows - 1));
                int ix2 = std::max(0, std::min((int)spot.x2, lane.crop.cols));
                int iy2 = std::max(0, std::min((int)spot.y2, lane.crop.rows));
                
                int w = ix2 - ix1;
                int h = iy2 - iy1;
                
                double intensity = 0.0;
                if (w > 0 && h > 0) {
                    cv::Mat spot_crop = lane.crop(cv::Rect(ix1, iy1, w, h));
                    cv::Mat gray;
                    cv::cvtColor(spot_crop, gray, cv::COLOR_BGR2GRAY);
                    cv::Scalar mean_val = cv::mean(255 - gray);
                    intensity = mean_val[0] / 255.0; // Normalized to [0, 1]
                }

                double spot_center_y = (spot.y1 + spot.y2) / 2.0;
                double rf = spot_center_y / l_height;

                FinalSpot fs;
                fs.id = 0; // Assigned after sorting
                fs.x1 = spot.x1;
                fs.y1 = spot.y1;
                fs.x2 = spot.x2;
                fs.y2 = spot.y2;
                fs.abs_x1 = spot.x1 + lane.x1;
                fs.abs_y1 = spot.y1 + lane.y1;
                fs.abs_x2 = spot.x2 + lane.x1;
                fs.abs_y2 = spot.y2 + lane.y1;
                fs.confidence = spot.confidence;
                fs.rf = rf;
                fs.intensity = intensity;
                fs.auc = 0.0;

                lane.spots.push_back(fs);
            }

            // Sort spots by Rf ascending
            std::sort(lane.spots.begin(), lane.spots.end(), [](const FinalSpot& a, const FinalSpot& b) {
                return a.rf < b.rf;
            });

            // Assign spot IDs and calculate AUC
            for (size_t k = 0; k < lane.spots.size(); ++k) {
                lane.spots[k].id = (int)(k + 1);
                
                std::vector<double> peak_x;
                std::vector<double> peak_y;
                double box_width = lane.spots[k].x2 - lane.spots[k].x1;
                double box_height = lane.spots[k].y2 - lane.spots[k].y1;
                double area = box_width * box_height;

                AUCCalculator::generate_peak_data(lane.spots[k].rf, lane.spots[k].intensity, area, peak_x, peak_y);
                lane.spots[k].auc = AUCCalculator::calculate_auc(peak_x, peak_y);
            }
        }

        // 5. Drawing Multi-Lane Densitogram Plot using OpenCV
        int plot_w = 800;
        int plot_h = 500;
        cv::Mat plot(plot_h, plot_w, CV_8UC3, cv::Scalar(255, 255, 255));

        int margin_l = 80;
        int margin_r = 150; // Increased right margin for legend
        int margin_t = 50;
        int margin_b = 60;

        int graph_w = plot_w - margin_l - margin_r;
        int graph_h = plot_h - margin_t - margin_b;

        // Find max intensity across all lanes to scale the plot
        double max_intensity = 0.0;
        for (const auto& lane : lanes) {
            for (const auto& spot : lane.spots) {
                if (spot.intensity > max_intensity) {
                    max_intensity = spot.intensity;
                }
            }
        }
        if (max_intensity == 0.0) {
            max_intensity = 1.0;
        }

        // Determine step size and rounded Y-axis limit (e.g. max_intensity 0.669 -> limit 0.7)
        double step_y = 0.1;
        if (max_intensity <= 0.1) {
            step_y = 0.02;
        } else if (max_intensity <= 0.2) {
            step_y = 0.05;
        } else if (max_intensity <= 0.5) {
            step_y = 0.1;
        } else {
            step_y = 0.1;
        }
        double y_limit = std::ceil(max_intensity / step_y) * step_y;
        if (y_limit == 0.0) {
            y_limit = step_y;
        }

        // Draw background grid lines (horizontal and vertical)
        // Horizontal grid at clean intervals (0.0, 0.1, 0.2, etc.)
        for (double val = 0.0; val <= y_limit + 0.0001; val += step_y) {
            double ratio = val / y_limit;
            int y = plot_h - margin_b - (int)(ratio * graph_h);
            cv::line(plot, cv::Point(margin_l, y), cv::Point(plot_w - margin_r, y), cv::Scalar(230, 230, 230), 1);
            
            // Y Label (formatted as 0.0, 0.1, 0.2...)
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << val;
            cv::putText(plot, ss.str(), cv::Point(15, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(50, 50, 50), 1);
        }

        // Vertical grid (6 lines)
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
        cv::putText(plot, "Multi-Lane TLC Densitogram", cv::Point(margin_l + graph_w / 2 - 120, margin_t - 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(plot, "Rf Value", cv::Point(margin_l + graph_w / 2 - 30, plot_h - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        cv::putText(plot, "Intensity", cv::Point(margin_l - 15, margin_t - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        // Define colors for up to 5 lanes (resets if more than 5)
        std::vector<cv::Scalar> lane_colors = {
            cv::Scalar(0, 0, 255),    // Red (BGR: Red)
            cv::Scalar(255, 0, 0),    // Blue (BGR: Blue)
            cv::Scalar(0, 255, 0),    // Green (BGR: Green)
            cv::Scalar(0, 165, 255),  // Orange (BGR: Orange)
            cv::Scalar(255, 0, 255)   // Purple (BGR: Purple)
        };

        // Draw filled peaks in a semi-transparent overlay
        cv::Mat overlay = plot.clone();
        for (size_t i = 0; i < lanes.size(); ++i) {
            cv::Scalar color = lane_colors[i % lane_colors.size()];
            for (const auto& spot : lanes[i].spots) {
                std::vector<double> peak_x;
                std::vector<double> peak_y;
                double box_width = spot.x2 - spot.x1;
                double box_height = spot.y2 - spot.y1;
                double area = box_width * box_height;

                AUCCalculator::generate_peak_data(spot.rf, spot.intensity, area, peak_x, peak_y);

                std::vector<cv::Point> pts;
                pts.reserve(peak_x.size() + 2);

                // Start point on baseline
                double first_x = peak_x.front();
                int first_px = margin_l + (int)((first_x - 0.0) / 1.0 * graph_w);
                pts.push_back(cv::Point(first_px, plot_h - margin_b));

                for (size_t j = 0; j < peak_x.size(); ++j) {
                    int px = margin_l + (int)((peak_x[j] - 0.0) / 1.0 * graph_w);
                    int py = plot_h - margin_b - (int)((peak_y[j] - 0.0) / y_limit * graph_h);
                    px = std::max(margin_l, std::min(px, plot_w - margin_r));
                    py = std::max(margin_t, std::min(py, plot_h - margin_b));
                    pts.push_back(cv::Point(px, py));
                }

                // End point on baseline
                double last_x = peak_x.back();
                int last_px = margin_l + (int)((last_x - 0.0) / 1.0 * graph_w);
                pts.push_back(cv::Point(last_px, plot_h - margin_b));

                std::vector<std::vector<cv::Point>> fill_pts = { pts };
                cv::fillPoly(overlay, fill_pts, color);
            }
        }

        // Blend transparent peaks
        cv::addWeighted(overlay, 0.3, plot, 0.7, 0, plot);

        // Draw solid lines for peak borders
        for (size_t i = 0; i < lanes.size(); ++i) {
            cv::Scalar color = lane_colors[i % lane_colors.size()];
            for (const auto& spot : lanes[i].spots) {
                std::vector<double> peak_x;
                std::vector<double> peak_y;
                double box_width = spot.x2 - spot.x1;
                double box_height = spot.y2 - spot.y1;
                double area = box_width * box_height;

                AUCCalculator::generate_peak_data(spot.rf, spot.intensity, area, peak_x, peak_y);

                std::vector<cv::Point> line_pts;
                line_pts.reserve(peak_x.size());

                for (size_t j = 0; j < peak_x.size(); ++j) {
                    int px = margin_l + (int)((peak_x[j] - 0.0) / 1.0 * graph_w);
                    int py = plot_h - margin_b - (int)((peak_y[j] - 0.0) / y_limit * graph_h);
                    px = std::max(margin_l, std::min(px, plot_w - margin_r));
                    py = std::max(margin_t, std::min(py, plot_h - margin_b));
                    line_pts.push_back(cv::Point(px, py));
                }

                cv::polylines(plot, line_pts, false, color, 2);
            }
        }

        // Draw Legend
        int legend_x = plot_w - margin_r + 20;
        int legend_y = margin_t + 10;
        for (size_t i = 0; i < lanes.size(); ++i) {
            cv::Scalar color = lane_colors[i % lane_colors.size()];
            cv::rectangle(plot, cv::Rect(legend_x, legend_y + i * 20, 15, 10), color, -1);
            
            std::stringstream ss;
            ss << "Lane " << lanes[i].id;
            cv::putText(plot, ss.str(), cv::Point(legend_x + 25, legend_y + 9 + i * 20), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
        }

        // Save plot
        cv::imwrite(plot_output_path, plot);
        if (verbose) {
            std::cout << "Densitogram plot saved to: " << plot_output_path << std::endl;
        }

        // 6. Output Table Results
        std::cout << "\n================= TLC DENSITOMETRY REPORT =================\n";
        std::printf("%-8s %-8s %-10s %-12s %-10s\n", "Lane ID", "Spot #", "Rf Value", "Intensity", "AUC");
        for (const auto& lane : lanes) {
            for (const auto& spot : lane.spots) {
                std::printf("%-8d %-8d %-10.3f %-12.3f %-10.4f\n", lane.id, spot.id, spot.rf, spot.intensity, spot.auc);
            }
        }

        // 7. Structured JSON Output
        std::cout << "\nJSON_OUTPUT_START\n";
        std::cout << "{\n";
        std::cout << "  \"lanes\": [\n";
        for (size_t i = 0; i < lanes.size(); ++i) {
            const auto& lane = lanes[i];
            std::cout << "    {\n";
            std::cout << "      \"lane_id\": " << lane.id << ",\n";
            std::cout << "      \"box\": [" << lane.x1 << ", " << lane.y1 << ", " << lane.x2 << ", " << lane.y2 << "],\n";
            std::cout << "      \"confidence\": " << lane.confidence << ",\n";
            std::cout << "      \"spots\": [\n";
            for (size_t j = 0; j < lane.spots.size(); ++j) {
                const auto& spot = lane.spots[j];
                std::cout << "        {\n";
                std::cout << "          \"id\": " << spot.id << ",\n";
                std::cout << "          \"rf\": " << spot.rf << ",\n";
                std::cout << "          \"intensity\": " << spot.intensity << ",\n";
                std::cout << "          \"auc\": " << spot.auc << ",\n";
                std::cout << "          \"confidence\": " << spot.confidence << ",\n";
                std::cout << "          \"box_relative\": [" << spot.x1 << ", " << spot.y1 << ", " << spot.x2 << ", " << spot.y2 << "],\n";
                std::cout << "          \"box_absolute\": [" << spot.abs_x1 << ", " << spot.abs_y1 << ", " << spot.abs_x2 << ", " << spot.abs_y2 << "]\n";
                std::cout << "        }" << (j == lane.spots.size() - 1 ? "" : ",") << "\n";
            }
            std::cout << "      ]\n";
            std::cout << "    }" << (i == lanes.size() - 1 ? "" : ",") << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
        std::cout << "JSON_OUTPUT_END\n";

        // Display results if UI is enabled
        if (!headless) {
            cv::imshow("TLC Multi-Lane Densitogram", plot);
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