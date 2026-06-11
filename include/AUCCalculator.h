#pragma once

#include <vector>

namespace AUCCalculator
{
    // Generate 100 points for x and y of a peak given its rf (xi) and intensity (yi)
    void generate_peak_data(double xi, double yi, std::vector<double>& peak_x, std::vector<double>& peak_y);

    // Calculate Area Under Curve (AUC) using trapezoidal rule
    double calculate_auc(const std::vector<double>& x, const std::vector<double>& y);
}