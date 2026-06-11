#pragma once

#include <vector>
#include <utility>
#include "SpotDetector.h"

namespace RFCalculator
{
    // Calculate Rf value and return pair of (rf_value, confidence)
    std::vector<std::pair<double, double>> calculate_rf(const std::vector<Spot>& spots, double lane_height);
}