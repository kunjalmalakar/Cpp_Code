#include "../include/RFCalculator.h"

namespace RFCalculator
{
    std::vector<std::pair<double, double>> calculate_rf(const std::vector<Spot>& spots, double lane_height)
    {
        std::vector<std::pair<double, double>> rf_values;
        rf_values.reserve(spots.size());
        for (const auto& spot : spots)
        {
            double spot_center_y = (spot.y1 + spot.y2) / 2.0;
            double rf = spot_center_y / lane_height;
            rf_values.push_back({rf, (double)spot.confidence});
        }
        return rf_values;
    }
}