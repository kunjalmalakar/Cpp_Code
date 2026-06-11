#include "../include/AUCCalculator.h"
#include <cmath>

namespace AUCCalculator
{
    void generate_peak_data(double xi, double yi, std::vector<double>& peak_x, std::vector<double>& peak_y)
    {
        peak_x.resize(100);
        peak_y.resize(100);
        
        double start_x = xi - 0.02;
        double end_x = xi + 0.02;
        double step = (end_x - start_x) / 99.0; // 100 points means 99 intervals
        
        for (int i = 0; i < 100; ++i)
        {
            double cur_x = start_x + i * step;
            double diff = cur_x - xi;
            double cur_y = yi * std::exp(-(diff * diff) / (2.0 * 0.005 * 0.005));
            peak_x[i] = cur_x;
            peak_y[i] = cur_y;
        }
    }

    double calculate_auc(const std::vector<double>& x, const std::vector<double>& y)
    {
        if (x.size() != y.size() || x.size() < 2)
        {
            return 0.0;
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < x.size() - 1; ++i)
        {
            double dx = x[i+1] - x[i];
            double mean_y = 0.5 * (y[i] + y[i+1]);
            sum += dx * mean_y;
        }
        return sum;
    }
}