#include <spdlog/spdlog.h>
#include <fstream>

#include <symforce/opt/factor.h>
#include "./gen/Residual_factor.h"

namespace event_res {

void unit_test() {
    Eigen::Matrix<double, 3, 3> R_;
    Eigen::Matrix<double, 3, 1> t_;
    Eigen::Matrix<double, 6, 1> x;
    Eigen::Matrix<double, 3, 1> point;
    Eigen::Matrix<double, 3, 3> R_cam;
    Eigen::Matrix<double, 3, 1> t_cam;
    const double* mask;
    const double* TsObs;
    double width_ = 240;
    double height_ = 180;
    double mask_col = 240;
    double Ts_col = 240;
    double Ts_row = 180;

    R_cam << 156.925, 0, 108.167,
            0, 156.925, 78.4205,
            0, 0, 1;

    t_cam << 0, 0, 0;

    // Get values from txt file
    std::ifstream file("/home/ckengjwe/ThisIsMe.txt");

    // Read in R_ values
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            file >> R_(i, j);
        }
    }
    // spdlog::info(R_);

    // Read in t_ values
    file >> t_(0) >> t_(1) >> t_(2);

    // spdlog::info(t_);

    // Read in x
    for (int i = 0; i < 6; i++) {
        file >> x(i);
    }

    // spdlog::info(x);

    // Read in Mask values
    std::string string;
    // std::vector<double> mask_values;
    // std::getline(file, string);
    // string.erase(std::remove(string.begin(), string.end(), '['), string.end());
    // string.erase(std::remove(string.begin(), string.end(), ']'), string.end());

    // // Split line to individual pixel values
    // std::istringstream iss(string);
    // double value;
    // while (iss >> value) {
    //     mask_values.push_back(value);
    //     // iss.ignore();
    // }
    // mask = mask_values.data();

    std::vector<double> mask_values;
    double val;
    for (int i = 0; i < 43200; i++) {
        file >> val;
        mask_values.push_back(val);
    }
    mask = mask_values.data();
    // for(int j = 0; j < 43200; j++) {
    //     spdlog::info(mask[j]);
    // }
    
    // Get TsObs negatives
    std::vector<double> TsObs_values;
    std::getline(file, string);
    std::getline(file, string);
    // spdlog::info(string);
    string.erase(std::remove(string.begin(), string.end(), '['), string.end());
    string.erase(std::remove(string.begin(), string.end(), ']'), string.end());

    // Split line to individual pixel values
    std::istringstream iss1(string);
    double value1;
    while (iss1 >> value1) {
        TsObs_values.push_back(value1);
        iss1.ignore();
    }
    TsObs = TsObs_values.data();


    Eigen::Matrix<double, 1, 1> res;
    Eigen::Matrix<double, 1, 6> jacobian;
    Eigen::Matrix<double, 6, 6> hessian;
    Eigen::Matrix<double, 6, 1> rhs;

    // spdlog::info("Start");
    std::ofstream o("/home/ckengjwe/autogen_fvec.txt", std::ios::app);
    std::ofstream jac_o("/home/ckengjwe/autogen_fjac.txt", std::ios::app);
    for (int i = 0; i < 200; i++) {
        // Read in the 3D point
        file >> point(0) >> point(1) >> point(2);
        // spdlog::info(point);

        event_res::ResidualFactor(
        R_, t_, x, point, R_cam, t_cam, width_, height_,
        mask_col, Ts_col, Ts_row, mask, TsObs, &res, &jacobian, &hessian, &rhs
        );
        jac_o << jacobian(0, 0) << " " << jacobian(0, 1) << " " << jacobian(0, 2) << " " << jacobian(0, 3) << " " << jacobian(0, 4) << " " << jacobian(0, 5) << std::endl; 
        o << "fvec " << i << ": " << res(0, 0) << std::endl;
    }
     
    // spdlog::info("Complete");
    file.close();
    o.close();
    jac_o.close();
}

}