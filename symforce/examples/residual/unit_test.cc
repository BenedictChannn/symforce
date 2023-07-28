#include <spdlog/spdlog.h>
#include <fstream>
#include <chrono>

#include <symforce/opt/factor.h>
#include "./gen/Residual_factor.h"

namespace event_res {

void unit_test() {
    Eigen::Matrix<float, 3, 3> R_;
    Eigen::Matrix<float, 3, 1> t_;
    Eigen::Matrix<float, 6, 1> x;
    Eigen::Matrix<float, 3, 1> point;
    Eigen::Matrix<float, 3, 3> R_cam;
    Eigen::Matrix<float, 3, 1> t_cam;
    const float* mask;
    const float* TsObs;

    // /////////////////////////////////////////////////////////////
    // // rpg dataset
    // float width_ = 240;
    // float height_ = 180;
    // float mask_col = 240;
    // float Ts_col = 240;
    // float Ts_row = 180;

    // R_cam << 156.925, 0, 108.167,
    //         0, 156.925, 78.4205,
    //         0, 0, 1;

    // t_cam << 0, 0, 0;

    // /////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////
    // upenn dataset
    float width_ = 346;
    float height_ = 260;
    float mask_col = 346;
    float Ts_col = 346;
    float Ts_row = 260;

    R_cam << 199.653, 0, 177.433,
            0, 199.653, 126.812,
            0, 0, 1;

    t_cam << 0, 0, 0;

    ///////////////////////////////////////////////////////////////


    // Get values from txt file
    std::ifstream file("/home/ckengjwe/dso/upenntxt/input_values.txt");

    


    Eigen::Matrix<float, 1, 1> res;
    Eigen::Matrix<float, 1, 6> jacobian;
    Eigen::Matrix<float, 6, 6> hessian;
    Eigen::Matrix<float, 6, 1> rhs;

    // spdlog::info("Start");
    std::ofstream res_o("/home/ckengjwe/dso/upenntxt/autogen_fvec.txt", std::ios::app);
    // std::ofstream jac_o("/home/ckengjwe/dso/rpgtxt/autogen_fjac.txt", std::ios::app); 
    for (int j = 0; j < 10; j++) {
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

        std::vector<float> mask_values;
        float val;
        // i here depends on size of mask and TsObs, rpg: 43200, upenn: 89960
        for (int i = 0; i < 89960; i++) {
            file >> val;
            mask_values.push_back(val);
        }
        mask = mask_values.data();
        // for(int j = 0; j < 43200; j++) {
        //     spdlog::info(mask[j]);
        // }
    
        // Get TsObs negatives
        std::vector<float> TsObs_values;
        std::getline(file, string);
        std::getline(file, string);
        // spdlog::info(string);
        string.erase(std::remove(string.begin(), string.end(), '['), string.end());
        string.erase(std::remove(string.begin(), string.end(), ']'), string.end());

        // Split line to individual pixel values
        std::istringstream iss1(string);
        float value1;
        while (iss1 >> value1) {
            TsObs_values.push_back(value1);
            iss1.ignore();
        }
        TsObs = TsObs_values.data();

        // double totalDuration = 0.0;

        for (int i = 0; i < 200; i++) {
            // Read in the 3D point
            file >> point(0) >> point(1) >> point(2);
            // spdlog::info(point);

            // auto startTime = std::chrono::high_resolution_clock::now();

            event_res::ResidualFactor(
                R_, t_, x, point, R_cam, t_cam, width_, height_,
                mask_col, Ts_col, Ts_row, mask, TsObs, &res, &jacobian, &hessian, &rhs
            );

            // auto endTime = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
            // totalDuration += duration;

            // jac_o << jacobian << std::endl;
            res_o << res(0, 0) << std::endl; 
        }
        // spdlog::info("Time taken: {}", totalDuration);
    }

    // spdlog::info("Complete");
    file.close();
    res_o.close();
    // jac_o.close();
}

}