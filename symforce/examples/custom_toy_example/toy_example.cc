#include <spdlog/spdlog.h>

#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>

#include "./gen/toy_residual_with_jacobians.h"

namespace toy_example {

    // Create a factor object
    sym::Factord BuildFactor() {
        // factors.push_back(sym::Factord::Jacobian(
        //     [](double x, sym::Vector1d* res, Eigen::Matrix<double, 1, 1>* jac) {
        //         (*res)[0] = x*x + 5*x;

        //         if(jac) {
        //             (*jac)(0, 0) = 2*x + 5;
        //         }
        //     },
        //     {'x'}
        // ));
        return sym::Factord::Jacobian(toy_example::ToyResidualWithJacobians<double>, {'x', 'y'});
    }



    void find_gt() {
        // Create inital value
        sym::Valuesd values;
        values.Set('x', 0.0);
        values.Set('y', 0.0);
        spdlog::info("Initial x: {} y: {}", values.At<double>('x'), values.At<double>('y'));

        // Set optimizer parameters
        sym::optimizer_params_t params = sym::DefaultOptimizerParams();
        params.initial_lambda = 10.0;
        params.lambda_up_factor = 3.0;
        params.lambda_down_factor = 1.0 / 3.0;
        params.lambda_lower_bound = 0.01;
        params.lambda_upper_bound = 1000.0;
        params.early_exit_min_reduction = 1e-9; // Error tolerance -> directly affects number of iterations i.e convergence criteria
        params.use_diagonal_damping = true;
        params.use_unit_damping = true;

        // Create solver
        sym::Optimizerd optimizer(params, {BuildFactor()}, sym::kDefaultEpsilond);
        sym::OptimizationStatsd stats = optimizer.Optimize(values);

        const auto& iteration_stats = stats.iterations;
        const auto& first_iter = iteration_stats.front();
        // const auto& last_iter = iteration_stats.back();
        const auto& best_iter = iteration_stats[stats.best_index];

        // Actual x, y values {9.854, -17.708} from wolfram alpha
        spdlog::info("Optimized x: {} y: {}", values.At<double>('x'), values.At<double>('y'));

        spdlog::info("Initial error: {}", first_iter.new_error);
        spdlog::info("Final error: {}", best_iter.new_error);
    }


} // namespace toy_example