//  -----------------------------------------------------------------------------
// This file was autogenerated by symforce. Do NOT modify by hand.
// -----------------------------------------------------------------------------
#pragma once

#include <vector>
#include <Eigen/Dense>

#include <geo/rot3.h>

namespace geo {
namespace rot3 {

/**
 * C++ StorageOps implementation for <class 'symforce.geo.rot3.Rot3'>.
 */
template <typename Scalar>
struct StorageOps {
  static constexpr int32_t StorageDim() {
    return 4;
  }

  static void ToStorage(const geo::Rot3<Scalar>& a, std::vector<Scalar>* vec);
  static geo::Rot3<Scalar> FromStorage(const std::vector<Scalar>& vec);
};

}  // namespace rot3

// Wrapper to specialize the public concept

template <>
struct StorageOps<Rot3<double>> : public rot3::StorageOps<double> {};
template <>
struct StorageOps<Rot3<float>> : public rot3::StorageOps<float> {};

}  // namespace geo