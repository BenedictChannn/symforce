//  ----------------------------------------------------------------------------
// This file was autogenerated by symforce. Do NOT modify by hand.
// -----------------------------------------------------------------------------
#pragma once

#include <Eigen/Dense>

#include <geo/pose2.h>

namespace geo {
namespace pose2 {

/**
 * C++ GroupOps implementation for <class 'symforce.geo.pose2.Pose2'>.
 */
template <typename Scalar>
struct GroupOps {
  static Pose2<Scalar> Identity();
  static Pose2<Scalar> Inverse(const Pose2<Scalar>& a);
  static Pose2<Scalar> Compose(const Pose2<Scalar>& a, const Pose2<Scalar>& b);
  static Pose2<Scalar> Between(const Pose2<Scalar>& a, const Pose2<Scalar>& b);

};

}  // namespace pose2

// Wrapper to specialize the public concept

template <>
struct GroupOps<Pose2<double>> : public pose2::GroupOps<double> {};
template <>
struct GroupOps<Pose2<float>> : public pose2::GroupOps<float> {};

}  // namespace geo