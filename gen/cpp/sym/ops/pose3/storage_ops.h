// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/ops/CLASS/storage_ops.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <lcmtypes/sym/type_t.hpp>

#include <sym/pose3.h>

namespace sym {

/**
 * C++ StorageOps implementation for <class 'symforce.geo.pose3.Pose3'>.
 */
template <typename ScalarType>
struct StorageOps<Pose3<ScalarType>> {
  using T = Pose3<ScalarType>;
  using Scalar = typename Pose3<ScalarType>::Scalar;

  static constexpr int32_t StorageDim() {
    return 7;
  }

  static void ToStorage(const T& a, ScalarType* out);
  static T FromStorage(const ScalarType* data);

  static constexpr type_t TypeEnum() {
    return type_t::POSE3;
  }

  template <typename Generator>
  static T Random(Generator& gen) {
    return T::Random(gen);
  }
};

}  // namespace sym

// Explicit instantiation
extern template struct sym::StorageOps<sym::Pose3<double>>;
extern template struct sym::StorageOps<sym::Pose3<float>>;
