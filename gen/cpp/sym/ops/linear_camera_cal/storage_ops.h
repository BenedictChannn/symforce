// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/ops/CLASS/storage_ops.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <sym/linear_camera_cal.h>

namespace sym {

/**
 * C++ StorageOps implementation for <class 'symforce.cam.linear_camera_cal.LinearCameraCal'>.
 */
template <typename ScalarType>
struct StorageOps<LinearCameraCal<ScalarType>> {
  using T = LinearCameraCal<ScalarType>;
  using Scalar = typename LinearCameraCal<ScalarType>::Scalar;

  static constexpr int32_t StorageDim() {
    return 4;
  }

  static void ToStorage(const T& a, ScalarType* out);
  static T FromStorage(const ScalarType* data);
};

}  // namespace sym

// Explicit instantiation
extern template struct sym::StorageOps<sym::LinearCameraCal<double>>;
extern template struct sym::StorageOps<sym::LinearCameraCal<float>>;
