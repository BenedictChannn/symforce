// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/ops/CLASS/storage_ops.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <sym/atan_camera_cal.h>

namespace sym {

/**
 * C++ StorageOps implementation for <class 'symforce.cam.atan_camera_cal.ATANCameraCal'>.
 */
template <typename ScalarType>
struct StorageOps<ATANCameraCal<ScalarType>> {
  using T = ATANCameraCal<ScalarType>;
  using Scalar = typename ATANCameraCal<ScalarType>::Scalar;

  static constexpr int32_t StorageDim() {
    return 5;
  }

  static void ToStorage(const T& a, ScalarType* out);
  static T FromStorage(const ScalarType* data);
};

}  // namespace sym

// Explicit instantiation
extern template struct sym::StorageOps<sym::ATANCameraCal<double>>;
extern template struct sym::StorageOps<sym::ATANCameraCal<float>>;
