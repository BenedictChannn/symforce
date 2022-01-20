// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/ops/CLASS/storage_ops.cc.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#include "./storage_ops.h"

namespace sym {

template <typename ScalarType>
void StorageOps<EquidistantEpipolarCameraCal<ScalarType>>::ToStorage(
    const EquidistantEpipolarCameraCal<ScalarType>& a, ScalarType* out) {
  assert(out != nullptr);
  std::copy_n(a.Data().data(), a.StorageDim(), out);
}

template <typename ScalarType>
EquidistantEpipolarCameraCal<ScalarType>
StorageOps<EquidistantEpipolarCameraCal<ScalarType>>::FromStorage(const ScalarType* data) {
  assert(data != nullptr);
  return EquidistantEpipolarCameraCal<ScalarType>(
      Eigen::Map<const typename EquidistantEpipolarCameraCal<ScalarType>::DataVec>(data));
}

}  // namespace sym

// Explicit instantiation
template struct sym::StorageOps<sym::EquidistantEpipolarCameraCal<double>>;
template struct sym::StorageOps<sym::EquidistantEpipolarCameraCal<float>>;
