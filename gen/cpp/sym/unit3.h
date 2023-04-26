// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     geo_package/CLASS.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <ostream>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "./ops/group_ops.h"
#include "./ops/lie_group_ops.h"
#include "./ops/storage_ops.h"
#include "./rot3.h"
#include "./util/epsilon.h"

namespace sym {

/**
 * Autogenerated C++ implementation of <class 'symforce.geo.unit3.Unit3'>.
 *
 * Direction in R^3, represented as a Rot3 that transforms [0, 0, 1] to the desired direction.
 * The storage is therefore a quaternion and the tangent space is 2 dimensional.
 * Most operations are implemented using operations from Rot3.
 *
 * Note: an alternative implementation could directly store a unit vector and define its boxplus
 * manifold as described in Appendix B.2 of [Hertzberg 2013]. This can be done by finding the
 * Householder reflector of x and use it to transform the exponential map of delta, which is a
 * small perturbation in the tangent space (R^2). Namely,
 *
 * x.retract(delta) = x [+] delta = Rx * Exp(delta), where
 * Exp(delta) = [sinc(||delta||) * delta, cos(||delta||)], and
 * Rx = (I - 2 vv^T / (v^Tv))X, v = x - e_z != 0, X is a matrix negating 2nd vector component
 *    = I                     , x = e_z
 *
 * [Hertzberg 2013] Integrating Generic Sensor Fusion Algorithms with Sound State Representations
 * through Encapsulation of Manifolds
 */
template <typename ScalarType>
class Unit3 {
 public:
  // Typedefs
  using Scalar = ScalarType;
  using Self = Unit3<Scalar>;
  using DataVec = Eigen::Matrix<Scalar, 4, 1>;
  using TangentVec = Eigen::Matrix<Scalar, 2, 1>;
  using SelfJacobian = Eigen::Matrix<Scalar, 2, 2>;

  // Construct from data vec
  explicit Unit3(const DataVec& data) : data_(data) {}

  // Default construct to identity
  Unit3() : Unit3(GroupOps<Self>::Identity()) {}

  // Access underlying storage as const
  inline const DataVec& Data() const {
    return data_;
  }

  // Matrix type aliases
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

  // --------------------------------------------------------------------------
  // Handwritten methods included from "custom_methods/unit3.h.jinja"
  // --------------------------------------------------------------------------

  Unit3(const Rot3<Scalar>& rotation) {
    data_ = rotation.Data();
  }

  // Generate a random element in Unit3
  template <typename Generator>
  static Unit3 Random(Generator& gen) {
    return FromVector(Rot3<Scalar>::Random(gen) * Vector3(0, 0, 1.0), kDefaultEpsilon<Scalar>);
  }

  // --------------------------------------------------------------------------
  // Custom generated methods
  // --------------------------------------------------------------------------

  const static sym::Unit3<Scalar> FromVector(const Vector3& a, const Scalar epsilon);

  const Vector3 ToUnitVector() const;

  const sym::Rot3<Scalar> ToRotation() const;

  // --------------------------------------------------------------------------
  // StorageOps concept
  // --------------------------------------------------------------------------

  static constexpr int32_t StorageDim() {
    return StorageOps<Self>::StorageDim();
  }

  void ToStorage(Scalar* const vec) const {
    return StorageOps<Self>::ToStorage(*this, vec);
  }

  static Unit3 FromStorage(const Scalar* const vec) {
    return StorageOps<Self>::FromStorage(vec);
  }

  // --------------------------------------------------------------------------
  // GroupOps concept
  // --------------------------------------------------------------------------

  static Self Identity() {
    return GroupOps<Self>::Identity();
  }

  Self Inverse() const {
    return GroupOps<Self>::Inverse(*this);
  }

  Self Compose(const Self& b) const {
    return GroupOps<Self>::Compose(*this, b);
  }

  Self Between(const Self& b) const {
    return GroupOps<Self>::Between(*this, b);
  }

  Self InverseWithJacobian(SelfJacobian* const res_D_a = nullptr) const {
    return GroupOps<Self>::InverseWithJacobian(*this, res_D_a);
  }

  Self ComposeWithJacobians(const Self& b, SelfJacobian* const res_D_a = nullptr,
                            SelfJacobian* const res_D_b = nullptr) const {
    return GroupOps<Self>::ComposeWithJacobians(*this, b, res_D_a, res_D_b);
  }

  Self BetweenWithJacobians(const Self& b, SelfJacobian* const res_D_a = nullptr,
                            SelfJacobian* const res_D_b = nullptr) const {
    return GroupOps<Self>::BetweenWithJacobians(*this, b, res_D_a, res_D_b);
  }

  // Compose shorthand
  template <typename Other>
  auto operator*(const Other& b) const -> decltype(Compose(b)) {
    return Compose(b);
  }

  // --------------------------------------------------------------------------
  // LieGroupOps concept
  // --------------------------------------------------------------------------

  static constexpr int32_t TangentDim() {
    return LieGroupOps<Self>::TangentDim();
  }

  static Self FromTangent(const TangentVec& vec, const Scalar epsilon = kDefaultEpsilon<Scalar>) {
    return LieGroupOps<Self>::FromTangent(vec, epsilon);
  }

  TangentVec ToTangent(const Scalar epsilon = kDefaultEpsilon<Scalar>) const {
    return LieGroupOps<Self>::ToTangent(*this, epsilon);
  }

  Self Retract(const TangentVec& vec, const Scalar epsilon = kDefaultEpsilon<Scalar>) const {
    return LieGroupOps<Self>::Retract(*this, vec, epsilon);
  }

  TangentVec LocalCoordinates(const Self& b, const Scalar epsilon = kDefaultEpsilon<Scalar>) const {
    return LieGroupOps<Self>::LocalCoordinates(*this, b, epsilon);
  }

  Self Interpolate(const Self b, const Scalar alpha,
                   const Scalar epsilon = kDefaultEpsilon<Scalar>) const {
    return LieGroupOps<Self>::Interpolate(*this, b, alpha, epsilon);
  }

  // --------------------------------------------------------------------------
  // General Helpers
  // --------------------------------------------------------------------------

  bool IsApprox(const Self& b, const Scalar tol) const {
    // isApprox is multiplicative so we check the norm for the exact zero case
    // https://eigen.tuxfamily.org/dox/classEigen_1_1DenseBase.html#ae8443357b808cd393be1b51974213f9c
    if (b.Data() == DataVec::Zero()) {
      return Data().norm() < tol;
    }

    return Data().isApprox(b.Data(), tol);
  }

  template <typename ToScalar>
  Unit3<ToScalar> Cast() const {
    return Unit3<ToScalar>(Data().template cast<ToScalar>());
  }

  bool operator==(const Unit3& rhs) const {
    return data_ == rhs.Data();
  }

 protected:
  DataVec data_;
};

// Shorthand for scalar types
using Unit3d = Unit3<double>;
using Unit3f = Unit3<float>;

// Print definitions
std::ostream& operator<<(std::ostream& os, const Unit3<double>& a);
std::ostream& operator<<(std::ostream& os, const Unit3<float>& a);

}  // namespace sym

// Externs to reduce duplicate instantiation
extern template class sym::Unit3<double>;
extern template class sym::Unit3<float>;

// Concept implementations for this class
#include "./ops/unit3/group_ops.h"
#include "./ops/unit3/lie_group_ops.h"
#include "./ops/unit3/storage_ops.h"
