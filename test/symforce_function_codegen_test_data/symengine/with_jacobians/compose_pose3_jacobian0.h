// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     cpp_templates/function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>
#include <sym/pose3.h>

namespace sym {

/**
 * Composition of two elements in the group.
 *
 * Returns:
 *     res_D_a: (6x6) jacobian of res (6) wrt arg a (6)
 */
template <typename Scalar>
Eigen::Matrix<Scalar, 6, 6> ComposePose3Jacobian0(const sym::Pose3<Scalar>& a,
                                                  const sym::Pose3<Scalar>& b) {
  // Total ops: 253

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _a = a.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _b = b.Data();

  // Intermediate terms (84)
  const Scalar _tmp0 = _a[1] * _b[2];
  const Scalar _tmp1 = (Scalar(1) / Scalar(2)) * _tmp0;
  const Scalar _tmp2 = _a[0] * _b[3];
  const Scalar _tmp3 = (Scalar(1) / Scalar(2)) * _tmp2;
  const Scalar _tmp4 = -_tmp3;
  const Scalar _tmp5 = _a[3] * _b[0];
  const Scalar _tmp6 = (Scalar(1) / Scalar(2)) * _tmp5;
  const Scalar _tmp7 = _a[2] * _b[1];
  const Scalar _tmp8 = -Scalar(1) / Scalar(2) * _tmp7;
  const Scalar _tmp9 = -_tmp6 + _tmp8;
  const Scalar _tmp10 = _tmp1 + _tmp4 + _tmp9;
  const Scalar _tmp11 = 2 * _tmp0 + 2 * _tmp2 + 2 * _tmp5 - 2 * _tmp7;
  const Scalar _tmp12 = _a[1] * _b[3];
  const Scalar _tmp13 = (Scalar(1) / Scalar(2)) * _tmp12;
  const Scalar _tmp14 = -_tmp13;
  const Scalar _tmp15 = _a[0] * _b[2];
  const Scalar _tmp16 = -Scalar(1) / Scalar(2) * _tmp15;
  const Scalar _tmp17 = _a[3] * _b[1];
  const Scalar _tmp18 = (Scalar(1) / Scalar(2)) * _tmp17;
  const Scalar _tmp19 = _a[2] * _b[0];
  const Scalar _tmp20 = (Scalar(1) / Scalar(2)) * _tmp19;
  const Scalar _tmp21 = -_tmp20;
  const Scalar _tmp22 = _tmp14 + _tmp16 + _tmp18 + _tmp21;
  const Scalar _tmp23 = 2 * _tmp12 - 2 * _tmp15 + 2 * _tmp17 + 2 * _tmp19;
  const Scalar _tmp24 = _a[1] * _b[0];
  const Scalar _tmp25 = -Scalar(1) / Scalar(2) * _tmp24;
  const Scalar _tmp26 = _a[0] * _b[1];
  const Scalar _tmp27 = (Scalar(1) / Scalar(2)) * _tmp26;
  const Scalar _tmp28 = -_tmp27;
  const Scalar _tmp29 = _a[3] * _b[2];
  const Scalar _tmp30 = (Scalar(1) / Scalar(2)) * _tmp29;
  const Scalar _tmp31 = -_tmp30;
  const Scalar _tmp32 = _a[2] * _b[3];
  const Scalar _tmp33 = (Scalar(1) / Scalar(2)) * _tmp32;
  const Scalar _tmp34 = _tmp25 + _tmp28 + _tmp31 + _tmp33;
  const Scalar _tmp35 = -2 * _tmp24 + 2 * _tmp26 + 2 * _tmp29 + 2 * _tmp32;
  const Scalar _tmp36 = _a[0] * _b[0];
  const Scalar _tmp37 = (Scalar(1) / Scalar(2)) * _tmp36;
  const Scalar _tmp38 = _a[2] * _b[2];
  const Scalar _tmp39 = (Scalar(1) / Scalar(2)) * _tmp38;
  const Scalar _tmp40 = _a[1] * _b[1];
  const Scalar _tmp41 = (Scalar(1) / Scalar(2)) * _tmp40;
  const Scalar _tmp42 = _a[3] * _b[3];
  const Scalar _tmp43 = (Scalar(1) / Scalar(2)) * _tmp42;
  const Scalar _tmp44 = _tmp41 + _tmp43;
  const Scalar _tmp45 = -_tmp37 + _tmp39 + _tmp44;
  const Scalar _tmp46 = -2 * _tmp36 - 2 * _tmp38 - 2 * _tmp40 + 2 * _tmp42;
  const Scalar _tmp47 = _tmp16 - _tmp18;
  const Scalar _tmp48 = _tmp14 + _tmp20 + _tmp47;
  const Scalar _tmp49 = -_tmp1;
  const Scalar _tmp50 = _tmp3 + _tmp49 + _tmp9;
  const Scalar _tmp51 = _tmp37 + _tmp39 - _tmp41 + _tmp43;
  const Scalar _tmp52 = _tmp25 - _tmp33;
  const Scalar _tmp53 = _tmp28 + _tmp30 + _tmp52;
  const Scalar _tmp54 = _tmp27 + _tmp31 + _tmp52;
  const Scalar _tmp55 = _tmp37 - _tmp39 + _tmp44;
  const Scalar _tmp56 = _tmp4 + _tmp49 + _tmp6 + _tmp8;
  const Scalar _tmp57 = _tmp13 + _tmp21 + _tmp47;
  const Scalar _tmp58 = 2 * _a[0] * _a[1];
  const Scalar _tmp59 = -_tmp58;
  const Scalar _tmp60 = 2 * _a[3];
  const Scalar _tmp61 = _a[2] * _tmp60;
  const Scalar _tmp62 = _a[1] * _tmp60;
  const Scalar _tmp63 = 2 * _a[2];
  const Scalar _tmp64 = _a[0] * _tmp63;
  const Scalar _tmp65 = std::pow(_a[0], Scalar(2));
  const Scalar _tmp66 = std::pow(_a[2], Scalar(2));
  const Scalar _tmp67 = -_tmp66;
  const Scalar _tmp68 = _tmp65 + _tmp67;
  const Scalar _tmp69 = std::pow(_a[1], Scalar(2));
  const Scalar _tmp70 = -_tmp69;
  const Scalar _tmp71 = std::pow(_a[3], Scalar(2));
  const Scalar _tmp72 = _tmp70 + _tmp71;
  const Scalar _tmp73 = -_tmp62;
  const Scalar _tmp74 = -_tmp64;
  const Scalar _tmp75 = -_tmp71;
  const Scalar _tmp76 = _tmp69 + _tmp75;
  const Scalar _tmp77 = -_tmp65;
  const Scalar _tmp78 = _tmp66 + _tmp77;
  const Scalar _tmp79 = -_tmp61;
  const Scalar _tmp80 = _a[0] * _tmp60;
  const Scalar _tmp81 = -_tmp80;
  const Scalar _tmp82 = _a[1] * _tmp63;
  const Scalar _tmp83 = -_tmp82;

  // Output terms (1)
  Eigen::Matrix<Scalar, 6, 6> _res_D_a;

  _res_D_a(0, 0) = -_tmp10 * _tmp11 - _tmp22 * _tmp23 + _tmp34 * _tmp35 + _tmp45 * _tmp46;
  _res_D_a(0, 1) = -_tmp11 * _tmp48 - _tmp23 * _tmp50 + _tmp35 * _tmp51 + _tmp46 * _tmp53;
  _res_D_a(0, 2) = -_tmp11 * _tmp54 - _tmp23 * _tmp55 + _tmp35 * _tmp56 + _tmp46 * _tmp57;
  _res_D_a(0, 3) = 0;
  _res_D_a(0, 4) = 0;
  _res_D_a(0, 5) = 0;
  _res_D_a(1, 0) = -_tmp10 * _tmp23 + _tmp11 * _tmp22 + _tmp34 * _tmp46 - _tmp35 * _tmp45;
  _res_D_a(1, 1) = _tmp11 * _tmp50 - _tmp23 * _tmp48 - _tmp35 * _tmp53 + _tmp46 * _tmp51;
  _res_D_a(1, 2) = _tmp11 * _tmp55 - _tmp23 * _tmp54 - _tmp35 * _tmp57 + _tmp46 * _tmp56;
  _res_D_a(1, 3) = 0;
  _res_D_a(1, 4) = 0;
  _res_D_a(1, 5) = 0;
  _res_D_a(2, 0) = -_tmp10 * _tmp35 - _tmp11 * _tmp34 + _tmp22 * _tmp46 + _tmp23 * _tmp45;
  _res_D_a(2, 1) = -_tmp11 * _tmp51 + _tmp23 * _tmp53 - _tmp35 * _tmp48 + _tmp46 * _tmp50;
  _res_D_a(2, 2) = -_tmp11 * _tmp56 + _tmp23 * _tmp57 - _tmp35 * _tmp54 + _tmp46 * _tmp55;
  _res_D_a(2, 3) = 0;
  _res_D_a(2, 4) = 0;
  _res_D_a(2, 5) = 0;
  _res_D_a(3, 0) = _b[5] * (_tmp62 + _tmp64) + _b[6] * (_tmp59 + _tmp61);
  _res_D_a(3, 1) = _b[4] * (_tmp73 + _tmp74) + _b[6] * (_tmp68 + _tmp72);
  _res_D_a(3, 2) = _b[4] * (_tmp58 + _tmp79) + _b[5] * (_tmp76 + _tmp78);
  _res_D_a(3, 3) = 1;
  _res_D_a(3, 4) = 0;
  _res_D_a(3, 5) = 0;
  _res_D_a(4, 0) = _b[5] * (_tmp81 + _tmp82) + _b[6] * (_tmp65 + _tmp66 + _tmp70 + _tmp75);
  _res_D_a(4, 1) = _b[4] * (_tmp80 + _tmp83) + _b[6] * (_tmp58 + _tmp61);
  _res_D_a(4, 2) = _b[4] * (_tmp67 + _tmp69 + _tmp71 + _tmp77) + _b[5] * (_tmp59 + _tmp79);
  _res_D_a(4, 3) = 0;
  _res_D_a(4, 4) = 1;
  _res_D_a(4, 5) = 0;
  _res_D_a(5, 0) = _b[5] * (_tmp72 + _tmp78) + _b[6] * (_tmp81 + _tmp83);
  _res_D_a(5, 1) = _b[4] * (_tmp68 + _tmp76) + _b[6] * (_tmp64 + _tmp73);
  _res_D_a(5, 2) = _b[4] * (_tmp80 + _tmp82) + _b[5] * (_tmp62 + _tmp74);
  _res_D_a(5, 3) = 0;
  _res_D_a(5, 4) = 0;
  _res_D_a(5, 5) = 1;

  return _res_D_a;
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym