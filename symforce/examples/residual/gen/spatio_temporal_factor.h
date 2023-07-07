// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Dense>

namespace event_res {

/**
 * This function was autogenerated from a symbolic function. Do not modify by hand.
 *
 * Symbolic function: spatio_temporal_residual
 *
 * Args:
 *     R_: Matrix33
 *     t_: Matrix31
 *     x: Matrix61
 *     point: Matrix31
 *     R_cam: Matrix33
 *     t_cam: Matrix31
 *     wx: Scalar
 *     wy: Scalar
 *     width_: Scalar
 *     height_: Scalar
 *     img_col: Scalar
 *     mask: Scalar
 *     TsObs: Scalar
 *
 * Outputs:
 *     res: Matrix11
 *     jacobian: (1x6) jacobian of res wrt arg x (6)
 *     hessian: (6x6) Gauss-Newton hessian for arg x (6)
 *     rhs: (6x1) Gauss-Newton rhs for arg x (6)
 */
template <typename Scalar>
void SpatioTemporalFactor(const Eigen::Matrix<Scalar, 3, 3>& R_,
                          const Eigen::Matrix<Scalar, 3, 1>& t_,
                          const Eigen::Matrix<Scalar, 6, 1>& x,
                          const Eigen::Matrix<Scalar, 3, 1>& point,
                          const Eigen::Matrix<Scalar, 3, 3>& R_cam,
                          const Eigen::Matrix<Scalar, 3, 1>& t_cam, const Scalar wx,
                          const Scalar wy, const Scalar width_, const Scalar height_,
                          const Scalar img_col, const Scalar* const mask, const Scalar* const TsObs,
                          Eigen::Matrix<Scalar, 1, 1>* const res = nullptr,
                          Eigen::Matrix<Scalar, 1, 6>* const jacobian = nullptr,
                          Eigen::Matrix<Scalar, 6, 6>* const hessian = nullptr,
                          Eigen::Matrix<Scalar, 6, 1>* const rhs = nullptr) {
  // Total ops: 305

  // Unused inputs
  (void)wx;
  (void)wy;
  (void)width_;
  (void)height_;
  (void)mask;

  // Input arrays

  // Intermediate terms (105)
  const Scalar _tmp0 = x(1, 0) * x(2, 0);
  const Scalar _tmp1 = _tmp0 - x(0, 0);
  const Scalar _tmp2 = std::pow(x(0, 0), Scalar(2));
  const Scalar _tmp3 = std::pow(x(2, 0), Scalar(2));
  const Scalar _tmp4 = std::pow(x(1, 0), Scalar(2));
  const Scalar _tmp5 = _tmp4 + Scalar(1.0);
  const Scalar _tmp6 = Scalar(1.0) / (_tmp2 + _tmp3 + _tmp5);
  const Scalar _tmp7 = Scalar(2.0) * _tmp6;
  const Scalar _tmp8 = R_(2, 0) * _tmp7;
  const Scalar _tmp9 = x(0, 0) * x(1, 0);
  const Scalar _tmp10 = _tmp9 + x(2, 0);
  const Scalar _tmp11 = R_(0, 0) * _tmp7;
  const Scalar _tmp12 = -_tmp3;
  const Scalar _tmp13 = -_tmp2;
  const Scalar _tmp14 = Scalar(1.0) * _tmp6;
  const Scalar _tmp15 = _tmp14 * (_tmp12 + _tmp13 + _tmp5);
  const Scalar _tmp16 = R_(1, 0) * _tmp15 + _tmp1 * _tmp8 + _tmp10 * _tmp11;
  const Scalar _tmp17 = x(0, 0) * x(2, 0);
  const Scalar _tmp18 = _tmp17 + x(1, 0);
  const Scalar _tmp19 = R_(2, 1) * _tmp7;
  const Scalar _tmp20 = _tmp7 * (_tmp9 - x(2, 0));
  const Scalar _tmp21 = Scalar(1.0) - _tmp4;
  const Scalar _tmp22 = _tmp14 * (_tmp12 + _tmp2 + _tmp21);
  const Scalar _tmp23 = R_(0, 1) * _tmp22 + R_(1, 1) * _tmp20 + _tmp18 * _tmp19;
  const Scalar _tmp24 = _tmp16 + _tmp23;
  const Scalar _tmp25 = _tmp7 * (_tmp0 + x(0, 0));
  const Scalar _tmp26 = _tmp17 - x(1, 0);
  const Scalar _tmp27 = R_(0, 2) * _tmp7;
  const Scalar _tmp28 = R_(2, 2) * _tmp6;
  const Scalar _tmp29 = Scalar(1.0) * _tmp13 + Scalar(1.0) * _tmp21 + Scalar(1.0) * _tmp3;
  const Scalar _tmp30 = R_(1, 2) * _tmp25 + _tmp26 * _tmp27 + _tmp28 * _tmp29;
  const Scalar _tmp31 = -_tmp30;
  const Scalar _tmp32 = R_(0, 1) * _tmp7;
  const Scalar _tmp33 = R_(1, 1) * _tmp15 + _tmp1 * _tmp19 + _tmp10 * _tmp32;
  const Scalar _tmp34 = R_(0, 0) * _tmp22 + R_(1, 0) * _tmp20 + _tmp18 * _tmp8;
  const Scalar _tmp35 = 1 - _tmp34;
  const Scalar _tmp36 = _tmp30 + _tmp33;
  const Scalar _tmp37 = -std::max<Scalar>(
      _tmp30, std::max<Scalar>(_tmp33, std::max<Scalar>(_tmp34, _tmp34 + _tmp36)));
  const Scalar _tmp38 = _tmp34 + _tmp37;
  const Scalar _tmp39 = 1 - std::max<Scalar>(0, -(((_tmp38) > 0) - ((_tmp38) < 0)));
  const Scalar _tmp40 = std::min<Scalar>(
      1 - std::max<Scalar>(0, _tmp39),
      1 - std::max<Scalar>(0, -(((_tmp33 + _tmp37) > 0) - ((_tmp33 + _tmp37) < 0))));
  const Scalar _tmp41 =
      -_tmp40 + std::sqrt(Scalar(std::max<Scalar>(0, _tmp31 + _tmp33 + _tmp35))) + 1;
  const Scalar _tmp42 = (Scalar(1) / Scalar(2)) * _tmp40;
  const Scalar _tmp43 = _tmp42 / _tmp41;
  const Scalar _tmp44 = Scalar(2.0) * _tmp28;
  const Scalar _tmp45 = R_(0, 2) * _tmp22 + R_(1, 2) * _tmp20 + _tmp18 * _tmp44;
  const Scalar _tmp46 = _tmp29 * _tmp6;
  const Scalar _tmp47 = R_(1, 0) * _tmp25 + R_(2, 0) * _tmp46 + _tmp11 * _tmp26;
  const Scalar _tmp48 = _tmp45 + _tmp47;
  const Scalar _tmp49 = std::min<Scalar>(
      1 - std::max<Scalar>(0, std::max<Scalar>(_tmp39, _tmp40)),
      1 - std::max<Scalar>(0, -(((_tmp30 + _tmp37) > 0) - ((_tmp30 + _tmp37) < 0))));
  const Scalar _tmp50 = -_tmp33;
  const Scalar _tmp51 =
      -_tmp49 + std::sqrt(Scalar(std::max<Scalar>(0, _tmp30 + _tmp35 + _tmp50))) + 1;
  const Scalar _tmp52 = (Scalar(1) / Scalar(2)) * _tmp49;
  const Scalar _tmp53 = _tmp52 / _tmp51;
  const Scalar _tmp54 = _tmp34 + 1;
  const Scalar _tmp55 =
      -_tmp39 + std::sqrt(Scalar(std::max<Scalar>(0, _tmp31 + _tmp50 + _tmp54))) + 1;
  const Scalar _tmp56 = (Scalar(1) / Scalar(2)) * _tmp39;
  const Scalar _tmp57 = R_(1, 1) * _tmp25 + R_(2, 1) * _tmp46 + _tmp26 * _tmp32;
  const Scalar _tmp58 = R_(1, 2) * _tmp15 + _tmp1 * _tmp44 + _tmp10 * _tmp27;
  const Scalar _tmp59 = -_tmp57 + _tmp58;
  const Scalar _tmp60 = std::min<Scalar>(
      1 - std::max<Scalar>(0, std::max<Scalar>(_tmp39, std::max<Scalar>(_tmp40, _tmp49))),
      1 - std::max<Scalar>(0, -(((_tmp36 + _tmp38) > 0) - ((_tmp36 + _tmp38) < 0))));
  const Scalar _tmp61 = -_tmp60 + std::sqrt(Scalar(std::max<Scalar>(0, _tmp36 + _tmp54))) + 1;
  const Scalar _tmp62 = (Scalar(1) / Scalar(2)) * _tmp60;
  const Scalar _tmp63 = _tmp62 / _tmp61;
  const Scalar _tmp64 = _tmp24 * _tmp43 + _tmp48 * _tmp53 + _tmp55 * _tmp56 + _tmp59 * _tmp63;
  const Scalar _tmp65 = -_tmp16 + _tmp23;
  const Scalar _tmp66 = _tmp56 / _tmp55;
  const Scalar _tmp67 = _tmp57 + _tmp58;
  const Scalar _tmp68 = _tmp43 * _tmp67 + _tmp48 * _tmp66 + _tmp51 * _tmp52 + _tmp63 * _tmp65;
  const Scalar _tmp69 = std::pow(_tmp68, Scalar(2));
  const Scalar _tmp70 = -_tmp45 + _tmp47;
  const Scalar _tmp71 = _tmp43 * _tmp70 + _tmp53 * _tmp65 + _tmp59 * _tmp66 + _tmp61 * _tmp62;
  const Scalar _tmp72 = _tmp24 * _tmp66 + _tmp41 * _tmp42 + _tmp53 * _tmp67 + _tmp63 * _tmp70;
  const Scalar _tmp73 = std::pow(_tmp72, Scalar(2));
  const Scalar _tmp74 = std::pow(_tmp64, Scalar(2));
  const Scalar _tmp75 = 2 / (_tmp69 + std::pow(_tmp71, Scalar(2)) + _tmp73 + _tmp74);
  const Scalar _tmp76 = _tmp64 * _tmp68 * _tmp75;
  const Scalar _tmp77 = _tmp71 * _tmp75;
  const Scalar _tmp78 = _tmp72 * _tmp77;
  const Scalar _tmp79 = _tmp76 + _tmp78;
  const Scalar _tmp80 = -_tmp69 * _tmp75;
  const Scalar _tmp81 = -_tmp73 * _tmp75 + 1;
  const Scalar _tmp82 = _tmp80 + _tmp81;
  const Scalar _tmp83 = _tmp7 * t_(2, 0);
  const Scalar _tmp84 = _tmp18 * _tmp83 + _tmp20 * t_(1, 0) + _tmp22 * t_(0, 0) + x(3, 0);
  const Scalar _tmp85 = _tmp68 * _tmp77;
  const Scalar _tmp86 = _tmp72 * _tmp75;
  const Scalar _tmp87 = _tmp64 * _tmp86;
  const Scalar _tmp88 = -_tmp85 + _tmp87;
  const Scalar _tmp89 = _tmp7 * t_(0, 0);
  const Scalar _tmp90 = _tmp25 * t_(1, 0) + _tmp26 * _tmp89 + _tmp46 * t_(2, 0) + x(5, 0);
  const Scalar _tmp91 = _tmp1 * _tmp83 + _tmp10 * _tmp89 + _tmp15 * t_(1, 0) + x(4, 0);
  const Scalar _tmp92 = -_tmp79 * _tmp90 + _tmp79 * point(2, 0) - _tmp82 * _tmp84 +
                        _tmp82 * point(0, 0) - _tmp88 * _tmp91 + _tmp88 * point(1, 0);
  const Scalar _tmp93 = _tmp85 + _tmp87;
  const Scalar _tmp94 = -_tmp74 * _tmp75;
  const Scalar _tmp95 = _tmp80 + _tmp94 + 1;
  const Scalar _tmp96 = _tmp68 * _tmp86;
  const Scalar _tmp97 = _tmp64 * _tmp77;
  const Scalar _tmp98 = _tmp96 - _tmp97;
  const Scalar _tmp99 = -_tmp84 * _tmp93 - _tmp90 * _tmp98 - _tmp91 * _tmp95 +
                        _tmp93 * point(0, 0) + _tmp95 * point(1, 0) + _tmp98 * point(2, 0);
  const Scalar _tmp100 = _tmp76 - _tmp78;
  const Scalar _tmp101 = _tmp96 + _tmp97;
  const Scalar _tmp102 = _tmp81 + _tmp94;
  const Scalar _tmp103 = -_tmp100 * _tmp84 + _tmp100 * point(0, 0) - _tmp101 * _tmp91 +
                         _tmp101 * point(1, 0) - _tmp102 * _tmp90 + _tmp102 * point(2, 0);
  const Scalar _tmp104 = Scalar(1.0) / (R_cam(2, 0) * _tmp92 + R_cam(2, 1) * _tmp99 +
                                        R_cam(2, 2) * _tmp103 + t_cam(2, 0));

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 1, 1>& _res = (*res);

    _res(0, 0) = TsObs[static_cast<size_t>(
        img_col * std::floor(_tmp104 * (R_cam(1, 0) * _tmp92 + R_cam(1, 1) * _tmp99 +
                                        R_cam(1, 2) * _tmp103 + t_cam(1, 0))) +
        std::floor(_tmp104 * (R_cam(0, 0) * _tmp92 + R_cam(0, 1) * _tmp99 + R_cam(0, 2) * _tmp103 +
                              t_cam(0, 0))))];
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 1, 6>& _jacobian = (*jacobian);

    _jacobian.setZero();
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _hessian = (*hessian);

    _hessian.setZero();
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 6, 1>& _rhs = (*rhs);

    _rhs.setZero();
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace event_res
