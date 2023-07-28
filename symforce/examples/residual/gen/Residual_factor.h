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
 *     width_: Scalar
 *     height_: Scalar
 *     mask_col: Scalar
 *     Ts_col: Scalar
 *     Ts_row: Scalar
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
void ResidualFactor(const Eigen::Matrix<Scalar, 3, 3>& R_, const Eigen::Matrix<Scalar, 3, 1>& t_,
                    const Eigen::Matrix<Scalar, 6, 1>& x, const Eigen::Matrix<Scalar, 3, 1>& point,
                    const Eigen::Matrix<Scalar, 3, 3>& R_cam,
                    const Eigen::Matrix<Scalar, 3, 1>& t_cam, const Scalar width_,
                    const Scalar height_, const Scalar mask_col, const Scalar Ts_col,
                    const Scalar Ts_row, const Scalar* const mask, const Scalar* const TsObs,
                    Eigen::Matrix<Scalar, 1, 1>* const res = nullptr,
                    Eigen::Matrix<Scalar, 1, 6>* const jacobian = nullptr,
                    Eigen::Matrix<Scalar, 6, 6>* const hessian = nullptr,
                    Eigen::Matrix<Scalar, 6, 1>* const rhs = nullptr) {
  // Total ops: 1098

  // Input arrays

  // Intermediate terms (322)
  const Scalar _tmp0 = 2 * x(0, 0);
  const Scalar _tmp1 = 2 * x(2, 0);
  const Scalar _tmp2 = _tmp1 * x(1, 0);
  const Scalar _tmp3 = _tmp0 + _tmp2;
  const Scalar _tmp4 = std::pow(x(0, 0), Scalar(2));
  const Scalar _tmp5 = std::pow(x(2, 0), Scalar(2));
  const Scalar _tmp6 = std::pow(x(1, 0), Scalar(2));
  const Scalar _tmp7 = _tmp4 + _tmp5 + _tmp6;
  const Scalar _tmp8 = _tmp7 + Scalar(1.0);
  const Scalar _tmp9 = Scalar(1.0) / (_tmp8);
  const Scalar _tmp10 = _tmp9 * t_(1, 0);
  const Scalar _tmp11 = 2 * x(1, 0);
  const Scalar _tmp12 = _tmp1 * x(0, 0);
  const Scalar _tmp13 = -_tmp11 + _tmp12;
  const Scalar _tmp14 = _tmp13 * _tmp9;
  const Scalar _tmp15 = Scalar(1.0) - _tmp7;
  const Scalar _tmp16 = _tmp15 + 2 * _tmp5;
  const Scalar _tmp17 = _tmp9 * t_(2, 0);
  const Scalar _tmp18 = _tmp10 * _tmp3 + _tmp14 * t_(0, 0) + _tmp16 * _tmp17 + x(5, 0);
  const Scalar _tmp19 = _tmp3 * _tmp9;
  const Scalar _tmp20 = R_(1, 2) * _tmp19;
  const Scalar _tmp21 = R_(0, 2) * _tmp14;
  const Scalar _tmp22 = _tmp16 * _tmp9;
  const Scalar _tmp23 = R_(2, 2) * _tmp22;
  const Scalar _tmp24 = _tmp20 + _tmp21 + _tmp23;
  const Scalar _tmp25 = _tmp11 + _tmp12;
  const Scalar _tmp26 = _tmp0 * x(1, 0);
  const Scalar _tmp27 = -_tmp1 + _tmp26;
  const Scalar _tmp28 = _tmp15 + 2 * _tmp4;
  const Scalar _tmp29 = _tmp9 * t_(0, 0);
  const Scalar _tmp30 = _tmp10 * _tmp27 + _tmp17 * _tmp25 + _tmp28 * _tmp29 + x(3, 0);
  const Scalar _tmp31 = _tmp25 * _tmp9;
  const Scalar _tmp32 = R_(2, 2) * _tmp31;
  const Scalar _tmp33 = _tmp27 * _tmp9;
  const Scalar _tmp34 = R_(1, 2) * _tmp33;
  const Scalar _tmp35 = _tmp28 * _tmp9;
  const Scalar _tmp36 = R_(0, 2) * _tmp35;
  const Scalar _tmp37 = _tmp32 + _tmp34 + _tmp36;
  const Scalar _tmp38 = -_tmp0 + _tmp2;
  const Scalar _tmp39 = _tmp38 * _tmp9;
  const Scalar _tmp40 = R_(2, 2) * _tmp39;
  const Scalar _tmp41 = _tmp1 + _tmp26;
  const Scalar _tmp42 = _tmp41 * _tmp9;
  const Scalar _tmp43 = R_(0, 2) * _tmp42;
  const Scalar _tmp44 = _tmp15 + 2 * _tmp6;
  const Scalar _tmp45 = _tmp44 * _tmp9;
  const Scalar _tmp46 = R_(1, 2) * _tmp45;
  const Scalar _tmp47 = _tmp40 + _tmp43 + _tmp46;
  const Scalar _tmp48 = _tmp17 * _tmp38 + _tmp29 * _tmp41 + _tmp45 * t_(1, 0) + x(4, 0);
  const Scalar _tmp49 = -_tmp18 * _tmp24 + _tmp24 * point(2, 0) - _tmp30 * _tmp37 +
                        _tmp37 * point(0, 0) - _tmp47 * _tmp48 + _tmp47 * point(1, 0);
  const Scalar _tmp50 = R_(2, 0) * _tmp9;
  const Scalar _tmp51 = _tmp38 * _tmp50;
  const Scalar _tmp52 = R_(0, 0) * _tmp42;
  const Scalar _tmp53 = R_(1, 0) * _tmp45;
  const Scalar _tmp54 = _tmp51 + _tmp52 + _tmp53;
  const Scalar _tmp55 = R_(1, 0) * _tmp19;
  const Scalar _tmp56 = R_(0, 0) * _tmp14;
  const Scalar _tmp57 = _tmp16 * _tmp50;
  const Scalar _tmp58 = _tmp55 + _tmp56 + _tmp57;
  const Scalar _tmp59 = _tmp25 * _tmp50;
  const Scalar _tmp60 = R_(1, 0) * _tmp33;
  const Scalar _tmp61 = R_(0, 0) * _tmp35;
  const Scalar _tmp62 = _tmp59 + _tmp60 + _tmp61;
  const Scalar _tmp63 = -_tmp18 * _tmp58 - _tmp30 * _tmp62 - _tmp48 * _tmp54 +
                        _tmp54 * point(1, 0) + _tmp58 * point(2, 0) + _tmp62 * point(0, 0);
  const Scalar _tmp64 = R_(1, 1) * _tmp19;
  const Scalar _tmp65 = R_(0, 1) * _tmp14;
  const Scalar _tmp66 = R_(2, 1) * _tmp22;
  const Scalar _tmp67 = _tmp64 + _tmp65 + _tmp66;
  const Scalar _tmp68 = R_(2, 1) * _tmp31;
  const Scalar _tmp69 = R_(1, 1) * _tmp33;
  const Scalar _tmp70 = R_(0, 1) * _tmp35;
  const Scalar _tmp71 = _tmp68 + _tmp69 + _tmp70;
  const Scalar _tmp72 = R_(2, 1) * _tmp39;
  const Scalar _tmp73 = R_(0, 1) * _tmp42;
  const Scalar _tmp74 = R_(1, 1) * _tmp45;
  const Scalar _tmp75 = _tmp72 + _tmp73 + _tmp74;
  const Scalar _tmp76 = -_tmp18 * _tmp67 - _tmp30 * _tmp71 - _tmp48 * _tmp75 +
                        _tmp67 * point(2, 0) + _tmp71 * point(0, 0) + _tmp75 * point(1, 0);
  const Scalar _tmp77 =
      R_cam(1, 0) * _tmp63 + R_cam(1, 1) * _tmp76 + R_cam(1, 2) * _tmp49 + t_cam(1, 0);
  const Scalar _tmp78 =
      R_cam(2, 0) * _tmp63 + R_cam(2, 1) * _tmp76 + R_cam(2, 2) * _tmp49 + t_cam(2, 0);
  const Scalar _tmp79 = Scalar(1.0) / (_tmp78);
  const Scalar _tmp80 = _tmp77 * _tmp79;
  const Scalar _tmp81 =
      R_cam(0, 0) * _tmp63 + R_cam(0, 1) * _tmp76 + R_cam(0, 2) * _tmp49 + t_cam(0, 0);
  const Scalar _tmp82 = _tmp79 * _tmp81;
  const Scalar _tmp83 = std::min<Scalar>(
      1 - std::max<Scalar>(0,
                           -(((mask[static_cast<size_t>(_tmp80 * mask_col + _tmp82)] - 125) > 0) -
                             ((mask[static_cast<size_t>(_tmp80 * mask_col + _tmp82)] - 125) < 0))),
      1 - std::max<Scalar>(
              0, std::max<Scalar>(
                     (((_tmp80 - height_ + 2) > 0) - ((_tmp80 - height_ + 2) < 0)),
                     std::max<Scalar>(
                         (((_tmp82 - width_ + 2) > 0) - ((_tmp82 - width_ + 2) < 0)),
                         std::max<Scalar>(
                             1 - std::max<Scalar>(0, (((_tmp80) > 0) - ((_tmp80) < 0))),
                             1 - std::max<Scalar>(0, (((_tmp82) > 0) - ((_tmp82) < 0))))))));
  const Scalar _tmp84 = std::floor(_tmp80);
  const Scalar _tmp85 = std::floor(_tmp82);
  const Scalar _tmp86 =
      1 - std::max<Scalar>(
              1 - std::max<Scalar>(0, -(((-Ts_col + _tmp85) > 0) - ((-Ts_col + _tmp85) < 0))),
              1 - std::max<Scalar>(0, -(((-Ts_row + _tmp84) > 0) - ((-Ts_row + _tmp84) < 0))));
  const Scalar _tmp87 = _tmp82 - _tmp85;
  const Scalar _tmp88 = Ts_col * _tmp84 + _tmp85;
  const Scalar _tmp89 = TsObs[static_cast<size_t>(_tmp88 + 1)];
  const Scalar _tmp90 = -_tmp82 + _tmp85 + 1;
  const Scalar _tmp91 = TsObs[static_cast<size_t>(_tmp88)];
  const Scalar _tmp92 = _tmp87 * _tmp89 + _tmp90 * _tmp91;
  const Scalar _tmp93 = _tmp80 - _tmp84;
  const Scalar _tmp94 = _tmp84 + 1;
  const Scalar _tmp95 = Ts_col * _tmp94 + _tmp85;
  const Scalar _tmp96 = TsObs[static_cast<size_t>(_tmp95 + 1)];
  const Scalar _tmp97 = TsObs[static_cast<size_t>(_tmp95)];
  const Scalar _tmp98 = _tmp87 * _tmp96 + _tmp90 * _tmp97;
  const Scalar _tmp99 = -_tmp80 + _tmp94;
  const Scalar _tmp100 =
      _tmp83 * (_tmp86 * (_tmp92 * _tmp93 + _tmp98 * _tmp99 + Scalar(2.2204460492503099e-15)) -
                Scalar(255.0) * _tmp86 + Scalar(255.0)) -
      Scalar(255.0) * _tmp83 + Scalar(255.0);
  const Scalar _tmp101 = Scalar(50.0) / _tmp100;
  const Scalar _tmp102 =
      std::max<Scalar>(0, (((_tmp100 + Scalar(-50.0)) > 0) - ((_tmp100 + Scalar(-50.0)) < 0)));
  const Scalar _tmp103 = _tmp100 * _tmp102;
  const Scalar _tmp104 = 1 - _tmp102;
  const Scalar _tmp105 = _tmp100 * _tmp104;
  const Scalar _tmp106 = std::max<Scalar>(0, std::sqrt(_tmp101) * _tmp103 + _tmp105);
  const Scalar _tmp107 = std::pow(_tmp8, Scalar(-2));
  const Scalar _tmp108 = _tmp0 * _tmp107;
  const Scalar _tmp109 = R_(2, 0) * _tmp108;
  const Scalar _tmp110 = _tmp108 * _tmp27;
  const Scalar _tmp111 = R_(0, 0) * _tmp108;
  const Scalar _tmp112 = _tmp0 * _tmp9;
  const Scalar _tmp113 = _tmp11 * _tmp9;
  const Scalar _tmp114 = R_(0, 0) * _tmp112 + R_(1, 0) * _tmp113 + _tmp1 * _tmp50;
  const Scalar _tmp115 = -R_(1, 0) * _tmp110 - _tmp109 * _tmp25 - _tmp111 * _tmp28 + _tmp114;
  const Scalar _tmp116 = 2 * _tmp9;
  const Scalar _tmp117 = R_(1, 0) * _tmp116;
  const Scalar _tmp118 = _tmp108 * _tmp3;
  const Scalar _tmp119 = _tmp1 * _tmp9;
  const Scalar _tmp120 = R_(0, 0) * _tmp119;
  const Scalar _tmp121 = _tmp0 * _tmp50;
  const Scalar _tmp122 =
      -R_(1, 0) * _tmp118 - _tmp109 * _tmp16 - _tmp111 * _tmp13 + _tmp117 + _tmp120 - _tmp121;
  const Scalar _tmp123 = _tmp108 * t_(1, 0);
  const Scalar _tmp124 = _tmp116 * t_(2, 0);
  const Scalar _tmp125 = _tmp38 * t_(2, 0);
  const Scalar _tmp126 = _tmp41 * t_(0, 0);
  const Scalar _tmp127 = _tmp0 * _tmp10;
  const Scalar _tmp128 = _tmp11 * _tmp29;
  const Scalar _tmp129 =
      -_tmp108 * _tmp125 - _tmp108 * _tmp126 - _tmp123 * _tmp44 - _tmp124 - _tmp127 + _tmp128;
  const Scalar _tmp130 = 2 * _tmp50;
  const Scalar _tmp131 = _tmp108 * _tmp44;
  const Scalar _tmp132 = R_(0, 0) * _tmp113;
  const Scalar _tmp133 = R_(1, 0) * _tmp112;
  const Scalar _tmp134 =
      -R_(1, 0) * _tmp131 - _tmp109 * _tmp38 - _tmp111 * _tmp41 - _tmp130 + _tmp132 - _tmp133;
  const Scalar _tmp135 = _tmp108 * _tmp25;
  const Scalar _tmp136 = _tmp0 * _tmp29 + _tmp1 * _tmp17 + _tmp10 * _tmp11;
  const Scalar _tmp137 =
      -_tmp108 * _tmp28 * t_(0, 0) - _tmp110 * t_(1, 0) - _tmp135 * t_(2, 0) + _tmp136;
  const Scalar _tmp138 = _tmp108 * _tmp16;
  const Scalar _tmp139 = _tmp116 * t_(1, 0);
  const Scalar _tmp140 = _tmp13 * t_(0, 0);
  const Scalar _tmp141 = _tmp1 * _tmp29;
  const Scalar _tmp142 = _tmp0 * _tmp17;
  const Scalar _tmp143 =
      -_tmp108 * _tmp140 - _tmp123 * _tmp3 - _tmp138 * t_(2, 0) + _tmp139 + _tmp141 - _tmp142;
  const Scalar _tmp144 = -_tmp115 * _tmp30 + _tmp115 * point(0, 0) - _tmp122 * _tmp18 +
                         _tmp122 * point(2, 0) - _tmp129 * _tmp54 - _tmp134 * _tmp48 +
                         _tmp134 * point(1, 0) - _tmp137 * _tmp62 - _tmp143 * _tmp58;
  const Scalar _tmp145 = R_(2, 2) * _tmp116;
  const Scalar _tmp146 = R_(0, 2) * _tmp41;
  const Scalar _tmp147 = R_(2, 2) * _tmp38;
  const Scalar _tmp148 = R_(0, 2) * _tmp113;
  const Scalar _tmp149 = R_(1, 2) * _tmp112;
  const Scalar _tmp150 =
      -R_(1, 2) * _tmp131 - _tmp108 * _tmp146 - _tmp108 * _tmp147 - _tmp145 + _tmp148 - _tmp149;
  const Scalar _tmp151 = R_(2, 2) * _tmp25;
  const Scalar _tmp152 = R_(0, 2) * _tmp108;
  const Scalar _tmp153 = R_(0, 2) * _tmp112 + R_(1, 2) * _tmp113 + R_(2, 2) * _tmp119;
  const Scalar _tmp154 = -R_(1, 2) * _tmp110 - _tmp108 * _tmp151 - _tmp152 * _tmp28 + _tmp153;
  const Scalar _tmp155 = R_(1, 2) * _tmp116;
  const Scalar _tmp156 = R_(0, 2) * _tmp119;
  const Scalar _tmp157 = R_(2, 2) * _tmp112;
  const Scalar _tmp158 =
      -R_(1, 2) * _tmp118 - R_(2, 2) * _tmp138 - _tmp13 * _tmp152 + _tmp155 + _tmp156 - _tmp157;
  const Scalar _tmp159 = -_tmp129 * _tmp47 - _tmp137 * _tmp37 - _tmp143 * _tmp24 -
                         _tmp150 * _tmp48 + _tmp150 * point(1, 0) - _tmp154 * _tmp30 +
                         _tmp154 * point(0, 0) - _tmp158 * _tmp18 + _tmp158 * point(2, 0);
  const Scalar _tmp160 = R_(0, 1) * _tmp108;
  const Scalar _tmp161 = R_(0, 1) * _tmp112 + R_(1, 1) * _tmp113 + R_(2, 1) * _tmp119;
  const Scalar _tmp162 = -R_(1, 1) * _tmp110 - R_(2, 1) * _tmp135 - _tmp160 * _tmp28 + _tmp161;
  const Scalar _tmp163 = R_(1, 1) * _tmp116;
  const Scalar _tmp164 = R_(2, 1) * _tmp16;
  const Scalar _tmp165 = R_(0, 1) * _tmp119;
  const Scalar _tmp166 = R_(2, 1) * _tmp112;
  const Scalar _tmp167 =
      -R_(1, 1) * _tmp118 - _tmp108 * _tmp164 - _tmp13 * _tmp160 + _tmp163 + _tmp165 - _tmp166;
  const Scalar _tmp168 = R_(2, 1) * _tmp116;
  const Scalar _tmp169 = R_(2, 1) * _tmp38;
  const Scalar _tmp170 = R_(0, 1) * _tmp113;
  const Scalar _tmp171 = R_(1, 1) * _tmp112;
  const Scalar _tmp172 =
      -R_(1, 1) * _tmp131 - _tmp108 * _tmp169 - _tmp160 * _tmp41 - _tmp168 + _tmp170 - _tmp171;
  const Scalar _tmp173 = -_tmp129 * _tmp75 - _tmp137 * _tmp71 - _tmp143 * _tmp67 -
                         _tmp162 * _tmp30 + _tmp162 * point(0, 0) - _tmp167 * _tmp18 +
                         _tmp167 * point(2, 0) - _tmp172 * _tmp48 + _tmp172 * point(1, 0);
  const Scalar _tmp174 =
      _tmp79 * (R_cam(0, 0) * _tmp144 + R_cam(0, 1) * _tmp173 + R_cam(0, 2) * _tmp159);
  const Scalar _tmp175 = std::pow(_tmp78, Scalar(-2));
  const Scalar _tmp176 =
      _tmp175 * (R_cam(2, 0) * _tmp144 + R_cam(2, 1) * _tmp173 + R_cam(2, 2) * _tmp159);
  const Scalar _tmp177 = _tmp176 * _tmp81;
  const Scalar _tmp178 = -_tmp174 + _tmp177;
  const Scalar _tmp179 = _tmp174 - _tmp177;
  const Scalar _tmp180 =
      _tmp79 * (R_cam(1, 0) * _tmp144 + R_cam(1, 1) * _tmp173 + R_cam(1, 2) * _tmp159);
  const Scalar _tmp181 = _tmp176 * _tmp77;
  const Scalar _tmp182 =
      _tmp92 * (_tmp180 - _tmp181) + _tmp93 * (_tmp178 * _tmp91 + _tmp179 * _tmp89) +
      _tmp98 * (-_tmp180 + _tmp181) + _tmp99 * (_tmp178 * _tmp97 + _tmp179 * _tmp96);
  const Scalar _tmp183 = std::sqrt(_tmp101);
  const Scalar _tmp184 = _tmp83 * _tmp86;
  const Scalar _tmp185 = _tmp102 * _tmp184;
  const Scalar _tmp186 = _tmp183 * _tmp185;
  const Scalar _tmp187 = _tmp104 * _tmp184;
  const Scalar _tmp188 = (Scalar(1) / Scalar(2)) * _tmp101 * _tmp185 / _tmp183;
  const Scalar _tmp189 = _tmp182 * _tmp186 + _tmp182 * _tmp187 - _tmp182 * _tmp188;
  const Scalar _tmp190 =
      (((_tmp103 * _tmp183 + _tmp105) > 0) - ((_tmp103 * _tmp183 + _tmp105) < 0)) + 1;
  const Scalar _tmp191 = (Scalar(1) / Scalar(2)) * _tmp190;
  const Scalar _tmp192 = _tmp189 * _tmp191;
  const Scalar _tmp193 = _tmp107 * _tmp11;
  const Scalar _tmp194 = R_(1, 2) * _tmp193;
  const Scalar _tmp195 = _tmp193 * _tmp28;
  const Scalar _tmp196 =
      -R_(0, 2) * _tmp195 + _tmp145 - _tmp148 + _tmp149 - _tmp151 * _tmp193 - _tmp194 * _tmp27;
  const Scalar _tmp197 = _tmp193 * _tmp44;
  const Scalar _tmp198 = -R_(1, 2) * _tmp197 - _tmp146 * _tmp193 - _tmp147 * _tmp193 + _tmp153;
  const Scalar _tmp199 = R_(0, 2) * _tmp116;
  const Scalar _tmp200 = _tmp16 * _tmp193;
  const Scalar _tmp201 = R_(2, 2) * _tmp113;
  const Scalar _tmp202 = R_(1, 2) * _tmp119;
  const Scalar _tmp203 = -R_(0, 2) * _tmp13 * _tmp193 - R_(2, 2) * _tmp200 - _tmp194 * _tmp3 -
                         _tmp199 - _tmp201 + _tmp202;
  const Scalar _tmp204 = _tmp116 * t_(0, 0);
  const Scalar _tmp205 = _tmp193 * t_(1, 0);
  const Scalar _tmp206 = _tmp1 * _tmp10;
  const Scalar _tmp207 = _tmp11 * _tmp17;
  const Scalar _tmp208 =
      -_tmp140 * _tmp193 - _tmp200 * t_(2, 0) - _tmp204 - _tmp205 * _tmp3 + _tmp206 - _tmp207;
  const Scalar _tmp209 = _tmp25 * t_(2, 0);
  const Scalar _tmp210 =
      _tmp124 + _tmp127 - _tmp128 - _tmp193 * _tmp209 - _tmp195 * t_(0, 0) - _tmp205 * _tmp27;
  const Scalar _tmp211 = -_tmp125 * _tmp193 - _tmp126 * _tmp193 + _tmp136 - _tmp197 * t_(1, 0);
  const Scalar _tmp212 = -_tmp18 * _tmp203 - _tmp196 * _tmp30 + _tmp196 * point(0, 0) -
                         _tmp198 * _tmp48 + _tmp198 * point(1, 0) + _tmp203 * point(2, 0) -
                         _tmp208 * _tmp24 - _tmp210 * _tmp37 - _tmp211 * _tmp47;
  const Scalar _tmp213 = R_(0, 1) * _tmp193;
  const Scalar _tmp214 = -R_(1, 1) * _tmp197 + _tmp161 - _tmp169 * _tmp193 - _tmp213 * _tmp41;
  const Scalar _tmp215 = R_(0, 1) * _tmp116;
  const Scalar _tmp216 = R_(1, 1) * _tmp193;
  const Scalar _tmp217 = R_(2, 1) * _tmp113;
  const Scalar _tmp218 = R_(1, 1) * _tmp119;
  const Scalar _tmp219 =
      -_tmp13 * _tmp213 - _tmp164 * _tmp193 - _tmp215 - _tmp216 * _tmp3 - _tmp217 + _tmp218;
  const Scalar _tmp220 = R_(2, 1) * _tmp25;
  const Scalar _tmp221 =
      -R_(0, 1) * _tmp195 + _tmp168 - _tmp170 + _tmp171 - _tmp193 * _tmp220 - _tmp216 * _tmp27;
  const Scalar _tmp222 = -_tmp18 * _tmp219 - _tmp208 * _tmp67 - _tmp210 * _tmp71 -
                         _tmp211 * _tmp75 - _tmp214 * _tmp48 + _tmp214 * point(1, 0) +
                         _tmp219 * point(2, 0) - _tmp221 * _tmp30 + _tmp221 * point(0, 0);
  const Scalar _tmp223 = R_(2, 0) * _tmp25;
  const Scalar _tmp224 = R_(1, 0) * _tmp193;
  const Scalar _tmp225 =
      -R_(0, 0) * _tmp195 + _tmp130 - _tmp132 + _tmp133 - _tmp193 * _tmp223 - _tmp224 * _tmp27;
  const Scalar _tmp226 = R_(0, 0) * _tmp193;
  const Scalar _tmp227 =
      -R_(1, 0) * _tmp197 - R_(2, 0) * _tmp193 * _tmp38 + _tmp114 - _tmp226 * _tmp41;
  const Scalar _tmp228 = R_(0, 0) * _tmp116;
  const Scalar _tmp229 = _tmp11 * _tmp50;
  const Scalar _tmp230 = R_(1, 0) * _tmp119;
  const Scalar _tmp231 =
      -R_(2, 0) * _tmp200 - _tmp13 * _tmp226 - _tmp224 * _tmp3 - _tmp228 - _tmp229 + _tmp230;
  const Scalar _tmp232 = -_tmp18 * _tmp231 - _tmp208 * _tmp58 - _tmp210 * _tmp62 -
                         _tmp211 * _tmp54 - _tmp225 * _tmp30 + _tmp225 * point(0, 0) -
                         _tmp227 * _tmp48 + _tmp227 * point(1, 0) + _tmp231 * point(2, 0);
  const Scalar _tmp233 =
      _tmp175 * (R_cam(2, 0) * _tmp232 + R_cam(2, 1) * _tmp222 + R_cam(2, 2) * _tmp212);
  const Scalar _tmp234 = _tmp233 * _tmp81;
  const Scalar _tmp235 =
      _tmp79 * (R_cam(0, 0) * _tmp232 + R_cam(0, 1) * _tmp222 + R_cam(0, 2) * _tmp212);
  const Scalar _tmp236 = _tmp234 - _tmp235;
  const Scalar _tmp237 = -_tmp234 + _tmp235;
  const Scalar _tmp238 = _tmp233 * _tmp77;
  const Scalar _tmp239 =
      _tmp79 * (R_cam(1, 0) * _tmp232 + R_cam(1, 1) * _tmp222 + R_cam(1, 2) * _tmp212);
  const Scalar _tmp240 =
      _tmp92 * (-_tmp238 + _tmp239) + _tmp93 * (_tmp236 * _tmp91 + _tmp237 * _tmp89) +
      _tmp98 * (_tmp238 - _tmp239) + _tmp99 * (_tmp236 * _tmp97 + _tmp237 * _tmp96);
  const Scalar _tmp241 = _tmp186 * _tmp240 + _tmp187 * _tmp240 - _tmp188 * _tmp240;
  const Scalar _tmp242 = _tmp191 * _tmp241;
  const Scalar _tmp243 = _tmp1 * _tmp107;
  const Scalar _tmp244 = R_(1, 1) * _tmp243;
  const Scalar _tmp245 = _tmp243 * _tmp28;
  const Scalar _tmp246 =
      -R_(0, 1) * _tmp245 - _tmp163 - _tmp165 + _tmp166 - _tmp220 * _tmp243 - _tmp244 * _tmp27;
  const Scalar _tmp247 = _tmp243 * t_(1, 0);
  const Scalar _tmp248 =
      -_tmp139 - _tmp141 + _tmp142 - _tmp209 * _tmp243 - _tmp245 * t_(0, 0) - _tmp247 * _tmp27;
  const Scalar _tmp249 = _tmp243 * _tmp41;
  const Scalar _tmp250 =
      -R_(0, 1) * _tmp249 - _tmp169 * _tmp243 + _tmp215 + _tmp217 - _tmp218 - _tmp244 * _tmp44;
  const Scalar _tmp251 = _tmp13 * _tmp243;
  const Scalar _tmp252 = -R_(0, 1) * _tmp251 + _tmp161 - _tmp164 * _tmp243 - _tmp244 * _tmp3;
  const Scalar _tmp253 = _tmp16 * _tmp243;
  const Scalar _tmp254 = _tmp136 - _tmp140 * _tmp243 - _tmp247 * _tmp3 - _tmp253 * t_(2, 0);
  const Scalar _tmp255 =
      -_tmp125 * _tmp243 - _tmp126 * _tmp243 + _tmp204 - _tmp206 + _tmp207 - _tmp247 * _tmp44;
  const Scalar _tmp256 = -_tmp18 * _tmp252 - _tmp246 * _tmp30 + _tmp246 * point(0, 0) -
                         _tmp248 * _tmp71 - _tmp250 * _tmp48 + _tmp250 * point(1, 0) +
                         _tmp252 * point(2, 0) - _tmp254 * _tmp67 - _tmp255 * _tmp75;
  const Scalar _tmp257 = R_(1, 2) * _tmp243;
  const Scalar _tmp258 =
      -R_(0, 2) * _tmp245 - _tmp151 * _tmp243 - _tmp155 - _tmp156 + _tmp157 - _tmp257 * _tmp27;
  const Scalar _tmp259 = _tmp243 * _tmp38;
  const Scalar _tmp260 =
      -R_(2, 2) * _tmp259 - _tmp146 * _tmp243 + _tmp199 + _tmp201 - _tmp202 - _tmp257 * _tmp44;
  const Scalar _tmp261 = -R_(0, 2) * _tmp251 - R_(2, 2) * _tmp253 + _tmp153 - _tmp257 * _tmp3;
  const Scalar _tmp262 = -_tmp18 * _tmp261 - _tmp24 * _tmp254 - _tmp248 * _tmp37 -
                         _tmp255 * _tmp47 - _tmp258 * _tmp30 + _tmp258 * point(0, 0) -
                         _tmp260 * _tmp48 + _tmp260 * point(1, 0) + _tmp261 * point(2, 0);
  const Scalar _tmp263 = R_(1, 0) * _tmp243;
  const Scalar _tmp264 =
      -R_(0, 0) * _tmp245 - _tmp117 - _tmp120 + _tmp121 - _tmp223 * _tmp243 - _tmp263 * _tmp27;
  const Scalar _tmp265 =
      -R_(0, 0) * _tmp249 - R_(2, 0) * _tmp259 + _tmp228 + _tmp229 - _tmp230 - _tmp263 * _tmp44;
  const Scalar _tmp266 = -R_(0, 0) * _tmp251 - R_(2, 0) * _tmp253 + _tmp114 - _tmp263 * _tmp3;
  const Scalar _tmp267 = -_tmp18 * _tmp266 - _tmp248 * _tmp62 - _tmp254 * _tmp58 -
                         _tmp255 * _tmp54 - _tmp264 * _tmp30 + _tmp264 * point(0, 0) -
                         _tmp265 * _tmp48 + _tmp265 * point(1, 0) + _tmp266 * point(2, 0);
  const Scalar _tmp268 =
      _tmp175 * (R_cam(2, 0) * _tmp267 + R_cam(2, 1) * _tmp256 + R_cam(2, 2) * _tmp262);
  const Scalar _tmp269 = _tmp268 * _tmp81;
  const Scalar _tmp270 =
      _tmp79 * (R_cam(0, 0) * _tmp267 + R_cam(0, 1) * _tmp256 + R_cam(0, 2) * _tmp262);
  const Scalar _tmp271 = -_tmp269 + _tmp270;
  const Scalar _tmp272 = _tmp269 - _tmp270;
  const Scalar _tmp273 = _tmp268 * _tmp77;
  const Scalar _tmp274 =
      _tmp79 * (R_cam(1, 0) * _tmp267 + R_cam(1, 1) * _tmp256 + R_cam(1, 2) * _tmp262);
  const Scalar _tmp275 =
      _tmp92 * (-_tmp273 + _tmp274) + _tmp93 * (_tmp271 * _tmp89 + _tmp272 * _tmp91) +
      _tmp98 * (_tmp273 - _tmp274) + _tmp99 * (_tmp271 * _tmp96 + _tmp272 * _tmp97);
  const Scalar _tmp276 = _tmp186 * _tmp275 + _tmp187 * _tmp275 - _tmp188 * _tmp275;
  const Scalar _tmp277 = _tmp191 * _tmp276;
  const Scalar _tmp278 = -_tmp32 - _tmp34 - _tmp36;
  const Scalar _tmp279 = -_tmp59 - _tmp60 - _tmp61;
  const Scalar _tmp280 = -_tmp68 - _tmp69 - _tmp70;
  const Scalar _tmp281 =
      _tmp79 * (R_cam(0, 0) * _tmp279 + R_cam(0, 1) * _tmp280 + R_cam(0, 2) * _tmp278);
  const Scalar _tmp282 =
      _tmp175 * (R_cam(2, 0) * _tmp279 + R_cam(2, 1) * _tmp280 + R_cam(2, 2) * _tmp278);
  const Scalar _tmp283 = _tmp282 * _tmp81;
  const Scalar _tmp284 = -_tmp281 + _tmp283;
  const Scalar _tmp285 = _tmp281 - _tmp283;
  const Scalar _tmp286 =
      _tmp79 * (R_cam(1, 0) * _tmp279 + R_cam(1, 1) * _tmp280 + R_cam(1, 2) * _tmp278);
  const Scalar _tmp287 = _tmp282 * _tmp77;
  const Scalar _tmp288 =
      _tmp92 * (_tmp286 - _tmp287) + _tmp93 * (_tmp284 * _tmp91 + _tmp285 * _tmp89) +
      _tmp98 * (-_tmp286 + _tmp287) + _tmp99 * (_tmp284 * _tmp97 + _tmp285 * _tmp96);
  const Scalar _tmp289 = _tmp186 * _tmp288 + _tmp187 * _tmp288 - _tmp188 * _tmp288;
  const Scalar _tmp290 = _tmp191 * _tmp289;
  const Scalar _tmp291 = -_tmp40 - _tmp43 - _tmp46;
  const Scalar _tmp292 = -_tmp72 - _tmp73 - _tmp74;
  const Scalar _tmp293 = -_tmp51 - _tmp52 - _tmp53;
  const Scalar _tmp294 =
      _tmp79 * (R_cam(0, 0) * _tmp293 + R_cam(0, 1) * _tmp292 + R_cam(0, 2) * _tmp291);
  const Scalar _tmp295 =
      _tmp175 * (R_cam(2, 0) * _tmp293 + R_cam(2, 1) * _tmp292 + R_cam(2, 2) * _tmp291);
  const Scalar _tmp296 = _tmp295 * _tmp81;
  const Scalar _tmp297 = _tmp294 - _tmp296;
  const Scalar _tmp298 = -_tmp294 + _tmp296;
  const Scalar _tmp299 =
      _tmp79 * (R_cam(1, 0) * _tmp293 + R_cam(1, 1) * _tmp292 + R_cam(1, 2) * _tmp291);
  const Scalar _tmp300 = _tmp295 * _tmp77;
  const Scalar _tmp301 =
      _tmp92 * (_tmp299 - _tmp300) + _tmp93 * (_tmp297 * _tmp89 + _tmp298 * _tmp91) +
      _tmp98 * (-_tmp299 + _tmp300) + _tmp99 * (_tmp297 * _tmp96 + _tmp298 * _tmp97);
  const Scalar _tmp302 = _tmp186 * _tmp301 + _tmp187 * _tmp301 - _tmp188 * _tmp301;
  const Scalar _tmp303 = _tmp191 * _tmp302;
  const Scalar _tmp304 = -_tmp64 - _tmp65 - _tmp66;
  const Scalar _tmp305 = -_tmp55 - _tmp56 - _tmp57;
  const Scalar _tmp306 = -_tmp20 - _tmp21 - _tmp23;
  const Scalar _tmp307 =
      _tmp79 * (R_cam(0, 0) * _tmp305 + R_cam(0, 1) * _tmp304 + R_cam(0, 2) * _tmp306);
  const Scalar _tmp308 =
      _tmp175 * (R_cam(2, 0) * _tmp305 + R_cam(2, 1) * _tmp304 + R_cam(2, 2) * _tmp306);
  const Scalar _tmp309 = _tmp308 * _tmp81;
  const Scalar _tmp310 = _tmp307 - _tmp309;
  const Scalar _tmp311 = -_tmp307 + _tmp309;
  const Scalar _tmp312 =
      _tmp79 * (R_cam(1, 0) * _tmp305 + R_cam(1, 1) * _tmp304 + R_cam(1, 2) * _tmp306);
  const Scalar _tmp313 = _tmp308 * _tmp77;
  const Scalar _tmp314 =
      _tmp92 * (_tmp312 - _tmp313) + _tmp93 * (_tmp310 * _tmp89 + _tmp311 * _tmp91) +
      _tmp98 * (-_tmp312 + _tmp313) + _tmp99 * (_tmp310 * _tmp96 + _tmp311 * _tmp97);
  const Scalar _tmp315 = _tmp186 * _tmp314 + _tmp187 * _tmp314 - _tmp188 * _tmp314;
  const Scalar _tmp316 = _tmp191 * _tmp315;
  const Scalar _tmp317 = (Scalar(1) / Scalar(4)) * std::pow(_tmp190, Scalar(2));
  const Scalar _tmp318 = _tmp189 * _tmp317;
  const Scalar _tmp319 = _tmp289 * _tmp317;
  const Scalar _tmp320 = _tmp302 * _tmp317;
  const Scalar _tmp321 = _tmp315 * _tmp317;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 1, 1>& _res = (*res);

    _res(0, 0) = _tmp106;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 1, 6>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp192;
    _jacobian(0, 1) = _tmp242;
    _jacobian(0, 2) = _tmp277;
    _jacobian(0, 3) = _tmp290;
    _jacobian(0, 4) = _tmp303;
    _jacobian(0, 5) = _tmp316;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 6, 6>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp189, Scalar(2)) * _tmp317;
    _hessian(1, 0) = _tmp241 * _tmp318;
    _hessian(2, 0) = _tmp276 * _tmp318;
    _hessian(3, 0) = _tmp189 * _tmp319;
    _hessian(4, 0) = _tmp302 * _tmp318;
    _hessian(5, 0) = _tmp315 * _tmp318;
    _hessian(0, 1) = 0;
    _hessian(1, 1) = std::pow(_tmp241, Scalar(2)) * _tmp317;
    _hessian(2, 1) = _tmp241 * _tmp276 * _tmp317;
    _hessian(3, 1) = _tmp241 * _tmp319;
    _hessian(4, 1) = _tmp241 * _tmp320;
    _hessian(5, 1) = _tmp241 * _tmp321;
    _hessian(0, 2) = 0;
    _hessian(1, 2) = 0;
    _hessian(2, 2) = std::pow(_tmp276, Scalar(2)) * _tmp317;
    _hessian(3, 2) = _tmp276 * _tmp319;
    _hessian(4, 2) = _tmp276 * _tmp320;
    _hessian(5, 2) = _tmp276 * _tmp321;
    _hessian(0, 3) = 0;
    _hessian(1, 3) = 0;
    _hessian(2, 3) = 0;
    _hessian(3, 3) = std::pow(_tmp289, Scalar(2)) * _tmp317;
    _hessian(4, 3) = _tmp302 * _tmp319;
    _hessian(5, 3) = _tmp315 * _tmp319;
    _hessian(0, 4) = 0;
    _hessian(1, 4) = 0;
    _hessian(2, 4) = 0;
    _hessian(3, 4) = 0;
    _hessian(4, 4) = std::pow(_tmp302, Scalar(2)) * _tmp317;
    _hessian(5, 4) = _tmp302 * _tmp321;
    _hessian(0, 5) = 0;
    _hessian(1, 5) = 0;
    _hessian(2, 5) = 0;
    _hessian(3, 5) = 0;
    _hessian(4, 5) = 0;
    _hessian(5, 5) = std::pow(_tmp315, Scalar(2)) * _tmp317;
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 6, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp106 * _tmp192;
    _rhs(1, 0) = _tmp106 * _tmp242;
    _rhs(2, 0) = _tmp106 * _tmp277;
    _rhs(3, 0) = _tmp106 * _tmp290;
    _rhs(4, 0) = _tmp106 * _tmp303;
    _rhs(5, 0) = _tmp106 * _tmp316;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace event_res