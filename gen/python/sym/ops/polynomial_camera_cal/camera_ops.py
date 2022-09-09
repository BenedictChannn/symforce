# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     cam_package/ops/CLASS/camera_ops.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

import math
import numpy
import typing as T

import sym  # pylint: disable=unused-import


class CameraOps(object):
    """
    Python CameraOps implementation for <class 'symforce.cam.polynomial_camera_cal.PolynomialCameraCal'>.
    """

    @staticmethod
    def focal_length(self):
        # type: (sym.PolynomialCameraCal) -> numpy.ndarray
        """
        Return the focal length.
        """

        # Total ops: 0

        # Input arrays
        _self = self.data

        # Intermediate terms (0)

        # Output terms
        _focal_length = numpy.zeros((2, 1))
        _focal_length[0, 0] = _self[0]
        _focal_length[1, 0] = _self[1]
        return _focal_length

    @staticmethod
    def principal_point(self):
        # type: (sym.PolynomialCameraCal) -> numpy.ndarray
        """
        Return the principal point.
        """

        # Total ops: 0

        # Input arrays
        _self = self.data

        # Intermediate terms (0)

        # Output terms
        _principal_point = numpy.zeros((2, 1))
        _principal_point[0, 0] = _self[2]
        _principal_point[1, 0] = _self[3]
        return _principal_point

    @staticmethod
    def pixel_from_camera_point(self, point, epsilon):
        # type: (sym.PolynomialCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float]
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Return:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds else 0
        """

        # Total ops: 32

        # Input arrays
        _self = self.data
        if len(point.shape) == 1:
            point = point.reshape((3, 1))

        # Intermediate terms (4)
        _tmp0 = max(epsilon, point[2, 0])
        _tmp1 = _tmp0 ** (-2)
        _tmp2 = _tmp1 * point[0, 0] ** 2 + _tmp1 * point[1, 0] ** 2 + epsilon
        _tmp3 = (
            1.0 * _self[5] * _tmp2 + 1.0 * _self[6] * _tmp2 ** 2 + 1.0 * _self[7] * _tmp2 ** 3 + 1.0
        ) / _tmp0

        # Output terms
        _pixel = numpy.zeros((2, 1))
        _pixel[0, 0] = _self[0] * _tmp3 * point[0, 0] + _self[2]
        _pixel[1, 0] = _self[1] * _tmp3 * point[1, 0] + _self[3]
        _is_valid = max(
            0,
            min(
                (0.0 if point[2, 0] == 0 else math.copysign(1, point[2, 0])),
                (
                    0.0
                    if _self[4] - math.sqrt(_tmp2) == 0
                    else math.copysign(1, _self[4] - math.sqrt(_tmp2))
                ),
            ),
        )
        return _pixel, _is_valid

    @staticmethod
    def pixel_from_camera_point_with_jacobians(self, point, epsilon):
        # type: (sym.PolynomialCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float, numpy.ndarray, numpy.ndarray]
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Return:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds else 0
            pixel_D_cal: Derivative of pixel with respect to intrinsic calibration parameters
            pixel_D_point: Derivative of pixel with respect to point
        """

        # Total ops: 103

        # Input arrays
        _self = self.data
        if len(point.shape) == 1:
            point = point.reshape((3, 1))

        # Intermediate terms (35)
        _tmp0 = point[1, 0] ** 2
        _tmp1 = max(epsilon, point[2, 0])
        _tmp2 = _tmp1 ** (-2)
        _tmp3 = _tmp0 * _tmp2
        _tmp4 = point[0, 0] ** 2
        _tmp5 = _tmp2 * _tmp4
        _tmp6 = _tmp3 + _tmp5 + epsilon
        _tmp7 = 1.0 * _tmp6 ** 3
        _tmp8 = _tmp6 ** 2
        _tmp9 = 1.0 * _tmp8
        _tmp10 = 1.0 * _self[5]
        _tmp11 = _self[6] * _tmp9 + _self[7] * _tmp7 + _tmp10 * _tmp6 + 1.0
        _tmp12 = _tmp1 ** (-1)
        _tmp13 = _tmp11 * _tmp12
        _tmp14 = _self[0] * _tmp13
        _tmp15 = _self[1] * _tmp13
        _tmp16 = _self[0] * point[0, 0]
        _tmp17 = _tmp12 * (1.0 * _tmp3 + 1.0 * _tmp5 + 1.0 * epsilon)
        _tmp18 = _self[1] * point[1, 0]
        _tmp19 = _tmp12 * _tmp9
        _tmp20 = _tmp12 * _tmp7
        _tmp21 = _tmp2 * point[0, 0]
        _tmp22 = 2.0 * _self[5]
        _tmp23 = _self[7] * _tmp8
        _tmp24 = 6.0 * _tmp23
        _tmp25 = _self[6] * _tmp6
        _tmp26 = 4.0 * _tmp25
        _tmp27 = _tmp12 * (_tmp21 * _tmp22 + _tmp21 * _tmp24 + _tmp21 * _tmp26)
        _tmp28 = _tmp2 * point[1, 0]
        _tmp29 = _tmp12 * (_tmp22 * _tmp28 + _tmp24 * _tmp28 + _tmp26 * _tmp28)
        _tmp30 = (
            0.0 if -epsilon + point[2, 0] == 0 else math.copysign(1, -epsilon + point[2, 0])
        ) + 1
        _tmp31 = (1.0 / 2.0) * _tmp11 * _tmp30
        _tmp32 = _tmp30 / _tmp1 ** 3
        _tmp33 = -_tmp0 * _tmp32 - _tmp32 * _tmp4
        _tmp34 = _tmp12 * (_tmp10 * _tmp33 + 3.0 * _tmp23 * _tmp33 + 2.0 * _tmp25 * _tmp33)

        # Output terms
        _pixel = numpy.zeros((2, 1))
        _pixel[0, 0] = _self[2] + _tmp14 * point[0, 0]
        _pixel[1, 0] = _self[3] + _tmp15 * point[1, 0]
        _is_valid = max(
            0,
            min(
                (0.0 if point[2, 0] == 0 else math.copysign(1, point[2, 0])),
                (
                    0.0
                    if _self[4] - math.sqrt(_tmp6) == 0
                    else math.copysign(1, _self[4] - math.sqrt(_tmp6))
                ),
            ),
        )
        _pixel_D_cal = numpy.zeros((2, 7))
        _pixel_D_cal[0, 0] = _tmp13 * point[0, 0]
        _pixel_D_cal[1, 0] = 0
        _pixel_D_cal[0, 1] = 0
        _pixel_D_cal[1, 1] = _tmp13 * point[1, 0]
        _pixel_D_cal[0, 2] = 1
        _pixel_D_cal[1, 2] = 0
        _pixel_D_cal[0, 3] = 0
        _pixel_D_cal[1, 3] = 1
        _pixel_D_cal[0, 4] = _tmp16 * _tmp17
        _pixel_D_cal[1, 4] = _tmp17 * _tmp18
        _pixel_D_cal[0, 5] = _tmp16 * _tmp19
        _pixel_D_cal[1, 5] = _tmp18 * _tmp19
        _pixel_D_cal[0, 6] = _tmp16 * _tmp20
        _pixel_D_cal[1, 6] = _tmp18 * _tmp20
        _pixel_D_point = numpy.zeros((2, 3))
        _pixel_D_point[0, 0] = _tmp14 + _tmp16 * _tmp27
        _pixel_D_point[1, 0] = _tmp18 * _tmp27
        _pixel_D_point[0, 1] = _tmp16 * _tmp29
        _pixel_D_point[1, 1] = _tmp15 + _tmp18 * _tmp29
        _pixel_D_point[0, 2] = -_self[0] * _tmp21 * _tmp31 + _tmp16 * _tmp34
        _pixel_D_point[1, 2] = -_self[1] * _tmp28 * _tmp31 + _tmp18 * _tmp34
        return _pixel, _is_valid, _pixel_D_cal, _pixel_D_point
