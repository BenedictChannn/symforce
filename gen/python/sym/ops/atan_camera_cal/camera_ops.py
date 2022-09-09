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
    Python CameraOps implementation for <class 'symforce.cam.atan_camera_cal.ATANCameraCal'>.
    """

    @staticmethod
    def focal_length(self):
        # type: (sym.ATANCameraCal) -> numpy.ndarray
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
        # type: (sym.ATANCameraCal) -> numpy.ndarray
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
        # type: (sym.ATANCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float]
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Return:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds else 0
        """

        # Total ops: 25

        # Input arrays
        _self = self.data
        if len(point.shape) == 1:
            point = point.reshape((3, 1))

        # Intermediate terms (4)
        _tmp0 = max(epsilon, point[2, 0])
        _tmp1 = _tmp0 ** (-2)
        _tmp2 = math.sqrt(_tmp1 * point[0, 0] ** 2 + _tmp1 * point[1, 0] ** 2 + epsilon)
        _tmp3 = math.atan(2 * _tmp2 * math.tan(0.5 * _self[4])) / (_self[4] * _tmp0 * _tmp2)

        # Output terms
        _pixel = numpy.zeros((2, 1))
        _pixel[0, 0] = _self[0] * _tmp3 * point[0, 0] + _self[2]
        _pixel[1, 0] = _self[1] * _tmp3 * point[1, 0] + _self[3]
        _is_valid = max(0, (0.0 if point[2, 0] == 0 else math.copysign(1, point[2, 0])))
        return _pixel, _is_valid

    @staticmethod
    def pixel_from_camera_point_with_jacobians(self, point, epsilon):
        # type: (sym.ATANCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float, numpy.ndarray, numpy.ndarray]
        """
        Project a 3D point in the camera frame into 2D pixel coordinates.

        Return:
            pixel: (x, y) coordinate in pixels if valid
            is_valid: 1 if the operation is within bounds else 0
            pixel_D_cal: Derivative of pixel with respect to intrinsic calibration parameters
            pixel_D_point: Derivative of pixel with respect to point
        """

        # Total ops: 113

        # Input arrays
        _self = self.data
        if len(point.shape) == 1:
            point = point.reshape((3, 1))

        # Intermediate terms (46)
        _tmp0 = 0.5 * _self[4]
        _tmp1 = math.tan(_tmp0)
        _tmp2 = point[1, 0] ** 2
        _tmp3 = max(epsilon, point[2, 0])
        _tmp4 = _tmp3 ** (-2)
        _tmp5 = point[0, 0] ** 2
        _tmp6 = _tmp2 * _tmp4 + _tmp4 * _tmp5 + epsilon
        _tmp7 = math.sqrt(_tmp6)
        _tmp8 = 2 * _tmp7
        _tmp9 = math.atan(_tmp1 * _tmp8)
        _tmp10 = _tmp7 ** (-1)
        _tmp11 = _self[4] ** (-1)
        _tmp12 = _tmp3 ** (-1)
        _tmp13 = _tmp11 * _tmp12
        _tmp14 = _tmp10 * _tmp13
        _tmp15 = _tmp14 * _tmp9
        _tmp16 = _self[0] * _tmp15
        _tmp17 = _self[1] * _tmp15
        _tmp18 = math.tan(_tmp0)
        _tmp19 = math.atan(_tmp18 * _tmp8)
        _tmp20 = _tmp14 * _tmp19
        _tmp21 = _self[0] * point[0, 0]
        _tmp22 = _tmp10 * _tmp21
        _tmp23 = _tmp12 * _tmp19 / _self[4] ** 2
        _tmp24 = _tmp18 ** 2
        _tmp25 = 4 * _tmp6
        _tmp26 = 1.0 * (_tmp24 + 1) / (_tmp24 * _tmp25 + 1)
        _tmp27 = _tmp13 * _tmp21
        _tmp28 = _self[1] * point[1, 0]
        _tmp29 = _tmp10 * _tmp28
        _tmp30 = _tmp13 * _tmp28
        _tmp31 = _self[0] * _tmp5
        _tmp32 = _tmp3 ** (-3)
        _tmp33 = _tmp11 * _tmp32
        _tmp34 = _tmp1 / (_tmp6 * (_tmp1 ** 2 * _tmp25 + 1))
        _tmp35 = 2 * _tmp33 * _tmp34
        _tmp36 = _tmp9 / _tmp6 ** (3.0 / 2.0)
        _tmp37 = _tmp33 * _tmp36
        _tmp38 = _self[1] * _tmp37
        _tmp39 = _tmp21 * point[1, 0]
        _tmp40 = (
            0.0 if -epsilon + point[2, 0] == 0 else math.copysign(1, -epsilon + point[2, 0])
        ) + 1
        _tmp41 = _tmp32 * _tmp40
        _tmp42 = -_tmp2 * _tmp41 - _tmp41 * _tmp5
        _tmp43 = _tmp34 * _tmp42
        _tmp44 = (1.0 / 2.0) * _tmp36 * _tmp42
        _tmp45 = (1.0 / 2.0) * _tmp11 * _tmp4 * _tmp40 * _tmp9

        # Output terms
        _pixel = numpy.zeros((2, 1))
        _pixel[0, 0] = _self[2] + _tmp16 * point[0, 0]
        _pixel[1, 0] = _self[3] + _tmp17 * point[1, 0]
        _is_valid = max(0, (0.0 if point[2, 0] == 0 else math.copysign(1, point[2, 0])))
        _pixel_D_cal = numpy.zeros((2, 5))
        _pixel_D_cal[0, 0] = _tmp20 * point[0, 0]
        _pixel_D_cal[1, 0] = 0
        _pixel_D_cal[0, 1] = 0
        _pixel_D_cal[1, 1] = _tmp20 * point[1, 0]
        _pixel_D_cal[0, 2] = 1
        _pixel_D_cal[1, 2] = 0
        _pixel_D_cal[0, 3] = 0
        _pixel_D_cal[1, 3] = 1
        _pixel_D_cal[0, 4] = -_tmp22 * _tmp23 + _tmp26 * _tmp27
        _pixel_D_cal[1, 4] = -_tmp23 * _tmp29 + _tmp26 * _tmp30
        _pixel_D_point = numpy.zeros((2, 3))
        _pixel_D_point[0, 0] = _tmp16 + _tmp31 * _tmp35 - _tmp31 * _tmp37
        _pixel_D_point[1, 0] = _tmp28 * _tmp35 * point[0, 0] - _tmp38 * point[0, 0] * point[1, 0]
        _pixel_D_point[0, 1] = _tmp35 * _tmp39 - _tmp37 * _tmp39
        _pixel_D_point[1, 1] = _self[1] * _tmp2 * _tmp35 + _tmp17 - _tmp2 * _tmp38
        _pixel_D_point[0, 2] = -_tmp22 * _tmp45 + _tmp27 * _tmp43 - _tmp27 * _tmp44
        _pixel_D_point[1, 2] = -_tmp29 * _tmp45 + _tmp30 * _tmp43 - _tmp30 * _tmp44
        return _pixel, _is_valid, _pixel_D_cal, _pixel_D_point

    @staticmethod
    def camera_ray_from_pixel(self, pixel, epsilon):
        # type: (sym.ATANCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float]
        """
        Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

        TODO(hayk): Add a normalize boolean argument? Like in `cam.Camera`

        Return:
            camera_ray: The ray in the camera frame (NOT normalized)
            is_valid: 1 if the operation is within bounds else 0
        """

        # Total ops: 27

        # Input arrays
        _self = self.data
        if len(pixel.shape) == 1:
            pixel = pixel.reshape((2, 1))

        # Intermediate terms (5)
        _tmp0 = -_self[2] + pixel[0, 0]
        _tmp1 = -_self[3] + pixel[1, 0]
        _tmp2 = math.sqrt(epsilon + _tmp1 ** 2 / _self[1] ** 2 + _tmp0 ** 2 / _self[0] ** 2)
        _tmp3 = _self[4] * _tmp2
        _tmp4 = (1.0 / 2.0) * math.tan(_tmp3) / (_tmp2 * math.tan(0.5 * _self[4]))

        # Output terms
        _camera_ray = numpy.zeros((3, 1))
        _camera_ray[0, 0] = _tmp0 * _tmp4 / _self[0]
        _camera_ray[1, 0] = _tmp1 * _tmp4 / _self[1]
        _camera_ray[2, 0] = 1
        _is_valid = max(
            0,
            (
                0.0
                if -abs(_tmp3) + (1.0 / 2.0) * math.pi == 0
                else math.copysign(1, -abs(_tmp3) + (1.0 / 2.0) * math.pi)
            ),
        )
        return _camera_ray, _is_valid

    @staticmethod
    def camera_ray_from_pixel_with_jacobians(self, pixel, epsilon):
        # type: (sym.ATANCameraCal, numpy.ndarray, float) -> T.Tuple[numpy.ndarray, float, numpy.ndarray, numpy.ndarray]
        """
        Backproject a 2D pixel coordinate into a 3D ray in the camera frame.

        Return:
            camera_ray: The ray in the camera frame (NOT normalized)
            is_valid: 1 if the operation is within bounds else 0
            point_D_cal: Derivative of point with respect to intrinsic calibration parameters
            point_D_pixel: Derivation of point with respect to pixel
        """

        # Total ops: 133

        # Input arrays
        _self = self.data
        if len(pixel.shape) == 1:
            pixel = pixel.reshape((2, 1))

        # Intermediate terms (54)
        _tmp0 = -_self[2] + pixel[0, 0]
        _tmp1 = -_self[3] + pixel[1, 0]
        _tmp2 = _tmp1 ** 2
        _tmp3 = _self[1] ** (-2)
        _tmp4 = _tmp0 ** 2
        _tmp5 = _self[0] ** (-2)
        _tmp6 = _tmp2 * _tmp3 + _tmp4 * _tmp5 + epsilon
        _tmp7 = math.sqrt(_tmp6)
        _tmp8 = _self[4] * _tmp7
        _tmp9 = math.tan(_tmp8)
        _tmp10 = _tmp9 / _tmp7
        _tmp11 = 0.5 * _self[4]
        _tmp12 = math.tan(_tmp11) ** (-1)
        _tmp13 = _self[0] ** (-1)
        _tmp14 = (1.0 / 2.0) * _tmp13
        _tmp15 = _tmp12 * _tmp14
        _tmp16 = _tmp10 * _tmp15
        _tmp17 = _self[1] ** (-1)
        _tmp18 = (1.0 / 2.0) * _tmp17
        _tmp19 = _tmp1 * _tmp18
        _tmp20 = _tmp10 * _tmp12
        _tmp21 = _tmp0 ** 3 / _self[0] ** 4
        _tmp22 = math.tan(_tmp11)
        _tmp23 = _tmp22 ** (-1)
        _tmp24 = _tmp9 ** 2 + 1
        _tmp25 = _tmp23 * _tmp24
        _tmp26 = _self[4] / _tmp6
        _tmp27 = (1.0 / 2.0) * _tmp26
        _tmp28 = _tmp25 * _tmp27
        _tmp29 = _tmp9 / _tmp6 ** (3.0 / 2.0)
        _tmp30 = (1.0 / 2.0) * _tmp29
        _tmp31 = _tmp23 * _tmp30
        _tmp32 = _tmp10 * _tmp23
        _tmp33 = (1.0 / 2.0) * _tmp32
        _tmp34 = _tmp0 * _tmp5
        _tmp35 = _tmp4 / _self[0] ** 3
        _tmp36 = _tmp19 * _tmp25
        _tmp37 = _tmp26 * _tmp36
        _tmp38 = _tmp23 * _tmp29
        _tmp39 = _tmp19 * _tmp38
        _tmp40 = _tmp2 / _self[1] ** 3
        _tmp41 = _tmp0 * _tmp14
        _tmp42 = _tmp25 * _tmp41
        _tmp43 = _tmp26 * _tmp42
        _tmp44 = _tmp38 * _tmp41
        _tmp45 = _tmp1 ** 3 / _self[1] ** 4
        _tmp46 = _tmp1 * _tmp3
        _tmp47 = _tmp22 ** 2
        _tmp48 = 0.25 * _tmp10 * (_tmp47 + 1) / _tmp47
        _tmp49 = _tmp12 * _tmp30
        _tmp50 = _tmp12 * _tmp24
        _tmp51 = _tmp27 * _tmp50
        _tmp52 = _tmp19 * _tmp34
        _tmp53 = _tmp0 * _tmp15 * _tmp46

        # Output terms
        _camera_ray = numpy.zeros((3, 1))
        _camera_ray[0, 0] = _tmp0 * _tmp16
        _camera_ray[1, 0] = _tmp19 * _tmp20
        _camera_ray[2, 0] = 1
        _is_valid = max(
            0,
            (
                0.0
                if -abs(_tmp8) + (1.0 / 2.0) * math.pi == 0
                else math.copysign(1, -abs(_tmp8) + (1.0 / 2.0) * math.pi)
            ),
        )
        _point_D_cal = numpy.zeros((3, 5))
        _point_D_cal[0, 0] = -_tmp21 * _tmp28 + _tmp21 * _tmp31 - _tmp33 * _tmp34
        _point_D_cal[1, 0] = -_tmp35 * _tmp37 + _tmp35 * _tmp39
        _point_D_cal[2, 0] = 0
        _point_D_cal[0, 1] = -_tmp40 * _tmp43 + _tmp40 * _tmp44
        _point_D_cal[1, 1] = -_tmp28 * _tmp45 + _tmp31 * _tmp45 - _tmp33 * _tmp46
        _point_D_cal[2, 1] = 0
        _point_D_cal[0, 2] = -_tmp14 * _tmp32 - _tmp28 * _tmp35 + _tmp31 * _tmp35
        _point_D_cal[1, 2] = -_tmp34 * _tmp37 + _tmp34 * _tmp39
        _point_D_cal[2, 2] = 0
        _point_D_cal[0, 3] = -_tmp43 * _tmp46 + _tmp44 * _tmp46
        _point_D_cal[1, 3] = -_tmp18 * _tmp32 - _tmp28 * _tmp40 + _tmp31 * _tmp40
        _point_D_cal[2, 3] = 0
        _point_D_cal[0, 4] = -_tmp0 * _tmp13 * _tmp48 + _tmp42
        _point_D_cal[1, 4] = -_tmp1 * _tmp17 * _tmp48 + _tmp36
        _point_D_cal[2, 4] = 0
        _point_D_pixel = numpy.zeros((3, 2))
        _point_D_pixel[0, 0] = _tmp16 - _tmp35 * _tmp49 + _tmp35 * _tmp51
        _point_D_pixel[1, 0] = -_tmp12 * _tmp29 * _tmp52 + _tmp26 * _tmp50 * _tmp52
        _point_D_pixel[2, 0] = 0
        _point_D_pixel[0, 1] = _tmp24 * _tmp26 * _tmp53 - _tmp29 * _tmp53
        _point_D_pixel[1, 1] = _tmp18 * _tmp20 - _tmp40 * _tmp49 + _tmp40 * _tmp51
        _point_D_pixel[2, 1] = 0
        return _camera_ray, _is_valid, _point_D_cal, _point_D_pixel
