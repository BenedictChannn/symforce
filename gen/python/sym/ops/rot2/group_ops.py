# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     ops/CLASS/group_ops.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

import math
import numpy
import typing as T

import sym  # pylint: disable=unused-import


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.rot2.Rot2'>.
    """

    @staticmethod
    def identity():
        # type: () -> sym.Rot2

        # Total ops: 0

        # Input arrays

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = 1
        _res[1] = 0
        return sym.Rot2.from_storage(_res)

    @staticmethod
    def inverse(a):
        # type: (sym.Rot2) -> sym.Rot2

        # Total ops: 1

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0]
        _res[1] = -_a[1]
        return sym.Rot2.from_storage(_res)

    @staticmethod
    def compose(a, b):
        # type: (sym.Rot2, sym.Rot2) -> sym.Rot2

        # Total ops: 6

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0] * _b[0] - _a[1] * _b[1]
        _res[1] = _a[0] * _b[1] + _a[1] * _b[0]
        return sym.Rot2.from_storage(_res)

    @staticmethod
    def between(a, b):
        # type: (sym.Rot2, sym.Rot2) -> sym.Rot2

        # Total ops: 6

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0] * _b[0] + _a[1] * _b[1]
        _res[1] = _a[0] * _b[1] - _a[1] * _b[0]
        return sym.Rot2.from_storage(_res)

    @staticmethod
    def inverse_with_jacobian(a):
        # type: (sym.Rot2) -> T.Tuple[sym.Rot2, numpy.ndarray]

        # Total ops: 5

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0]
        _res[1] = -_a[1]
        _res_D_a = numpy.zeros((1, 1))
        _res_D_a[0, 0] = -_a[0] ** 2 - _a[1] ** 2
        return sym.Rot2.from_storage(_res), _res_D_a

    @staticmethod
    def compose_with_jacobians(a, b):
        # type: (sym.Rot2, sym.Rot2) -> T.Tuple[sym.Rot2, numpy.ndarray, numpy.ndarray]

        # Total ops: 11

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (5)
        _tmp0 = _a[0] * _b[0] - _a[1] * _b[1]
        _tmp1 = _a[0] * _b[1]
        _tmp2 = _a[1] * _b[0]
        _tmp3 = _tmp1 + _tmp2
        _tmp4 = _tmp0 ** 2 - _tmp3 * (-_tmp1 - _tmp2)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _tmp0
        _res[1] = _tmp3
        _res_D_a = numpy.zeros((1, 1))
        _res_D_a[0, 0] = _tmp4
        _res_D_b = numpy.zeros((1, 1))
        _res_D_b[0, 0] = _tmp4
        return sym.Rot2.from_storage(_res), _res_D_a, _res_D_b

    @staticmethod
    def between_with_jacobians(a, b):
        # type: (sym.Rot2, sym.Rot2) -> T.Tuple[sym.Rot2, numpy.ndarray, numpy.ndarray]

        # Total ops: 15

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (6)
        _tmp0 = _a[0] * _b[0]
        _tmp1 = _a[1] * _b[1]
        _tmp2 = _tmp0 + _tmp1
        _tmp3 = _a[0] * _b[1]
        _tmp4 = _a[1] * _b[0]
        _tmp5 = _tmp3 - _tmp4

        # Output terms
        _res = [0.0] * 2
        _res[0] = _tmp2
        _res[1] = _tmp5
        _res_D_a = numpy.zeros((1, 1))
        _res_D_a[0, 0] = _tmp2 * (-_tmp0 - _tmp1) - _tmp5 ** 2
        _res_D_b = numpy.zeros((1, 1))
        _res_D_b[0, 0] = _tmp2 ** 2 - _tmp5 * (-_tmp3 + _tmp4)
        return sym.Rot2.from_storage(_res), _res_D_a, _res_D_b
