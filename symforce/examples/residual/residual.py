import symforce
symforce.set_epsilon_to_symbol()

import numpy as np
import symforce.symbolic as sf
import sympy as sp
from pathlib import Path

from symforce import codegen
from symforce.ops import StorageOps

def cayley2rot(cayley: sf.V3) -> sf.M33:
    # TODO
    # Rewrite this using the eqn R = ((1 - C^T * C) * eye + 2 * C * C^T + 2[c]) / (1 + C^T * C)
    # Where[c] is the skew symmetric matrix of c: can use skew_symmetric(a: sf.V3) function from matrix.py 
    c0, c1, c2 = cayley
    rot = sf.M33(
        [
            [
                1.0 + (c0 * c0) - (c1 * c1) - (c2 * c2),
                2 * (c0 * c1 - c2),
                2 * (c0 * c2 + c1),
            ],
            [
                2 * (c0 * c1 + c2),
                1.0 - (c0 * c0) + (c1 * c1) - (c2 * c2),
                2 * (c1 * c2 - c0),
            ],
            [
                2 * (c0 * c2 - c1),
                2 * (c1 * c2 + c0),
                1.0 - (c0 * c0) - (c1 * c1) + (c2 * c2),
            ],
        ]
    )

    rot = (1.0 / (1.0 + (c0 * c0) + (c1 * c1) + (c2 * c2))) * rot
    return rot

def get_warp_transformation(
        x: sf.V6,
        R_: sf.M33,
        t_: sf.V3,
) -> sf.M34:
    dc = x[:3] # cayley parameters
    dt = x[3:] # translation

    dR = cayley2rot(dc)
    R_cur_ref = R_.transpose() * dR.transpose() 
    # Normalize
    q_cur_ref = sf.Rot3.from_rotation_matrix(R_cur_ref) 
    q_cur_ref_normalized = sf.Rot3(q_cur_ref.q / sf.sqrt(q_cur_ref.q.squared_norm()))
    R_cur_ref_normalized = q_cur_ref_normalized.to_rotation_matrix()

    warpingTransformation = sf.M34()
    warpingTransformation[:3, :3] = R_cur_ref_normalized
    warpingTransformation[:3, 3] = -R_cur_ref_normalized * (dt + dR * t_)
    return warpingTransformation
    



def is_valid(
        wx: sf.Scalar,
        wy: sf.Scalar,
        width_: sf.Scalar,
        height_: sf.Scalar,
        patchCenter: sf.V2,
        mask: sf.DataBuffer("cameraMask"),
) -> bool:
    if patchCenter[0] < (wx - 1) / 2 or patchCenter[0] > width_ - (wx - 1) / 2 - 1 \
        or patchCenter[1] < (wy - 1) / 2 or patchCenter[1] > height_ - (wy - 1) / 2 - 1:
        return False
    colSize = mask.shape[1]
    if mask[(patchCenter[1] - (wy - 1) / 2) * colSize + (patchCenter[0] - (wx - 1) / 2)] < 125:
        return False
    if mask[(patchCenter[1] - (wy - 1) / 2) * colSize + (patchCenter[0] + (wx - 1) / 2)] < 125:
        return False
    if mask[(patchCenter[1] + (wy - 1) / 2) * colSize + (patchCenter[0] - (wx - 1) / 2)] < 125:
        return False
    if mask[(patchCenter[1] + (wy - 1) / 2) * colSize + (patchCenter[0] + (wx - 1) / 2)] < 125:
        return False
    return True



# def patchInterpolation(
#         img_row: sf.Scalar, img_col: sf.Scalar, 
#         wx: sf.Scalar, wy: sf.Scalar,
#         TsObs: sf.DataBuffer('TimeSurfaceNegative'),
#         pixel: sf.V2,
# ) -> sf.Matrix:
#     SrcPatch_UpLeft = sf.V2(sf.floor(pixel[0] - (wx - 1) / 2), sf.floor(pixel[1] - (wy - 1) / 2))
#     SrcPatch_DownRight = sf.V2(sf.floor(pixel[0] + (wx - 1) / 2), sf.floor(pixel[1] + (wy - 1) / 2))

#     # Check if patch containing pixel is within the boundaries of the TS negative
#     if SrcPatch_UpLeft[0] < 0 or SrcPatch_UpLeft[1] < 0:
#         return sf.Matrix(wy, wx) # Return a zero matrix for False
#     if SrcPatch_DownRight[0] >= img_col or SrcPatch_DownRight[1] >= img_row:
#         return sf.Matrix(wy, wx)
#     if SrcPatch_UpLeft[1] + wy >= img_row or SrcPatch_UpLeft[0] + wx >= img_col:
#         return sf.Matrix(wy, wx)
    
#     # Calculating the weights
#     q1 = (sf.floor(pixel[0]) + 1) - pixel[0]
#     q2 = pixel[0] - sf.floor(pixel[0])
#     q3 = (sf.floor(pixel[1]) + 1) - pixel[1]
#     q4 = pixel[1] - sf.floor(pixel[1])

#     wx2, wy2 = wx + 1, wy + 1
#     # Convert DataBuffer to a Matrix
#     cropped_Ts_list = []
#     for i in range(wy2):
#         Ts_row = []
#         for j in range(wx2):
#             TsPixel = TsObs[(SrcPatch_UpLeft[1] + i) * wx2 + (SrcPatch_UpLeft[0] + j)]
#             Ts_row.append(TsPixel)

#         cropped_Ts_list.extend(Ts_row)
#     # Extract the relevant patch of TS negative and convert to matrix
#     cropped_Ts = sf.Matrix(wy2, wx2, cropped_Ts_list)

#     # Compute patch values
#     R = q1 * cropped_Ts[:wy2, :wx] + q2 * cropped_Ts[:wy2, 1:wx + 1]
#     patch = q3 * R[:wy, :wx] + q4 * R[1:wy + 1, :wx]
#     return patch

def patch_interpolation_1x1(
        img_cols: sf.Scalar,
        TsObs: sf.DataBuffer("TimeSurfaceObservation"),
        pixel: sf.V2,
) -> sf.Scalar:
    pixel_x, pixel_y = sf.floor(pixel[0]), sf.floor(pixel[1])
    ts_value = TsObs[img_cols * pixel_y + pixel_x]
    return ts_value



def spatio_temporal_residual(
        R_: sf.M33,
        t_: sf.V3,
        x: sf.V6,
        point: sf.V3,
        R_cam: sf.M33,
        t_cam: sf.V3,
        wx: sf.Scalar, wy: sf.Scalar,
        width_: sf.Scalar, height_: sf.Scalar,
        img_col: sf.Scalar,
        mask: sf.DataBuffer("cameraMask"),
        TsObs: sf.DataBuffer("TimeSurfaceObservation")
) -> sf.V1:
    warp_transform = get_warp_transformation(x, R_, t_)
    R_warp = warp_transform[:3, :3]
    t_warp = warp_transform[:3, 3]

    point_left_curr = R_warp * point + t_warp # Transform 3D point to the left camera 
    # world2cam
    pixel_left_curr_hom = R_cam * point_left_curr + t_cam
    pixel_left_curr = pixel_left_curr_hom[:2] / pixel_left_curr_hom[2] # 3D to 2D pixel

    # flag = True # status for patch interpolation
    # if not is_valid(wx, wy, width_, height_, pixel_left_curr, mask):
    #     flag = False # Let pixel value of warped TS negative be max value 255
    # else:
    #     #TODO
    #     patch_values = patchInterpolation(img_row, img_col, wx, wy, TsObs, pixel_left_curr)

    #     if patch_values != sf.Matrix(wy, wx):
    #         res = sf.Matrix(list(patch_values))
    #         return res


    ts_value = patch_interpolation_1x1(img_col, TsObs, pixel_left_curr)

    # # Huber norm implementation
    # huber_threshold = 50.0
    # sign = (sf.Min(sf.sign(ts_value - huber_threshold), 0) + 1)
    # ts_value = sign * sf.sqrt(huber_threshold / ts_value) * ts_value + (1 - sign) * ts_value

    return sf.V1(ts_value)

def generate(output_dir: Path) -> None:
    codegen.Codegen.function(spatio_temporal_residual, config = codegen.CppConfig()).with_linearization(
        which_args = ['x']
    ).generate_function(output_dir = output_dir, namespace = 'event_res', skip_directory_nesting = True)


def main():
    generate("/home/ckengjwe/dso/symforce/symforce/examples/residual/gen")

if __name__ == "__main__":
    main()
