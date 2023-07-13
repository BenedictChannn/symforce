import symforce
symforce.set_epsilon_to_symbol()

import cv2 as cv
import symforce.symbolic as sf
from pathlib import Path

from symforce import codegen
from symforce.ops import StorageOps

# def cayley2rot(cayley: sf.V3) -> sf.M33:
#     # TODO
#     # Rewrite this using the eqn R = ((1 - C^T * C) * eye + 2 * C * C^T + 2[c]) / (1 + C^T * C)
#     # Where[c] is the skew symmetric matrix of c: can use skew_symmetric(a: sf.V3) function from matrix.py 
#     c0, c1, c2 = cayley
#     rot = sf.M33(
#         [
#             [
#                 1.0 + (c0 * c0) - (c1 * c1) - (c2 * c2),
#                 2 * (c0 * c1 - c2),
#                 2 * (c0 * c2 + c1),
#             ],
#             [
#                 2 * (c0 * c1 + c2),
#                 1.0 - (c0 * c0) + (c1 * c1) - (c2 * c2),
#                 2 * (c1 * c2 - c0),
#             ],
#             [
#                 2 * (c0 * c2 - c1),
#                 2 * (c1 * c2 + c0),
#                 1.0 - (c0 * c0) - (c1 * c1) + (c2 * c2),
#             ],
#         ]
#     )

#     rot = (1.0 / (1.0 + (c0 * c0) + (c1 * c1) + (c2 * c2))) * rot
#     return rot

def cayley2rot(cayley: sf.V3) -> sf.M33:
    eye = sf.Matrix.eye(3, 3)
    skew_c = sf.Matrix.skew_symmetric(cayley)
    c0, c1, c2 = cayley
    AtA = c0 * c0 + c1 * c1 + c2 * c2 # Tried to use compute_AtA but output is a sf.M11 and (1 +- sf.M11) will have error 

    # R = ((1 - C^T * C) * eye + 2 * C * C^T + 2[c]) / (1 + C^T * C)
    R = sf.M33()
    R = ((1 - AtA) * eye + 2 * cayley * cayley.transpose() + 2 * skew_c) / (1 + AtA)
    R = StorageOps.simplify(R) # E.g. (1 - (x0^2 + x1^2 + x2^2) + 2 x0^2) ---simplify--> (1 + x0^2 - x1^2 - x2^2)
    # Take note "symbolic.simplify():493 WARNING -- Converting to sympy to use .simplify"
    # This could slow down/ cause some errors in future by not using symengine

    return R

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
    



# def is_valid(
#         wx: sf.Scalar,
#         wy: sf.Scalar,
#         width_: sf.Scalar,
#         height_: sf.Scalar,
#         patchCenter: sf.V2,
#         mask: sf.DataBuffer("cameraMask"),
# ) -> bool:
#     if patchCenter[0] < (wx - 1) / 2 or patchCenter[0] > width_ - (wx - 1) / 2 - 1 \
#         or patchCenter[1] < (wy - 1) / 2 or patchCenter[1] > height_ - (wy - 1) / 2 - 1:
#         return False
#     colSize = mask.shape[1] # This wont work for sf.DataBuffer since it is of (n, 1) i.e. 1D array
#     if mask[(patchCenter[1] - (wy - 1) / 2) * colSize + (patchCenter[0] - (wx - 1) / 2)] < 125:
#         return False
#     if mask[(patchCenter[1] - (wy - 1) / 2) * colSize + (patchCenter[0] + (wx - 1) / 2)] < 125:
#         return False
#     if mask[(patchCenter[1] + (wy - 1) / 2) * colSize + (patchCenter[0] - (wx - 1) / 2)] < 125:
#         return False
#     if mask[(patchCenter[1] + (wy - 1) / 2) * colSize + (patchCenter[0] + (wx - 1) / 2)] < 125:
#         return False
#     return True

def is_valid_1x1(
        width_: sf.Scalar, height_: sf.Scalar,
        img_col: sf.Scalar,
        patch_center: sf.V2,
        mask: sf.DataBuffer("CameraMask"),
) -> sf.Scalar:
    patch_x, patch_y = patch_center
    # Take note that the sf.logical_or may return "max(True, False)" instead of "True"
    # Even when there is at least one truth statement in itS
    
    return sf.logical_and(
        sf.logical_not(
            sf.logical_or(
                sf.is_nonpositive(patch_x), sf.greater(patch_x, width_ - 2), sf.is_nonpositive(patch_y), sf.greater(patch_y, height_ - 2),
                unsafe = True
            ),
            unsafe = True
        ),
        sf.less(mask[patch_y * img_col + patch_x], 125),
        unsafe = True
    )




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

def bilinear_interpolation_1x1(
        img_cols: sf.Scalar, 
        TsObs: sf.DataBuffer("TimeSurfaceObservation"),
        pixel: sf.V2,
) -> sf.Scalar:
    # Find the surrounding pixel coords
    pixel_x, pixel_y = pixel[0], pixel[1]
    pixel_x_floor, pixel_y_floor = sf.floor(pixel_x), sf.floor(pixel_y)

    q1 = (pixel_x_floor + 1) - pixel_x
    q2 = pixel_x - pixel_x_floor
    q3 = (pixel_y_floor + 1) - pixel_y
    q4 = pixel_y - pixel_y_floor

    linear_y0 = q1 * TsObs[pixel_y_floor * img_cols + pixel_x_floor] + q2 * TsObs[pixel_y_floor * img_cols + pixel_x_floor + 1]
    linear_y1 = q1 * TsObs[(pixel_y_floor + 1) * img_cols + pixel_x_floor] + q2 * TsObs[(pixel_y_floor + 1) * img_cols + pixel_x_floor + 1]
    ts_value = q4 * linear_y0 + q3 * linear_y1

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
        mask_col: sf.Scalar, Ts_col: sf.Scalar, 
        mask: sf.DataBuffer("cameraMask"),
        TsObs: sf.DataBuffer("TimeSurfaceObservation")
) -> sf.V1:
    warp_transform = get_warp_transformation(x, R_, t_)
    R_warp = warp_transform[:3, :3]
    t_warp = warp_transform[:3, 3]

    point_left_curr = R_warp * point + t_warp # Transform 3D point to the curr frame
    # world2cam
    pixel_left_curr_hom = R_cam * point_left_curr + t_cam
    pixel_left_curr = pixel_left_curr_hom[:2] / pixel_left_curr_hom[2] # 3D to 2D pixel

    # Find the corresponding TS value
    ts_value = bilinear_interpolation_1x1(Ts_col, TsObs, pixel_left_curr)

    # Huber norm implementation
    huber_threshold = 50.0
    condition = sf.greater(ts_value, huber_threshold) # Returns 1 if ts_value > huber threshold, 0 otherwise
    ts_value = condition * sf.sqrt(huber_threshold / ts_value) * ts_value + (1 - condition) * ts_value

    # Check if reprojection previously was valid
    # max_ts_value = 255.0
    # valid_flag = is_valid_1x1(width_, height_, mask_col, pixel_left_curr, mask)
    # ts_value = valid_flag * ts_value + (1 - valid_flag) * max_ts_value

    return sf.V1(ts_value)


def generate(output_dir: Path) -> None:
    codegen.Codegen.function(spatio_temporal_residual, config = codegen.CppConfig()).with_linearization(
        which_args = ['x']
    ).generate_function(output_dir = output_dir, namespace = 'event_res', skip_directory_nesting = True)


def main():
    generate("/home/ckengjwe/dso/symforce/symforce/examples/residual/gen")

if __name__ == "__main__":
    main()
