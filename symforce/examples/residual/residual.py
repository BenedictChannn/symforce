import symforce
symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf
from pathlib import Path

from symforce import codegen
from symforce.ops import StorageOps


def cayley2rot(cayley: sf.V3) -> sf.M33:
    eye = sf.M33.eye()
    skew_c = sf.Matrix.skew_symmetric(cayley)
    c0, c1, c2 = cayley
    AtA = c0 * c0 + c1 * c1 + c2 * c2 # Tried to use compute_AtA but output is a sf.M11 and (1 +- sf.M11) will have error 

    # R = ((1 - C^T * C) * eye + 2 * C * C^T + 2[c]) / (1 + C^T * C)
    R = sf.M33.eye()
    R = ((1.0 - AtA) * eye + 2 * cayley * cayley.transpose() + 2 * skew_c) / (1.0 + AtA)

    return R

def get_warp_transformation(
        x: sf.V6,
        R_: sf.M33,
        t_: sf.V3,
) -> sf.M44:
    dc = x[:3] # cayley parameters    
    dt = x[3:] # translation

    dR = cayley2rot(dc)
    R_cur_ref = R_.transpose() * dR.transpose() 

    # # Normalize
    # q_cur_ref = sf.Rot3.from_rotation_matrix(R_cur_ref)
    # q_cur_ref_normalized = sf.Rot3(q_cur_ref.q / (sf.sqrt(q_cur_ref.q.squared_norm())))
    # R_cur_ref_normalized = q_cur_ref_normalized.to_rotation_matrix()

    warpingTransformation = sf.M44.eye()
    warpingTransformation[:3, :3] = R_cur_ref
    warpingTransformation[:3, 3] = -R_cur_ref * (dt + dR * t_)
    return warpingTransformation
    


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
                sf.is_nonpositive(patch_x), sf.greater(patch_x, width_ - 2), 
                sf.is_nonpositive(patch_y), sf.greater(patch_y, height_ - 2),
                unsafe = True
            ),
            unsafe = True
        ),
        sf.greater_equal(mask[patch_y * img_col + patch_x], 125),
        unsafe = True
    )


def valid_interpolation(
        pixel: sf.V2,
        img_cols, img_rows,
) -> sf.Scalar:
    pixel_x, pixel_y = sf.floor(pixel[0]), sf.floor(pixel[1])

    return sf.logical_not(
        sf.logical_or(
            sf.greater_equal(pixel_x, img_cols),
            sf.greater_equal(pixel_y, img_rows),
            unsafe = True            
        ),
        unsafe = True
    )

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
    ts_value = q3 * linear_y1 + q4 * linear_y0

    return ts_value



def spatio_temporal_residual(
        R_: sf.M33,
        t_: sf.V3,
        x: sf.V6,
        point: sf.V3,
        R_cam: sf.M33,
        t_cam: sf.V3,
        width_: sf.Scalar, height_: sf.Scalar,
        mask_col: sf.Scalar, 
        Ts_col: sf.Scalar, Ts_row: sf.Scalar,
        mask: sf.DataBuffer("cameraMask"),
        TsObs: sf.DataBuffer("TimeSurfaceObservation")
) -> sf.V1:
    warp_transform = get_warp_transformation(x, R_, t_)
    R_warp = warp_transform[:3, :3]
    t_warp = warp_transform[:3, 3]

    point_left_curr = R_warp * point + t_warp # Transform 3D point to the curr frame
    # world2cam
    pixel_left_curr_hom = R_cam * point_left_curr + t_cam
    pixel_left_curr = pixel_left_curr_hom[:2] / pixel_left_curr_hom[2]# 3D to 2D pixel

    # Find the corresponding TS value
    ts_value = bilinear_interpolation_1x1(Ts_col, TsObs, pixel_left_curr) + sf.numeric_epsilon


    max_ts_value = 255.0
    # Check if interpolation was valid
    flag = valid_interpolation(pixel_left_curr, Ts_col, Ts_row)
    ts_value = flag * ts_value + (1 - flag) * max_ts_value

    # Check if reprojection previously was valid
    valid_flag = is_valid_1x1(width_, height_, mask_col, pixel_left_curr, mask)
    ts_value = valid_flag * ts_value + (1 - valid_flag) * max_ts_value

    # Huber norm implementation
    huber_threshold = 50.0
    condition = sf.greater(ts_value, huber_threshold) # Returns 1 if ts_value > huber threshold, 0 otherwise
    ts_value = condition * sf.sqrt(huber_threshold / ts_value) * ts_value + (1 - condition) * ts_value

    return sf.V1(sf.Max(ts_value, 0)) # TEMPORARY FIX by adding sf.Max
    # Somehow there is one residual with -nan for our own code; that same residual has value of 0 in original code


def generate(output_dir: Path) -> None:
    codegen.Codegen.function(spatio_temporal_residual, config = codegen.CppConfig(), name = "Residual").with_linearization(
        which_args = ['x']
    ).generate_function(output_dir = output_dir, namespace = 'event_res', skip_directory_nesting = True)


def main():
    generate("/home/ckengjwe/dso/symforce/symforce/examples/residual/gen")

if __name__ == "__main__":
    main()
