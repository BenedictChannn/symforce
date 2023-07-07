import symforce
symforce.set_epsilon_to_symbol()

import symforce.symbolic as sf 
from pathlib import Path

from symforce import codegen
from symforce import typing as T


# def toy_residual(x: sf.Scalar, y: sf.Scalar, c: sf.Scalar) -> sf.V1:
#     # # Consider y = x^2 function
#     # y = x**2 + 5*x
#     z = 3 -0.5 * sf.sin((x - 2) / 5) + 1.0 * sf.sin((y + 2) / 10.)
#     z = sf.sign_no_zero(c) * z
#     return sf.V1(z)

def toy_residual(x: sf.Scalar, y: sf.DataBuffer('y')) -> sf.V1:
    return x*y[x]

def generate(output_dir:Path) -> None:
    codegen.Codegen.function(toy_residual, codegen.CppConfig()).with_linearization(
        which_args = ['x'], linearization_mode = "stacked_jacobian",
    ).generate_function(output_dir = output_dir, namespace = 'toy_example', skip_directory_nesting = True)

def main():
    generate("/home/ckengjwe/dso/symforce/symforce/examples/custom_toy_example/gen")

if __name__ == "__main__":
    main()