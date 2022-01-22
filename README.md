# Fit measured points to the faces of a 3D body

This program addresses a common measurement problem.  Using CMM (coordinate measuring machine), one gets a set of 3d coordinates of points that lie on the surface of a part, accurately.  However, the origin and axes of the CMM measurement and the origin of the model in the computer often do not coincide.  It is useful to globally fit the measured points to the faces of the model while simultaneously fixing the 6 free parameters for rotation and translation.

The penalty function for the fit is the sum of squares of the shortest distances from all points to their respective faces.  The measurement uncertainties (sigma) can be supplied, which will be used as weights in the penalty function.  A nonlinear least-squares fitting function in [GSL](https://www.gnu.org/software/gsl/) is used.  After the fit, the fit parameters, errors, and residual distances of all points are reported.

  - A `FACE_LINE` is a cylinder with `r=0` and no probe radius (`pr`) offset.
  - A `FACE_POINT` is a sphere with `r=0` and no probe radius (`pr`) offset.

Positive `faceid` indicates that measurements are from outside (cylinder and sphere) where negative `faceid` means the measurements are from the inside.

## Input file format
See `examples/`.

## Get parameters of faces
Place the entire folder of `GetFaceParameters` under the `Scripts` folder of Autodesk Fusion 360: `~/Library/Application\ Support/Autodesk/Autodesk\ Fusion\ 360/API/Scripts/`.  This script will display the parameters of a selected face in the `TEXT COMMANDS` window.
