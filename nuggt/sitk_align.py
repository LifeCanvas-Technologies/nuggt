""""sitk_align - Align using simple ITK and Elastix

Partially derived from elastix_auto.py by Jae Cho
"""
import argparse
import json
import SimpleITK as sitk
import numpy as np
import os
import re
import shutil
import tempfile
from ._sitk_align import parse as parse_pts_file

FINAL_GRID_SPACING_IN_VOXELS = ['32', '32', '32']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving-file",
                        help="Path to file to be aligned",
                        required=True)
    parser.add_argument("--fixed-file",
                        help="Path to the reference file",
                        required=True)
    parser.add_argument("--fixed-point-file",
                        help="Path to the marked points on the reference image")
    parser.add_argument("--alignment-point-file",
                        help="Path to the file for nuggt-align's initial pts",
                        default=None)
    parser.add_argument("--xyz",
                        help="Coordinates in --fixed-point-file are in "
                        "X, Y, Z form, not Z, Y, X (e.g. the output of nuggt)",
                        action="store_true",
                        default=False)
    parser.add_argument("--aligned-file",
                        help="Path to the alignment image file to be written.",
                        default=None)
    parser.add_argument("--final-grid-spacing",
                        help="The spacing between voxels for the bspline grid",
                        default=None)
    parser.add_argument("--transform-parameters-folder",
                        help="sitk-align will save the TransformParameters.txt"
                        " files in this directory if specified.",
                        default=None)

    parser.add_argument(
        "--custom-points-registration",
        help="Option to using points files in SITK registration",
        default=False)

    parser.add_argument(
        "--custom-reference-points",
        help="Provide the reference points files for SITK points registration",
        default=None)

    parser.add_argument(
        "--custom-moving-points",
        help="Provide the moving points files for SITK points registration",
        default=None)

    return parser.parse_args()


def getParameterMap(rigid=True, affine=True, bspline=True):
    parameterMapVector = sitk.VectorOfParameterMap()
    if rigid:
        rigidMap = sitk.GetDefaultParameterMap("rigid")
        rigidMap['MaximumNumberOfIterations'] = ['1000']
        rigidMap['NumberOfHistogramBins'] = ['8','16','32']
        rigidMap['NumberOfResolutions'] = ['6']
        rigidMap['MaximumStepLength'] = ['0.5']
        rigidMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
        rigidMap['Interpolator'] = ['BSplineInterpolator']
        parameterMapVector.append(rigidMap)
    if affine:
        affineMap = sitk.GetDefaultParameterMap("affine")
        affineMap['MaximumNumberOfIterations'] = ['1000']
        affineMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
        affineMap['Interpolator'] = ['BSplineInterpolator']
        affineMap['BSplineInterpolationOrder'] = ['3']
        affineMap['FinalBSplineInterpolationOrder'] = ['3']
        affineMap['NumberOfHistogramBins'] = ['4','8','16','32']
        affineMap['NumberOfResolutions'] = ['5']
        affineMap['MaximumStepLength'] = ['0.5']
        parameterMapVector.append(affineMap)
    if bspline:
        bsplineMap = sitk.GetDefaultParameterMap("bspline")
        bsplineMap['MaximumNumberOfIterations'] = ['5000']
        bsplineMap['Registration'] = ['MultiMetricMultiResolutionRegistration']
        bsplineMap['Interpolator'] = ['BSplineInterpolator']
        bsplineMap['BSplineInterpolationOrder'] = ['3']
        # increasing make atlas deform more, decrease deform less. should be odd numbers from 3
        bsplineMap['FinalBSplineInterpolationOrder'] = ['3']
        # increasing make atlas deform more, decrease deform less. should be odd numbers from 3
        #bsplineMap['FinalGridSpacingInVoxels'] = ['8']
        bsplineMap['FinalGridSpacingInVoxels'] = FINAL_GRID_SPACING_IN_VOXELS
        # increasing make atlas deform less, decrease deform more., current issue might be the gridspacing issue
        bsplineMap['NumberOfHistogramBins'] = ['4','8','16','32','64']
        bsplineMap['NumberOfResolutions'] = ['6']
        bsplineMap['MaximumStepLength'] = ['1']
        # default : 1
        bsplineMap['ResultImagePixelType'] = ['int']
        bsplineMap['Metric0Weight'] = ['4']
        bsplineMap['Metric1Weight'] = ['1000']
        bsplineMap['Metric2Weight'] = ['1']
        bsplineMap.erase('FinalGridSpacingInPhysicalUnits')
        bsplineMap.erase('GridSpacingSchedule')
        parameterMapVector.append(bsplineMap)
    return parameterMapVector


def align(fixed_image, moving_image, aligned_image_path,
          points_registration,  moving_points, reference_points,
          transform_parameter_folder = None):
    """Align the files

    :param fixed_image: the SimpleITK image for the fixed image
    :param moving_path: the SimpleITK moving image
    :param aligned_image_path: path to write the image after alignment or
        None if user does not want the image
    :param points_registration: boolean to determine if points
        registration parameter is added to parameter vector
    :param moving_points: txt file containing moving image points
    :param reference_points: txt file containing reference image points
    :param transform_parameter_folder: where to store the transform
    parameter files (this is a side-effect of running
    ElastixImageFilter.Execute in the transfer_parameter_folder directory)
    :returns: a transform map
    """
    selx = sitk.ElastixImageFilter()
    parameterMapVector = getParameterMap(False,True,True)
    selx.SetParameterMap(parameterMapVector)

    if points_registration is True:
        pm = selx.GetParameterMap()
        for i in pm:
            m = list(i['Metric'])
            m.append("CorrespondingPointsEuclideanDistanceMetric")
            i['Metric'] = tuple(m)
            l = len(i['Metric'])
            c = 0
            while c < l:
                i['Metric' + str(c) + 'Weight'] = '1'
                if (l == 2) and (c == 1):
                    i['Metric' + str(c) + 'Weight'] = str('')
                    new = list(i['Metric' + str(c) + 'Weight'])
                    new.append(str(1000))
                    i['Metric' + str(c) + 'Weight'] = tuple(new)

                if (l == 3) and (c == 2):
                    i['Metric' + str(c) + 'Weight'] = str('')
                    new = list(i['Metric' + str(c) + 'Weight'])
                    new.append(str(1000))
                    i['Metric' + str(c) + 'Weight'] = tuple(new)
                c += 1

    selx.SetFixedImage(fixed_image)
    selx.SetMovingImage(moving_image)

    if moving_points is not None:
        selx.SetFixedPointSetFileName(reference_points)
        selx.SetMovingPointSetFileName(moving_points)

    curdir = os.path.abspath(os.getcwd())
    try:
        if transform_parameter_folder is not None:
            os.chdir(transform_parameter_folder)
        selx.Execute()
    finally:
        os.chdir(curdir)
    if aligned_image_path is not None:
        sitk.WriteImage(selx.GetResultImage(), aligned_image_path)
    return selx.GetTransformParameterMap()


def write_point_set(filename, points):
    """Write a point set file, the way Elastix wants

    The format of the file:
    "point" or "index"
    # of points
    x y z
    ...

    :param filename: Write to this file
    :param points: an Nx3 array of points to write
    """
    with open(filename, "w") as fd:
        fd.write("point\n")
        fd.write("%d\n" % len(points))
        for z, y, x in points:
            fd.write("%f %f %f\n" % (x, y, z))


def read_point_set(filename):
    """Read the point set file that's the output of the transformation

    :param filename: The location of the point set file
    :returns: an Nx3 array of the output points
    """
    pattern = \
        "%s\\s*=\\s*\\[\\s*(-?\\d+.?\\d*)\\s+(-?\\d+.?\\d*)\\s+(-?\\d+.?\\d*)\\s*\\]"
    outpoints = []
    with open(filename) as fd:
        for line in fd:
            match = re.search(pattern % "OutputPoint", line)
            outpoints.append(list(reversed([float(_) for _ in match.groups()])))
    return outpoints


def transform(points, moving_image, transform_parameter_map):
    """Transform the points in the fixed coordinate space to moving

    :param points: Points in the fixed coordinate space
    :param moving_image: The moving image as loaded by SimpleITK (needed
           by the transformation to find the image dimensions)
    :param transform_parameter_map: The transform parameter map produced
    by ElastixImageFilter after running.
    :returns: the point coordinates in the moving coordinate space
    """
    temp_dir = tempfile.mkdtemp()
    try:
        fixed_point_set_path = os.path.join(temp_dir, "fixed_points.txt")
        write_point_set(fixed_point_set_path, points)
        tif = sitk.TransformixImageFilter()
        tif.SetTransformParameterMap(transform_parameter_map)
        tif.SetFixedPointSetFileName(fixed_point_set_path)
        tif.SetMovingImage(moving_image)
        tif.SetOutputDirectory(temp_dir)
        tif.LogToConsoleOn()
        tif.Execute()
        output_path = os.path.join(temp_dir, "outputpoints.txt")
        out_a = np.memmap(output_path, np.uint8, mode="r")
        shape = (len(points), len(points[0]))
        result = np.zeros(shape, np.float32)
        parse_pts_file(out_a, result)
        return result[:,::-1]
    finally:
        shutil.rmtree(temp_dir)


def main():
    global FINAL_GRID_SPACING_IN_VOXELS
    args = parse_args()
    if args.final_grid_spacing is not None:
        FINAL_GRID_SPACING_IN_VOXELS = args.final_grid_spacing.split(",")

    fixed_image = sitk.ReadImage(args.fixed_file)
    moving_image = sitk.ReadImage(args.moving_file)
    aligned_file = args.aligned_file
    fixed_point_file = args.fixed_point_file
    alignment_point_file = args.alignment_point_file
    transform_pm = align(
        fixed_image, moving_image, aligned_file, 
        points_registration=args.custom_points_registration,
        moving_points=args.custom_moving_points, 
        reference_points=args.custom_reference_points,
        transform_parameter_folder=args.transform_parameters_folder)
    if alignment_point_file is not None:
        with open(fixed_point_file) as fd:
            points = json.load(fd)
            if args.xyz:
                points = [_[::-1] for _ in points]
        out_points = transform(points, moving_image, transform_pm)
        out_points = out_points.astype(float).tolist()
        with open(alignment_point_file, "w") as fd:
            json.dump(dict(reference=points,
                           moving=out_points), fd, indent=2)

if __name__=="__main__":
    main()
