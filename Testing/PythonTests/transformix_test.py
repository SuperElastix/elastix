# =========================================================================
#
#  Copyright UMC Utrecht and contributors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# =========================================================================

"""transformix test module."""

import os
import filecmp
import pathlib
import random
import subprocess
import sys
import unittest
import itk
import SimpleITK as sitk
import numpy as np

FLOAT32_MAX = 3.402823e38
OUTPUTPOINTS_FILENAME = "outputpoints.txt"


class TransformixTestCase(unittest.TestCase):
    """Tests transformix from https://elastix.lumc.nl"""

    version_string = "5.1.0"
    transformix_exe_file_path = pathlib.Path(os.environ["TRANSFORMIX_EXE"])
    temporary_directory_path = pathlib.Path(os.environ["TRANSFORMIX_TEST_TEMP_DIR"])

    def get_name_of_current_function(self):
        """Returns the name of the current function"""

        return sys._getframe(1).f_code.co_name

    def create_test_function_output_directory(self):
        """Creates an output directory for the current test function, and returns its path."""

        directory_path = self.temporary_directory_path / sys._getframe(1).f_code.co_name
        directory_path.mkdir(exist_ok=True)
        return directory_path

    def assert_equal_image_info(self, actual: sitk.Image, expected: sitk.Image) -> None:
        """Asserts that the actual image has the same image information (size, spacing, pixel type,
        etc) as the expected image."""

        self.assertEqual(actual.GetDimension(), expected.GetDimension())
        self.assertEqual(actual.GetSize(), expected.GetSize())
        self.assertEqual(actual.GetSpacing(), expected.GetSpacing())
        self.assertEqual(actual.GetOrigin(), expected.GetOrigin())
        self.assertEqual(actual.GetDirection(), expected.GetDirection())
        self.assertEqual(
            actual.GetPixelIDTypeAsString(), expected.GetPixelIDTypeAsString()
        )

    def create_image_with_sequence_of_natural_numbers(
        self, number_of_columns, number_of_rows, pixel_type
    ):
        """Creates an image, having an incremental sequence of natural numbers (1, 2, 3, ...) as pixel values"""

        image = sitk.Image(number_of_columns, number_of_rows, pixel_type)
        pixel_value = 0
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                pixel_value += 1
                image.SetPixel([column, row], pixel_value)
        return image

    def assert_equal_mesh(self, actual, expected) -> None:
        """Asserts that the actual mesh is equal to the expected one."""
        number_of_points = expected.GetNumberOfPoints()
        self.assertEqual(actual.GetNumberOfPoints(), number_of_points)

        for i in range(number_of_points):
            self.assertEqual(actual.GetPoint(i), expected.GetPoint(i))

    def random_finite_float32(self):
        """Returns a pseudo-random float, for testing purposes"""

        return random.uniform(-FLOAT32_MAX, FLOAT32_MAX)

    def test_without_arguments(self) -> None:
        """Tests executing transformix without arguments"""

        completed = subprocess.run(
            [str(self.transformix_exe_file_path)], capture_output=True, check=True
        )
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(
            completed.stdout.decode().strip(),
            'Use "transformix --help" for information about transformix-usage.',
        )

    def test_help(self) -> None:
        """Tests --help"""

        completed = subprocess.run(
            [str(self.transformix_exe_file_path), "--help"],
            capture_output=True,
            check=True,
        )
        self.assertEqual(completed.stderr, b"")
        self.assertTrue(
            "transformix applies a transform on an input image and/or"
            in completed.stdout.decode()
        )

    def test_version(self) -> None:
        """Tests --version"""

        completed = subprocess.run(
            [str(self.transformix_exe_file_path), "--version"],
            capture_output=True,
            check=True,
        )
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(
            completed.stdout.decode().strip(),
            "transformix version: " + self.version_string,
        )

    def test_extended_version(self) -> None:
        """Tests --extended-version"""

        completed = subprocess.run(
            [str(self.transformix_exe_file_path), "--extended-version"],
            capture_output=True,
            check=True,
        )
        self.assertEqual(completed.stderr, b"")

        output: str = completed.stdout.decode()
        self.assertTrue("transformix version: " in output)
        self.assertTrue("Git revision SHA: " in output)
        self.assertTrue("Git revision date: " in output)
        self.assertTrue("Memory address size: " in output)
        self.assertTrue("CMake version: " in output)
        self.assertTrue("ITK version: " in output)

    def test_missing_tp_commandline_option(self) -> None:
        """Tests missing -tp commandline option"""

        completed = subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-in",
                "InputImageFile.ext",
                "-out",
                str(self.create_test_function_output_directory()),
            ],
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(
            completed.stderr.decode().strip(),
            'ERROR: No CommandLine option "-tp" given!',
        )

    def test_missing_out_commandline_option(self) -> None:
        """Tests missing -out commandline options"""

        completed = subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-in",
                "InputImageFile.ext",
                "-tp",
                "TransformParameters.txt",
            ],
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(
            completed.stderr.decode().strip(),
            'ERROR: No CommandLine option "-out" given!',
        )

    def test_missing_input_commandline_option(self) -> None:
        """Tests missing input commandline option"""

        completed = subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-tp",
                "TransformParameters.txt",
                "-out",
                str(self.create_test_function_output_directory()),
            ],
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(
            completed.stderr.decode().strip(),
            'ERROR: At least one of the CommandLine options "-in", "-def", "-jac", or "-jacmat" should be given!',
        )

    def test_translation_of_images(self) -> None:
        """Tests translation of images"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()
        data_directory_path = source_directory_path / ".." / "Data"
        parameter_directory_path = source_directory_path / "TransformParameters"

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-in",
                str(data_directory_path / "2D_2x2_square_object_at_(2,1).mhd"),
                "-tp",
                str(parameter_directory_path / "Translation(1,-2).txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        expected_image = sitk.ReadImage(
            str(data_directory_path / "2D_2x2_square_object_at_(1,3).mhd")
        )
        actual_image = sitk.ReadImage(str(output_directory_path / "result.mhd"))

        self.assert_equal_image_info(actual_image, expected_image)

        actual_pixel_data = sitk.GetArrayFromImage(expected_image)
        expected_pixel_data = sitk.GetArrayFromImage(actual_image)

        max_absolute_difference = 3.0878078e-16
        np.testing.assert_allclose(
            actual_pixel_data, expected_pixel_data, atol=max_absolute_difference, rtol=0
        )

    def test_custom_result_image_name(self) -> None:
        """Tests the ResultImageName parameter"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()
        data_directory_path = source_directory_path / ".." / "Data"
        input_file_path = data_directory_path / "2D_2x2_square_object_at_(2,1).mhd"
        parameter_directory_path = source_directory_path / "TransformParameters"

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-in",
                str(input_file_path),
                "-tp",
                str(parameter_directory_path / "Translation(1,-2).txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        expected_image = sitk.ReadImage(str(output_directory_path / "result.mhd"))

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-in",
                str(input_file_path),
                "-tp",
                str(parameter_directory_path / "Translation(1,-2)-CustomResultImageName.txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        actual_image = sitk.ReadImage(str(output_directory_path / "CustomResultImageName.mhd"))

        self.assert_equal_image_info(actual_image, expected_image)
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(actual_image),
            sitk.GetArrayFromImage(expected_image),
        )

    def test_translation_of_points(self) -> None:
        """Tests translation of points"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()

        data_directory_path = source_directory_path / ".." / "Data"
        parameter_directory_path = source_directory_path / "TransformParameters"

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-def",
                str(data_directory_path / "2D_unit_square_corner_points.txt"),
                "-tp",
                str(parameter_directory_path / "Translation(1,-2).txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        self.assertTrue(
            filecmp.cmp(
                output_directory_path / OUTPUTPOINTS_FILENAME,
                source_directory_path
                / "ExpectedOutput"
                / self.get_name_of_current_function()
                / OUTPUTPOINTS_FILENAME,
                shallow=False,
            )
        )

    def test_translation_of_images_and_points(self) -> None:
        """Tests translation of images and points together"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()
        data_directory_path = source_directory_path / ".." / "Data"
        parameter_directory_path = source_directory_path / "TransformParameters"

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-in",
                str(data_directory_path / "2D_2x2_square_object_at_(2,1).mhd"),
                "-def",
                str(data_directory_path / "2D_unit_square_corner_points.txt"),
                "-tp",
                str(parameter_directory_path / "Translation(1,-2).txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        expected_image = sitk.ReadImage(
            str(data_directory_path / "2D_2x2_square_object_at_(1,3).mhd")
        )
        actual_image = sitk.ReadImage(str(output_directory_path / "result.mhd"))

        self.assert_equal_image_info(actual_image, expected_image)

        actual_pixel_data = sitk.GetArrayFromImage(expected_image)
        expected_pixel_data = sitk.GetArrayFromImage(actual_image)

        max_absolute_difference = 3.0878078e-16
        np.testing.assert_allclose(
            actual_pixel_data, expected_pixel_data, atol=max_absolute_difference, rtol=0
        )

        self.assertTrue(
            filecmp.cmp(
                output_directory_path / OUTPUTPOINTS_FILENAME,
                source_directory_path
                / "ExpectedOutput"
                / self.get_name_of_current_function()
                / OUTPUTPOINTS_FILENAME,
                shallow=False,
            )
        )

    def test_transform_image_to_int(self) -> None:
        """Tests transformation of an image to an int pixel type"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()
        parameter_directory_path = source_directory_path / "TransformParameters"

        # Create an image that has a different value for each pixel.
        number_of_columns = 5
        number_of_rows = 6

        for pixel_type in [sitk.sitkInt32, sitk.sitkFloat32]:
            input_image = self.create_image_with_sequence_of_natural_numbers(
                number_of_columns, number_of_rows, pixel_type
            )
            sitk.WriteImage(input_image, str(output_directory_path / "input.mhd"))
            subprocess.run(
                [
                    str(self.transformix_exe_file_path),
                    "-in",
                    str(output_directory_path / "input.mhd"),
                    "-tp",
                    str(parameter_directory_path / "Transform_to_int_5x6.txt"),
                    "-out",
                    str(output_directory_path),
                ],
                capture_output=True,
                check=True,
            )

            actual_image = sitk.ReadImage(str(output_directory_path / "result.mhd"))
            expected_image = self.create_image_with_sequence_of_natural_numbers(
                number_of_columns, number_of_rows, sitk.sitkInt32
            )

            self.assert_equal_image_info(actual_image, expected_image)
            np.testing.assert_array_equal(
                sitk.GetArrayFromImage(actual_image),
                sitk.GetArrayFromImage(expected_image),
            )

    def test_transform_image_to_float(self) -> None:
        """Tests transformation of an image to a float pixel type"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()
        parameter_directory_path = source_directory_path / "TransformParameters"

        # Create an image that has a different value for each pixel.
        number_of_columns = 5
        number_of_rows = 6

        for pixel_type in [sitk.sitkInt32, sitk.sitkFloat32]:
            input_image = self.create_image_with_sequence_of_natural_numbers(
                number_of_columns, number_of_rows, pixel_type
            )
            sitk.WriteImage(input_image, str(output_directory_path / "input.mhd"))
            subprocess.run(
                [
                    str(self.transformix_exe_file_path),
                    "-in",
                    str(output_directory_path / "input.mhd"),
                    "-tp",
                    str(parameter_directory_path / "Transform_to_float_5x6.txt"),
                    "-out",
                    str(output_directory_path),
                ],
                capture_output=True,
                check=True,
            )

            actual_image = sitk.ReadImage(str(output_directory_path / "result.mhd"))
            expected_image = self.create_image_with_sequence_of_natural_numbers(
                number_of_columns, number_of_rows, sitk.sitkFloat32
            )

            self.assert_equal_image_info(actual_image, expected_image)
            np.testing.assert_array_equal(
                sitk.GetArrayFromImage(actual_image),
                sitk.GetArrayFromImage(expected_image),
            )

    def test_zero_translation_of_vtk_3d_points(self) -> None:
        """Tests zero-translation of VTK points in 3D"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()

        parameter_directory_path = source_directory_path / "TransformParameters"

        input_mesh = itk.Mesh[itk.D, 3].New()
        for i in range(4):
            input_mesh.SetPoint(
                i,
                (
                    self.random_finite_float32(),
                    self.random_finite_float32(),
                    self.random_finite_float32(),
                ),
            )

        itk.meshwrite(input_mesh, str(output_directory_path / "inputpoints.vtk"))

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-def",
                str(output_directory_path / "inputpoints.vtk"),
                "-tp",
                str(parameter_directory_path / "Translation(0,0,0).txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        output_mesh = itk.meshread(str(output_directory_path / "outputpoints.vtk"))

        self.assert_equal_mesh(output_mesh, input_mesh)

    def test_zero_translation_of_vtk_2d_points(self) -> None:
        """Tests zero-translation of VTK points in 2D"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()

        parameter_directory_path = source_directory_path / "TransformParameters"

        input_mesh = itk.Mesh[itk.D, 2].New()
        for i in range(4):
            input_mesh.SetPoint(
                i, (self.random_finite_float32(), self.random_finite_float32())
            )

        itk.meshwrite(input_mesh, str(output_directory_path / "inputpoints.vtk"))

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-def",
                str(output_directory_path / "inputpoints.vtk"),
                "-tp",
                str(parameter_directory_path / "Translation(0,0).txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        # Note that itk.meshread does not work, as the following produces a 3D mesh, instead of a 2D
        # mesh.
        #
        # output_mesh = itk.meshread(str(output_directory_path / "outputpoints.vtk"))
        reader = itk.MeshFileReader[itk.Mesh[itk.D, 2]].New()
        reader.SetFileName(str(output_directory_path / "outputpoints.vtk"))
        reader.Update()
        output_mesh = reader.GetOutput()
        self.assert_equal_mesh(output_mesh, input_mesh)

    def test_translation_deformation_field(self) -> None:
        """Tests zero-translation of VTK points in 2D"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()

        parameter_directory_path = source_directory_path / "TransformParameters"

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-def",
                "all",
                "-tp",
                str(parameter_directory_path / "Translation(1,-2).txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        number_of_columns = 5
        number_of_rows = 6
        translation_vector = (1, -2)

        actual_image = sitk.ReadImage(
            str(output_directory_path / "deformationField.mhd")
        )
        expected_image = sitk.Image(
            number_of_columns, number_of_rows, sitk.sitkVectorFloat32
        )
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                expected_image.SetPixel([column, row], translation_vector)

        self.assert_equal_image_info(actual_image, expected_image)

        actual_pixel_data = sitk.GetArrayFromImage(expected_image)
        expected_pixel_data = sitk.GetArrayFromImage(actual_image)

        np.testing.assert_allclose(actual_pixel_data, expected_pixel_data, rtol=0)

    def test_jacobian_determinant(self) -> None:
        """Tests determinant of the spatial Jacobian"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()

        parameter_directory_path = source_directory_path / "TransformParameters"

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-jac",
                "all",
                "-tp",
                str(parameter_directory_path / "Translation(1,-2).txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        number_of_columns = 5
        number_of_rows = 6

        actual_image = sitk.ReadImage(
            str(output_directory_path / "spatialJacobian.mhd")
        )
        expected_image = sitk.Image(number_of_columns, number_of_rows, sitk.sitkFloat32)
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                expected_image.SetPixel([column, row], 1)

        self.assert_equal_image_info(actual_image, expected_image)

        actual_pixel_data = sitk.GetArrayFromImage(expected_image)
        expected_pixel_data = sitk.GetArrayFromImage(actual_image)

        np.testing.assert_allclose(actual_pixel_data, expected_pixel_data, rtol=0)

    def test_jacobian_matrix(self) -> None:
        """Tests determinant of the spatial Jacobian"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()

        parameter_directory_path = source_directory_path / "TransformParameters"

        subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-jacmat",
                "all",
                "-tp",
                str(parameter_directory_path / "Translation(1,-2).txt"),
                "-out",
                str(output_directory_path),
            ],
            capture_output=True,
            check=True,
        )

        number_of_columns = 5
        number_of_rows = 6

        actual_image = sitk.ReadImage(
            str(output_directory_path / "fullSpatialJacobian.mhd")
        )
        self.assertEqual(actual_image.GetDimension(), 2)
        self.assertEqual(actual_image.GetSize(), (5, 6))
        self.assertEqual(actual_image.GetSpacing(), (1.0, 1.0))
        self.assertEqual(actual_image.GetOrigin(), (0.0, 0.0))
        self.assertEqual(actual_image.GetDirection(), (1.0, 0.0, 0.0, 1.0))
        self.assertEqual(
            actual_image.GetPixelIDTypeAsString(), "vector of 32-bit float"
        )
        for row in range(number_of_rows):
            for column in range(number_of_columns):
                self.assertEqual(
                    actual_image.GetPixel([column, row]), (1.0, 0.0, 0.0, 1.0)
                )


if __name__ == "__main__":
    # Specify argv to avoid sys.argv to be used directly by unittest.main
    # Note: Use '--verbose' option just as long as the output fits the screen!
    unittest.main(argv=["TransformixTest", "--verbose"])
