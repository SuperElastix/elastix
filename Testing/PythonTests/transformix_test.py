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

import os
import filecmp
import pathlib
import subprocess
import sys
import unittest
import SimpleITK as sitk
import numpy as np


class TransformixTestCase(unittest.TestCase):
    """Tests transformix from https://elastix.lumc.nl"""

    version_string = "5.0.1"
    transformix_exe_file_path = pathlib.Path(sys.argv[1])
    temporary_directory_path = pathlib.Path(sys.argv[2])

    def create_test_function_output_directory(self):
        directory_path = self.temporary_directory_path / sys._getframe(1).f_code.co_name
        directory_path.mkdir(exist_ok=True)
        return directory_path

    def assert_equal_image_info(self, actual: sitk.Image, expected: sitk.Image) -> None:
        """Asserts that the actual image has the same image information (size, spacing, pixel type, etc) as the expected image."""

        self.assertEqual(actual.GetDimension(), expected.GetDimension())
        self.assertEqual(actual.GetSize(), expected.GetSize())
        self.assertEqual(actual.GetSpacing(), expected.GetSpacing())
        self.assertEqual(actual.GetOrigin(), expected.GetOrigin())
        self.assertEqual(actual.GetDirection(), expected.GetDirection())
        self.assertEqual(
            actual.GetPixelIDTypeAsString(), expected.GetPixelIDTypeAsString()
        )

    def test_without_arguments(self) -> None:
        """Tests executing transformix without arguments"""

        completed = subprocess.run(
            [str(self.transformix_exe_file_path)], capture_output=True
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(
            completed.stdout.decode().strip(),
            'Use "transformix --help" for information about transformix-usage.',
        )

    def test_help(self) -> None:
        """Tests --help"""

        completed = subprocess.run(
            [str(self.transformix_exe_file_path), "--help"], capture_output=True
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, b"")
        self.assertTrue(
            "transformix applies a transform on an input image and/or"
            in completed.stdout.decode()
        )

    def test_version(self) -> None:
        """Tests --version"""

        completed = subprocess.run(
            [str(self.transformix_exe_file_path), "--version"], capture_output=True
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(
            completed.stdout.decode().strip(),
            "transformix version: " + self.version_string,
        )

    def test_translation(self) -> None:
        """Tests translation of images"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        output_directory_path = self.create_test_function_output_directory()
        data_directory_path = source_directory_path / ".." / "Data"
        parameter_directory_path = source_directory_path / "TransformParameters"

        completed = subprocess.run(
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
        )
        self.assertEqual(completed.returncode, 0)

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

    def test_translation_of_points(self) -> None:
        """Tests translation of points"""

        source_directory_path = pathlib.Path(__file__).resolve().parent
        test_function_output_directory_path = (
            self.create_test_function_output_directory()
        )
        out_directory_path = test_function_output_directory_path / "out"
        out_directory_path.mkdir(exist_ok=True)

        data_directory_path = source_directory_path / ".." / "Data"
        parameter_directory_path = source_directory_path / "TransformParameters"

        completed = subprocess.run(
            [
                str(self.transformix_exe_file_path),
                "-def",
                str(data_directory_path / "2D_unit_square_corner_points.txt"),
                "-tp",
                str(parameter_directory_path / "Translation(1,-2).txt"),
                "-out",
                str(out_directory_path),
            ],
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0)

        outputpoints_filename = "outputpoints.txt"
        self.assertTrue(
            filecmp.cmp(
                out_directory_path / outputpoints_filename,
                source_directory_path / "ExpectedOutput" / outputpoints_filename,
                shallow=False,
            )
        )


if __name__ == "__main__":
    # Specify argv to avoid sys.argv to be used directly by unittest.main
    # Note: Use '--verbose' option just as long as the output fits the screen!
    unittest.main(argv=["TransformixTest", "--verbose"])
