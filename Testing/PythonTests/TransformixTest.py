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

import pathlib
import subprocess
import sys
import unittest

class TransformixTestCase(unittest.TestCase):
    """Tests transformix from https://elastix.lumc.nl"""

    version_string = "5.0.1"
    transformix_exe_file_path = pathlib.Path(sys.argv[1])

    def test_version(self):
        """Tests --version"""
    
        completed = subprocess.run([str(self.transformix_exe_file_path), "--version"], capture_output=True)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, b'')
        self.assertEqual(completed.stdout.decode().strip(), "transformix version: " + self.version_string)

if __name__ == '__main__':
    # Specify argv, just to avoid sys.argv to be used directly by unittest.main
    unittest.main(argv=['TransformixTest'])
