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

if(python_executable)
  add_test(NAME TransformixTest COMMAND ${python_executable}
    "${CMAKE_CURRENT_LIST_DIR}/transformix_test.py")
  set_tests_properties(TransformixTest PROPERTIES ENVIRONMENT
    "TRANSFORMIX_EXE=$<TARGET_FILE_DIR:transformix_exe>/transformix;TRANSFORMIX_TEST_TEMP_DIR=${CMAKE_CURRENT_BINARY_DIR}")
endif()
