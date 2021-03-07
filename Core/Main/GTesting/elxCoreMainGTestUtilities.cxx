/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "elxCoreMainGTestUtilities.h"

// GoogleTest header file:
#include <gtest/gtest.h>

namespace elastix
{
std::string
CoreMainGTestUtilities::GetDataDirectoryPath()
{
  static_assert(ELX_CMAKE_SOURCE_DIR != nullptr, "The CMAKE_SOURCE_DIR must not be null");
  const std::string sourceDirectoryPath = ELX_CMAKE_SOURCE_DIR;
  [&sourceDirectoryPath] { ASSERT_FALSE(sourceDirectoryPath.empty()); }();
  return sourceDirectoryPath + (sourceDirectoryPath.back() == '/' ? "" : "/") + "Testing/Data";
}

} // namespace elastix
