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

#include <cassert>
#include <type_traits> // For is_same.


namespace elastix
{
std::string
CoreMainGTestUtilities::GetDataDirectoryPath()
{
  constexpr auto sourceDirectoryPath = ELX_CMAKE_SOURCE_DIR;
  static_assert(std::is_same<decltype(sourceDirectoryPath), const char * const>(),
                "CMAKE_SOURCE_DIR must be a character string!");
  static_assert(sourceDirectoryPath != nullptr, "CMAKE_SOURCE_DIR must not be null!");
  static_assert(*sourceDirectoryPath != '\0', "CMAKE_SOURCE_DIR must not be empty!");

  const std::string str = sourceDirectoryPath;
  return str + ((str.back() == '/') ? "" : "/") + "Testing/Data";
}

std::string
CoreMainGTestUtilities::GetBinaryDirectoryPath()
{
  constexpr auto binaryDirectoryPath = ELX_CMAKE_BINARY_DIR;
  static_assert(std::is_same<decltype(binaryDirectoryPath), const char * const>(),
                "CMAKE_BINARY_DIR must be a character string!");
  static_assert(binaryDirectoryPath != nullptr, "CMAKE_BINARY_DIR must not be null!");
  static_assert(*binaryDirectoryPath != '\0', "CMAKE_BINARY_DIR must not be empty!");

  const std::string str = binaryDirectoryPath;
  const char        back = str.back();

  return (back == '/') || (back == '\\') ? std::string(str.cbegin(), str.cend() - 1) : str;
}

} // namespace elastix
