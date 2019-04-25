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


// First include the header file to be tested:
#include "itkImageMaskSpatialObject2.h"

#include <gtest/gtest.h>

namespace
{
  template <unsigned NDimension>
  void Expect_New_returns_valid_ImageMaskSpatialObject2_Pointer()
  {
    using Type = itk::ImageMaskSpatialObject2<NDimension>;

    static_assert(
      std::is_same<decltype(Type::New()), typename Type::Pointer>::value,
      "Type::New() should return a Type::Pointer");

    EXPECT_NE(Type::New(), nullptr);
  }
}

GTEST_TEST(ImageMaskSpatialObject2, NewReturnsValidPointer)
{
  Expect_New_returns_valid_ImageMaskSpatialObject2_Pointer<1>();
  Expect_New_returns_valid_ImageMaskSpatialObject2_Pointer<2>();
  Expect_New_returns_valid_ImageMaskSpatialObject2_Pointer<3>();
}

