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
#include "elxBaseComponent.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>
#include <vector>

#include <itkImageBase.h>
#include <itkIndex.h>
#include <itkOptimizerParameters.h>
#include <itkPoint.h>
#include <itkSize.h>
#include <itkVector.h>

namespace
{
template <typename TContainer>
void
Expect_GetNumberOfElements_returns_size(const TContainer & container)
{
  EXPECT_EQ(elx::BaseComponent::GetNumberOfElements(container), container.size());
}
} // namespace


GTEST_TEST(BaseComponent, GetNumberOfParameters)
{
  Expect_GetNumberOfElements_returns_size(itk::Size<>{});
  Expect_GetNumberOfElements_returns_size(itk::Index<>{});

  // Note: Currently we still support ITK 5.1.1, which does not yet have a
  // size() member function for itk::Point and itk::Vector.
  EXPECT_EQ(elx::BaseComponent::GetNumberOfElements(itk::Point<double>{}), 3);
  EXPECT_EQ(elx::BaseComponent::GetNumberOfElements(itk::Vector<double>{}), 3);

  for (std::size_t i{}; i <= 2; ++i)
  {
    Expect_GetNumberOfElements_returns_size(itk::OptimizerParameters<double>(i));
  }
}


GTEST_TEST(BaseComponent, BoolToString)
{
  // Tests that BoolToString can be evaluated at compile-time.
  static_assert((elx::BaseComponent::BoolToString(false) != nullptr) &&
                  (elx::BaseComponent::BoolToString(true) != nullptr),
                "BoolToString(bool) does not return nullptr");

  EXPECT_EQ(elx::BaseComponent::BoolToString(false), std::string{ "false" });
  EXPECT_EQ(elx::BaseComponent::BoolToString(true), std::string{ "true" });
}


GTEST_TEST(BaseComponent, ToString)
{
  // Note that this is different from std::to_string(false) and
  // std::to_string(true), which return "0" and "1", respecively.
  EXPECT_EQ(elx::BaseComponent::ToString(false), "false");
  EXPECT_EQ(elx::BaseComponent::ToString(true), "true");

  EXPECT_EQ(elx::BaseComponent::ToString(0), "0");
  EXPECT_EQ(elx::BaseComponent::ToString(1), "1");
  EXPECT_EQ(elx::BaseComponent::ToString(-1), "-1");

  EXPECT_EQ(elx::BaseComponent::ToString(std::numeric_limits<std::int64_t>::min()), "-9223372036854775808");
  EXPECT_EQ(elx::BaseComponent::ToString(std::numeric_limits<std::uint64_t>::max()), "18446744073709551615");

  // Note that this is different from std::to_string(0.5), which returns "0.500000"
  EXPECT_EQ(elx::BaseComponent::ToString(0.5), "0.5");
}


GTEST_TEST(BaseComponent, ToVectorOfStrings)
{
  using VectorOfStrings = std::vector<std::string>;
  using ArrayOfDoubles = std::array<double, 3>;

  EXPECT_EQ(elx::BaseComponent::ToVectorOfStrings(itk::Size<>{ 1, 2 }), VectorOfStrings({ "1", "2" }));
  EXPECT_EQ(elx::BaseComponent::ToVectorOfStrings(itk::Index<>{ 1, 2 }), VectorOfStrings({ "1", "2" }));
  EXPECT_EQ(elx::BaseComponent::ToVectorOfStrings(itk::Point<double>(ArrayOfDoubles{ -0.5, 0.0, 0.25 })),
            VectorOfStrings({ "-0.5", "0", "0.25" }));
  EXPECT_EQ(elx::BaseComponent::ToVectorOfStrings(itk::Vector<double>(ArrayOfDoubles{ -0.5, 0.0, 0.25 }.data())),
            VectorOfStrings({ "-0.5", "0", "0.25" }));
  EXPECT_EQ(elx::BaseComponent::ToVectorOfStrings(itk::OptimizerParameters<double>{}), VectorOfStrings{});
  EXPECT_EQ(elx::BaseComponent::ToVectorOfStrings(itk::ImageBase<>::DirectionType{}),
            VectorOfStrings({ "0", "0", "0", "0" }));
}
