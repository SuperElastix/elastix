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
#include "elxConversion.h"

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

// The class to be tested.
using elastix::Conversion;

namespace
{
template <typename TContainer>
void
Expect_GetNumberOfElements_returns_size(const TContainer & container)
{
  EXPECT_EQ(Conversion::GetNumberOfElements(container), container.size());
}
} // namespace


GTEST_TEST(Conversion, GetNumberOfParameters)
{
  Expect_GetNumberOfElements_returns_size(itk::Size<>{});
  Expect_GetNumberOfElements_returns_size(itk::Index<>{});

  // Note: Currently we still support ITK 5.1.1, which does not yet have a
  // size() member function for itk::Point and itk::Vector.
  EXPECT_EQ(Conversion::GetNumberOfElements(itk::Point<double>{}), 3);
  EXPECT_EQ(Conversion::GetNumberOfElements(itk::Vector<double>{}), 3);

  for (std::size_t i{}; i <= 2; ++i)
  {
    Expect_GetNumberOfElements_returns_size(itk::OptimizerParameters<double>(i));
  }
}


GTEST_TEST(Conversion, BoolToString)
{
  // Tests that BoolToString can be evaluated at compile-time.
  static_assert((Conversion::BoolToString(false) != nullptr) && (Conversion::BoolToString(true) != nullptr),
                "BoolToString(bool) does not return nullptr");

  EXPECT_EQ(Conversion::BoolToString(false), std::string{ "false" });
  EXPECT_EQ(Conversion::BoolToString(true), std::string{ "true" });
}


GTEST_TEST(Conversion, ParameterMapToString)
{
  EXPECT_EQ(Conversion::ParameterMapToString({}), std::string{});
  EXPECT_EQ(Conversion::ParameterMapToString({ { "A", {} } }), "(A)\n");
  EXPECT_EQ(Conversion::ParameterMapToString({ { "Numbers", { "0", "1" } } }), "(Numbers 0 1)\n");
  EXPECT_EQ(Conversion::ParameterMapToString({ { "Letters", { "a", "z" } } }), "(Letters \"a\" \"z\")\n");

  // A realistic example:
  const std::string expectedString = R"((Direction 0 0 0 0)
(FixedImageDimension 2)
(FixedInternalImagePixelType "float")
(HowToCombineTransforms "Compose")
(Index 0 0)
(InitialTransformParametersFileName "NoInitialTransform")
(MovingImageDimension 2)
(MovingInternalImagePixelType "float")
(NumberOfParameters 2)
(Origin 0 0)
(Size 0 0)
(Spacing 1 1)
(Transform "TranslationTransform")
(TransformParameters 0 0)
(UseBinaryFormatForTransformationParameters "false")
(UseDirectionCosines "true")
)";

  EXPECT_EQ(Conversion::ParameterMapToString({ { "Direction", { "0", "0", "0", "0" } },
                                               { "FixedImageDimension", { "2" } },
                                               { "FixedInternalImagePixelType", { "float" } },
                                               { "HowToCombineTransforms", { "Compose" } },
                                               { "Index", { "0", "0" } },
                                               { "InitialTransformParametersFileName", { "NoInitialTransform" } },
                                               { "MovingImageDimension", { "2" } },
                                               { "MovingInternalImagePixelType", { "float" } },
                                               { "NumberOfParameters", { "2" } },
                                               { "Origin", { "0", "0" } },
                                               { "Size", { "0", "0" } },
                                               { "Spacing", { "1", "1" } },
                                               { "Transform", { "TranslationTransform" } },
                                               { "TransformParameters", { "0", "0" } },
                                               { "UseBinaryFormatForTransformationParameters", { "false" } },
                                               { "UseDirectionCosines", { "true" } } }),
            expectedString);
}


GTEST_TEST(Conversion, ToString)
{
  // Note that this is different from std::to_string(false) and
  // std::to_string(true), which return "0" and "1", respecively.
  EXPECT_EQ(Conversion::ToString(false), "false");
  EXPECT_EQ(Conversion::ToString(true), "true");

  EXPECT_EQ(Conversion::ToString(0), "0");
  EXPECT_EQ(Conversion::ToString(1), "1");
  EXPECT_EQ(Conversion::ToString(-1), "-1");

  EXPECT_EQ(Conversion::ToString(std::numeric_limits<std::int64_t>::min()), "-9223372036854775808");
  EXPECT_EQ(Conversion::ToString(std::numeric_limits<std::uint64_t>::max()), "18446744073709551615");

  // Extensive tests of conversion from double to string:

  using DoubleLimits = std::numeric_limits<double>;

  // Note that this is different from std::to_string(0.5), which returns "0.500000"
  EXPECT_EQ(Conversion::ToString(0.5), "0.5");

  constexpr auto expectedPrecision = 16;
  static_assert(expectedPrecision == std::numeric_limits<double>::digits10 + 1,
                "The expected precision for double floating point numbers");
  const auto expectedString = "0." + std::string(expectedPrecision, '3');

  EXPECT_EQ(Conversion::ToString(+0.0), "0");
  EXPECT_EQ(Conversion::ToString(-0.0), "0");
  EXPECT_EQ(Conversion::ToString(+1.0), "1");
  EXPECT_EQ(Conversion::ToString(-1.0), "-1");
  EXPECT_EQ(Conversion::ToString(0.1), "0.1");
  EXPECT_EQ(Conversion::ToString(1.0 / 3.0), "0." + std::string(expectedPrecision, '3'));

  for (std::uint8_t exponent{ 20 }; exponent > 0; --exponent)
  {
    const auto power_of_ten = std::pow(10.0, exponent);

    // Test +/- 1000...000
    EXPECT_EQ(Conversion::ToString(power_of_ten), '1' + std::string(exponent, '0'));
    EXPECT_EQ(Conversion::ToString(-power_of_ten), "-1" + std::string(exponent, '0'));
  }

  for (std::uint8_t exponent{ 15 }; exponent > 0; --exponent)
  {
    const auto power_of_ten = std::pow(10.0, exponent);

    // Test +/- 999...999
    EXPECT_EQ(Conversion::ToString(power_of_ten - 1), std::string(exponent, '9'));
    EXPECT_EQ(Conversion::ToString(1 - power_of_ten), '-' + std::string(exponent, '9'));
  }

  for (std::int8_t exponent{ -6 }; exponent < 0; ++exponent)
  {
    const auto power_of_ten = std::pow(10.0, exponent);

    // Test +/- 0.000...001
    EXPECT_EQ(Conversion::ToString(power_of_ten), "0." + std::string(-1 - exponent, '0') + '1');
    EXPECT_EQ(Conversion::ToString(-power_of_ten), "-0." + std::string(-1 - exponent, '0') + '1');
  }

  for (std::int8_t exponent{ -16 }; exponent < 0; ++exponent)
  {
    const auto power_of_ten = std::pow(10.0, exponent);

    // Test +/- 0.999...999
    EXPECT_EQ(Conversion::ToString(1 - power_of_ten), "0." + std::string(-exponent, '9'));
    EXPECT_EQ(Conversion::ToString(power_of_ten - 1), "-0." + std::string(-exponent, '9'));
  }

  // The first powers of ten that are represented by scientific "e" notation:
  EXPECT_EQ(Conversion::ToString(1e+21), "1e+21");
  EXPECT_EQ(Conversion::ToString(1e-7), "1e-7");

  // Test the most relevant constants from <limits>:
  EXPECT_EQ(Conversion::ToString(DoubleLimits::epsilon()), "2.220446049250313e-16");
  EXPECT_EQ(Conversion::ToString(DoubleLimits::min()), "2.2250738585072014e-308");
  EXPECT_EQ(Conversion::ToString(DoubleLimits::lowest()), "-1.7976931348623157e+308");
  EXPECT_EQ(Conversion::ToString(DoubleLimits::max()), "1.7976931348623157e+308");
  EXPECT_EQ(Conversion::ToString(DoubleLimits::quiet_NaN()), "NaN");
  EXPECT_EQ(Conversion::ToString(-DoubleLimits::quiet_NaN()), "NaN");
  EXPECT_EQ(Conversion::ToString(DoubleLimits::infinity()), "Infinity");
  EXPECT_EQ(Conversion::ToString(-DoubleLimits::infinity()), "-Infinity");
  EXPECT_EQ(Conversion::ToString(DoubleLimits::denorm_min()), "5e-324");
  EXPECT_EQ(Conversion::ToString(-DoubleLimits::denorm_min()), "-5e-324");
}


GTEST_TEST(Conversion, ToVectorOfStrings)
{
  using VectorOfStrings = std::vector<std::string>;
  using ArrayOfDoubles = std::array<double, 3>;

  EXPECT_EQ(Conversion::ToVectorOfStrings(itk::Size<>{ 1, 2 }), VectorOfStrings({ "1", "2" }));
  EXPECT_EQ(Conversion::ToVectorOfStrings(itk::Index<>{ 1, 2 }), VectorOfStrings({ "1", "2" }));
  EXPECT_EQ(Conversion::ToVectorOfStrings(itk::Point<double>(ArrayOfDoubles{ -0.5, 0.0, 0.25 })),
            VectorOfStrings({ "-0.5", "0", "0.25" }));
  EXPECT_EQ(Conversion::ToVectorOfStrings(itk::Vector<double>(ArrayOfDoubles{ -0.5, 0.0, 0.25 }.data())),
            VectorOfStrings({ "-0.5", "0", "0.25" }));
  EXPECT_EQ(Conversion::ToVectorOfStrings(itk::OptimizerParameters<double>{}), VectorOfStrings{});
  EXPECT_EQ(Conversion::ToVectorOfStrings(itk::ImageBase<>::DirectionType{}), VectorOfStrings({ "0", "0", "0", "0" }));
}


GTEST_TEST(Conversion, ConcatenateVectors)
{
  using VectorType = std::vector<std::string>;

  const auto expectConcatenation = [](const VectorType & vector1, const VectorType & vector2) {
    auto expectedResult = vector1;

    for (const auto & element : vector2)
    {
      expectedResult.push_back(element);
    }
    EXPECT_EQ(Conversion::ConcatenateVectors(vector1, vector2), expectedResult);
  };

  expectConcatenation({}, {});
  expectConcatenation({ "1" }, {});
  expectConcatenation({}, { "1" });
  expectConcatenation({ "1" }, { "2", "3" });
}


GTEST_TEST(Conversion, IsNumberReturnsFalseOnNonNumericString)
{
  const auto expect_IsNumber_returns_false = [](const std::string & str) {
    SCOPED_TRACE(str);
    EXPECT_FALSE(Conversion::IsNumber(str));
  };

  for (const auto str : { "", " ", "a", "A B C", "true", "false", "nan", "NaN" })
  {
    expect_IsNumber_returns_false(str);
  }

  for (const auto transformName : { "AffineDTITransform",
                                    "AffineLogStackTransform",
                                    "AffineLogTransform",
                                    "AffineTransform",
                                    "BSplineStackTransform",
                                    "BSplineTransform",
                                    "BSplineTransformWithDiffusion",
                                    "DeformationFieldTransform",
                                    "EulerStackTransform",
                                    "EulerTransform",
                                    "MultiBSplineTransformWithNormal",
                                    "RecursiveBSplineTransform",
                                    "SimilarityTransform",
                                    "SplineKernelTransform",
                                    "TranslationStackTransform",
                                    "TranslationTransform",
                                    "WeightedCombinationTransform" })
  {
    expect_IsNumber_returns_false(transformName);
  }
}


GTEST_TEST(Conversion, IsNumberReturnsTrueOnNumericString)
{
  const auto expect_IsNumber_returns_true = [](const std::string & str) {
    SCOPED_TRACE(str);
    EXPECT_TRUE(Conversion::IsNumber(str));
  };

  for (int i{ -10 }; i < 10; ++i)
  {
    expect_IsNumber_returns_true(Conversion::ToString(i));
  }

  expect_IsNumber_returns_true("1e+21");
  expect_IsNumber_returns_true("1e-7");

  for (double d{ -1.0 }; d < 1.0; d += 0.1)
  {
    expect_IsNumber_returns_true(Conversion::ToString(d));
  }

  using limits = std::numeric_limits<double>;

  for (const auto limit : { limits::min(), limits::max(), limits::epsilon(), limits::lowest(), limits::denorm_min() })
  {
    expect_IsNumber_returns_true(Conversion::ToString(limit));
  }
}
