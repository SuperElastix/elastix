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

#include "itkParameterFileParser.h"
#include "itkParameterMapInterface.h"
#include "../Core/Main/GTesting/elxCoreMainGTestUtilities.h"

#include <gtest/gtest.h>

#include <initializer_list>
#include <limits>
#include <numeric> // For iota.
#include <string>
#include <type_traits> // For is_floating_point.
#include <vector>

#include <itkImageBase.h>
#include <itkIndex.h>
#include <itkOptimizerParameters.h>
#include <itkPoint.h>
#include <itkSize.h>
#include <itkVector.h>

// The class to be tested.
using elastix::Conversion;

// Using-declaration:
using elx::CoreMainGTestUtilities::CheckNew;
using ParameterMapType = itk::ParameterMapInterface::ParameterMapType;

namespace
{

// A realistic test example of a parameter map and its text string representation:
namespace TestExample
{
const ParameterMapType parameterMap = { { "Direction", { "0", "0", "0", "0" } },
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
                                        { "UseDirectionCosines", { "true" } } };

const std::string parameterMapTextString = R"((Direction 0 0 0 0)
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
(UseDirectionCosines "true")
)";

} // namespace TestExample


template <typename TParameterValue>
TParameterValue
Expect_successful_round_trip_of_parameter_value(const TParameterValue & parameterValue)
{
  const std::string                                  parameterName("Key");
  const itk::ParameterMapInterface::ParameterMapType parameterMap{ { parameterName,
                                                                     { Conversion::ToString(parameterValue) } } };
  const auto                                         parameterMapInterface = CheckNew<itk::ParameterMapInterface>();

  parameterMapInterface->SetParameterMap(parameterMap);

  TParameterValue actualParameterValue{};

  std::string errorMessage;
  try
  {
    EXPECT_TRUE(parameterMapInterface->ReadParameter(actualParameterValue, parameterName, 0, errorMessage));
  }
  catch (const itk::ExceptionObject & exceptionObject)
  {
    EXPECT_EQ(exceptionObject.what(), std::string{});
  }
  EXPECT_EQ(errorMessage, std::string{});
  return actualParameterValue;
}


template <typename TParameterValue>
void
Expect_lossless_round_trip_of_parameter_value(const TParameterValue & parameterValue)
{
  const TParameterValue & roundTrippedParameterValue = Expect_successful_round_trip_of_parameter_value(parameterValue);
  EXPECT_EQ(roundTrippedParameterValue, parameterValue);
}


template <typename TParameterValue>
void
Expect_lossless_round_trip_of_parameter_values(const std::initializer_list<TParameterValue> & parameterValues)
{
  for (const auto parameterValue : parameterValues)
  {
    Expect_lossless_round_trip_of_parameter_value(parameterValue);
  }
}


template <typename TParameterValue>
void
Expect_lossless_round_trip_of_positive_and_negative_parameter_values(
  const std::initializer_list<TParameterValue> & parameterValues)
{
  for (const auto parameterValue : parameterValues)
  {
    Expect_lossless_round_trip_of_parameter_value<TParameterValue>(+parameterValue);
    Expect_lossless_round_trip_of_parameter_value<TParameterValue>(-parameterValue);
  }
}


template <typename TUnsignedInteger>
void
Expect_lossless_round_trip_of_unsigned_parameter_values()
{
  using NumericLimits = std::numeric_limits<TUnsignedInteger>;
  static_assert(!NumericLimits::is_signed, "This function is only meant to test unsigned types!");

  // Note: the static_cast for `NumericLimits::max() - 1` is a workaround for
  // the following compile error from Visual C++ 2017:
  // > error C2398: Element '3': conversion from 'int' to 'TUnsignedInteger' requires a narrowing conversion
  Expect_lossless_round_trip_of_parameter_values<TUnsignedInteger>(
    { 0, 1, static_cast<TUnsignedInteger>(NumericLimits::max() - 1), NumericLimits::max() });
}


template <typename TSignedInteger>
void
Expect_lossless_round_trip_of_signed_integer_parameter_values()
{
  using NumericLimits = std::numeric_limits<TSignedInteger>;

  static_assert(NumericLimits::is_signed && NumericLimits::is_integer,
                "This function is only meant to test signed integer types!");

  const auto minValue = NumericLimits::min();
  const auto maxValue = NumericLimits::max();

  Expect_lossless_round_trip_of_parameter_values<TSignedInteger>(
    { minValue, minValue + 1, -1, 0, 1, maxValue - 1, maxValue });
}


template <typename TInteger>
void
Expect_parameter_with_decimal_point_and_trailing_zeros_can_be_read_as_integer()
{
  using NumericLimits = std::numeric_limits<TInteger>;

  const auto minValue = NumericLimits::min();
  const auto maxValue = NumericLimits::max();

  for (const TInteger integer : { minValue,
                                  static_cast<TInteger>(minValue + 1),
                                  TInteger{},
                                  TInteger{ 1 },
                                  static_cast<TInteger>(maxValue - 1),
                                  maxValue })
  {
    for (unsigned numberOfTrailingZeros{}; numberOfTrailingZeros <= 2; ++numberOfTrailingZeros)
    {
      const std::string parameterName("Key");
      const std::string parameterStringValue = std::to_string(integer) + '.' + std::string(numberOfTrailingZeros, '0');

      const auto parameterMapInterface = CheckNew<itk::ParameterMapInterface>();
      parameterMapInterface->SetParameterMap({ { parameterName, { parameterStringValue } } });

      TInteger actualParameterValue{};

      std::string errorMessage;
      EXPECT_TRUE(parameterMapInterface->ReadParameter(actualParameterValue, parameterName, 0, errorMessage));
      EXPECT_EQ(errorMessage, std::string{});
      EXPECT_EQ(actualParameterValue, integer);
    }
  }
}


template <typename TInteger>
void
Expect_parameter_with_decimal_point_and_non_zero_trailing_chars_can_not_be_read_as_integer()
{
  using NumericLimits = std::numeric_limits<TInteger>;

  for (const TInteger integer : { NumericLimits::min(), TInteger{}, TInteger{ 1 }, NumericLimits::max() })
  {
    for (const auto trail : { " ", " 0", "1", "10", "A", "F" })
    {
      const std::string parameterName("Key");
      const std::string parameterStringValue = std::to_string(integer) + '.' + trail;

      const auto parameterMapInterface = CheckNew<itk::ParameterMapInterface>();
      parameterMapInterface->SetParameterMap({ { parameterName, { parameterStringValue } } });

      TInteger actualParameterValue{};

      std::string errorMessage;
      EXPECT_THROW(parameterMapInterface->ReadParameter(actualParameterValue, parameterName, 0, errorMessage),
                   itk::ExceptionObject);
    }
  }
}

template <typename TFloatingPoint>
void
Expect_lossless_round_trip_of_floating_point_parameter_values()
{
  static_assert(std::is_floating_point<TFloatingPoint>::value,
                "This function is only meant to test floating point types!");

  using NumericLimits = std::numeric_limits<TFloatingPoint>;

  Expect_lossless_round_trip_of_positive_and_negative_parameter_values<TFloatingPoint>({ 0,
                                                                                         NumericLimits::denorm_min(),
                                                                                         NumericLimits::min(),
                                                                                         NumericLimits::epsilon(),
                                                                                         1,
                                                                                         NumericLimits::max(),
                                                                                         NumericLimits::infinity() });

  const TFloatingPoint & roundTrippedNaN = Expect_successful_round_trip_of_parameter_value(NumericLimits::quiet_NaN());
  EXPECT_TRUE(std::isnan(roundTrippedNaN));
}

} // namespace


GTEST_TEST(Conversion, BoolToString)
{
  // Tests that BoolToString can be evaluated at compile-time.
  static_assert((Conversion::BoolToString(false) != nullptr) && (Conversion::BoolToString(true) != nullptr),
                "BoolToString(bool) does not return nullptr");

  EXPECT_EQ(Conversion::BoolToString(false), std::string{ "false" });
  EXPECT_EQ(Conversion::BoolToString(true), std::string{ "true" });
}


GTEST_TEST(Conversion, ToOptimizerParameters)
{
  using OptimizerParametersType = itk::OptimizerParameters<double>;
  using StdVectorType = std::vector<double>;

  EXPECT_EQ(Conversion::ToOptimizerParameters(StdVectorType{}), OptimizerParametersType{});

  // Sanity check: _not_ just every result from ToOptimizerParameters compares equal to any OptimizerParameters object!
  EXPECT_NE(Conversion::ToOptimizerParameters(StdVectorType{ 0.0 }), OptimizerParametersType{});

  for (const double value : { -1.0, 0.0, 1.0, DBL_MIN, DBL_MAX })
  {
    EXPECT_EQ(Conversion::ToOptimizerParameters(StdVectorType{ value }), OptimizerParametersType(1U, value));
  }

  StdVectorType stdVector(10U);
  std::iota(stdVector.begin(), stdVector.end(), 1.0);

  const auto optimizerParameters = Conversion::ToOptimizerParameters(stdVector);

  ASSERT_EQ(optimizerParameters.size(), stdVector.size());
  EXPECT_TRUE(std::equal(optimizerParameters.begin(), optimizerParameters.end(), stdVector.cbegin()));
}


GTEST_TEST(Conversion, ParameterMapToString)
{
  EXPECT_EQ(Conversion::ParameterMapToString({}), std::string{});
  EXPECT_EQ(Conversion::ParameterMapToString({ { "A", {} } }), "(A)\n");
  EXPECT_EQ(Conversion::ParameterMapToString({ { "Numbers", { "0", "1" } } }), "(Numbers 0 1)\n");
  EXPECT_EQ(Conversion::ParameterMapToString({ { "Letters", { "a", "z" } } }), "(Letters \"a\" \"z\")\n");
  EXPECT_EQ(Conversion::ParameterMapToString(TestExample::parameterMap), TestExample::parameterMapTextString);
}


GTEST_TEST(ParameterFileParser, ConvertTextToParameterMap)
{
  using itk::ParameterFileParser;

  EXPECT_EQ(ParameterFileParser::ConvertToParameterMap(""), ParameterMapType{});
  EXPECT_EQ(ParameterFileParser::ConvertToParameterMap("(Numbers 0 1)\n"),
            ParameterMapType({ { "Numbers", { "0", "1" } } }));
  EXPECT_EQ(ParameterFileParser::ConvertToParameterMap("(Letters \"a\" \"z\")\n"),
            ParameterMapType({ { "Letters", { "a", "z" } } }));
  EXPECT_EQ(ParameterFileParser::ConvertToParameterMap(TestExample::parameterMapTextString), TestExample::parameterMap);
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

  EXPECT_EQ(Conversion::ToString(char{}), "0");
  EXPECT_EQ(Conversion::ToString(char{ 2 }), "2");
  EXPECT_EQ(Conversion::ToString(std::numeric_limits<signed char>::min()), "-128");
  EXPECT_EQ(Conversion::ToString(std::numeric_limits<unsigned char>::max()), "255");

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

  // Even though the IEEE float and double representations of 0.1 both have a
  // small rounding error, we would rather not have 0.1 converted to a string
  // like "0.10000000000000001", or even "0.10000000149011612".
  EXPECT_EQ(Conversion::ToString(0.1), "0.1");
  EXPECT_EQ(Conversion::ToString(0.1f), "0.1");
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


GTEST_TEST(Conversion, LosslessRoundTripOfParameterValue)
{
  Expect_lossless_round_trip_of_unsigned_parameter_values<unsigned>();
  Expect_lossless_round_trip_of_unsigned_parameter_values<std::uint8_t>();
  Expect_lossless_round_trip_of_unsigned_parameter_values<std::uint16_t>();
  Expect_lossless_round_trip_of_unsigned_parameter_values<std::uintmax_t>();

  Expect_lossless_round_trip_of_signed_integer_parameter_values<int>();
  Expect_lossless_round_trip_of_signed_integer_parameter_values<std::int8_t>();
  Expect_lossless_round_trip_of_signed_integer_parameter_values<std::int16_t>();
  Expect_lossless_round_trip_of_signed_integer_parameter_values<std::intmax_t>();

  // Note that the C++ Standard (C++11) does not specify whether `char` is a
  // signed or an unsigned type, so it is tested here separately from
  // `signed char` (int8_t) and `unsigned char` (uint8_t).
  for (const char parameterValue : { std::numeric_limits<char>::min(), '\0', '\1', std::numeric_limits<char>::max() })
  {
    Expect_lossless_round_trip_of_parameter_value<char>(parameterValue);
  }

  Expect_lossless_round_trip_of_floating_point_parameter_values<float>();
  Expect_lossless_round_trip_of_floating_point_parameter_values<double>();

  for (const bool parameterValue : { false, true })
  {
    Expect_lossless_round_trip_of_parameter_value<bool>(parameterValue);
  }
}


GTEST_TEST(Conversion, ToNativePathNameSeparators)
{
  EXPECT_EQ(Conversion::ToNativePathNameSeparators(std::string{}), std::string{});
  EXPECT_EQ(Conversion::ToNativePathNameSeparators(" "), std::string{ " " });

  constexpr bool isBackslashNativeSeparator =
#ifdef _WIN32
    true;
#else
    false;
#endif

  constexpr char nativeSeparator = isBackslashNativeSeparator ? '\\' : '/';
  constexpr char nonNativeSeparator = isBackslashNativeSeparator ? '/' : '\\';

  EXPECT_EQ(Conversion::ToNativePathNameSeparators({ nativeSeparator }), std::string{ nativeSeparator });
  EXPECT_EQ(Conversion::ToNativePathNameSeparators({ nonNativeSeparator }), std::string{ nativeSeparator });
  EXPECT_EQ(Conversion::ToNativePathNameSeparators({ nonNativeSeparator, nativeSeparator }),
            std::string(std::size_t{ 2 }, nativeSeparator));
  EXPECT_EQ(Conversion::ToNativePathNameSeparators({ nativeSeparator, nonNativeSeparator }),
            std::string(std::size_t{ 2 }, nativeSeparator));
}


GTEST_TEST(ParameterMapInterface, ParameterWithDecimalPointAndTrailingZerosCanBeReadAsInteger)
{
  Expect_parameter_with_decimal_point_and_trailing_zeros_can_be_read_as_integer<int>();
  Expect_parameter_with_decimal_point_and_trailing_zeros_can_be_read_as_integer<std::int8_t>();
  Expect_parameter_with_decimal_point_and_trailing_zeros_can_be_read_as_integer<std::intmax_t>();
  Expect_parameter_with_decimal_point_and_trailing_zeros_can_be_read_as_integer<char>();
  Expect_parameter_with_decimal_point_and_trailing_zeros_can_be_read_as_integer<unsigned>();
  Expect_parameter_with_decimal_point_and_trailing_zeros_can_be_read_as_integer<std::uint8_t>();
  Expect_parameter_with_decimal_point_and_trailing_zeros_can_be_read_as_integer<std::uintmax_t>();
}

GTEST_TEST(ParameterMapInterface, ParameterWithDecimalPointAndNonZeroTrailingCharsCanNotBeReadAsInteger)
{
  Expect_parameter_with_decimal_point_and_non_zero_trailing_chars_can_not_be_read_as_integer<int>();
}
