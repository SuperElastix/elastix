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

#include "elxConversion.h"

#include <itkNumberToString.h>
#include <itkOptimizerParameters.h>

#include <double-conversion.h>

// Standard C++ header files:
#include <cassert>
#include <cmath>   // For fmod, fpclassify and FP_SUBNORMAL.
#include <iomanip> // For setprecision.
#include <limits>
#include <numeric> // For accumulate.
#include <regex>
#include <sstream>     // For ostringstream.
#include <type_traits> // For is_floating_point.


namespace
{

template <typename TFloatingPoint>
TFloatingPoint
ConvertStringToFloatingPoint(const double_conversion::StringToDoubleConverter &, const char *, int, int *);

template <>
float
ConvertStringToFloatingPoint(const double_conversion::StringToDoubleConverter & converter,
                             const char * const                                 buffer,
                             const int                                          length,
                             int * const                                        processed_characters_count)
{
  return converter.StringToFloat(buffer, length, processed_characters_count);
}

template <>
double
ConvertStringToFloatingPoint(const double_conversion::StringToDoubleConverter & converter,
                             const char * const                                 buffer,
                             const int                                          length,
                             int * const                                        processed_characters_count)
{
  return converter.StringToDouble(buffer, length, processed_characters_count);
}


template <typename TFloatingPoint>
bool
StringToFloatingPointValue(const std::string & str, TFloatingPoint & value)
{
  static_assert(std::is_floating_point<TFloatingPoint>::value,
                "This function template only supports floating point types.");

  using NumericLimits = std::numeric_limits<TFloatingPoint>;

  if (str == "NaN")
  {
    value = NumericLimits::quiet_NaN();
    return true;
  }
  if (str == "Infinity")
  {
    value = NumericLimits::infinity();
    return true;
  }
  if (str == "-Infinity")
  {
    value = -NumericLimits::infinity();
    return true;
  }
  const auto numberOfChars = str.size();

  if (numberOfChars > std::numeric_limits<int>::max())
  {
    return false;
  }

  constexpr auto double_NaN = std::numeric_limits<double>::quiet_NaN();
  int            processed_characters_count{};

  const double_conversion::StringToDoubleConverter converter(0, double_NaN, double_NaN, "inf", "nan");

  const auto conversionResult = ConvertStringToFloatingPoint<TFloatingPoint>(
    converter, str.c_str(), static_cast<int>(numberOfChars), &processed_characters_count);

  if (std::isnan(conversionResult) || (processed_characters_count != static_cast<int>(numberOfChars)))
  {
    // Conversion failed: the result is NaN (while `str` is not
    // "NaN"), or the converter did not process all characters.
    return false;
  }
  value = conversionResult;
  return true;
}


template <typename TChar>
bool
StringToCharValue(const std::string & str, TChar & value)
{
  static_assert(sizeof(TChar) < sizeof(int), "StringToValueCharType only supports character types smaller than int");

  int temp{};

  if (elastix::Conversion::StringToValue<int>(str, temp))
  {
    // Check that `temp` can be copied losslessly to `value`.
    if ((temp >= std::numeric_limits<TChar>::lowest()) && (temp <= std::numeric_limits<TChar>::max()))
    {
      value = static_cast<TChar>(temp);
      return true;
    }
  }
  return false;
}


} // namespace

namespace elastix
{

/**
 * ****************** SecondsToDHMS ****************************
 */

std::string
Conversion::SecondsToDHMS(const double totalSeconds, const unsigned int precision)
{
  /** Define days, hours, minutes. */
  const std::size_t secondsPerMinute = 60;
  const std::size_t secondsPerHour = 60 * secondsPerMinute;
  const std::size_t secondsPerDay = 24 * secondsPerHour;

  /** Convert total seconds. */
  std::size_t       iSeconds = static_cast<std::size_t>(totalSeconds);
  const std::size_t days = iSeconds / secondsPerDay;

  iSeconds %= secondsPerDay;
  const std::size_t hours = iSeconds / secondsPerHour;

  iSeconds %= secondsPerHour;
  const std::size_t minutes = iSeconds / secondsPerMinute;

  // iSeconds %= secondsPerMinute;
  // const std::size_t seconds = iSeconds;
  const double dSeconds = fmod(totalSeconds, 60.0);

  /** Create a string in days, hours, minutes and seconds. */
  bool               nonzero = false;
  std::ostringstream make_string;
  if (days != 0)
  {
    make_string << days << "d";
    nonzero = true;
  }
  if (hours != 0 || nonzero)
  {
    make_string << hours << "h";
    nonzero = true;
  }
  if (minutes != 0 || nonzero)
  {
    make_string << minutes << "m";
  }
  make_string << std::showpoint << std::fixed << std::setprecision(precision);
  make_string << dSeconds << "s";

  /** Return a value. */
  return make_string.str();

} // end SecondsToDHMS()


/**
 * ****************** ToOptimizerParameters ****************************
 */

itk::OptimizerParameters<double>
Conversion::ToOptimizerParameters(const std::vector<double> & stdVector)
{
  return itk::OptimizerParameters<double>(stdVector.data(), stdVector.size());
};


/**
 * ****************** ToString ****************************
 */

std::string
Conversion::ParameterMapToString(const ParameterMapType & parameterMap)
{
  const auto expectedNumberOfChars = std::accumulate(
    parameterMap.cbegin(),
    parameterMap.cend(),
    std::size_t{},
    [](const std::size_t numberOfChars, const std::pair<std::string, ParameterValuesType> & parameter) {
      return numberOfChars +
             std::accumulate(parameter.second.cbegin(),
                             parameter.second.cend(),
                             // Two parentheses and a linebreak are added for each parameter.
                             parameter.first.size() + 3,
                             [](const std::size_t numberOfCharsPerParameter, const std::string & value) {
                               // A space character is added for each of the values.
                               // Plus two double-quotes, if the value is not a number.
                               return numberOfCharsPerParameter + value.size() + (Conversion::IsNumber(value) ? 1 : 3);
                             });
    });

  std::string result;
  result.reserve(expectedNumberOfChars);

  for (const auto & parameter : parameterMap)
  {
    result.push_back('(');
    result.append(parameter.first);

    for (const auto & value : parameter.second)
    {
      result.push_back(' ');

      if (Conversion::IsNumber(value))
      {
        result.append(value);
      }
      else
      {
        result.push_back('"');
        result.append(value);
        result.push_back('"');
      }
    }
    result.append(")\n");
  }

  // Assert that the correct number of characters was reserved.
  assert(result.size() == expectedNumberOfChars);
  return result;
}


std::string
Conversion::ToString(const double scalar)
{
  return itk::NumberToString<double>{}(scalar);
}


std::string
Conversion::ToString(const float scalar)
{
  return itk::NumberToString<float>{}(scalar);
}


bool
Conversion::IsNumber(const std::string & str)
{
  auto       iter = str.cbegin();
  const auto end = str.cend();

  if (iter == end)
  {
    return false;
  }
  if (*iter == '-')
  {
    // Skip minus sign.
    ++iter;

    if (iter == end)
    {
      return false;
    }
  }

  const auto isDigit = [](const char ch) { return (ch >= '0') && (ch <= '9'); };

  if (!(isDigit(*iter) && isDigit(str.back())))
  {
    // Any number must start and end with a digit.
    return false;
  }
  ++iter;

  const auto numberOfChars = end - iter;
  const auto numberOfDigits = std::count_if(iter, end, isDigit);

  if (numberOfDigits == numberOfChars)
  {
    // Whole (integral) number, e.g.: 1234567890
    return true;
  }

  if ((std::find(iter, end, '.') != end) && (numberOfDigits == (numberOfChars - 1)))
  {
    // Decimal notation, e.g.: 12345.67890
    return true;
  }
  // Scientific notation, e.g.: -1.23e-89 (Note: `iter` has already parsed the optional minus sign and the first digit.
  return std::regex_match(iter, end, std::regex("(\\.\\d+)?e[+-]\\d+"));
}


std::string
Conversion::ToNativePathNameSeparators(const std::string & pathName)
{
  constexpr char separators[] = { '/', '\\' };

  constexpr auto nativateSeparatorIndex =
#ifdef _WIN32
    1;
#else
    0;
#endif

  constexpr char nativeSeparator = separators[nativateSeparatorIndex];
  constexpr char nonNativeSeparator = separators[1 - nativateSeparatorIndex];

  auto result = pathName;
  std::replace(result.begin(), result.end(), nonNativeSeparator, nativeSeparator);
  return result;
}

/**
 * **************** StringToValue ***************
 */

bool
Conversion::StringToValue(const std::string & str, std::string & value)
{
  value = str;
  return true;
} // end StringToValue()


bool
Conversion::StringToValue(const std::string & str, float & value)
{
  return StringToFloatingPointValue(str, value);
}


bool
Conversion::StringToValue(const std::string & str, double & value)
{
  return StringToFloatingPointValue(str, value);
}


bool
Conversion::StringToValue(const std::string & str, char & value)
{
  return StringToCharValue(str, value);
}


bool
Conversion::StringToValue(const std::string & str, signed char & value)
{
  return StringToCharValue(str, value);
}


bool
Conversion::StringToValue(const std::string & str, unsigned char & value)
{
  return StringToCharValue(str, value);
}


bool
Conversion::StringToValue(const std::string & str, bool & value)
{
  if (str == "false")
  {
    value = false;
    return true;
  }
  if (str == "true")
  {
    value = true;
    return true;
  }
  return false;
}


} // end namespace elastix
