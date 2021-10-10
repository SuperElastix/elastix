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

#include "itkParameterMapInterface.h"

#include <double-conversion.h>

// Standard C++ header files:
#include <cmath> // For fpclassify and FP_SUBNORMAL.
#include <limits>
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

} // namespace


namespace itk
{

/**
 * **************** Constructor ***************
 */

ParameterMapInterface::ParameterMapInterface() = default;


/**
 * **************** Destructor ***************
 */

ParameterMapInterface ::~ParameterMapInterface() = default;


/**
 * **************** SetParameterMap ***************
 */

void
ParameterMapInterface::SetParameterMap(const ParameterMapType & parMap)
{
  if (!parMap.empty())
  {
    this->m_ParameterMap = parMap;
  }

} // end SetParameterMap()


/**
 * **************** CountNumberOfParameterEntries ***************
 */

std::size_t
ParameterMapInterface::CountNumberOfParameterEntries(const std::string & parameterName) const
{
  if (this->m_ParameterMap.count(parameterName))
  {
    return this->m_ParameterMap.find(parameterName)->second.size();
  }
  return 0;

} // end CountNumberOfParameterEntries()


/**
 * **************** ReadParameter ***************
 */

bool
ParameterMapInterface::ReadParameter(bool &              parameterValue,
                                     const std::string & parameterName,
                                     const unsigned int  entry_nr,
                                     const bool          printThisErrorMessage,
                                     std::string &       errorMessage) const
{
  /** Translate the default boolean to string. */
  std::string parameterValueString;
  if (parameterValue)
  {
    parameterValueString = "true";
  }
  else
  {
    parameterValueString = "false";
  }

  /** Read the boolean as a string. */
  bool dummy = this->ReadParameter(parameterValueString, parameterName, entry_nr, printThisErrorMessage, errorMessage);

  /** Translate the read-in string to boolean. */
  parameterValue = false;

  if (!StringCast(parameterValueString, parameterValue))
  {
    /** Trying to read a string other than "true" or "false" as a boolean. */
    std::stringstream ss;
    ss << "ERROR: Entry number " << entry_nr << " for the parameter \"" << parameterName
       << "\" should be a boolean, i.e. either \"true\" or \"false\", but it reads \"" << parameterValueString << "\".";

    itkExceptionMacro(<< ss.str());
  }

  return dummy;

} // end ReadParameter()


/**
 * **************** StringCast ***************
 */

bool
ParameterMapInterface::StringCast(const std::string & parameterValue, std::string & casted)
{
  casted = parameterValue;
  return true;
} // end StringCast()


template <typename TFloatingPoint>
bool
ParameterMapInterface::StringCastToFloatingPoint(const std::string & parameterValue, TFloatingPoint & casted)
{
  static_assert(std::is_floating_point<TFloatingPoint>::value,
                "This function template only supports floating point types.");

  using NumericLimits = std::numeric_limits<TFloatingPoint>;

  if (parameterValue == "NaN")
  {
    casted = NumericLimits::quiet_NaN();
    return true;
  }
  if (parameterValue == "Infinity")
  {
    casted = NumericLimits::infinity();
    return true;
  }
  if (parameterValue == "-Infinity")
  {
    casted = -NumericLimits::infinity();
    return true;
  }
  const auto numberOfChars = parameterValue.size();

  if (numberOfChars > std::numeric_limits<int>::max())
  {
    return false;
  }

  constexpr auto double_NaN = std::numeric_limits<double>::quiet_NaN();
  int            processed_characters_count{};

  const double_conversion::StringToDoubleConverter converter(0, double_NaN, double_NaN, "inf", "nan");

  const auto conversionResult = ConvertStringToFloatingPoint<TFloatingPoint>(
    converter, parameterValue.c_str(), static_cast<int>(numberOfChars), &processed_characters_count);

  if (std::isnan(conversionResult) || (processed_characters_count != static_cast<int>(numberOfChars)))
  {
    // Conversion failed: the result is NaN (while `parameterValue` is not
    // "NaN"), or the converter did not process all characters.
    return false;
  }
  casted = conversionResult;
  return true;

} // end StringCastToFloatingPoint()


bool
ParameterMapInterface::StringCast(const std::string & parameterValue, float & casted)
{
  return Self::StringCastToFloatingPoint(parameterValue, casted);
}


bool
ParameterMapInterface::StringCast(const std::string & parameterValue, double & casted)
{
  return Self::StringCastToFloatingPoint(parameterValue, casted);
}


template <typename TChar>
bool
ParameterMapInterface::StringCastToCharType(const std::string & parameterValue, TChar & casted)
{
  static_assert(sizeof(TChar) < sizeof(int), "StringCastCharType only supports character types smaller than int");

  int temp{};

  if (Self::StringCast<int>(parameterValue, temp))
  {
    // Check that `temp` can be copied losslessly to `casted`.
    if ((temp >= std::numeric_limits<TChar>::lowest()) && (temp <= std::numeric_limits<TChar>::max()))
    {
      casted = static_cast<TChar>(temp);
      return true;
    }
  }
  return false;

} // end StringCastCharType()


bool
ParameterMapInterface::StringCast(const std::string & parameterValue, char & casted)
{
  return StringCastToCharType(parameterValue, casted);
}


bool
ParameterMapInterface::StringCast(const std::string & parameterValue, signed char & casted)
{
  return StringCastToCharType(parameterValue, casted);
}


bool
ParameterMapInterface::StringCast(const std::string & parameterValue, unsigned char & casted)
{
  return StringCastToCharType(parameterValue, casted);
}


bool
ParameterMapInterface::StringCast(const std::string & parameterValue, bool & casted)
{
  if (parameterValue == "false")
  {
    casted = false;
    return true;
  }
  if (parameterValue == "true")
  {
    casted = true;
    return true;
  }
  return false;
}


/**
 * **************** ReadParameter ***************
 */

bool
ParameterMapInterface::ReadParameter(std::vector<std::string> & parameterValues,
                                     const std::string &        parameterName,
                                     const unsigned int         entry_nr_start,
                                     const unsigned int         entry_nr_end,
                                     const bool                 printThisErrorMessage,
                                     std::string &              errorMessage) const
{
  /** Reset the error message. */
  errorMessage = "";

  /** Get the number of entries. */
  std::size_t numberOfEntries = this->CountNumberOfParameterEntries(parameterName);

  /** Check if the requested parameter exists. */
  if (numberOfEntries == 0)
  {
    std::stringstream ss;
    ss << "WARNING: The parameter \"" << parameterName << "\", requested between entry numbers " << entry_nr_start
       << " and " << entry_nr_end << ", does not exist at all.\n"
       << "  The default values are used instead." << std::endl;
    if (printThisErrorMessage && this->m_PrintErrorMessages)
    {
      errorMessage = ss.str();
    }
    return false;
  }

  /** Check. */
  if (entry_nr_start > entry_nr_end)
  {
    std::stringstream ss;
    ss << "WARNING: The entry number start (" << entry_nr_start << ") should be smaller than entry number end ("
       << entry_nr_end << "). It was requested for parameter \"" << parameterName << "\"." << std::endl;

    /** Programming error: just throw an exception. */
    itkExceptionMacro(<< ss.str());
  }

  /** Check if it exists at the requested entry numbers. */
  if (entry_nr_end >= numberOfEntries)
  {
    std::stringstream ss;
    ss << "WARNING: The parameter \"" << parameterName << "\" does not exist at entry number " << entry_nr_end
       << ".\nThe default empty string \"\" is used instead." << std::endl;
    itkExceptionMacro(<< ss.str());
  }

  /** Get the vector of parameters. */
  const ParameterValuesType & vec = this->m_ParameterMap.find(parameterName)->second;

  /** Copy all parameters at once. */
  std::vector<std::string>::const_iterator it = vec.begin();
  parameterValues.clear();
  parameterValues.assign(it + entry_nr_start, it + entry_nr_end + 1);

  return true;
}


} // end namespace itk
