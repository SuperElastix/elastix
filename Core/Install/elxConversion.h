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
#ifndef elxConversion_h
#define elxConversion_h

#include "itkMatrix.h"

#include <vnl/vnl_vector.h>

#include <iterator>
#include <map>
#include <string>
#include <type_traits> // For is_integral and is_same.
#include <vector>

namespace itk
{
// Forward declaration from ITK header <itkOptimizerParameters.h>.
template <typename>
class ITK_TEMPLATE_EXPORT OptimizerParameters;
} // namespace itk

namespace elastix
{
/**
 * \class Conversion
 *
 * \brief A class that contains utility functions for the conversion of
 * number of seconds and parameter values to text.
 */
class Conversion
{
public:
  /** Corresponds with typedefs from the elastix class itk::ParameterFileParser. */
  using ParameterValuesType = std::vector<std::string>;
  using ParameterMapType = std::map<std::string, ParameterValuesType>;

  /** Overload set, similar to C++17 `std::size(const TContainer&)` (which can only be
   * used within the implementation of elastix is upgraded to C++17 or higher).
   */
  template <typename TContainer, unsigned NDimension = TContainer::Dimension>
  static std::size_t
  GetNumberOfElements(const TContainer &)
  {
    return NDimension;
  }

  template <typename TValue>
  static std::size_t
  GetNumberOfElements(const vnl_vector<TValue> & vnlVector)
  {
    return vnlVector.size();
  }


  /** Convenience function to convert seconds to day, hour, minute, second format. */
  static std::string
  SecondsToDHMS(const double totalSeconds, const unsigned int precision);

  /** Convenience function to convert a boolean to a text string. */
  static constexpr const char *
  BoolToString(const bool arg)
  {
    return arg ? "true" : "false";
  }


  /** Converts the specified `std::vector` to an OptimizerParameters object. */
  static itk::OptimizerParameters<double>
  ToOptimizerParameters(const std::vector<double> &);

  /** Converts the specified parameter map to a text string, according to the elastix parameter text file format. */
  static std::string
  ParameterMapToString(const ParameterMapType &);

  /** Convenience function overload to convert a Boolean to a text string. */
  static std::string
  ToString(const bool arg)
  {
    return BoolToString(arg);
  }

  /** Convenience function overload to convert a double precision floating point to a text string. */
  static std::string
  ToString(double);

  /** Convenience function overload to convert a single precision floating point to a text string. */
  static std::string
  ToString(float);

  /** Convenience function overload to convert an integer to a text string. */
  template <typename TInteger>
  static std::string
  ToString(const TInteger integerValue)
  {
    static_assert(std::is_integral<TInteger>::value, "An integer type expected!");
    static_assert(!std::is_same<TInteger, bool>::value, "No bool expected!");
    return std::to_string(integerValue);
  }


  /** Convenience function overload to convert a container to a vector of
   * text strings. The container may be an itk::Size, itk::Index,
   * itk::Point<double,N>, or itk::Vector<double,N>, or
   * itk::OptimizationParameters<double>.
   *
   * The C++ SFINAE idiom is being used to ensure that the argument type
   * supports standard C++ iteration.
   */
  template <typename TContainer, typename SFINAE = typename TContainer::iterator>
  static std::vector<std::string>
  ToVectorOfStrings(const TContainer & container)
  {
    std::vector<std::string> result;

    // Note: Uses TContainer::Dimension instead of container.size(),
    // because itk::FixedArray::size() is not yet included with ITK 5.1.1.
    result.reserve(GetNumberOfElements(container));

    for (const auto element : container)
    {
      result.push_back(Conversion::ToString(element));
    }
    return result;
  }

  /** Convenience function overload to convert a 2-D matrix to a vector of
   * text strings. Typically used for an itk::ImageBase::DirectionType.
   */
  template <typename T, unsigned int NRows, unsigned int NColumns>
  static std::vector<std::string>
  ToVectorOfStrings(const itk::Matrix<T, NRows, NColumns> & matrix)
  {
    std::vector<std::string> result;
    result.reserve(NColumns * NRows);

    for (unsigned column{}; column < NColumns; ++column)
    {
      for (unsigned row{}; row < NRows; ++row)
      {
        result.push_back(Conversion::ToString(matrix(row, column)));
      }
    }
    return result;
  }


  /** Convenience function to concatenate two vectors. */
  template <typename TValue>
  static std::vector<TValue>
  ConcatenateVectors(std::vector<TValue> vector1, std::vector<TValue> vector2)
  {
    vector1.insert(end(vector1), std::make_move_iterator(begin(vector2)), std::make_move_iterator(end(vector2)));
    return vector1;
  }


  /** Convenience function which tells whether the argument may represent a number (either fixed point, floating point,
   * or integer/whole number).
   * \note IsNumber("NaN") and IsNumber("nan") return false.
   */
  static bool
  IsNumber(const std::string &);


  /** Similar to Qt5 `QDir::toNativeSeparators(const QString &pathName)`.
   */
  static std::string
  ToNativePathNameSeparators(const std::string &);
};

} // end namespace elastix

#endif // end #ifndef elxConversion_h
