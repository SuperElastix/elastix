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

#ifndef elxCoreMainGTestUtilities_h
#define elxCoreMainGTestUtilities_h

#include <elxParameterObject.h>

#include <itkImage.h>
#include <itkImageRegionRange.h>
#include <itkIndex.h>
#include <itkSize.h>

#include <algorithm> // For fill.
#include <array>
#include <cmath> // For round.
#include <map>
#include <initializer_list>
#include <iterator> // For begin and end.
#include <string>
#include <vector>

// GoogleTest header file:
#include <gtest/gtest.h>


namespace elastix
{
namespace CoreMainGTestUtilities
{

/// Simple exception class, to be used by unit tests.
class Exception : public std::exception
{
  const char * m_message = "";

public:
  explicit Exception(const char * const message)
    : m_message(message)
  {}

  const char *
  what() const noexcept override
  {
    return m_message;
  }
};


/// Expect the specified condition to be false, and throw an exception if it is true.
#define ELX_GTEST_EXPECT_FALSE_AND_THROW_EXCEPTION_IF(condition)                                                       \
  if (condition)                                                                                                       \
  {                                                                                                                    \
    EXPECT_FALSE(true) << "Expected to be false: " #condition;                                                         \
    throw ::elastix::CoreMainGTestUtilities::Exception("Exception thrown because " #condition);                        \
  }                                                                                                                    \
  static_assert(true, "Expect a semi-colon ';' at the end of a macro call")

/// Dereferences the specified pointer. Throws an `Exception` instead, when the pointer is null.
template <typename T>
T &
Deref(T * ptr)
{
  if (ptr == nullptr)
  {
    throw Exception("Deref error: the pointer should not be null!");
  }
  return *ptr;
}


/// Returns a reference to the front of the specified container. Throws an
/// `Exception` instead, when the container is empty.
template <typename T>
decltype(T().front())
Front(T & container)
{
  if (container.empty())
  {
    throw Exception("Front error: the container should be non-empty!");
  }
  return container.front();
}


template <typename T>
itk::SmartPointer<T>
CheckNew()
{
  const auto ptr = T::New();
  if (ptr == nullptr)
  {
    throw Exception("New() error: should not return null!");
  }
  return ptr;
}

/// Fills the specified image region with pixel values 1.
template <typename TPixel, unsigned int VImageDimension>
void
FillImageRegion(itk::Image<TPixel, VImageDimension> & image,
                const itk::Index<VImageDimension> &   regionIndex,
                const itk::Size<VImageDimension> &    regionSize)
{
  const itk::ImageRegionRange<itk::Image<TPixel, VImageDimension>> imageRegionRange{
    image, itk::ImageRegion<VImageDimension>{ regionIndex, regionSize }
  };
  std::fill(std::begin(imageRegionRange), std::end(imageRegionRange), 1);
}


// Converts the specified strings to an array of double.
// Assumes that each string represents a floating point number.
template <unsigned VDimension>
std::array<double, VDimension>
ConvertStringsToArrayOfDouble(const std::vector<std::string> & strings)
{
  ELX_GTEST_EXPECT_FALSE_AND_THROW_EXCEPTION_IF(strings.size() != VDimension);

  std::array<double, VDimension> result;

  for (std::size_t i{}; i < VDimension; ++i)
  {
    const auto & str = strings[i];
    std::size_t  index{};
    result[i] = std::stod(str, &index);

    // Test that all characters have been processed, by std::stod.
    EXPECT_EQ(index, str.size());
  }

  return result;
}


// Converts the specified array of double to itk::Offset, by rounding each element.
template <std::size_t VDimension>
itk::Offset<VDimension>
ConvertArrayOfDoubleToOffset(const std::array<double, VDimension> & doubles)
{
  itk::Offset<VDimension> result;

  for (std::size_t i{}; i < VDimension; ++i)
  {
    const auto roundedValue = std::round(doubles[i]);

    EXPECT_GE(roundedValue, std::numeric_limits<itk::OffsetValueType>::min());
    EXPECT_LE(roundedValue, std::numeric_limits<itk::OffsetValueType>::max());

    result[i] = static_cast<itk::OffsetValueType>(roundedValue);
  }

  return result;
}


std::map<std::string, std::vector<std::string>> inline CreateParameterMap(
  std::initializer_list<std::pair<std::string, std::string>> initializerList)
{
  std::map<std::string, std::vector<std::string>> result;

  for (const auto & pair : initializerList)
  {
    EXPECT_TRUE(result.insert({ pair.first, { pair.second } }).second);
  }
  return result;
}


template <unsigned VImageDimension>
std::map<std::string, std::vector<std::string>>
CreateParameterMap(std::initializer_list<std::pair<std::string, std::string>> initializerList)
{
  std::map<std::string, std::vector<std::string>> result = CreateParameterMap(initializerList);

  for (const auto & key : { "FixedImageDimension", "MovingImageDimension" })
  {
    EXPECT_TRUE(result.insert({ key, { std::to_string(VImageDimension) } }).second);
  }
  return result;
}


ParameterObject::Pointer inline CreateParameterObject(
  std::initializer_list<std::pair<std::string, std::string>> initializerList)
{
  const auto parameterObject = ParameterObject::New();
  parameterObject->SetParameterMap(CreateParameterMap(initializerList));
  return parameterObject;
}


std::string
GetDataDirectoryPath();

} // namespace CoreMainGTestUtilities
} // namespace elastix


#endif
