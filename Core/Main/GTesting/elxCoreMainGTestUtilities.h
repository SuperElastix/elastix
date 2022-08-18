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

#include <elxBaseComponent.h> // For elx.
#include <elxParameterObject.h>

#include <itkImage.h>
#include <itkImageBufferRange.h>
#include <itkImageRegionRange.h>
#include <itkIndex.h>
#include <itkSize.h>

#include <algorithm> // For fill and transform.
#include <array>
#include <cmath> // For round.
#include <initializer_list>
#include <iterator> // For begin and end.
#include <map>
#include <numeric> // For iota.
#include <string>
#include <type_traits> // For is_pointer, is_same, and integral_constant.
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

/// Dereferences the specified raw pointer. Throws an `Exception` instead, when the pointer is null.
template <typename TRawPointer>
decltype(auto)
Deref(const TRawPointer ptr)
{
  static_assert(std::is_pointer<TRawPointer>::value, "For smart pointers, use DerefSmartPointer instead!");

  if (ptr == nullptr)
  {
    throw Exception("Deref error: the pointer should not be null!");
  }
  return *ptr;
}


template <typename TSmartPointer>
decltype(auto)
DerefSmartPointer(const TSmartPointer & ptr)
{
  static_assert(!std::is_pointer<TSmartPointer>::value, "For raw pointers, use Deref instead!");

  if (ptr == nullptr)
  {
    throw Exception("Deref error: the (smart) pointer should not be null!");
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
  static_assert(std::is_same<decltype(T::New()), itk::SmartPointer<T>>{},
                "T::New() must return an itk::SmartPointer<T>!");

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


// Converts the specified strings to a vector of double.
// Assumes that each string represents a floating point number.
inline std::vector<double>
ConvertStringsToVectorOfDouble(const std::vector<std::string> & strings)
{
  std::vector<double> result(strings.size());

  std::transform(strings.cbegin(), strings.cend(), result.begin(), [](const std::string & str) {
    std::size_t index{};
    const auto  result = std::stod(str, &index);

    // Test that all characters have been processed, by std::stod.
    EXPECT_EQ(index, str.size());
    return result;
  });

  return result;
}


// Converts the specified vector of double to itk::Offset, by rounding each element.
template <std::size_t VDimension>
itk::Offset<VDimension>
ConvertToOffset(const std::vector<double> & doubles)
{
  ELX_GTEST_EXPECT_FALSE_AND_THROW_EXCEPTION_IF(doubles.size() != VDimension);

  itk::Offset<VDimension> result;
  std::size_t             i{};

  for (const double value : doubles)
  {
    const auto roundedValue = std::round(value);

    EXPECT_GE(roundedValue, std::numeric_limits<itk::OffsetValueType>::min());
    EXPECT_LE(roundedValue, std::numeric_limits<itk::OffsetValueType>::max());

    result[i] = static_cast<itk::OffsetValueType>(roundedValue);
    ++i;
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


inline std::vector<double>
GetTransformParametersFromMaps(const std::vector<ParameterObject::ParameterMapType> & transformParameterMaps)
{
  // For the time being, only support a single parameter map here.
  EXPECT_EQ(transformParameterMaps.size(), 1);

  if (transformParameterMaps.empty())
  {
    throw Exception("Error: GetTransformParametersFromMaps should not return an empty ParameterMap!");
  }

  const auto & transformParameterMap = transformParameterMaps.front();
  const auto   found = transformParameterMap.find("TransformParameters");

  if (found == transformParameterMap.cend())
  {
    throw Exception("Error: GetTransformParametersFromMaps did not find TransformParameters!");
  }
  return ConvertStringsToVectorOfDouble(found->second);
}


template <typename TFilter>
std::vector<double>
GetTransformParametersFromFilter(TFilter & filter)
{
  const auto   transformParameterObject = filter.GetTransformParameterObject();
  const auto & transformParameterMaps = Deref(transformParameterObject).GetParameterMap();
  return GetTransformParametersFromMaps(transformParameterMaps);
}


// Creates a test image, filled with zero.
template <typename TPixel, unsigned VImageDimension>
auto
CreateImage(const itk::Size<VImageDimension> & imageSize)
{
  const auto image = itk::Image<TPixel, VImageDimension>::New();
  image->SetRegions(imageSize);
  image->Allocate(true);
  return image;
}


// Creates a test image, filled with a sequence of natural numbers, 1, 2, 3, ..., N.
template <typename TPixel, unsigned VImageDimension>
auto
CreateImageFilledWithSequenceOfNaturalNumbers(const itk::Size<VImageDimension> & imageSize)
{
  using ImageType = itk::Image<TPixel, VImageDimension>;
  const auto image = ImageType::New();
  image->SetRegions(imageSize);
  image->Allocate();
  const itk::ImageBufferRange<ImageType> imageBufferRange{ *image };
  std::iota(imageBufferRange.begin(), imageBufferRange.end(), TPixel{ 1 });
  return image;
}


std::string
GetDataDirectoryPath();

// Returns CMAKE_CURRENT_BINARY_DIR: the path to Core Main GTesting subdirectory of the elastix build tree (without
// trailing slash).
std::string
GetCurrentBinaryDirectoryPath();

// Returns the name of a test defined by `GTEST_TEST(TestSuiteName, TestName)` as "TestSuiteName_TestName_Test".
std::string
GetNameOfTest(const testing::Test &);

} // namespace CoreMainGTestUtilities
} // namespace elastix


#endif
