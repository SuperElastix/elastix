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
#include <elxConversion.h>

#include <itkImage.h>
#include <itkImageBase.h>
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
#include <random>
#include <string>
#include <type_traits> // For is_pointer, is_same, and integral_constant.
#include <vector>

// GoogleTest header file:
#include <gtest/gtest.h>


namespace elastix
{
namespace CoreMainGTestUtilities
{

/// Eases passing a type as argument to a generic lambda.
template <typename TNested>
struct TypeHolder
{
  using Type = TNested;
};

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
DerefRawPointer(const TRawPointer ptr)
{
  static_assert(std::is_pointer_v<TRawPointer>, "For smart pointers, use DerefSmartPointer instead!");

  if (ptr == nullptr)
  {
    throw Exception("DerefRawPointer error: the pointer should not be null!");
  }
  return *ptr;
}


template <typename TSmartPointer>
decltype(auto)
DerefSmartPointer(const TSmartPointer & ptr)
{
  static_assert(!std::is_pointer_v<TSmartPointer>, "For raw pointers, use DerefRawPointer instead!");

  if (ptr == nullptr)
  {
    throw Exception("DerefRawPointer error: the (smart) pointer should not be null!");
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
  std::initializer_list<std::pair<std::string, std::vector<std::string>>> initializerList)
{
  std::map<std::string, std::vector<std::string>> result;

  for (const auto & pair : initializerList)
  {
    EXPECT_TRUE(result.insert(pair).second);
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


ParameterObject::Pointer inline CreateParameterObject(const ParameterObject::ParameterMapType & parameterMap)
{
  const auto parameterObject = ParameterObject::New();
  parameterObject->SetParameterMap(parameterMap);
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
  const auto & transformParameterMaps = DerefRawPointer(transformParameterObject).GetParameterMaps();
  return GetTransformParametersFromMaps(transformParameterMaps);
}


// ITK's RecursiveSeparableImageFilter "requires a minimum of four pixels along the dimension to be processed", at
// https://github.com/InsightSoftwareConsortium/ITK/blob/v5.3.0/Modules/Filtering/ImageFilterBase/include/itkRecursiveSeparableImageFilter.hxx#L226
constexpr itk::SizeValueType minimumImageSizeValue{ 4 };


// The image domain. ITK calls it the "geometry" of an image. ("The geometry of an image is defined by its position,
// orientation, spacing, and extent", according to https://itk.org/Doxygen52/html/classitk_1_1ImageBase.html#details).
// The elastix manual (elastix-5.1.0-manual.pdf, January 16, 2023) simply calls it "the
// Size/Spacing/Origin/Index/Direction settings".
template <unsigned int VDimension>
struct ImageDomain
{
  using ImageBaseType = itk::ImageBase<VDimension>;

  using DirectionType = typename ImageBaseType::DirectionType;
  using IndexType = typename ImageBaseType::IndexType;
  using SizeType = typename ImageBaseType::SizeType;
  using SpacingType = typename ImageBaseType::SpacingType;
  using PointType = typename ImageBaseType::PointType;

  DirectionType direction{ DirectionType::GetIdentity() };
  IndexType     index{};
  SizeType      size{};
  SpacingType   spacing{ itk::MakeFilled<SpacingType>(1.0) };
  PointType     origin{};

  // Default-constructor
  ImageDomain() = default;

  // Explicit constructor
  explicit ImageDomain(const SizeType & size)
    : size(size)
  {}

  explicit ImageDomain(const ImageBaseType & image)
    : direction(image.GetDirection())
    , index(image.GetLargestPossibleRegion().GetIndex())
    , size(image.GetLargestPossibleRegion().GetSize())
    , spacing(image.GetSpacing())
    , origin(image.GetOrigin())
  {}

  // Constructor, allowing to explicitly specify all the settings of the domain.
  ImageDomain(const DirectionType & direction,
              const IndexType &     index,
              const SizeType &      size,
              const SpacingType &   spacing,
              const PointType &     origin)
    : direction(direction)
    , index(index)
    , size(size)
    , spacing(spacing)
    , origin(origin)
  {}

  // Puts the domain settings into the specified image.
  void
  ToImage(itk::ImageBase<VDimension> & image) const
  {
    image.SetDirection(direction);
    image.SetRegions({ index, size });
    image.SetSpacing(spacing);
    image.SetOrigin(origin);
  }

  // Returns the data of this image domain as an elastix/transformix parameter map.
  ParameterObject::ParameterMapType
  AsParameterMap() const
  {
    return {
      // Parameters in alphabetic order:
      { "Direction", elx::Conversion::ToVectorOfStrings(direction) },
      { "Index", elx::Conversion::ToVectorOfStrings(index) },
      { "Origin", elx::Conversion::ToVectorOfStrings(origin) },
      { "Size", elx::Conversion::ToVectorOfStrings(size) },
      { "Spacing", elx::Conversion::ToVectorOfStrings(spacing) },
    };
  }

  friend bool
  operator==(const ImageDomain & lhs, const ImageDomain & rhs)
  {
    return lhs.direction == rhs.direction && lhs.index == rhs.index && lhs.size == rhs.size &&
           lhs.spacing == rhs.spacing && lhs.origin == rhs.origin;
  }

  friend bool
  operator!=(const ImageDomain & lhs, const ImageDomain & rhs)
  {
    return !(lhs == rhs);
  }
};


template <typename TRandomNumberEngine>
int
GenerateRandomSign(TRandomNumberEngine & randomNumberEngine)
{
  return (randomNumberEngine() % 2 == 0) ? -1 : 1;
}


template <unsigned int VImageDimension>
auto
CreateRandomImageDomain(std::mt19937 & randomNumberEngine)
{
  using ImageDomainType = ImageDomain<VImageDimension>;

  const auto createRandomDirection = [&randomNumberEngine] {
    using DirectionType = typename ImageDomainType::DirectionType;
    auto randomDirection = DirectionType::GetIdentity();

    // For now, just a single random rotation
    const auto randomRotation = std::uniform_real_distribution<>{ -M_PI, M_PI }(randomNumberEngine);
    const auto cosRandomRotation = std::cos(randomRotation);
    const auto sinRandomRotation = std::sin(randomRotation);

    randomDirection[0][0] = cosRandomRotation;
    randomDirection[0][1] = sinRandomRotation;
    randomDirection[1][0] = -sinRandomRotation;
    randomDirection[1][1] = cosRandomRotation;

    return randomDirection;
  };
  const auto createRandomIndex = [&randomNumberEngine] {
    typename ImageDomainType::IndexType randomIndex{};
    // Originally tried `std::uniform_int_distribution<itk::IndexValueType>` with
    // `std::numeric_limits<itk::IndexValueType>`, but that caused errors from ImageSamplerBase::CropInputImageRegion(),
    // saying "ERROR: the bounding box of the mask lies entirely out of the InputImageRegion!"
    std::generate(randomIndex.begin(), randomIndex.end(), [&randomNumberEngine] {
      return std::uniform_int_distribution{ std::numeric_limits<int>::min() / 2,
                                            std::numeric_limits<int>::max() / 2 }(randomNumberEngine);
    });
    return randomIndex;
  };
  const auto createRandomSmallImageSize = [&randomNumberEngine] {
    typename ImageDomainType::SizeType randomImageSize{};
    std::generate(randomImageSize.begin(), randomImageSize.end(), [&randomNumberEngine] {
      return std::uniform_int_distribution<itk::SizeValueType>{ minimumImageSizeValue,
                                                                2 * minimumImageSizeValue }(randomNumberEngine);
    });
    return randomImageSize;
  };
  const auto createRandomSpacing = [&randomNumberEngine] {
    typename ImageDomainType::SpacingType randomSpacing{};
    std::generate(randomSpacing.begin(), randomSpacing.end(), [&randomNumberEngine] {
      // Originally tried the maximum interval from std::numeric_limits<itk::SpacePrecisionType>::min() to
      // std::numeric_limits<itk::SpacePrecisionType>::max(), but that caused errors during inverse matrix computation.
      return std::uniform_real_distribution<itk::SpacePrecisionType>{ 0.1, 10.0 }(randomNumberEngine);
    });
    return randomSpacing;
  };
  const auto createRandomPoint = [&randomNumberEngine] {
    typename ImageDomainType::PointType randomPoint{};
    std::generate(randomPoint.begin(), randomPoint.end(), [&randomNumberEngine] {
      // Originally tried an interval up to `std::numeric_limits<itk::SpacePrecisionType>::max() / 2.0`, but that caused
      // errors from ImageSamplerBase::CropInputImageRegion(), saying "ERROR: the bounding box of the mask lies entirely
      // out of the InputImageRegion!"
      return std::uniform_real_distribution<itk::SpacePrecisionType>{
        std::numeric_limits<int>::min(), std::numeric_limits<int>::max()
      }(randomNumberEngine);
    });
    return randomPoint;
  };

  return ImageDomainType{ createRandomDirection(),
                          createRandomIndex(),
                          createRandomSmallImageSize(),
                          createRandomSpacing(),
                          createRandomPoint() };
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

// Creates a test image, filled with zero.
template <typename TPixel, unsigned VImageDimension>
auto
CreateImage(const ImageDomain<VImageDimension> & imageDomain)
{
  const auto image = itk::Image<TPixel, VImageDimension>::New();
  imageDomain.ToImage(*image);
  image->Allocate(true);
  return image;
}


// Creates a test image, filled with a sequence of natural numbers, 1, 2, 3, ..., N.
template <typename TPixel, unsigned VImageDimension>
auto
CreateImageFilledWithSequenceOfNaturalNumbers(const ImageDomain<VImageDimension> & imageDomain)
{
  using ImageType = itk::Image<TPixel, VImageDimension>;
  const auto image = ImageType::New();
  imageDomain.ToImage(*image);
  image->Allocate();
  const itk::ImageBufferRange<ImageType> imageBufferRange{ *image };
  std::iota(imageBufferRange.begin(), imageBufferRange.end(), TPixel{ 1 });
  return image;
}


// Creates a test image, filled with a sequence of natural numbers, 1, 2, 3, ..., N.
template <typename TPixel, unsigned VImageDimension>
auto
CreateImageFilledWithSequenceOfNaturalNumbers(const itk::Size<VImageDimension> & imageSize)
{
  return CreateImageFilledWithSequenceOfNaturalNumbers<TPixel>(ImageDomain<VImageDimension>{ imageSize });
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
