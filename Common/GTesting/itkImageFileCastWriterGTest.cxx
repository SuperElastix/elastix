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
#include "itkImageFileCastWriter.h"
#include "GTesting/elxCoreMainGTestUtilities.h"
#include "elxPixelTypeToString.h"

// ITK header files:
#include <itkDeref.h>
#include <itkFileTools.h>
#include <itkImage.h>
#include <itkImageFileReader.h>

#include <gtest/gtest.h>

#include <map>
#include <type_traits>
#include <typeindex>

using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::minimumImageSizeValue;
using elx::CoreMainGTestUtilities::GetCurrentBinaryDirectoryPath;
using elx::CoreMainGTestUtilities::GetNameOfTest;
using elx::CoreMainGTestUtilities::TypeHolder;


namespace
{
template <typename... T>
auto
CreateMapOfTypeTraits()
{
  struct TypeTraits
  {
    bool        isSigned;
    bool        isFloatingPoint;
    std::size_t numberOfBits;
  };
  return std::map<std::type_index, TypeTraits>{
    { typeid(T), TypeTraits{ std::is_signed_v<T>, std::is_floating_point_v<T>, sizeof(T) * CHAR_BIT } }...
  };
}
} // namespace


// Test that ImageFileCastWriter stores the specified component type, when calling `WriteCastedImage(image, filename,
// outputComponentType, compress)`.
GTEST_TEST(ImageFileCastWriter, StoresComponentType)
{
  const std::string outputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(outputDirectoryPath);

  using InputPixelType = int;
  static constexpr unsigned int imageDimension{ 2 };

  using InputImageType = itk::Image<InputPixelType, imageDimension>;

  const auto inputImage = CreateImageFilledWithSequenceOfNaturalNumbers<InputPixelType>(
    itk::Size<imageDimension>::Filled(minimumImageSizeValue));

  // Note: ImageFileCastWriter does not support "long_long" or "unsigned_long_long" as `outputComponentType`. When using
  // "long" or "unsigned_long" as `outputComponentType` on a platform whose `long` integer type has 64 bits, it may be
  // read back as "long_long" or "unsigned_long_long", respecively.
  for (const std::string outputComponentType :
       { "char", "unsigned_char", "short", "unsigned_short", "int", "unsigned_int", "float", "double" })
  {
    const std::string filename = outputDirectoryPath + '/' + outputComponentType + ".mhd";
    itk::WriteCastedImage(*inputImage, filename, outputComponentType, false);

    // Read back the file that is written by WriteCastedImage.
    const auto reader = itk::ImageFileReader<InputImageType>::New();
    reader->SetFileName(filename);
    reader->Update();

    // Check that the component type is in the image file.
    const auto & imageIO = itk::Deref(reader->GetImageIO());
    EXPECT_EQ(itk::ImageIOBase::GetComponentTypeAsString(imageIO.GetComponentType()), outputComponentType);

    // Also check that the read image is equal to the written input image.
    EXPECT_EQ(itk::Deref(reader->GetOutput()), itk::Deref(inputImage.get()));
  }
}


// Test that ImageFileCastWriter supports using a fixed width notation to specify the component type, when calling
// `WriteCastedImage(image, filename, outputComponentType, compress)`.
GTEST_TEST(ImageFileCastWriter, SupportsFixedWidthComponentType)
{
  const std::string outputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(outputDirectoryPath);

  using InputPixelType = int;
  static constexpr unsigned int imageDimension{ 2 };

  const auto inputImage = CreateImageFilledWithSequenceOfNaturalNumbers<InputPixelType>(
    itk::Size<imageDimension>::Filled(minimumImageSizeValue));

  const auto mapOfTypeTraits = CreateMapOfTypeTraits<char,
                                                     unsigned char,
                                                     short,
                                                     unsigned short,
                                                     int,
                                                     unsigned int,
                                                     long,
                                                     unsigned long,
                                                     long long,
                                                     unsigned long long,
                                                     float,
                                                     double>();

  const auto check = [inputImage, &outputDirectoryPath, &mapOfTypeTraits](const auto componentTypeHolder) {
    using ComponentType = typename decltype(componentTypeHolder)::Type;
    const auto outputComponentType = elx::PixelTypeToFixedWidthString<ComponentType>();

    const std::string filename = outputDirectoryPath + '/' + outputComponentType + ".mhd";
    itk::WriteCastedImage(*inputImage, filename, outputComponentType, false);

    // Read back the file that is written by WriteCastedImage.
    const auto reader = itk::ImageFileReader<itk::Image<InputPixelType, imageDimension>>::New();
    reader->SetFileName(filename);
    reader->Update();

    // Check that the component type in the image file has the expected type_info.
    const auto & imageIO = itk::Deref(reader->GetImageIO());

    const auto typeTraits = mapOfTypeTraits.at(imageIO.GetComponentTypeInfo());

    EXPECT_EQ(typeTraits.isSigned, std::is_signed_v<ComponentType>);
    EXPECT_EQ(typeTraits.isFloatingPoint, std::is_floating_point_v<ComponentType>);
    EXPECT_EQ(typeTraits.numberOfBits, sizeof(ComponentType) * CHAR_BIT);

    // Also check that the read image is equal to the written input image.
    EXPECT_EQ(itk::Deref(reader->GetOutput()), itk::Deref(inputImage.get()));
  };

  check(TypeHolder<std::int8_t>{});
  check(TypeHolder<std::uint8_t>{});
  check(TypeHolder<std::int16_t>{});
  check(TypeHolder<std::uint16_t>{});
  check(TypeHolder<std::int32_t>{});
  check(TypeHolder<std::uint32_t>{});
  check(TypeHolder<std::int64_t>{});
  check(TypeHolder<std::uint64_t>{});
  check(TypeHolder<float>{});
  check(TypeHolder<double>{});
}


// Test that ImageFileCastWriter throws an exception, when calling `WriteCastedImage(image, filename,
// outputComponentType, compress)` with an invalid or unsupported component type.
GTEST_TEST(ImageFileCastWriter, ThrowsExceptionOnInvalidComponentType)
{
  const std::string outputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(outputDirectoryPath);

  using InputPixelType = int;
  static constexpr unsigned int imageDimension{ 2 };

  const auto inputImage = CreateImageFilledWithSequenceOfNaturalNumbers<InputPixelType>(
    itk::Size<imageDimension>::Filled(minimumImageSizeValue));

  for (const std::string invalidComponentType : {
         "",
         "not a proper type",
         "string",
       })
  {
    EXPECT_THROW(itk::WriteCastedImage(*inputImage, outputDirectoryPath + "/file.mhd", invalidComponentType, false),
                 itk::ExceptionObject);
  }
}
