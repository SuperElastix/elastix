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
#include <itkTransformixFilter.h>
#include <itkParameterFileParser.h>

#include "elxTransformIO.h"
#include "elxCoreMainGTestUtilities.h"

// ITK header files:
#include <itkImage.h>
#include <itkImageBufferRange.h>
#include <itkResampleImageFilter.h>
#include <itkTranslationTransform.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For equal and transform.
#include <map>
#include <string>


using ParameterMapType = itk::ParameterFileParser::ParameterMapType;
using ParameterValuesType = itk::ParameterFileParser::ParameterValuesType;

namespace
{
elx::ParameterObject::Pointer
CreateParameterObject(const ParameterMapType & parameterMap)
{
  const auto parameterObject = elx::ParameterObject::New();
  parameterObject->SetParameterMap(parameterMap);
  return parameterObject;
}


template <unsigned ImageDimension>
ParameterValuesType
CreateDefaultDirectionParameterValues()
{
  constexpr auto      numberOfValues = ImageDimension * ImageDimension;
  ParameterValuesType values(numberOfValues, "0");

  for (std::size_t i{}; i < numberOfValues; i += (ImageDimension + 1))
  {
    values[i] = "1";
  }
  return values;
}


template <typename T>
ParameterValuesType
ConvertToParameterValues(const T & container)
{
  ParameterValuesType parameterValues(container.size());

  std::transform(std::begin(container),
                 std::end(container),
                 parameterValues.begin(),
                 [](decltype(*std::begin(container)) inputValue) { return std::to_string(inputValue); });
  return parameterValues;
}


template <typename TImage>
itk::SmartPointer<TImage>
TranslateImage(TImage & image, const typename TImage::OffsetType & translationOffset)
{
  constexpr auto ImageDimension = TImage::ImageDimension;

  const auto filter = itk::TransformixFilter<TImage>::New();

  filter->SetMovingImage(&image);
  filter->SetTransformParameterObject(
    CreateParameterObject({ // Parameters in alphabetic order:
                            { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                            { "Index", ParameterValuesType(ImageDimension, "0") },
                            { "NumberOfParameters", { std::to_string(ImageDimension) } },
                            { "Origin", ParameterValuesType(ImageDimension, "0") },
                            { "Size", ConvertToParameterValues(image.GetRequestedRegion().GetSize()) },
                            { "Transform", ParameterValuesType{ "TranslationTransform" } },
                            { "TransformParameters", ConvertToParameterValues(translationOffset) },
                            { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
  filter->Update();

  return &elx::CoreMainGTestUtilities::Deref(filter->GetOutput());
}


template <unsigned NImageDimension>
void
ExpectEqualImageBases(const itk::ImageBase<NImageDimension> & actualImageBase,
                      const itk::ImageBase<NImageDimension> & expectedImageBase)
{
  EXPECT_EQ(actualImageBase.GetBufferedRegion(), expectedImageBase.GetBufferedRegion());
  EXPECT_EQ(actualImageBase.GetSpacing(), expectedImageBase.GetSpacing());
  EXPECT_EQ(actualImageBase.GetOrigin(), expectedImageBase.GetOrigin());
  EXPECT_EQ(actualImageBase.GetDirection(), expectedImageBase.GetDirection());
  EXPECT_EQ(actualImageBase.GetInverseDirection(), expectedImageBase.GetInverseDirection());
}


template <typename TImage>
void
ExpectEqualImages(const TImage & actualImage, const TImage & expectedImage)
{
  ExpectEqualImageBases(actualImage, expectedImage);

  const auto * const actualPixelContainer = actualImage.GetPixelContainer();
  const auto * const expectedPixelContainer = expectedImage.GetPixelContainer();

  if (actualPixelContainer != expectedPixelContainer)
  {
    ASSERT_NE(actualPixelContainer, nullptr);
    ASSERT_NE(expectedPixelContainer, nullptr);

    const auto * const actualBufferPointer = actualImage.GetBufferPointer();
    const auto * const expectedBufferPointer = expectedImage.GetBufferPointer();

    if (actualBufferPointer != expectedBufferPointer)
    {
      ASSERT_NE(actualBufferPointer, nullptr);
      ASSERT_NE(expectedBufferPointer, nullptr);

      const auto actualBufferSize = actualPixelContainer->Size();
      ASSERT_EQ(actualBufferSize, expectedPixelContainer->Size());

      EXPECT_TRUE(std::equal(actualBufferPointer, actualBufferPointer + actualBufferSize, expectedBufferPointer));
    }
  }
}


template <typename TImage>
void
ExpectAlmostEqualPixelValues(const TImage & actualImage, const TImage & expectedImage)
{
  // ImageBufferRange is to be moved from namespace itk::Experimental
  // to namespace itk with ITK version 5.2.
  using namespace itk;
  using namespace itk::Experimental;
  using ImageBufferRangeType = ImageBufferRange<const TImage>;

  const ImageBufferRangeType actualImageBufferRange(actualImage);
  const ImageBufferRangeType expectedImageBufferRange(expectedImage);

  ASSERT_EQ(actualImageBufferRange.size(), expectedImageBufferRange.size());

  auto expectedImageIterator = expectedImageBufferRange.cbegin();

  using PixelType = typename TImage::PixelType;

  for (const PixelType actualPixelValue : actualImageBufferRange)
  {
    // No rounding errors are expected for the value 1.0 (one), but some
    // small rounding errors around 0.0 (zero) are found acceptable.
    EXPECT_LT(std::abs(actualPixelValue - *expectedImageIterator), std::numeric_limits<PixelType>::epsilon());
    ++expectedImageIterator;
  }
}


template <typename TImage>
void
Expect_TransformixFilter_output_almost_same_as_ResampleImageFilter(
  TImage &                                                                       image,
  const itk::Transform<double, TImage::ImageDimension, TImage::ImageDimension> & itkTransform)
{
  constexpr auto ImageDimension = TImage::ImageDimension;
  const auto     imageSize = image.GetRequestedRegion().GetSize();
  const auto     transformClassName =
    elx::TransformIO::ConvertITKNameOfClassToElastixClassName(itkTransform.GetNameOfClass());

  const auto resampleImageFilter = itk::ResampleImageFilter<TImage, TImage>::New();
  ASSERT_NE(resampleImageFilter, nullptr);

  resampleImageFilter->SetInput(&image);
  resampleImageFilter->SetTransform(&itkTransform);
  resampleImageFilter->SetSize(imageSize);
  resampleImageFilter->Update();
  const auto & resampleImageFilterOutput = elx::CoreMainGTestUtilities::Deref(resampleImageFilter->GetOutput());

  const auto transformixFilter = itk::TransformixFilter<TImage>::New();
  ASSERT_NE(transformixFilter, nullptr);
  transformixFilter->SetMovingImage(&image);
  transformixFilter->SetTransformParameterObject(CreateParameterObject(
    { // Parameters in alphabetic order:
      { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
      { "Index", ParameterValuesType(ImageDimension, "0") },
      { "ITKTransformParameters", ConvertToParameterValues(itkTransform.GetParameters()) },
      { "ITKTransformFixedParameters", ConvertToParameterValues(itkTransform.GetFixedParameters()) },
      { "Origin", ParameterValuesType(ImageDimension, "0") },
      { "Size", ConvertToParameterValues(imageSize) },
      { "Transform", ParameterValuesType{ transformClassName } },
      { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
  transformixFilter->Update();

  const auto & transformixOutput = elx::CoreMainGTestUtilities::Deref(transformixFilter->GetOutput());
  ExpectEqualImageBases(transformixOutput, resampleImageFilterOutput);
  ExpectAlmostEqualPixelValues(transformixOutput, resampleImageFilterOutput);
}

} // namespace


// Tests translating a small (5x6) binary image, having a 2x2 white square.
GTEST_TEST(itkTransformixFilter, Translation2D)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;

  const itk::Offset<ImageDimension> translationOffset{ { 1, -2 } };
  const auto                        regionSize = SizeType::Filled(2);
  const SizeType                    imageSize{ { 5, 6 } };
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const auto transformedImage = TranslateImage(*movingImage, translationOffset);

  ExpectEqualImageBases(*transformedImage, *fixedImage);
  ExpectAlmostEqualPixelValues(*transformedImage, *fixedImage);
}


// Tests translating a small (5x7x9) binary 3D image, having a 2x2x2 white cube.
GTEST_TEST(itkTransformixFilter, Translation3D)
{
  constexpr auto ImageDimension = 3U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;

  const itk::Offset<ImageDimension> translationOffset{ { 1, 2, 3 } };
  const auto                        regionSize = SizeType::Filled(2);
  const SizeType                    imageSize{ { 5, 7, 9 } };
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 2, 3 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const auto transformedImage = TranslateImage(*movingImage, translationOffset);

  ExpectEqualImageBases(*transformedImage, *fixedImage);
  ExpectAlmostEqualPixelValues(*transformedImage, *fixedImage);
}


GTEST_TEST(itkTransformixFilter, TranslationViaExternalTransformFile)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;

  const itk::Offset<ImageDimension> translationOffset{ { 1, -2 } };
  const auto                        regionSize = SizeType::Filled(2);
  const SizeType                    imageSize{ { 5, 6 } };
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 3 } };

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const auto expectedOutputImage = TranslateImage(*movingImage, translationOffset);

  for (const std::string transformFileName :
       { "ITK-Transform.tfm", "ITK-HDF5-Transform.h5", "Special characters [(0-9,;!@#$%&)]/ITK-Transform.tfm" })
  {
    const auto transformFilePathName =
      elx::CoreMainGTestUtilities::GetDataDirectoryPath() + "/Translation(1,-2)/" + transformFileName;

    const auto filter = itk::TransformixFilter<ImageType>::New();
    ASSERT_NE(filter, nullptr);

    filter->SetMovingImage(movingImage);
    filter->SetTransformParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                              { "Index", ParameterValuesType(ImageDimension, "0") },
                              { "Origin", ParameterValuesType(ImageDimension, "0") },
                              { "Size", ConvertToParameterValues(imageSize) },
                              { "Transform", ParameterValuesType{ "File" } },
                              { "TransformFileName", { transformFilePathName } },
                              { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
    filter->Update();
    const auto * const outputImage = filter->GetOutput();
    ExpectEqualImages(elx::CoreMainGTestUtilities::Deref(outputImage), *expectedOutputImage);
  }
}


GTEST_TEST(itkTransformixFilter, ITKTranslationTransform2D)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;

  const itk::Offset<ImageDimension> translationOffset{ { 1, -2 } };
  const auto                        regionSize = SizeType::Filled(2);
  const SizeType                    imageSize{ { 5, 6 } };
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 3 } };

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const auto itkTransform = itk::TranslationTransform<double, ImageDimension>::New();
  ASSERT_NE(itkTransform, nullptr);
  itkTransform->SetOffset(itk::Vector<double, ImageDimension>(translationOffset.data()));

  Expect_TransformixFilter_output_almost_same_as_ResampleImageFilter(*movingImage, *itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKTranslationTransform3D)
{
  constexpr auto ImageDimension = 3U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;

  const itk::Offset<ImageDimension> translationOffset{ { 1, 2, 3 } };
  const auto                        regionSize = SizeType::Filled(2);
  const SizeType                    imageSize{ { 5, 7, 9 } };
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 2, 3 } };

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const auto itkTransform = itk::TranslationTransform<double, ImageDimension>::New();
  ASSERT_NE(itkTransform, nullptr);
  itkTransform->SetOffset(itk::Vector<double, ImageDimension>(translationOffset.data()));

  Expect_TransformixFilter_output_almost_same_as_ResampleImageFilter(*movingImage, *itkTransform);
}
