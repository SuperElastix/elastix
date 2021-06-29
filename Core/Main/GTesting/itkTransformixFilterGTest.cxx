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

#define _USE_MATH_DEFINES // For M_PI_4.

// First include the header file to be tested:
#include <itkTransformixFilter.h>
#include <itkParameterFileParser.h>

#include "elxCoreMainGTestUtilities.h"
#include "GTesting/elxGTestUtilities.h"

// ITK header files:
#include <itkAffineTransform.h>
#include <itkEuler2DTransform.h>
#include <itkEuler3DTransform.h>
#include <itkImage.h>
#include <itkImageBufferRange.h>
#include <itkNumberToString.h>
#include <itkResampleImageFilter.h>
#include <itkTranslationTransform.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For equal and transform.
#include <cmath>
#include <map>
#include <string>


using ParameterMapType = itk::ParameterFileParser::ParameterMapType;
using ParameterValuesType = itk::ParameterFileParser::ParameterValuesType;

template <typename TPixel, unsigned VImageDimension>
using ResampleImageFilterType =
  itk::ResampleImageFilter<itk::Image<TPixel, VImageDimension>, itk::Image<TPixel, VImageDimension>>;

using elx::GTestUtilities::MakePoint;
using elx::GTestUtilities::MakeSize;
using elx::GTestUtilities::MakeVector;
using elx::CoreMainGTestUtilities::Deref;

namespace
{
// Creates a test image, filled with a sequence of natural numbers, 1, 2, 3, ..., N.
template <typename TPixel, unsigned VImageDimension>
itk::SmartPointer<itk::Image<TPixel, VImageDimension>>
CreateImageFilledWithSequenceOfNaturalNumbers(const itk::Size<VImageDimension> & imageSize)
{
  const auto image = itk::Image<TPixel, VImageDimension>::New();
  image->SetRegions(imageSize);
  image->Allocate();
  const itk::ImageBufferRange<itk::Image<TPixel, VImageDimension>> imageBufferRange{ *image };
  std::iota(imageBufferRange.begin(), imageBufferRange.end(), TPixel{ 1 });
  return image;
}


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
                 [](decltype(*std::begin(container)) inputValue) { return itk::NumberToString<double>{}(inputValue); });
  return parameterValues;
}


// Translates an image by the specified offset, using itk::TransformixFilter,
// specifying "TranslationTransform" as Transform.
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
                            { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                            { "Size", ConvertToParameterValues(image.GetRequestedRegion().GetSize()) },
                            { "Transform", ParameterValuesType{ "TranslationTransform" } },
                            { "TransformParameters", ConvertToParameterValues(translationOffset) },
                            { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
  filter->Update();

  return &Deref(filter->GetOutput());
}


template <typename TPixel, unsigned int VImageDimension>
void
ExpectEqualImages(const itk::Image<TPixel, VImageDimension> & actualImage,
                  const itk::Image<TPixel, VImageDimension> & expectedImage)
{
  EXPECT_EQ(actualImage, expectedImage);
}


template <typename TImage>
bool
ImageBuffer_has_nonzero_pixel_values(const TImage & image)
{
  const itk::ImageBufferRange<const TImage> imageBufferRange(image);
  return std::any_of(imageBufferRange.cbegin(),
                     imageBufferRange.cend(),
                     [](const typename TImage::PixelType pixelValue) { return pixelValue != 0; });
}


template <typename TPixel, unsigned VImageDimension>
itk::SmartPointer<itk::TransformixFilter<itk::Image<TPixel, VImageDimension>>>
CreateTransformixFilter(itk::Image<TPixel, VImageDimension> &                            image,
                        const itk::Transform<double, VImageDimension, VImageDimension> & itkTransform)
{
  const auto filter = itk::TransformixFilter<itk::Image<TPixel, VImageDimension>>::New();
  filter->SetMovingImage(&image);

  std::string transformName = itkTransform.GetNameOfClass();

  const auto dimensionPosition = transformName.find(std::to_string(VImageDimension) + "DTransform");
  if (dimensionPosition != std::string::npos)
  {
    // Erase "2D" or "3D".
    transformName.erase(dimensionPosition, 2);
  }

  filter->SetTransformParameterObject(CreateParameterObject(
    { // Parameters in alphabetic order:
      { "Direction", CreateDefaultDirectionParameterValues<VImageDimension>() },
      { "Index", ParameterValuesType(VImageDimension, "0") },
      { "ITKTransformParameters", ConvertToParameterValues(itkTransform.GetParameters()) },
      { "ITKTransformFixedParameters", ConvertToParameterValues(itkTransform.GetFixedParameters()) },
      { "Origin", ParameterValuesType(VImageDimension, "0") },
      { "ResampleInterpolator", { "FinalLinearInterpolator" } },
      { "Size", ConvertToParameterValues(image.GetBufferedRegion().GetSize()) },
      { "Transform", { transformName } },
      { "Spacing", ParameterValuesType(VImageDimension, "1") } }));
  filter->Update();
  return filter;
}


template <typename TPixel, unsigned VImageDimension>
itk::SmartPointer<ResampleImageFilterType<TPixel, VImageDimension>>
CreateResampleImageFilter(const itk::Image<TPixel, VImageDimension> &                      image,
                          const itk::Transform<double, VImageDimension, VImageDimension> & itkTransform)
{
  const auto filter = ResampleImageFilterType<TPixel, VImageDimension>::New();
  filter->SetInput(&image);
  filter->SetTransform(&itkTransform);
  filter->SetSize(image.GetBufferedRegion().GetSize());
  filter->Update();
  return filter;
}


template <typename TPixel, unsigned VImageDimension>
void
Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
  itk::Image<TPixel, VImageDimension> &                            inputImage,
  const itk::Transform<double, VImageDimension, VImageDimension> & itkTransform)
{
  const auto resampleImageFilter = CreateResampleImageFilter(inputImage, itkTransform);
  const auto transformixFilter = CreateTransformixFilter(inputImage, itkTransform);

  const auto & resampleImageFilterOutput = Deref(Deref(resampleImageFilter.GetPointer()).GetOutput());
  const auto & transformixFilterOutput = Deref(Deref(transformixFilter.GetPointer()).GetOutput());

  // Check that the ResampleImageFilter output isn't equal to the input image,
  // otherwise the test itself would be less interesting.
  EXPECT_NE(resampleImageFilterOutput, inputImage);

  // Check that the output is not simply a black image, otherwise the test
  // itself would be less interesting.
  EXPECT_TRUE(ImageBuffer_has_nonzero_pixel_values(transformixFilterOutput));

  ExpectEqualImages(transformixFilterOutput, resampleImageFilterOutput);
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

  ExpectEqualImages(*transformedImage, *fixedImage);
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

  ExpectEqualImages(*transformedImage, *fixedImage);
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
                              { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                              { "Size", ConvertToParameterValues(imageSize) },
                              { "Transform", ParameterValuesType{ "File" } },
                              { "TransformFileName", { transformFilePathName } },
                              { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
    filter->Update();
    const auto * const outputImage = filter->GetOutput();
    ExpectEqualImages(Deref(outputImage), *expectedOutputImage);
  }
}


GTEST_TEST(itkTransformixFilter, ITKTranslationTransform2D)
{
  constexpr auto ImageDimension = 2U;

  const auto itkTransform = itk::TranslationTransform<double, ImageDimension>::New();
  itkTransform->SetOffset(MakeVector(1.0, -2.0));

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(MakeSize(5, 6)), *itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKTranslationTransform3D)
{
  constexpr auto ImageDimension = 3U;

  const auto itkTransform = itk::TranslationTransform<double, ImageDimension>::New();
  itkTransform->SetOffset(MakeVector(1.0, -2.0, 3.0));

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(MakeSize(5, 6, 7)), *itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKAffineTransform2D)
{
  constexpr auto ImageDimension = 2U;

  const auto itkTransform = itk::AffineTransform<double, ImageDimension>::New();
  itkTransform->SetTranslation(MakeVector(1.0, -2.0));
  itkTransform->SetCenter(MakePoint(2.5, 3.0));
  itkTransform->Rotate2D(M_PI_4);

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(MakeSize(5, 6)), *itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKAffineTransform3D)
{
  constexpr auto ImageDimension = 3U;

  const auto itkTransform = itk::AffineTransform<double, ImageDimension>::New();
  itkTransform->SetTranslation(MakeVector(1.0, 2.0, 3.0));
  itkTransform->SetCenter(MakePoint(3.0, 2.0, 1.0));
  itkTransform->Rotate3D(itk::Vector<double, ImageDimension>(1.0), M_PI_4);

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(MakeSize(5, 6, 7)), *itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKEulerTransform2D)
{
  const auto itkTransform = itk::Euler2DTransform<double>::New();
  itkTransform->SetTranslation(MakeVector(1.0, -2.0));
  itkTransform->SetCenter(MakePoint(2.5, 3.0));

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(MakeSize(5, 6)), *itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKEulerTransform3D)
{
  const auto itkTransform = itk::Euler3DTransform<double>::New();
  itkTransform->SetTranslation(MakeVector(1.0, -2.0, 3.0));
  itkTransform->SetCenter(MakePoint(3.0, 2.0, 1.0));

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(MakeSize(5, 6, 7)), *itkTransform);
}
