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

#include "elxTransformIO.h"
#include "elxCoreMainGTestUtilities.h"

// ITK header files:
#include <itkAffineTransform.h>
#include <itkImage.h>
#include <itkImageBufferRange.h>
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

using elx::CoreMainGTestUtilities::Deref;

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


template <typename TPixel, unsigned int VImageDimension>
void
ExpectEqualImages(const itk::Image<TPixel, VImageDimension> & actualImage,
                  const itk::Image<TPixel, VImageDimension> & expectedImage)
{
  EXPECT_EQ(actualImage, expectedImage);
}


template <typename TImage>
void
ExpectAlmostEqualPixelValues(const TImage & actualImage, const TImage & expectedImage, const double tolerance)
{
  // Expect the specified tolerance value to be greater than zero, otherwise
  // `ExpectEqualImages` should have been called instead.
  EXPECT_GT(tolerance, 0.0);

  using ImageBufferRangeType = itk::ImageBufferRange<const TImage>;

  const ImageBufferRangeType actualImageBufferRange(actualImage);
  const ImageBufferRangeType expectedImageBufferRange(expectedImage);

  ASSERT_EQ(actualImageBufferRange.size(), expectedImageBufferRange.size());

  const auto beginOfExpectedImageBuffer = expectedImageBufferRange.cbegin();

  // First expect that _not_ all pixel values are not _exactly_ equal,
  // otherwise `ExpectEqualImages` should probably have been called instead!
  EXPECT_FALSE(std::equal(actualImageBufferRange.cbegin(), actualImageBufferRange.cend(), beginOfExpectedImageBuffer));

  auto expectedImageIterator = beginOfExpectedImageBuffer;

  for (const typename TImage::PixelType actualPixelValue : actualImageBufferRange)
  {
    EXPECT_LE(std::abs(actualPixelValue - *expectedImageIterator), tolerance);
    ++expectedImageIterator;
  }
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
  filter->SetTransformParameterObject(CreateParameterObject(
    { // Parameters in alphabetic order:
      { "Direction", CreateDefaultDirectionParameterValues<VImageDimension>() },
      { "Index", ParameterValuesType(VImageDimension, "0") },
      { "ITKTransformParameters", ConvertToParameterValues(itkTransform.GetParameters()) },
      { "ITKTransformFixedParameters", ConvertToParameterValues(itkTransform.GetFixedParameters()) },
      { "Origin", ParameterValuesType(VImageDimension, "0") },
      { "ResampleInterpolator", { "FinalLinearInterpolator" } },
      { "Size", ConvertToParameterValues(image.GetBufferedRegion().GetSize()) },
      { "Transform", { elx::TransformIO::ConvertITKNameOfClassToElastixClassName(itkTransform.GetNameOfClass()) } },
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
  itk::Image<TPixel, VImageDimension> &                            image,
  const itk::Transform<double, VImageDimension, VImageDimension> & itkTransform)
{
  const auto resampleImageFilter = CreateResampleImageFilter(image, itkTransform);
  const auto transformixFilter = CreateTransformixFilter(image, itkTransform);

  const auto & resampleImageFilterOutput = Deref(Deref(resampleImageFilter.GetPointer()).GetOutput());
  const auto & transformixFilterOutput = Deref(Deref(transformixFilter.GetPointer()).GetOutput());

  // First just test that the output is not simply a black image, otherwise the
  // test itself would be less interesting.
  EXPECT_TRUE(ImageBuffer_has_nonzero_pixel_values(transformixFilterOutput));

  ExpectEqualImages(transformixFilterOutput, resampleImageFilterOutput);
}


template <typename TPixel, unsigned VImageDimension>
void
Expect_TransformixFilter_output_almost_equals_ResampleImageFilter_output(
  itk::Image<TPixel, VImageDimension> &                            image,
  const itk::Transform<double, VImageDimension, VImageDimension> & itkTransform,
  const double                                                     tolerance)
{
  const auto resampleImageFilter = CreateResampleImageFilter(image, itkTransform);
  const auto transformixFilter = CreateTransformixFilter(image, itkTransform);

  const auto & resampleImageFilterOutput = Deref(Deref(resampleImageFilter.GetPointer()).GetOutput());
  const auto & transformixFilterOutput = Deref(Deref(transformixFilter.GetPointer()).GetOutput());

  // First just test that the output is not simply a black image, otherwise the
  // test itself would be less interesting.
  EXPECT_TRUE(ImageBuffer_has_nonzero_pixel_values(transformixFilterOutput));
  EXPECT_TRUE(ImageBuffer_has_nonzero_pixel_values(resampleImageFilterOutput));

  ExpectEqualImageBases(transformixFilterOutput, resampleImageFilterOutput);
  ExpectAlmostEqualPixelValues(transformixFilterOutput, resampleImageFilterOutput, tolerance);
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

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(*movingImage, *itkTransform);
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

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(*movingImage, *itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKAffineTransform2D)
{
  constexpr auto                    ImageDimension = 2U;
  const itk::Offset<ImageDimension> translationOffset{ { 1, -2 } };
  const itk::Size<ImageDimension>   regionSize = itk::Size<ImageDimension>::Filled(2);
  const itk::Size<ImageDimension>   imageSize{ { 5, 6 } };
  const itk::Index<ImageDimension>  index{ { 1, 3 } };

  const auto image = itk::Image<float, ImageDimension>::New();
  image->SetRegions(imageSize);
  image->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*image, index + translationOffset, regionSize);

  const auto itkTransform = itk::AffineTransform<double, ImageDimension>::New();
  ASSERT_NE(itkTransform, nullptr);
  itkTransform->SetTranslation(
    itk::Vector<double, ImageDimension>(std::array<double, ImageDimension>{ 1.0, -2.0 }.data()));
  itkTransform->SetCenter(itk::Point<double, ImageDimension>(std::array<double, ImageDimension>{ 2.5, 3.0 }));
  itkTransform->Rotate2D(M_PI_4);

  // A tolerance value that is just high enough to avoid test failures.
  constexpr auto tolerance = 1.4e-06F;
  Expect_TransformixFilter_output_almost_equals_ResampleImageFilter_output(*image, *itkTransform, tolerance);
}


GTEST_TEST(itkTransformixFilter, ITKAffineTransform3D)
{
  constexpr auto                    ImageDimension = 3U;
  const itk::Offset<ImageDimension> translationOffset{ { 1, 2, 3 } };
  const itk::Size<ImageDimension>   regionSize = itk::Size<ImageDimension>::Filled(2);
  const itk::Size<ImageDimension>   imageSize{ { 5, 7, 9 } };
  const itk::Index<ImageDimension>  index{ { 1, 2, 3 } };

  const auto image = itk::Image<float, ImageDimension>::New();
  image->SetRegions(imageSize);
  image->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*image, index + translationOffset, regionSize);

  const auto itkTransform = itk::AffineTransform<double, ImageDimension>::New();
  ASSERT_NE(itkTransform, nullptr);
  itkTransform->SetTranslation(
    itk::Vector<double, ImageDimension>(std::array<double, ImageDimension>{ 1.0, 2.0, 3.0 }.data()));
  itkTransform->SetCenter(itk::Point<double, ImageDimension>(std::array<double, ImageDimension>{ 3.0, 2.0, 1.0 }));
  itkTransform->Rotate3D(itk::Vector<double, ImageDimension>(1.0), M_PI_4);

  // A tolerance value that is just high enough to avoid test failures.
  constexpr auto tolerance = 2.0e-06F;
  Expect_TransformixFilter_output_almost_equals_ResampleImageFilter_output(*image, *itkTransform, tolerance);
}
