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
#include "itkComputeImageExtremaFilter.h"
#include "../Core/Main/GTesting/elxCoreMainGTestUtilities.h"
#include "elxDefaultConstruct.h"

#include <itkImage.h>
#include <itkImageBufferRange.h>

#include <gtest/gtest.h>

namespace itk
{
template class ComputeImageExtremaFilter<itk::Image<int>>;
template class ComputeImageExtremaFilter<itk::Image<double, 3>>;
} // namespace itk

using itk::ComputeImageExtremaFilter;

// Using-declaration:
using elx::CoreMainGTestUtilities::CheckNew;
using elx::CoreMainGTestUtilities::CreateImage;

namespace
{
using MaskPixelType = unsigned char;

template <typename TValues>
void
AssertAllValuesGreaterThanZero(const TValues & values)
{
  for (const auto & value : values)
  {
    ASSERT_GT(value, 0);
  }
}


template <typename TImage>
void
Expect_uniform_image(const typename TImage::SizeType & imageSize)
{
  // Sanity check before testing what is expected.
  AssertAllValuesGreaterThanZero(imageSize);

  using PixelType = typename TImage::PixelType;
  const auto image = TImage::New();

  image->SetRegions(imageSize);
  image->Allocate();

  const auto ExpectForPixelValue = [image](const PixelType pixelValue) {
    image->FillBuffer(pixelValue);

    const auto filter = CheckNew<ComputeImageExtremaFilter<TImage>>();
    filter->SetInput(image);
    filter->Update();

    EXPECT_EQ(filter->GetMinimum(), pixelValue);
    EXPECT_EQ(filter->GetMaximum(), pixelValue);
  };

  using PixelLimitsType = std::numeric_limits<PixelType>;

  ExpectForPixelValue(0);
  ExpectForPixelValue(1);
  ExpectForPixelValue(2);
  ExpectForPixelValue(PixelLimitsType::lowest());
  ExpectForPixelValue(PixelLimitsType::min());
  ExpectForPixelValue(PixelLimitsType::max());

  // Note: infinity values are not tested, as these are
  // not actively supported by the filter.
}


template <typename TImage>
void
Expect_one_positive_pixel_value(const typename TImage::SizeType & imageSize)
{
  // Sanity check before testing what is expected.
  AssertAllValuesGreaterThanZero(imageSize);

  using PixelType = typename TImage::PixelType;
  using IndexType = typename TImage::IndexType;

  const auto image = TImage::New();
  image->SetRegions(imageSize);
  image->Allocate();

  const auto ExpectForPositivePixelValue = [image](const PixelType pixelValue) {
    ASSERT_TRUE(pixelValue > 0);

    image->FillBuffer(0);
    image->SetPixel(IndexType(), pixelValue);

    const auto filter = CheckNew<ComputeImageExtremaFilter<TImage>>();
    filter->SetInput(image);
    filter->Update();

    EXPECT_EQ(filter->GetMinimum(), 0);
    EXPECT_EQ(filter->GetMaximum(), pixelValue);
  };

  ExpectForPositivePixelValue(1);
  ExpectForPositivePixelValue(2);
  ExpectForPositivePixelValue(std::numeric_limits<PixelType>::max());
}


template <typename TImage>
void
Expect_one_negative_pixel_value(const typename TImage::SizeType & imageSize)
{
  // Sanity check before testing what is expected.
  AssertAllValuesGreaterThanZero(imageSize);

  using PixelType = typename TImage::PixelType;
  using IndexType = typename TImage::IndexType;

  const auto image = TImage::New();
  image->SetRegions(imageSize);
  image->Allocate();

  const auto ExpectForNegativePixelValue = [image](const PixelType pixelValue) {
    ASSERT_TRUE(pixelValue < 0);

    image->FillBuffer(0);
    image->SetPixel(IndexType(), pixelValue);

    const auto filter = CheckNew<ComputeImageExtremaFilter<TImage>>();
    filter->SetInput(image);
    filter->Update();

    EXPECT_EQ(filter->GetMinimum(), pixelValue);
    EXPECT_EQ(filter->GetMaximum(), 0);
  };

  using PixelLimitsType = std::numeric_limits<PixelType>;

  ExpectForNegativePixelValue(-1);
  ExpectForNegativePixelValue(-2);
  ExpectForNegativePixelValue(PixelLimitsType::lowest());
}


template <typename TImage>
void
Expect_one_non_zero_pixel_value_masked_in(const typename TImage::SizeType & imageSize)
{
  // Sanity check before testing what is expected.
  AssertAllValuesGreaterThanZero(imageSize);

  using PixelType = typename TImage::PixelType;
  using IndexType = typename TImage::IndexType;
  using FilterType = ComputeImageExtremaFilter<TImage>;
  using ImageSpatialMaskType = typename FilterType::ImageSpatialMaskType;

  static constexpr auto ImageDimension = TImage::ImageDimension;

  const auto image = TImage::New();
  image->SetRegions(imageSize);
  image->Allocate();

  const auto maskImage = itk::Image<MaskPixelType, ImageDimension>::New();
  maskImage->SetRegions(imageSize);
  maskImage->AllocateInitialized();
  maskImage->SetPixel(IndexType(), 1);

  const auto maskSpatialObject = ImageSpatialMaskType::New();
  maskSpatialObject->SetImage(maskImage);
  maskSpatialObject->Update();

  const auto ExpectForPixelValue = [image, maskSpatialObject](const PixelType pixelValue) {
    image->FillBuffer(0);
    image->SetPixel(IndexType(), pixelValue);

    const auto filter = CheckNew<FilterType>();
    filter->SetInput(image);
    filter->SetImageSpatialMask(maskSpatialObject);
    filter->Update();

    EXPECT_EQ(filter->GetMinimum(), pixelValue);
    EXPECT_EQ(filter->GetMaximum(), pixelValue);
  };

  using PixelLimitsType = std::numeric_limits<PixelType>;

  ExpectForPixelValue(1);
  ExpectForPixelValue(2);
  ExpectForPixelValue(PixelLimitsType::lowest());
  ExpectForPixelValue(PixelLimitsType::min());
  ExpectForPixelValue(PixelLimitsType::max());
}

template <typename TImage>
void
Expect_one_positive_pixel_value_all_pixels_masked_in(const typename TImage::SizeType & imageSize)
{
  // Sanity check before testing what is expected.
  AssertAllValuesGreaterThanZero(imageSize);

  using PixelType = typename TImage::PixelType;
  using IndexType = typename TImage::IndexType;
  using FilterType = ComputeImageExtremaFilter<TImage>;
  using ImageSpatialMaskType = typename FilterType::ImageSpatialMaskType;

  static constexpr auto ImageDimension = TImage::ImageDimension;

  const auto image = TImage::New();
  image->SetRegions(imageSize);
  image->Allocate();

  const auto maskImage = itk::Image<MaskPixelType, ImageDimension>::New();
  maskImage->SetRegions(imageSize);
  maskImage->Allocate();
  maskImage->FillBuffer(1);
  const auto maskSpatialObject = ImageSpatialMaskType::New();
  maskSpatialObject->SetImage(maskImage);
  maskSpatialObject->Update();

  const auto ExpectForPositivePixelValue = [image, maskSpatialObject](const PixelType pixelValue) {
    ASSERT_TRUE(pixelValue > 0);

    image->FillBuffer(0);
    image->SetPixel(IndexType(), pixelValue);

    const auto filter = CheckNew<FilterType>();
    filter->SetInput(image);
    filter->SetImageSpatialMask(maskSpatialObject);
    filter->Update();

    EXPECT_EQ(filter->GetMinimum(), 0);
    EXPECT_EQ(filter->GetMaximum(), pixelValue);
  };

  ExpectForPositivePixelValue(1);
  ExpectForPositivePixelValue(2);
  ExpectForPositivePixelValue(std::numeric_limits<PixelType>::max());
}

} // End of namespace.


GTEST_TEST(ComputeImageExtremaFilter, UniformImage)
{
  Expect_uniform_image<itk::Image<float, 2>>({ { 2, 3 } });
  Expect_uniform_image<itk::Image<short, 3>>({ { 2, 3, 4 } });
  Expect_uniform_image<itk::Image<unsigned char, 4>>({ { 2, 3, 4, 5 } });
}


GTEST_TEST(ComputeImageExtremaFilter, OnePositivePixelValue)
{
  Expect_one_positive_pixel_value<itk::Image<float, 2>>({ { 2, 3 } });
  Expect_one_positive_pixel_value<itk::Image<short, 3>>({ { 2, 3, 4 } });
  Expect_one_positive_pixel_value<itk::Image<unsigned char, 4>>({ { 2, 3, 4, 5 } });
}


GTEST_TEST(ComputeImageExtremaFilter, OneNegativePixelValue)
{
  Expect_one_negative_pixel_value<itk::Image<float, 2>>({ { 2, 3 } });
  Expect_one_negative_pixel_value<itk::Image<short, 3>>({ { 2, 3, 4 } });
}


GTEST_TEST(ComputeImageExtremaFilter, OnePositivePixelValueAllPixelsMaskedIn)
{
  Expect_one_positive_pixel_value_all_pixels_masked_in<itk::Image<float, 2>>({ { 2, 3 } });
  Expect_one_positive_pixel_value_all_pixels_masked_in<itk::Image<short, 3>>({ { 2, 3, 4 } });
  Expect_one_positive_pixel_value_all_pixels_masked_in<itk::Image<unsigned char, 4>>({ { 2, 3, 4, 5 } });
}


GTEST_TEST(ComputeImageExtremaFilter, OneNonZeroPixelValueMaskedIn)
{
  Expect_one_non_zero_pixel_value_masked_in<itk::Image<float, 2>>({ { 2, 3 } });
  Expect_one_non_zero_pixel_value_masked_in<itk::Image<short, 3>>({ { 2, 3, 4 } });
  Expect_one_non_zero_pixel_value_masked_in<itk::Image<unsigned char, 4>>({ { 2, 3, 4, 5 } });
}


// Changes of the mask Spacing and Direction with respect to the input image may affect the results.
GTEST_TEST(ComputeImageExtremaFilter, MaskSpacingAndDirectionAffectResults)
{
  using PixelType = int;
  static constexpr auto Dimension = 2U;
  enum class MaskChange
  {
    None,
    Spacing,
    Direction
  };

  using ImageType = itk::Image<PixelType, Dimension>;
  using FilterType = ComputeImageExtremaFilter<ImageType>;
  using ImageSpatialMaskType = FilterType::ImageSpatialMaskType;
  using MaskImageType = ImageSpatialMaskType::ImageType;

  const auto computeMaximum = [](const MaskChange maskChange) {
    const auto imageSize = itk::Size<Dimension>::Filled(4);

    const auto image = CreateImage<PixelType>(imageSize);
    const auto maskImage = CreateImage<MaskPixelType>(imageSize);

    // Set specific non-zero values at the first and the last pixel. All other pixels have value zero.
    *(itk::ImageBufferRange{ *image }.begin()) = PixelType{ 1 };
    *(itk::ImageBufferRange{ *image }.rbegin()) = PixelType{ 2 };

    // Include only the first and the last pixel under the mask. All other mask entries are zero (false).
    *(itk::ImageBufferRange{ *maskImage }.begin()) = MaskPixelType{ 1 };
    *(itk::ImageBufferRange{ *maskImage }.rbegin()) = MaskPixelType{ 1 };

    if (maskChange == MaskChange::Spacing)
    {
      // Change the spacing of the mask, so that the last pixel is no longer "under the mask".
      maskImage->SetSpacing(itk::MakeFilled<MaskImageType::SpacingType>(0.5));
    }
    else
    {
      if (maskChange == MaskChange::Direction)
      {
        // Reverse the direction of the mask, so that the last pixel is no longer "under the mask".
        maskImage->SetDirection(image->GetDirection() * -1.0);
      }
    }

    const auto maskSpatialObject = ImageSpatialMaskType::New();
    maskSpatialObject->SetImage(maskImage);
    maskSpatialObject->Update();

    elx::DefaultConstruct<FilterType> filter{};
    filter.SetInput(image);
    filter.SetImageSpatialMask(maskSpatialObject);
    filter.Update();
    return filter.GetMaximum();
  };

  const auto maximumWithoutMaskChanges = computeMaximum(MaskChange::None);

  EXPECT_NE(computeMaximum(MaskChange::Spacing), maximumWithoutMaskChanges);
  EXPECT_NE(computeMaximum(MaskChange::Direction), maximumWithoutMaskChanges);
}
