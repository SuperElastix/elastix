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
#include <elxElastixFilter.h>

#include "elxCoreMainGTestUtilities.h"

// ITK header file:
#include <itkImage.h>
#include <itkIndexRange.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For transform
#include <map>
#include <string>
#include <utility> // For pair


// Using-declarations:
using elx::CoreMainGTestUtilities::CheckNew;
using elx::CoreMainGTestUtilities::ConvertArrayOfDoubleToOffset;
using elx::CoreMainGTestUtilities::ConvertStringsToArrayOfDouble;
using elx::CoreMainGTestUtilities::CreateParameterObject;
using elx::CoreMainGTestUtilities::FillImageRegion;


// Tests registering two small (5x6) binary images, which are translated with respect to each other.
GTEST_TEST(ElastixFilter, Translation)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->Allocate(true);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->Allocate(true);
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const auto filter = CheckNew<elx::ElastixFilter<ImageType, ImageType>>();

  filter->SetFixedImage(fixedImage);
  filter->SetMovingImage(movingImage);
  filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                     { "ImageSampler", "Full" },
                                                     { "MaximumNumberOfIterations", "2" },
                                                     { "Metric", "AdvancedNormalizedCorrelation" },
                                                     { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                     { "Transform", "TranslationTransform" } }));
  filter->Update();

  const auto   transformParameterObject = filter->GetTransformParameterObject();
  const auto & transformParameterMaps = transformParameterObject->GetParameterMap();

  ASSERT_TRUE(!transformParameterMaps.empty());
  EXPECT_EQ(transformParameterMaps.size(), 1);

  const auto & transformParameterMap = transformParameterMaps.front();
  const auto   found = transformParameterMap.find("TransformParameters");
  ASSERT_NE(found, transformParameterMap.cend());

  const auto transformParameters = ConvertStringsToArrayOfDouble<ImageDimension>(found->second);
  EXPECT_EQ(ConvertArrayOfDoubleToOffset(transformParameters), translationOffset);
}


// Tests registering two images, having "WriteResultImage" set.
GTEST_TEST(ElastixFilter, WriteResultImage)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->Allocate(true);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->Allocate(true);
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  for (const bool writeResultImage : { true, false })
  {
    const auto filter = CheckNew<elx::ElastixFilter<ImageType, ImageType>>();

    filter->SetFixedImage(fixedImage);
    filter->SetMovingImage(movingImage);
    filter->SetParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "ImageSampler", "Full" },
                              { "MaximumNumberOfIterations", "2" },
                              { "Metric", "AdvancedNormalizedCorrelation" },
                              { "Optimizer", "AdaptiveStochasticGradientDescent" },
                              { "Transform", "TranslationTransform" },
                              { "WriteResultImage", (writeResultImage ? "true" : "false") } }));
    filter->Update();

    const auto * const output = filter->GetOutput();
    ASSERT_NE(output, nullptr);

    const auto &       outputImageSize = output->GetBufferedRegion().GetSize();
    const auto * const outputBufferPointer = output->GetBufferPointer();

    if (writeResultImage)
    {
      EXPECT_EQ(outputImageSize, imageSize);
      ASSERT_NE(outputBufferPointer, nullptr);

      // When "WriteResultImage" is true, expect an output image that is very much like the fixed image.
      for (const auto index : itk::ZeroBasedIndexRange<ImageDimension>(imageSize))
      {
        EXPECT_EQ(std::round(output->GetPixel(index)), std::round(fixedImage->GetPixel(index)));
      }
    }
    else
    {
      // When "WriteResultImage" is false, expect an empty output image.
      EXPECT_EQ(outputImageSize, ImageType::SizeType());
      EXPECT_EQ(outputBufferPointer, nullptr);
    }

    const auto   transformParameterObject = filter->GetTransformParameterObject();
    const auto & transformParameterMaps = transformParameterObject->GetParameterMap();

    ASSERT_TRUE(!transformParameterMaps.empty());
    EXPECT_EQ(transformParameterMaps.size(), 1);

    const auto & transformParameterMap = transformParameterMaps.front();
    const auto   found = transformParameterMap.find("TransformParameters");
    ASSERT_NE(found, transformParameterMap.cend());

    const auto transformParameters = ConvertStringsToArrayOfDouble<ImageDimension>(found->second);
    EXPECT_EQ(ConvertArrayOfDoubleToOffset(transformParameters), translationOffset);
  }
}
