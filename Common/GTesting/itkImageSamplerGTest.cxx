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
#include "itkImageSample.h"
#include "GTesting/elxCoreMainGTestUtilities.h"
#include <itkImage.h>
#include <gtest/gtest.h>


GTEST_TEST(ImageSample, IsEqualityComparable)
{
  using ImageSampleType = itk::ImageSample<itk::Image<int>>;
  using PointType = ImageSampleType::PointType;
  using PixelType = ImageSampleType::PixelType;

  EXPECT_TRUE(ImageSampleType{} == ImageSampleType{});
  EXPECT_FALSE(ImageSampleType{} != ImageSampleType{});

  const auto checkComparisonsToNonDefaultImageSample = [](const ImageSampleType nonDefaultImageSample) {
    EXPECT_TRUE(nonDefaultImageSample == nonDefaultImageSample);
    EXPECT_FALSE(nonDefaultImageSample != nonDefaultImageSample);

    EXPECT_TRUE(ImageSampleType{} != nonDefaultImageSample);
    EXPECT_TRUE(nonDefaultImageSample != ImageSampleType{});
    EXPECT_FALSE(ImageSampleType{} == nonDefaultImageSample);
    EXPECT_FALSE(nonDefaultImageSample == ImageSampleType{});
  };

  checkComparisonsToNonDefaultImageSample(ImageSampleType{ itk::MakeFilled<PointType>(1.0), PixelType{} });
  checkComparisonsToNonDefaultImageSample(ImageSampleType{ PointType{}, PixelType{ 1 } });
  checkComparisonsToNonDefaultImageSample(ImageSampleType{ itk::MakeFilled<PointType>(1.0), PixelType{ 1 } });
}
