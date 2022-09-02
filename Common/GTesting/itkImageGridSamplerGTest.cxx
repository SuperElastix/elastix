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
#include "itkImageGridSampler.h"
#include "elxDefaultConstruct.h"
#include <itkImage.h>

#include <gtest/gtest.h>

// The class to be tested.
using itk::ImageGridSampler;


GTEST_TEST(ImageGridSampler, DefaultConstructFillsSampleGridSpacingWithOne)
{
  using ImageType = itk::Image<int>;
  using ImageGridSamplerType = itk::ImageGridSampler<ImageType>;
  const elastix::DefaultConstruct<ImageGridSamplerType> imageGridSampler{};

  EXPECT_EQ(imageGridSampler.GetSampleGridSpacing(), itk::MakeFilled<ImageGridSamplerType::SampleGridSpacingType>(1));
}
