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
#include "itkImageFullSampler.h"
#include "elxDefaultConstruct.h"
#include <itkImage.h>

#include <gtest/gtest.h>

#include "GTesting/elxCoreMainGTestUtilities.h"


// The class to be tested.
using itk::ImageGridSampler;

using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::CreateRandomImageDomain;
using elx::CoreMainGTestUtilities::DerefRawPointer;


GTEST_TEST(ImageGridSampler, DefaultConstructFillsSampleGridSpacingWithOne)
{
  using ImageType = itk::Image<int>;
  using SamplerType = ImageGridSampler<ImageType>;
  const elx::DefaultConstruct<SamplerType> sampler{};

  EXPECT_EQ(sampler.GetSampleGridSpacing(), itk::MakeFilled<SamplerType::SampleGridSpacingType>(1));
}


// Tests that ImageGridSampler has the same output as ImageFullSampler, by default (when having the default
// SampleGridSpacing).
GTEST_TEST(ImageGridSampler, HasSameOutputAsFullSamplerByDefault)
{
  using PixelType = int;
  constexpr auto Dimension = 2U;
  using ImageType = itk::Image<PixelType, Dimension>;

  std::mt19937 randomNumberEngine{};
  const auto   imageDomain = CreateRandomImageDomain<Dimension>(randomNumberEngine);
  const auto   image = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);

  const auto generateSamples = [image](auto && sampler) {
    sampler.SetInput(image);
    sampler.Update();
    return std::move(DerefRawPointer(sampler.GetOutput()).CastToSTLContainer());
  };

  const auto samples = generateSamples(elx::DefaultConstruct<ImageGridSampler<ImageType>>{});

  EXPECT_EQ(samples.size(), itk::ImageBufferRange{ *image }.size());
  EXPECT_EQ(samples, generateSamples(elx::DefaultConstruct<itk::ImageFullSampler<ImageType>>{}));
}
