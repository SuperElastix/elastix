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
#include "itkImageFullSampler.h"
#include "GTesting/elxCoreMainGTestUtilities.h"
#include "elxDefaultConstruct.h"
#include <itkImage.h>
#include <itkImageMaskSpatialObject.h>
#include <gtest/gtest.h>


using elx::CoreMainGTestUtilities::CreateImage;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::CreateRandomImageDomain;
using elx::CoreMainGTestUtilities::DerefRawPointer;
using elx::CoreMainGTestUtilities::ImageDomain;


GTEST_TEST(ImageFullSampler, OutputHasSameSequenceOfPixelValuesAsInput)
{
  using PixelType = std::uint8_t;
  constexpr auto Dimension = 2U;
  using ImageType = itk::Image<PixelType, Dimension>;
  using ImageFullSamplerType = itk::ImageFullSampler<ImageType>;

  std::mt19937 randomNumberEngine{};
  const auto   imageDomain = CreateRandomImageDomain<Dimension>(randomNumberEngine);
  const auto   image = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);
  elx::DefaultConstruct<ImageFullSamplerType> sampler{};

  sampler.SetInput(image);
  sampler.Update();

  const auto &                output = DerefRawPointer(sampler.GetOutput());
  const itk::ImageBufferRange imageBufferRange(*image);
  const std::size_t           numberOfSamples{ output.size() };

  ASSERT_EQ(numberOfSamples, imageBufferRange.size());

  for (std::size_t i{}; i < numberOfSamples; ++i)
  {
    EXPECT_EQ(output[i].m_ImageValue, imageBufferRange[i]);
  }
}

GTEST_TEST(ImageFullSampler, HasSameOutputWhenUsingMultiThread)
{
  using PixelType = int;
  constexpr auto Dimension = 2U;
  using ImageType = itk::Image<PixelType, Dimension>;
  using SamplerType = itk::ImageFullSampler<ImageType>;

  std::mt19937 randomNumberEngine{};
  const auto   imageDomain = CreateRandomImageDomain<Dimension>(randomNumberEngine);
  const auto   image = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);

  const auto generateSamples = [image](const bool useMultiThread) {
    elx::DefaultConstruct<SamplerType> sampler{};
    sampler.SetUseMultiThread(useMultiThread);
    sampler.SetInput(image);
    sampler.Update();
    return std::move(DerefRawPointer(sampler.GetOutput()).CastToSTLContainer());
  };

  EXPECT_EQ(generateSamples(true), generateSamples(false));
}


// Tests that the sampler produces the same output when using a mask that is fully filled with ones as when using no
// mask at all.
GTEST_TEST(ImageFullSampler, HasSameOutputWhenUsingFullyFilledMask)
{
  using PixelType = int;
  enum
  {
    Dimension = 2U
  };
  using SamplerType = itk::ImageFullSampler<itk::Image<PixelType, Dimension>>;

  const ImageDomain<Dimension> imageDomain(itk::Size<Dimension>::Filled(4));
  const auto                   image = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);

  const auto generateSamples = [image](const bool useMask) {
    elx::DefaultConstruct<SamplerType> sampler{};

    sampler.SetInput(image);

    if (useMask)
    {
      using MaskSpatialObjectType = itk::ImageMaskSpatialObject<Dimension>;
      const auto maskImage = CreateImage<MaskSpatialObjectType::PixelType>(ImageDomain(*image));
      maskImage->FillBuffer(1);

      const auto maskSpatialObject = MaskSpatialObjectType::New();
      maskSpatialObject->SetImage(maskImage);
      maskSpatialObject->Update();

      sampler.SetMask(maskSpatialObject);
    }

    sampler.Update();
    return std::move(DerefRawPointer(sampler.GetOutput()).CastToSTLContainer());
  };

  EXPECT_EQ(generateSamples(true), generateSamples(false));
}