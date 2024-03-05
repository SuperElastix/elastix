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
#include "itkImageRandomSamplerSparseMask.h"
#include "elxDefaultConstruct.h"
#include "GTesting/elxCoreMainGTestUtilities.h"

// ITK header files:
#include <itkImage.h>
#include <itkMersenneTwisterRandomVariateGenerator.h>

#include <gtest/gtest.h>
#include <array>
#include <itkImageMaskSpatialObject.h>

// Using-declarations:
using elx::CoreMainGTestUtilities::DerefRawPointer;
using elx::CoreMainGTestUtilities::DerefSmartPointer;
using elx::CoreMainGTestUtilities::minimumImageSizeValue;
using elx::CoreMainGTestUtilities::CreateImage;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::FillImageRegion;
using elx::CoreMainGTestUtilities::ImageDomain;
using itk::Statistics::MersenneTwisterRandomVariateGenerator;


GTEST_TEST(ImageRandomSamplerSparseMask, CheckImageValuesOfSamples)
{
  using PixelType = int;
  static constexpr auto Dimension = 2;
  using ImageType = itk::Image<PixelType, Dimension>;
  using MaskSpatialObjectType = itk::ImageMaskSpatialObject<Dimension>;

  // Use a fixed seed, in order to have a reproducible sampler output.
  DerefSmartPointer(MersenneTwisterRandomVariateGenerator::GetInstance()).SetSeed(1);

  const auto imageSize = ImageType::SizeType::Filled(minimumImageSizeValue);
  const auto image = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageSize);

  const auto maskImage = CreateImage<MaskSpatialObjectType::PixelType>(imageSize);
  FillImageRegion(*maskImage, itk::Index<Dimension>::Filled(1), ImageType::SizeType::Filled(minimumImageSizeValue - 1));

  const auto maskSpatialObject = MaskSpatialObjectType::New();
  maskSpatialObject->SetImage(maskImage);
  maskSpatialObject->Update();

  elx::DefaultConstruct<itk::ImageRandomSamplerSparseMask<ImageType>> sampler{};

  const size_t numberOfSamples{ 3 };
  sampler.SetInput(image);
  sampler.SetMask(maskSpatialObject);
  sampler.SetNumberOfSamples(numberOfSamples);
  sampler.Update();

  const auto & samples = DerefRawPointer(sampler.GetOutput()).CastToSTLConstContainer();

  ASSERT_EQ(samples.size(), numberOfSamples);

  // The image values that appeared during the development of the test.
  const std::array<PixelType, numberOfSamples> expectedImageValues = { 12, 16, 12 };

  for (size_t i{}; i < numberOfSamples; ++i)
  {
    EXPECT_EQ(samples[i].m_ImageValue, expectedImageValues[i]);
  }
}


GTEST_TEST(ImageRandomSamplerSparseMask, SetSeedMakesRandomizationDeterministic)
{
  using PixelType = int;
  static constexpr auto Dimension = 2;
  using ImageType = itk::Image<PixelType, Dimension>;
  using SamplerType = itk::ImageRandomSamplerSparseMask<ImageType>;
  using MaskSpatialObjectType = itk::ImageMaskSpatialObject<Dimension>;

  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(ImageType::SizeType::Filled(minimumImageSizeValue));

  const auto maskImage = CreateImage<MaskSpatialObjectType::PixelType>(ImageDomain(*image));
  FillImageRegion(*maskImage, itk::Index<Dimension>::Filled(1), ImageType::SizeType::Filled(minimumImageSizeValue - 1));

  const auto maskSpatialObject = MaskSpatialObjectType::New();
  maskSpatialObject->SetImage(maskImage);
  maskSpatialObject->Update();

  for (const SamplerType::SeedIntegerType seed : { 0, 1 })
  {
    const auto generateSamples = [seed, image, maskSpatialObject] {
      elx::DefaultConstruct<SamplerType> sampler{};

      DerefSmartPointer(MersenneTwisterRandomVariateGenerator::GetInstance()).SetSeed(seed);
      sampler.SetInput(image);
      sampler.SetMask(maskSpatialObject);
      sampler.Update();
      return std::move(DerefRawPointer(sampler.GetOutput()).CastToSTLContainer());
    };

    const auto samples = generateSamples();

    // The test would be trivial (uninteresting) if there were no samples. Note that itk::ImageSamplerBase does
    // zero-initialize m_NumberOfSamples, but itk::ImageRandomSamplerBase does m_NumberOfSamples = 1000 afterwards.
    EXPECT_FALSE(samples.empty());

    // Do the same test another time, to check that the result remains the same.
    EXPECT_EQ(generateSamples(), samples);
  }
}


GTEST_TEST(ImageRandomSamplerSparseMask, HasSameOutputWhenUsingMultiThread)
{
  using PixelType = int;
  static constexpr auto Dimension = 2;
  using ImageType = itk::Image<PixelType, Dimension>;
  using SamplerType = itk::ImageRandomSamplerSparseMask<ImageType>;
  using MaskSpatialObjectType = itk::ImageMaskSpatialObject<Dimension>;

  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(ImageType::SizeType::Filled(minimumImageSizeValue));

  const auto maskImage = CreateImage<MaskSpatialObjectType::PixelType>(ImageDomain(*image));
  FillImageRegion(*maskImage, itk::Index<Dimension>::Filled(1), ImageType::SizeType::Filled(minimumImageSizeValue - 1));

  const auto maskSpatialObject = MaskSpatialObjectType::New();
  maskSpatialObject->SetImage(maskImage);
  maskSpatialObject->Update();

  const auto generateSamples = [image, maskSpatialObject](const bool useMultiThread) {
    DerefSmartPointer(MersenneTwisterRandomVariateGenerator::GetInstance()).SetSeed(1);
    elx::DefaultConstruct<SamplerType> sampler{};
    sampler.SetUseMultiThread(useMultiThread);
    sampler.SetInput(image);
    sampler.SetMask(maskSpatialObject);
    sampler.Update();
    return std::move(DerefRawPointer(sampler.GetOutput()).CastToSTLContainer());
  };

  EXPECT_EQ(generateSamples(true), generateSamples(false));
}
