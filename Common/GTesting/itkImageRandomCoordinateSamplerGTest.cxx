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
#include "itkImageRandomCoordinateSampler.h"
#include "elxDefaultConstruct.h"
#include "GTesting/elxCoreMainGTestUtilities.h"

// ITK header files:
#include <itkDeref.h>
#include <itkImage.h>
#include <itkMersenneTwisterRandomVariateGenerator.h>

#include <gtest/gtest.h>
#include <array>

// Using-declarations:
using elx::CoreMainGTestUtilities::DerefSmartPointer;
using elx::CoreMainGTestUtilities::minimumImageSizeValue;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using itk::Deref;
using itk::Statistics::MersenneTwisterRandomVariateGenerator;


GTEST_TEST(ImageRandomCoordinateSampler, CheckImageValuesOfSamples)
{
  using PixelType = int;
  using ImageType = itk::Image<PixelType>;
  using SamplerType = itk::ImageRandomCoordinateSampler<ImageType>;

  // Use a fixed seed, in order to have a reproducible sampler output.
  DerefSmartPointer(MersenneTwisterRandomVariateGenerator::GetInstance()).SetSeed(1);

  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(ImageType::SizeType::Filled(minimumImageSizeValue));

  elx::DefaultConstruct<SamplerType> sampler{};

  const size_t numberOfSamples{ 3 };
  sampler.SetNumberOfSamples(numberOfSamples);
  sampler.SetInput(image);
  sampler.Update();

  const auto & samples = Deref(sampler.GetOutput()).CastToSTLConstContainer();

  ASSERT_EQ(samples.size(), numberOfSamples);

  // The image values that appeared during the development of the test.
  const std::array<SamplerType::ImageSampleType::RealType, numberOfSamples> expectedImageValues = { 14.269278,
                                                                                                    14.93714,
                                                                                                    1.882026 };

  for (size_t i{}; i < numberOfSamples; ++i)
  {
    EXPECT_FLOAT_EQ(samples[i].m_ImageValue, expectedImageValues[i]);
  }
}


GTEST_TEST(ImageRandomCoordinateSampler, SetSeedMakesRandomizationDeterministic)
{
  using PixelType = int;
  using ImageType = itk::Image<PixelType>;
  using SamplerType = itk::ImageRandomCoordinateSampler<ImageType>;

  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(ImageType::SizeType::Filled(minimumImageSizeValue));

  for (const SamplerType::SeedIntegerType seed : { 0, 1 })
  {
    const auto generateSamples = [seed, image] {
      elx::DefaultConstruct<SamplerType> sampler{};

      DerefSmartPointer(MersenneTwisterRandomVariateGenerator::GetInstance()).SetSeed(seed);
      sampler.SetInput(image);
      sampler.Update();
      return std::move(Deref(sampler.GetOutput()).CastToSTLContainer());
    };

    const auto samples = generateSamples();

    // The test would be trivial (uninteresting) if there were no samples. Note that itk::ImageSamplerBase does
    // zero-initialize m_NumberOfSamples, but itk::ImageRandomSamplerBase does m_NumberOfSamples = 1000 afterwards.
    EXPECT_FALSE(samples.empty());

    // Do the same test another time, to check that the result remains the same.
    EXPECT_EQ(generateSamples(), samples);
  }
}


GTEST_TEST(ImageRandomCoordinateSampler, HasSameOutputWhenUsingMultiThread)
{
  using PixelType = int;
  using ImageType = itk::Image<PixelType>;
  using SamplerType = itk::ImageRandomCoordinateSampler<ImageType>;

  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(ImageType::SizeType::Filled(minimumImageSizeValue));

  const auto generateSamples = [image](const bool useMultiThread) {
    DerefSmartPointer(MersenneTwisterRandomVariateGenerator::GetInstance()).SetSeed(1);

    elx::DefaultConstruct<SamplerType> sampler{};
    sampler.SetUseMultiThread(useMultiThread);
    sampler.SetInput(image);
    sampler.Update();
    return std::move(Deref(sampler.GetOutput()).CastToSTLContainer());
  };

  EXPECT_EQ(generateSamples(true), generateSamples(false));
}
