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
#include <itkImageMaskSpatialObject.h>

#include <gtest/gtest.h>

#include "GTesting/elxCoreMainGTestUtilities.h"


// The class to be tested.
using itk::ImageGridSampler;

using elx::CoreMainGTestUtilities::minimumImageSizeValue;
using elx::CoreMainGTestUtilities::CreateImage;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::CreateRandomImageDomain;
using elx::CoreMainGTestUtilities::DerefRawPointer;
using elx::CoreMainGTestUtilities::ImageDomain;

namespace
{
template <unsigned int VDimension>
auto
GetMiddleIndex(const ImageDomain<VDimension> & imageDomain)
{
  auto index = imageDomain.index;

  for (unsigned int i = 0; i < VDimension; ++i)
  {
    index[i] += static_cast<itk::IndexValueType>((imageDomain.size[i] - 1) / 2);
  }
  return index;
}

} // namespace


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


// Tests that ImageGridSampler when having the maximum SampleGridSpacing.
GTEST_TEST(ImageGridSampler, MaxSampleGridSpacing)
{
  using PixelType = int;
  constexpr auto Dimension = 2U;
  using ImageType = itk::Image<PixelType, Dimension>;
  using SamplerType = ImageGridSampler<ImageType>;

  std::mt19937 randomNumberEngine{};
  const auto   imageDomain = CreateRandomImageDomain<Dimension>(randomNumberEngine);
  const auto   image = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);

  elx::DefaultConstruct<ImageGridSampler<ImageType>> sampler{};

  sampler.SetSampleGridSpacing(itk::MakeFilled<SamplerType::SampleGridSpacingType>(
    std::numeric_limits<SamplerType::SampleGridSpacingValueType>::max()));
  sampler.SetInput(image);
  sampler.SetUseMultiThread(false);
  sampler.Update();
  const auto & samples = DerefRawPointer(sampler.GetOutput()).CastToSTLContainer();

  ASSERT_FALSE(samples.empty());
  EXPECT_EQ(samples.size(), 1);
  EXPECT_EQ(samples.front().m_ImageValue, image->GetPixel(GetMiddleIndex(imageDomain)));
}


// Tests that ImageGridSampler when having the SampleGridSpacing equal to Image Size.
GTEST_TEST(ImageGridSampler, SampleGridSpacingGreaterEqualToImageSize)
{
  using PixelType = int;
  constexpr auto Dimension = 2U;
  using ImageType = itk::Image<PixelType, Dimension>;
  using SamplerType = ImageGridSampler<ImageType>;

  std::mt19937 randomNumberEngine{};
  const auto   imageDomain = CreateRandomImageDomain<Dimension>(randomNumberEngine);
  const auto   image = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);

  elx::DefaultConstruct<ImageGridSampler<ImageType>> sampler{};

  SamplerType::SampleGridSpacingType sampleGridSpacing{};

  for (unsigned int i{}; i < Dimension; ++i)
  {
    sampleGridSpacing[i] = static_cast<SamplerType::SampleGridSpacingValueType>(imageDomain.size[i]);
  }

  sampler.SetSampleGridSpacing(sampleGridSpacing);
  sampler.SetInput(image);
  sampler.Update();
  const auto & samples = DerefRawPointer(sampler.GetOutput()).CastToSTLContainer();

  ASSERT_FALSE(samples.empty());
  EXPECT_EQ(samples.size(), 1);
  EXPECT_EQ(samples.front().m_ImageValue, image->GetPixel(GetMiddleIndex(imageDomain)));
}


// Tests that ImageGridSampler when having the SampleGridSpacing equal to Image Size - 1 in each direction.
GTEST_TEST(ImageGridSampler, SampleGridSpacingOneLessThanImageSize)
{
  using PixelType = int;
  constexpr auto Dimension = 2U;
  using ImageType = itk::Image<PixelType, Dimension>;
  using SamplerType = ImageGridSampler<ImageType>;

  std::mt19937 randomNumberEngine{};
  const auto   imageDomain = CreateRandomImageDomain<Dimension>(randomNumberEngine);
  const auto   image = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);
  const itk::ImageBufferRange<const ImageType> imageBufferRange(*image);

  elx::DefaultConstruct<ImageGridSampler<ImageType>> sampler{};

  SamplerType::SampleGridSpacingType sampleGridSpacing{};

  for (unsigned int i{}; i < Dimension; ++i)
  {
    sampleGridSpacing[i] = static_cast<SamplerType::SampleGridSpacingValueType>(imageDomain.size[i] - 1);
  }

  sampler.SetSampleGridSpacing(sampleGridSpacing);
  sampler.SetInput(image);
  sampler.Update();
  const auto & samples = DerefRawPointer(sampler.GetOutput()).CastToSTLContainer();

  ASSERT_FALSE(samples.empty());
  EXPECT_EQ(samples.size(), 4);
  EXPECT_EQ(samples.front().m_ImageValue, *(imageBufferRange.cbegin()));
  EXPECT_EQ(samples.back().m_ImageValue, *(imageBufferRange.crbegin()));
}


// Tests that ImageGridSampler when having a SampleGridSpacing of 2 in each direction.
GTEST_TEST(ImageGridSampler, SampleGridSpacingTwo)
{
  using PixelType = int;
  constexpr auto Dimension = 2U;
  using ImageType = itk::Image<PixelType, Dimension>;
  using SamplerType = ImageGridSampler<ImageType>;

  std::mt19937 randomNumberEngine{};
  const auto   imageDomain = CreateRandomImageDomain<Dimension>(randomNumberEngine);
  const auto   image = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);
  const itk::ImageBufferRange<const ImageType> imageBufferRange(*image);

  elx::DefaultConstruct<ImageGridSampler<ImageType>> sampler{};

  sampler.SetSampleGridSpacing(itk::MakeFilled<SamplerType::SampleGridSpacingType>(2));
  sampler.SetInput(image);
  sampler.Update();
  const auto & samples = DerefRawPointer(sampler.GetOutput()).CastToSTLContainer();

  ASSERT_FALSE(samples.empty());

  size_t expectedNumberOfSamples{ 1 };

  for (unsigned int i{}; i < Dimension; ++i)
  {
    expectedNumberOfSamples *= (imageDomain.size[i] + 1) / 2;
  }

  EXPECT_EQ(samples.size(), expectedNumberOfSamples);
  EXPECT_EQ(samples.front().m_ImageValue, *(imageBufferRange.cbegin()));
}


GTEST_TEST(ImageGridSampler, HasSameOutputWhenUsingMultiThread)
{
  using PixelType = int;
  constexpr auto Dimension = 2U;
  using ImageType = itk::Image<PixelType, Dimension>;
  using SamplerType = itk::ImageGridSampler<ImageType>;

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
  const auto samplesGeneratedUsingMultiThreading = generateSamples(true);

  // The test would be trivial (uninteresting) if there were no samples. Note that itk::ImageSamplerBase does
  // zero-initialize m_NumberOfSamples, but itk::ImageGridSampler does "full sampling" by default anyway.
  EXPECT_FALSE(samplesGeneratedUsingMultiThreading.empty());

  EXPECT_EQ(samplesGeneratedUsingMultiThreading, generateSamples(false));
}


// Tests that the sampler produces the same output when using a mask that is fully filled with ones as when using no
// mask at all.
GTEST_TEST(ImageGridSampler, HasSameOutputWhenUsingFullyFilledMask)
{
  using PixelType = int;
  enum
  {
    Dimension = 2U
  };
  using SamplerType = itk::ImageGridSampler<itk::Image<PixelType, Dimension>>;

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

  const auto samplesGeneratedUsingFullyFilledMask = generateSamples(true);
  const auto samplesGeneratedWithoutMask = generateSamples(false);

  // The test would be trivial (uninteresting) if there were no samples.
  EXPECT_FALSE(samplesGeneratedUsingFullyFilledMask.empty());

  EXPECT_EQ(samplesGeneratedUsingFullyFilledMask.size(), samplesGeneratedWithoutMask.size());
  EXPECT_EQ(samplesGeneratedUsingFullyFilledMask, samplesGeneratedWithoutMask);
}
