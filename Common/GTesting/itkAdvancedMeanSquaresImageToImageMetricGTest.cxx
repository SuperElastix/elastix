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
#include "AdvancedMeanSquares/itkAdvancedMeanSquaresImageToImageMetric.h"
#include <itkNearestNeighborInterpolateImageFunction.h>
#include "itkAdvancedTranslationTransform.h"
#include "itkImageFullSampler.h"
#include "GTesting/elxCoreMainGTestUtilities.h"
#include "elxGTestUtilities.h"
#include "elxDefaultConstruct.h"
#include <itkImage.h>
#include <gtest/gtest.h>

// The template to be tested.
using itk::AdvancedMeanSquaresImageToImageMetric;

using elx::CoreMainGTestUtilities::CreateImage;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::DerefSmartPointer;
using elx::CoreMainGTestUtilities::minimumImageSizeValue;
using elx::GTestUtilities::InitializeMetric;
using elx::GTestUtilities::ValueAndDerivative;

namespace
{
template <typename TImage, typename TRandomNumberEngine>
void
RandomizePixelValues(TImage & image, TRandomNumberEngine && randomNumberEngine)
{
  using PixelType = typename TImage::PixelType;
  const itk::ImageBufferRange imageBufferRange(image);
  const auto                  maxValue = imageBufferRange.size();
  std::generate(imageBufferRange.begin(), imageBufferRange.end(), [&randomNumberEngine, maxValue] {
    return static_cast<PixelType>(std::uniform_int_distribution<std::size_t>{ 0, maxValue }(randomNumberEngine));
  });
};
} // namespace

// Checks if a default-constructed AdvancedMeanSquaresImageToImageMetric has the expected properties.
GTEST_TEST(AdvancedMeanSquaresImageToImageMetric, DefaultConstruct)
{
  static constexpr itk::SizeValueType defaultNumberOfFixedImageSamples{ 50000 };
  static constexpr double             defaultLimitRangeRatio{ 0.01 };

  static constexpr auto imageDimension = 3U;
  using ImageType = itk::Image<int, imageDimension>;
  using MetricType = AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType>;
  const elx::DefaultConstruct<MetricType> defaultConstructedMetric{};

  {
    const AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType> & advancedMeanSquaresImageToImageMetric =
      defaultConstructedMetric;

    // Note: m_NormalizationFactor cannot be tested this way, because there is no
    // `AdvancedMeanSquaresImageToImageMetric::GetNormalizationFactor()`.
    EXPECT_EQ(advancedMeanSquaresImageToImageMetric.GetUseNormalization(), false);
  }
  {
    const itk::AdvancedImageToImageMetric<ImageType, ImageType> & advancedImageToImageMetric = defaultConstructedMetric;

    EXPECT_EQ(advancedImageToImageMetric.GetImageSampler(), nullptr);

    // Note: The default-constructor of AdvancedMeanSquaresImageToImageMetric modifies UseImageSampler!
    EXPECT_EQ(advancedImageToImageMetric.GetUseImageSampler(), true);

    EXPECT_EQ(advancedImageToImageMetric.GetUseFixedImageLimiter(), false);
    EXPECT_EQ(advancedImageToImageMetric.GetUseMovingImageLimiter(), false);
    EXPECT_EQ(advancedImageToImageMetric.GetRequiredRatioOfValidSamples(), 0.25);
    EXPECT_EQ(advancedImageToImageMetric.GetUseMovingImageDerivativeScales(), false);
    EXPECT_EQ(advancedImageToImageMetric.GetScaleGradientWithRespectToMovingImageOrientation(), false);
    EXPECT_EQ(advancedImageToImageMetric.GetMovingImageDerivativeScales(),
              MetricType::MovingImageDerivativeScalesType::Filled(1.0));
    EXPECT_EQ(advancedImageToImageMetric.GetMovingImageLimiter(), nullptr);
    EXPECT_EQ(advancedImageToImageMetric.GetFixedImageLimiter(), nullptr);
    EXPECT_EQ(advancedImageToImageMetric.GetMovingLimitRangeRatio(), defaultLimitRangeRatio);
    EXPECT_EQ(advancedImageToImageMetric.GetFixedLimitRangeRatio(), defaultLimitRangeRatio);
    EXPECT_EQ(advancedImageToImageMetric.GetUseMetricSingleThreaded(), true);
    EXPECT_EQ(advancedImageToImageMetric.GetUseMultiThread(), false);
  }
  {
    const itk::ImageToImageMetric<ImageType, ImageType> & imageToImageMetric = defaultConstructedMetric;

    EXPECT_EQ(imageToImageMetric.GetComputeGradient(), false);
    EXPECT_EQ(imageToImageMetric.GetFixedImage(), nullptr);
    EXPECT_EQ(imageToImageMetric.GetMovingImage(), nullptr);
    EXPECT_EQ(imageToImageMetric.GetFixedImageMask(), nullptr);
    EXPECT_EQ(imageToImageMetric.GetMovingImageMask(), nullptr);
    EXPECT_EQ(imageToImageMetric.GetTransform(), nullptr);
    EXPECT_EQ(imageToImageMetric.GetInterpolator(), nullptr);
    EXPECT_EQ(imageToImageMetric.GetNumberOfPixelsCounted(), 0);
    EXPECT_EQ(imageToImageMetric.GetNumberOfFixedImageSamples(), defaultNumberOfFixedImageSamples);
    EXPECT_EQ(imageToImageMetric.GetFixedImageRegion(), itk::ImageRegion<imageDimension>{});
    EXPECT_EQ(imageToImageMetric.GetUseFixedImageIndexes(), false);
    EXPECT_GT(imageToImageMetric.GetNumberOfWorkUnits(), 0);
    EXPECT_EQ(imageToImageMetric.GetGradientImage(), nullptr);
    EXPECT_EQ(imageToImageMetric.GetFixedImageSamplesIntensityThreshold(), 0);
    EXPECT_EQ(imageToImageMetric.GetUseFixedImageSamplesIntensityThreshold(), false);
    EXPECT_EQ(imageToImageMetric.GetUseAllPixels(), false);
    EXPECT_EQ(imageToImageMetric.GetUseSequentialSampling(), false);
    EXPECT_EQ(imageToImageMetric.GetUseCachingOfBSplineWeights(), true);
    EXPECT_NE(imageToImageMetric.GetThreader(), nullptr);

    // Skipped:
    // - GetNumberOfMovingImageSamples(), as it is non-const, and it just returns GetNumberOfPixelsCounted().
    // - GetNumberOfSpatialSamples(), as it is non-const, and it just returns GetNumberOfFixedImageSamples().
    // - GetNumberOfParameters(), as it crashes while m_Transform is null.
    // - GetThreaderTransform(), as it is non-const.
  }
}


// Tests that the metric yields a zero-filled result (value and derivative) when fixed and moving image are equal, and
// an identity transform is used.
GTEST_TEST(AdvancedMeanSquaresImageToImageMetric, YieldsZeroWhenFixedAndMovingImageAreEqual)
{
  static constexpr auto imageDimension = 3U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, imageDimension>;

  const auto imageSize = itk::Size<imageDimension>::Filled(minimumImageSizeValue);
  const auto fixedImage = CreateImage<PixelType>(imageSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);

  RandomizePixelValues(*fixedImage, std::mt19937{});
  RandomizePixelValues(*movingImage, std::mt19937{});

  // Sanity check: after randomizing, the two images are still equal.
  EXPECT_EQ(*fixedImage, *movingImage);

  using MetricType = AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType>;

  elx::DefaultConstruct<itk::AdvancedTranslationTransform<double, imageDimension>> transform{};
  elx::DefaultConstruct<itk::NearestNeighborInterpolateImageFunction<ImageType>>   interpolator{};
  elx::DefaultConstruct<itk::ImageFullSampler<ImageType>>                          imageSampler{};

  elx::DefaultConstruct<MetricType> metric{};

  InitializeMetric(
    metric, *fixedImage, *movingImage, imageSampler, transform, interpolator, fixedImage->GetBufferedRegion());

  const auto valueAndDerivative = ValueAndDerivative::FromCostFunction(metric, transform.GetParameters());

  // Both value and derivative are zero-filled by GetValueAndDerivative, even when they are initialized by 1.
  EXPECT_EQ(valueAndDerivative.value, 0.0);
  EXPECT_EQ(valueAndDerivative.derivative, itk::Array<double>(itk::SizeValueType{ imageDimension }, 0.0));
}


// Tests that metric.SetUseMultiThread(false) and metric.SetUseMultiThread(true) both yield the same result (value and
// derivative).
GTEST_TEST(AdvancedMeanSquaresImageToImageMetric, MultiThreadResultEqualsSingleThreadResult)
{
  std::mt19937 randomNumberEngine{};

  static constexpr auto imageDimension = 3U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, imageDimension>;

  const auto imageSize = itk::Size<imageDimension>::Filled(minimumImageSizeValue);
  const auto fixedImage = CreateImage<PixelType>(imageSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);

  // Sanity check 1: before randomizing, the two images are equal.
  EXPECT_EQ(*fixedImage, *movingImage);

  RandomizePixelValues(*fixedImage, randomNumberEngine);
  RandomizePixelValues(*movingImage, randomNumberEngine);

  // Sanity check 2: after randomizing, the two images are no longer equal (hopefully).
  EXPECT_NE(*fixedImage, *movingImage);

  elx::DefaultConstruct<itk::AdvancedTranslationTransform<double, imageDimension>> transform{};
  elx::DefaultConstruct<itk::NearestNeighborInterpolateImageFunction<ImageType>>   interpolator{};
  elx::DefaultConstruct<itk::ImageFullSampler<ImageType>>                          imageSampler{};

  const auto getValueAndDerivative =
    [&fixedImage, &movingImage, &transform, &interpolator, &imageSampler](const bool useMultiThread) {
      using MetricType = AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType>;

      elx::DefaultConstruct<MetricType> metric{};

      metric.SetUseMultiThread(useMultiThread);
      InitializeMetric(
        metric, *fixedImage, *movingImage, imageSampler, transform, interpolator, fixedImage->GetBufferedRegion());

      return ValueAndDerivative::FromCostFunction(metric, transform.GetParameters());
    };

  const auto singleThreadResult = getValueAndDerivative(false);
  const auto multiThreadResult = getValueAndDerivative(true);

  EXPECT_EQ(multiThreadResult.value, singleThreadResult.value);
  EXPECT_EQ(multiThreadResult.derivative, singleThreadResult.derivative);
}
