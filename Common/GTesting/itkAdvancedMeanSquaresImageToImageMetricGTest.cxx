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
#include "GTesting/elxCoreMainGTestUtilities.h"
#include "elxDefaultConstruct.h"
#include <itkImage.h>
#include <gtest/gtest.h>

// The template to be tested.
using itk::AdvancedMeanSquaresImageToImageMetric;


// Checks if a default-constructed AdvancedMeanSquaresImageToImageMetric has the expected properties.
GTEST_TEST(AdvancedMeanSquaresImageToImageMetric, DefaultConstruct)
{
  constexpr itk::SizeValueType defaultNumberOfFixedImageSamples{ 50000 };
  constexpr double             defaultLimitRangeRatio{ 0.01 };

  constexpr auto imageDimension = 3U;
  using ImageType = itk::Image<int, imageDimension>;
  using MetricType = AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType>;
  const elx::DefaultConstruct<MetricType> defaultConstructedMetric{};

  {
    const AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType> & advancedMeanSquaresImageToImageMetric =
      defaultConstructedMetric;

    // Note: m_NormalizationFactor cannot be tested this way, because there is no
    // `AdvancedMeanSquaresImageToImageMetric::GetNormalizationFactor()`.
    EXPECT_EQ(advancedMeanSquaresImageToImageMetric.GetUseNormalization(), false);
    EXPECT_EQ(advancedMeanSquaresImageToImageMetric.GetSelfHessianSmoothingSigma(), 1.0);
    EXPECT_EQ(advancedMeanSquaresImageToImageMetric.GetSelfHessianNoiseRange(), 1.0);
    EXPECT_EQ(advancedMeanSquaresImageToImageMetric.GetNumberOfSamplesForSelfHessian(), 100000);
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
