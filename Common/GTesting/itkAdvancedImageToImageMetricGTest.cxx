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
#include "itkAdvancedImageToImageMetric.h"
#include "GTesting/elxCoreMainGTestUtilities.h"
#include "elxDefaultConstruct.h"
#include <itkImage.h>
#include <gtest/gtest.h>

// The template to be tested.
using itk::AdvancedImageToImageMetric;

namespace
{
constexpr itk::SizeValueType defaultNumberOfFixedImageSamples{ 50000 };
constexpr double             defaultLimitRangeRatio{ 0.01 };


template <typename TImage>
class TestMetric : public AdvancedImageToImageMetric<TImage, TImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TestMetric);

  using Self = TestMetric;
  using Superclass = AdvancedImageToImageMetric<TImage, TImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(TestMetric);

  void
  ExpectProtectedData() const
  {
    {
      using AdvancedImageToImageMetricType = itk::AdvancedImageToImageMetric<TImage, TImage>;

      // Superclass data members:
      EXPECT_EQ(AdvancedImageToImageMetricType::m_ImageSampler, nullptr);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_AdvancedTransform, nullptr);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_TransformIsBSpline, false);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_FixedImageTrueMin, 0);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_FixedImageTrueMax, 1);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_MovingImageTrueMin, 0);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_MovingImageTrueMax, 1);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_FixedImageMinLimit, 0);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_FixedImageMaxLimit, 1);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_MovingImageMinLimit, 0);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_MovingImageMaxLimit, 1);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_UseMetricSingleThreaded, true);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_UseMultiThread, false);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_ThreaderMetricParameters.st_Metric, this);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_ThreaderMetricParameters.st_DerivativePointer, nullptr);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_ThreaderMetricParameters.st_NormalizationFactor, 0.0);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_GetValueAndDerivativePerThreadVariables, nullptr);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_GetValueAndDerivativePerThreadVariablesSize, 0U);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_FixedLimitRangeRatio, defaultLimitRangeRatio);
      EXPECT_EQ(AdvancedImageToImageMetricType::m_MovingLimitRangeRatio, defaultLimitRangeRatio);
    }

    // Superclass::Superclass data members:
    using ImageToImageMetricType = itk::ImageToImageMetric<TImage, TImage>;

    EXPECT_EQ(ImageToImageMetricType::m_UseFixedImageIndexes, false);
    EXPECT_TRUE(ImageToImageMetricType::m_FixedImageIndexes.empty());
    EXPECT_EQ(ImageToImageMetricType::m_UseFixedImageSamplesIntensityThreshold, false);
    EXPECT_EQ(ImageToImageMetricType::m_FixedImageSamplesIntensityThreshold, 0);
    EXPECT_TRUE(ImageToImageMetricType::m_FixedImageSamples.empty());
    EXPECT_EQ(ImageToImageMetricType::m_NumberOfParameters, 0);
    EXPECT_EQ(ImageToImageMetricType::m_NumberOfFixedImageSamples, defaultNumberOfFixedImageSamples);
    EXPECT_EQ(ImageToImageMetricType::m_NumberOfPixelsCounted, 0);
    EXPECT_EQ(ImageToImageMetricType::m_FixedImage, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_MovingImage, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_Transform, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_ThreaderTransform, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_Interpolator, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_ComputeGradient, false);
    EXPECT_EQ(ImageToImageMetricType::m_GradientImage, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_FixedImageMask, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_MovingImageMask, nullptr);
    EXPECT_GT(ImageToImageMetricType::m_NumberOfWorkUnits, 0);
    EXPECT_EQ(ImageToImageMetricType::m_UseAllPixels, false);
    EXPECT_EQ(ImageToImageMetricType::m_UseSequentialSampling, false);
    EXPECT_EQ(ImageToImageMetricType::m_ReseedIterator, false);
    EXPECT_EQ(ImageToImageMetricType::m_RandomSeed, int{ ImageToImageMetricType::m_RandomSeed });
    EXPECT_EQ(ImageToImageMetricType::m_TransformIsBSpline, false);
    EXPECT_EQ(ImageToImageMetricType::m_NumBSplineWeights, 0);
    EXPECT_EQ(ImageToImageMetricType::m_BSplineTransform, nullptr);
    EXPECT_TRUE(ImageToImageMetricType::m_BSplineTransformWeightsArray.empty());
    EXPECT_TRUE(ImageToImageMetricType::m_BSplineTransformIndicesArray.empty());
    EXPECT_TRUE(ImageToImageMetricType::m_BSplinePreTransformPointsArray.empty());
    EXPECT_TRUE(ImageToImageMetricType::m_WithinBSplineSupportRegionArray.empty());
    EXPECT_EQ(ImageToImageMetricType::m_BSplineParametersOffset,
              typename ImageToImageMetricType::BSplineParametersOffsetType());
    EXPECT_EQ(ImageToImageMetricType::m_UseCachingOfBSplineWeights, true);
    EXPECT_EQ(ImageToImageMetricType::m_BSplineTransformWeights,
              typename ImageToImageMetricType::BSplineTransformWeightsType());
    EXPECT_EQ(ImageToImageMetricType::m_BSplineTransformIndices,
              typename ImageToImageMetricType::BSplineTransformIndexArrayType());
    EXPECT_EQ(ImageToImageMetricType::m_ThreaderBSplineTransformWeights, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_ThreaderBSplineTransformIndices, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_InterpolatorIsBSpline, false);
    EXPECT_EQ(ImageToImageMetricType::m_BSplineInterpolator, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_DerivativeCalculator, nullptr);
    EXPECT_NE(ImageToImageMetricType::m_Threader, nullptr);
    ASSERT_NE(ImageToImageMetricType::m_ConstSelfWrapper, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_ConstSelfWrapper->GetConstMetricPointer(), this);
    EXPECT_EQ(ImageToImageMetricType::m_ThreaderNumberOfMovingImageSamples, nullptr);
    EXPECT_EQ(ImageToImageMetricType::m_WithinThreadPreProcess, false);
    EXPECT_EQ(ImageToImageMetricType::m_WithinThreadPostProcess, false);
  }

protected:
  TestMetric() = default;

private:
  itk::SingleValuedCostFunction::MeasureType
  GetValue(const itk::SingleValuedCostFunction::ParametersType &) const override
  {
    return 0;
  }

  void
  GetDerivative(const itk::SingleValuedCostFunction::ParametersType &,
                itk::SingleValuedCostFunction::DerivativeType &) const override
  {}
};

} // namespace


// Checks if a default-constructed AdvancedImageToImageMetric has the expected properties.
GTEST_TEST(AdvancedImageToImageMetric, DefaultConstruct)
{
  static constexpr auto imageDimension = 3U;
  using ImageType = itk::Image<int, imageDimension>;
  using MetricType = TestMetric<ImageType>;
  const elx::DefaultConstruct<MetricType> defaultConstructedMetric{};

  defaultConstructedMetric.ExpectProtectedData();
  {
    const AdvancedImageToImageMetric<ImageType, ImageType> & advancedImageToImageMetric = defaultConstructedMetric;

    EXPECT_EQ(advancedImageToImageMetric.GetImageSampler(), nullptr);
    EXPECT_EQ(advancedImageToImageMetric.GetUseImageSampler(), false);
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
