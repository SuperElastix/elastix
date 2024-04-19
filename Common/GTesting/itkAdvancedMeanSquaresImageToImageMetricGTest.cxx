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
#include <itkBSplineInterpolateImageFunction.h>
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

  // Just use whole numbers, but ensure that each pixel may have a unique number, by maxValue = imageBufferRange.size().
  std::generate(
    imageBufferRange.begin(), imageBufferRange.end(), [&randomNumberEngine, maxValue = imageBufferRange.size()] {
      return static_cast<PixelType>(std::uniform_int_distribution<std::size_t>{ 0, maxValue }(randomNumberEngine));
    });
};


template <typename TInterpolator>
itk::SmartPointer<itk::InterpolateImageFunction<typename TInterpolator::InputImageType>>
CreateInterpolator()
{
  return TInterpolator::New().GetPointer();
}


template <typename TPixel, unsigned VImageDimension>
auto
CreateImageOfDistanceToPoint(const itk::Size<VImageDimension> &          imageSize,
                             const itk::Point<double, VImageDimension> & point)
{
  const auto                                        image = CreateImage<TPixel>(imageSize);
  const itk::ImageBufferRange                       imageBufferRange{ *image };
  const itk::ImageRegionIndexRange<VImageDimension> indexRange{ image->GetBufferedRegion() };

  auto indexIterator = indexRange.cbegin();

  for (TPixel & pixel : imageBufferRange)
  {
    pixel = point.EuclideanDistanceTo(image->template TransformIndexToPhysicalPoint<double>(*indexIterator));
    ++indexIterator;
  }

  return image;
}

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
  elx::DefaultConstruct<itk::ImageFullSampler<ImageType>>                          imageSampler{};

  for (const auto interpolator : { CreateInterpolator<itk::AdvancedLinearInterpolateImageFunction<ImageType>>(),
                                   CreateInterpolator<itk::BSplineInterpolateImageFunction<ImageType>>(),
                                   CreateInterpolator<itk::NearestNeighborInterpolateImageFunction<ImageType>>() })
  {
    const auto getValueAndDerivative =
      [&fixedImage, &movingImage, &transform, &interpolator, &imageSampler](const bool useMultiThread) {
        elx::DefaultConstruct<AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType>> metric{};
        metric.SetUseMultiThread(useMultiThread);
        // Test for one work unit, to avoid test failures caused by rounding errors
        metric.SetNumberOfWorkUnits(1);
        InitializeMetric(
          metric, *fixedImage, *movingImage, imageSampler, transform, *interpolator, fixedImage->GetBufferedRegion());
        return ValueAndDerivative::FromCostFunction(metric, transform.GetParameters());
      };

    const auto singleThreadResult = getValueAndDerivative(false);
    const auto multiThreadResult = getValueAndDerivative(true);
    EXPECT_EQ(multiThreadResult.value, singleThreadResult.value);
    EXPECT_EQ(multiThreadResult.derivative, singleThreadResult.derivative);
  }

  // Specifically test with NearestNeighbor interpolation, as it should not introduce rounding errors.
  elx::DefaultConstruct<itk::NearestNeighborInterpolateImageFunction<ImageType>> interpolator{};

  const auto getValueAndDerivative =
    [&fixedImage, &movingImage, &transform, &interpolator, &imageSampler](const bool useMultiThread) {
      elx::DefaultConstruct<AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType>> metric{};
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


// Tests that the MeanSquares value is as expected, for random images.
GTEST_TEST(AdvancedMeanSquaresImageToImageMetric, ValueIsAsExpected)
{
  std::mt19937 randomNumberEngine{};

  static constexpr auto imageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, imageDimension>;

  const auto imageSize = itk::Size<imageDimension>::Filled(minimumImageSizeValue);
  const auto fixedImage = CreateImage<PixelType>(imageSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);

  RandomizePixelValues(*fixedImage, randomNumberEngine);
  RandomizePixelValues(*movingImage, randomNumberEngine);

  const auto sumOfSquareDifferences = [fixedImage, movingImage] {
    const itk::ImageBufferRange fixedImageBufferRange(*fixedImage);
    const itk::ImageBufferRange movingImageBufferRange(*movingImage);

    const auto numberOfPixels = fixedImageBufferRange.size();
    EXPECT_EQ(numberOfPixels, movingImageBufferRange.size());

    double sum{};

    for (std::size_t i{}; i < numberOfPixels; ++i)
    {
      sum += vnl_math::sqr(movingImageBufferRange[i] - fixedImageBufferRange[i]);
    }
    return sum;
  }();

  // Sanity check: if the sum of squares is not greater than 0, the images are too similar for this test to be of
  // interest.
  EXPECT_GT(sumOfSquareDifferences, 0.0);

  const auto check = [fixedImage, movingImage, sumOfSquareDifferences](auto && transform) {
    for (const auto interpolator : { CreateInterpolator<itk::AdvancedLinearInterpolateImageFunction<ImageType>>(),
                                     CreateInterpolator<itk::BSplineInterpolateImageFunction<ImageType>>(),
                                     CreateInterpolator<itk::NearestNeighborInterpolateImageFunction<ImageType>>() })
    {
      elx::DefaultConstruct<itk::ImageFullSampler<ImageType>>                            imageSampler{};
      elx::DefaultConstruct<AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType>> metric{};

      InitializeMetric(
        metric, *fixedImage, *movingImage, imageSampler, transform, *interpolator, fixedImage->GetBufferedRegion());
      const auto value = ValueAndDerivative::FromCostFunction(metric, transform.GetParameters()).value;

      // Expect numberOfPixels times the estimated mean of square differences equals the sum of square differences.
      EXPECT_EQ(std::round(fixedImage->GetBufferedRegion().GetNumberOfPixels() * value), sumOfSquareDifferences);
    }
  };

  check(elx::DefaultConstruct<itk::AdvancedTranslationTransform<double, imageDimension>>{});

  // Check with bspline transform:
  {
    static constexpr unsigned int                                                                       splineOrder = 2;
    elx::DefaultConstruct<itk::AdvancedBSplineDeformableTransform<double, imageDimension, splineOrder>> transform{};

    transform.SetGridRegion(itk::ImageRegion<imageDimension>(itk::Size<imageDimension>::Filled(splineOrder + 1)));

    // The optimizer parameters "are assumed to be maintained by the caller", according to
    // `AdvancedBSplineDeformableTransformBase::WrapAsImages()`, at
    // https://github.com/SuperElastix/elastix/blob/5.1.0/Common/Transforms/itkAdvancedBSplineDeformableTransformBase.hxx#L378-L379
    // Note that transform.GetNumberOfParameters() must be called after SetGridRegion, because GetNumberOfParameters()
    // internally uses the size of the grid region.
    const itk::OptimizerParameters parameters(transform.GetNumberOfParameters(), 0.0);
    transform.SetParameters(parameters);
    check(transform);
  }
}


// Checks the derivative when using the TranslationTransform. The test input images are distance images, both
// representing the distance to a corner point.
GTEST_TEST(AdvancedMeanSquaresImageToImageMetric, DerivativeTranslation)
{
  static constexpr auto imageDimension = 3U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, imageDimension>;
  using PointType = itk::Point<double, imageDimension>;

  const auto imageSizeValue = minimumImageSizeValue;
  const auto imageSize = itk::Size<imageDimension>::Filled(imageSizeValue);

  // Pixel values represent the distance to the left upper corner point.
  const auto leftImage = CreateImageOfDistanceToPoint<PixelType>(imageSize, PointType{});

  // Pixel values represent the distance to the right bottom corner point.
  const auto rightImage =
    CreateImageOfDistanceToPoint<PixelType>(imageSize, itk::MakeFilled<PointType>(imageSizeValue - 1.0));

  const auto getDerivative = [](const auto & fixedImage, const auto & movingImage, auto & interpolator) {
    elx::DefaultConstruct<itk::AdvancedTranslationTransform<double, imageDimension>>   transform{};
    elx::DefaultConstruct<itk::ImageFullSampler<ImageType>>                            imageSampler{};
    elx::DefaultConstruct<AdvancedMeanSquaresImageToImageMetric<ImageType, ImageType>> metric{};
    InitializeMetric(
      metric, *fixedImage, *movingImage, imageSampler, transform, *interpolator, fixedImage->GetBufferedRegion());
    return ValueAndDerivative::FromCostFunction(metric, transform.GetParameters()).derivative;
  };

  for (const auto interpolator : { CreateInterpolator<itk::AdvancedLinearInterpolateImageFunction<ImageType>>(),
                                   CreateInterpolator<itk::BSplineInterpolateImageFunction<ImageType>>(),
                                   CreateInterpolator<itk::NearestNeighborInterpolateImageFunction<ImageType>>() })
  {
    for (const double derivativeValue : getDerivative(leftImage, rightImage, interpolator))
    {
      EXPECT_GT(derivativeValue, 0.0);
    }

    for (const double derivativeValue : getDerivative(rightImage, leftImage, interpolator))
    {
      EXPECT_LT(derivativeValue, 0.0);
    }
  }
}
