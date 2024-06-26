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
#ifndef elxGTestUtilities_h
#define elxGTestUtilities_h

#include <elxConfiguration.h>
#include <elxElastixBase.h>
#include <itkAdvancedImageToImageMetric.h>
#include <itkAdvancedTransform.h>
#include <itkImageSamplerBase.h>

// ITK header files:
#include <itkInterpolateImageFunction.h>
#include <itkOptimizerParameters.h>
#include <itkPoint.h>
#include <itkSingleValuedCostFunction.h>
#include <itkSize.h>
#include <itkSmartPointer.h>
#include <itkVector.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For generate_n.
#include <cassert>
#include <cfloat> // For DBL_MAX.
#include <limits>
#include <random>

namespace elastix
{
namespace GTestUtilities
{

/// Expect that all keys of both specified maps are unique.
template <typename TMap>
void
ExpectAllKeysUnique(const TMap & map1, const TMap & map2)
{
  const auto endOfMap2 = map2.end();

  for (const auto & keyValuePair : map1)
  {
    EXPECT_EQ(map2.find(keyValuePair.first), endOfMap2);
  }
}


/// Makes a map by merging its two arguments together.
template <typename TMap>
TMap
MakeMergedMap(TMap map1, const TMap & map2)
{
  // Note: This for-loop should be equivalent to C++17 `map1.merge(TMap{map2});`
  for (const auto & keyValuePair : map2)
  {
    map1.insert(keyValuePair);
  }
  return map1;
}


/// Creates a default `ElastixTemplate<FixedImageType, MovingImageType>` object.
/// for unit testing purposes.
template <typename TElastix>
itk::SmartPointer<TElastix>
CreateDefaultElastixObject()
{
  using FixedImageType = typename TElastix::FixedImageType;
  using MovingImageType = typename TElastix::MovingImageType;

  const auto elastixObject = TElastix::New();

  elastixObject->SetConfiguration(elx::Configuration::New());

  const auto fixedImageContainer = elx::ElastixBase::DataObjectContainerType::New();
  fixedImageContainer->push_back(FixedImageType::New());
  elastixObject->SetFixedImageContainer(fixedImageContainer);

  const auto movingImageContainer = elx::ElastixBase::DataObjectContainerType::New();
  movingImageContainer->push_back(MovingImageType::New());
  elastixObject->SetMovingImageContainer(movingImageContainer);

  return elastixObject;
}


/// Returns an `OptimizerParameters` object, filled with pseudo random floating point numbers between the specified
/// minimum and maximum value.
inline itk::OptimizerParameters<double>
GeneratePseudoRandomParameters(const unsigned numberOfParameters, const double minValue, const double maxValue = 1.0)
{
  assert(minValue < maxValue);
  assert((maxValue - minValue) <= DBL_MAX);

  itk::OptimizerParameters<double> parameters(numberOfParameters);

  std::mt19937 randomNumberEngine;

  std::generate_n(parameters.begin(), numberOfParameters, [&randomNumberEngine, minValue, maxValue] {
    return std::uniform_real_distribution<>{ minValue, maxValue }(randomNumberEngine);
  });
  return parameters;
}


/// Does set up and initialize the specified advanced metric.
template <typename TFixedImage, typename TMovingImage>
void
InitializeMetric(
  itk::AdvancedImageToImageMetric<TFixedImage, TMovingImage> &                                metric,
  const TFixedImage &                                                                         fixedImage,
  const TMovingImage &                                                                        movingImage,
  itk::ImageSamplerBase<TFixedImage> &                                                        imageSampler,
  itk::AdvancedTransform<double, TFixedImage::ImageDimension, TMovingImage::ImageDimension> & advancedTransform,
  itk::InterpolateImageFunction<TMovingImage> &                                               interpolator,
  const typename TFixedImage::RegionType &                                                    fixedImageRegion)
{
  // In elastix, this member function is just called by elx::MetricBase::SetAdvancedMetricImageSampler, at
  // https://github.com/SuperElastix/elastix/blob/5.1.0/Core/ComponentBaseClasses/elxMetricBase.hxx#L313
  metric.SetImageSampler(&imageSampler);

  // Similar to the six member function calls in `MultiResolutionImageRegistrationMethod2::Initialize()` "Setup the
  // metric", at
  // https://github.com/SuperElastix/elastix/blob/5.1.0/Common/itkMultiResolutionImageRegistrationMethod2.hxx#L118-L124
  metric.SetMovingImage(&movingImage);
  metric.SetFixedImage(&fixedImage);
  metric.SetTransform(&advancedTransform);
  metric.SetInterpolator(&interpolator);
  metric.SetFixedImageRegion(fixedImageRegion);
  metric.Initialize();
}


/// Represents the value and derivative retrieved from a metric (cost function).
struct ValueAndDerivative
{
  double             value;
  itk::Array<double> derivative;

  static ValueAndDerivative
  FromCostFunction(const itk::SingleValuedCostFunction &    costFunction,
                   const itk::OptimizerParameters<double> & optimizerParameters)
  {
    static constexpr auto quiet_NaN = std::numeric_limits<double>::quiet_NaN();

    ValueAndDerivative valueAndDerivative{ quiet_NaN, itk::Array<double>(optimizerParameters.size(), quiet_NaN) };
    costFunction.GetValueAndDerivative(optimizerParameters, valueAndDerivative.value, valueAndDerivative.derivative);
    return valueAndDerivative;
  }
};

} // namespace GTestUtilities
} // namespace elastix


#endif
