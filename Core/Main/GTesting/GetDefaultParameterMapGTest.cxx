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
#include "elxParameterObject.h"

// GoogleTest header file:
#include <gtest/gtest.h>

// Standard C++ header files:
#include <cmath>   // For pow.
#include <iomanip> // For quoted.
#include <map>
#include <string>

// Alias copied from "elastix\Core\Install\elxBaseComponent.h"
namespace elx = elastix;

// Type aliases:
using ParameterMapType = elx::ParameterObject::ParameterMapType;
using ParameterValueVectorType = elx::ParameterObject::ParameterValueVectorType;


//  Tests that ParameterObject::GetDefaultParameterMap("nonrigid") throws an exception.
GTEST_TEST(GetDefaultParameterMap, ThrowsExceptionOnTransformNameNonrigid)
{
  EXPECT_THROW(elx::ParameterObject::GetDefaultParameterMap("nonrigid"), itk::ExceptionObject);
}


// Tests that the GridSpacingSchedule of a map returned by GetDefaultParameterMap is as expected.
GTEST_TEST(GetDefaultParameterMap, GridSpacingSchedule)
{
  const unsigned int expectedNumberOfResolutions = 4;

  for (const std::string transformName : { "bspline", "groupwise" })
  {
    const ParameterMapType parameterMap = elx::ParameterObject::GetDefaultParameterMap(transformName);

    const ParameterValueVectorType & parameterValues = parameterMap.at("GridSpacingSchedule");
    ASSERT_EQ(parameterValues.size(), expectedNumberOfResolutions);

    for (unsigned int resolution{}; resolution < expectedNumberOfResolutions; ++resolution)
    {
      const std::string & parameterValue = parameterValues.at(resolution);

      std::size_t  numberOfProcessedChars{};
      const double floatingPointValue{ std::stod(parameterValue, &numberOfProcessedChars) };
      EXPECT_EQ(numberOfProcessedChars, parameterValue.size());

      EXPECT_FLOAT_EQ(floatingPointValue, std::pow(2.0, (expectedNumberOfResolutions - resolution - 1) / 2.0));
    }
  }
}


// Tests that the MaximumNumberOfIterations of a map returned by GetDefaultParameterMap is as expected.
GTEST_TEST(GetDefaultParameterMap, MaximumNumberOfIterations)
{
  static constexpr unsigned int expectedMaximumNumberOfIterations{ 256 };

  for (const std::string transformName : { "translation", "rigid", "affine", "bspline", "spline", "groupwise" })
  {
    const ParameterMapType           parameterMap = elx::ParameterObject::GetDefaultParameterMap(transformName);
    const ParameterValueVectorType & parameterValues = parameterMap.at("MaximumNumberOfIterations");

    ASSERT_EQ(parameterValues.size(), 1);
    EXPECT_EQ(parameterValues.at(0), std::to_string(expectedMaximumNumberOfIterations));
  }
}


// Tests that the maps returned by GetDefaultParameterMap(transformName) are as expected, for each supported
// `transformName`.
GTEST_TEST(GetDefaultParameterMap, CheckForEachTransformName)
{
  const std::map<std::string, ParameterMapType> expectedParameterMaps = {
    { "affine",
      {
        { "AutomaticParameterEstimation", { "true" } },
        { "AutomaticScalesEstimation", { "true" } },
        { "CheckNumberOfSamples", { "true" } },
        { "DefaultPixelValue", { "0" } },
        { "FinalBSplineInterpolationOrder", { "3" } },
        { "FixedImagePyramid", { "FixedSmoothingImagePyramid" } },
        { "ImageSampler", { "RandomCoordinate" } },
        { "Interpolator", { "LinearInterpolator" } },
        { "MaximumNumberOfIterations", { "256" } },
        { "MaximumNumberOfSamplingAttempts", { "8" } },
        { "Metric", { "AdvancedMattesMutualInformation" } },
        { "MovingImagePyramid", { "MovingSmoothingImagePyramid" } },
        { "NewSamplesEveryIteration", { "true" } },
        { "NumberOfResolutions", { "4" } },
        { "NumberOfSamplesForExactGradient", { "4096" } },
        { "NumberOfSpatialSamples", { "2048" } },
        { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
        { "Registration", { "MultiResolutionRegistration" } },
        { "ResampleInterpolator", { "FinalBSplineInterpolator" } },
        { "Resampler", { "DefaultResampler" } },
        { "ResultImageFormat", { "nii" } },
        { "Transform", { "AffineTransform" } },
        { "WriteIterationInfo", { "false" } },
        { "WriteResultImage", { "true" } },
      } },
    { "bspline",
      {
        { "AutomaticParameterEstimation", { "true" } },
        { "CheckNumberOfSamples", { "true" } },
        { "DefaultPixelValue", { "0" } },
        { "FinalBSplineInterpolationOrder", { "3" } },
        { "FinalGridSpacingInPhysicalUnits", { "10.000000" } },
        { "FixedImagePyramid", { "FixedSmoothingImagePyramid" } },
        { "GridSpacingSchedule", { "2.8284271247461903", "2", "1.4142135623730951", "1" } },
        { "ImageSampler", { "RandomCoordinate" } },
        { "Interpolator", { "LinearInterpolator" } },
        { "MaximumNumberOfIterations", { "256" } },
        { "MaximumNumberOfSamplingAttempts", { "8" } },
        { "Metric", { "AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty" } },
        { "Metric0Weight", { "1.0" } },
        { "Metric1Weight", { "1.0" } },
        { "MovingImagePyramid", { "MovingSmoothingImagePyramid" } },
        { "NewSamplesEveryIteration", { "true" } },
        { "NumberOfResolutions", { "4" } },
        { "NumberOfSamplesForExactGradient", { "4096" } },
        { "NumberOfSpatialSamples", { "2048" } },
        { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
        { "Registration", { "MultiMetricMultiResolutionRegistration" } },
        { "ResampleInterpolator", { "FinalBSplineInterpolator" } },
        { "Resampler", { "DefaultResampler" } },
        { "ResultImageFormat", { "nii" } },
        { "Transform", { "BSplineTransform" } },
        { "WriteIterationInfo", { "false" } },
        { "WriteResultImage", { "true" } },
      } },
    { "groupwise",
      {
        { "AutomaticParameterEstimation", { "true" } },
        { "CheckNumberOfSamples", { "true" } },
        { "DefaultPixelValue", { "0" } },
        { "FinalBSplineInterpolationOrder", { "3" } },
        { "FinalGridSpacingInPhysicalUnits", { "10.000000" } },
        { "FixedImagePyramid", { "FixedSmoothingImagePyramid" } },
        { "GridSpacingSchedule", { "2.8284271247461903", "2", "1.4142135623730951", "1" } },
        { "ImageSampler", { "RandomCoordinate" } },
        { "Interpolator", { "ReducedDimensionBSplineInterpolator" } },
        { "MaximumNumberOfIterations", { "256" } },
        { "MaximumNumberOfSamplingAttempts", { "8" } },
        { "Metric", { "VarianceOverLastDimensionMetric" } },
        { "MovingImagePyramid", { "MovingSmoothingImagePyramid" } },
        { "NewSamplesEveryIteration", { "true" } },
        { "NumberOfResolutions", { "4" } },
        { "NumberOfSamplesForExactGradient", { "4096" } },
        { "NumberOfSpatialSamples", { "2048" } },
        { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
        { "Registration", { "MultiResolutionRegistration" } },
        { "ResampleInterpolator", { "FinalReducedDimensionBSplineInterpolator" } },
        { "Resampler", { "DefaultResampler" } },
        { "ResultImageFormat", { "nii" } },
        { "Transform", { "BSplineStackTransform" } },
        { "WriteIterationInfo", { "false" } },
        { "WriteResultImage", { "true" } },
      } },
    { "rigid",
      {
        { "AutomaticParameterEstimation", { "true" } },
        { "AutomaticScalesEstimation", { "true" } },
        { "CheckNumberOfSamples", { "true" } },
        { "DefaultPixelValue", { "0" } },
        { "FinalBSplineInterpolationOrder", { "3" } },
        { "FixedImagePyramid", { "FixedSmoothingImagePyramid" } },
        { "ImageSampler", { "RandomCoordinate" } },
        { "Interpolator", { "LinearInterpolator" } },
        { "MaximumNumberOfIterations", { "256" } },
        { "MaximumNumberOfSamplingAttempts", { "8" } },
        { "Metric", { "AdvancedMattesMutualInformation" } },
        { "MovingImagePyramid", { "MovingSmoothingImagePyramid" } },
        { "NewSamplesEveryIteration", { "true" } },
        { "NumberOfResolutions", { "4" } },
        { "NumberOfSamplesForExactGradient", { "4096" } },
        { "NumberOfSpatialSamples", { "2048" } },
        { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
        { "Registration", { "MultiResolutionRegistration" } },
        { "ResampleInterpolator", { "FinalBSplineInterpolator" } },
        { "Resampler", { "DefaultResampler" } },
        { "ResultImageFormat", { "nii" } },
        { "Transform", { "EulerTransform" } },
        { "WriteIterationInfo", { "false" } },
        { "WriteResultImage", { "true" } },
      } },
    { "spline",
      {
        { "AutomaticParameterEstimation", { "true" } },
        { "CheckNumberOfSamples", { "true" } },
        { "DefaultPixelValue", { "0" } },
        { "FinalBSplineInterpolationOrder", { "3" } },
        { "FixedImagePyramid", { "FixedSmoothingImagePyramid" } },
        { "ImageSampler", { "RandomCoordinate" } },
        { "Interpolator", { "LinearInterpolator" } },
        { "MaximumNumberOfIterations", { "256" } },
        { "MaximumNumberOfSamplingAttempts", { "8" } },
        { "Metric", { "AdvancedMattesMutualInformation" } },
        { "MovingImagePyramid", { "MovingSmoothingImagePyramid" } },
        { "NewSamplesEveryIteration", { "true" } },
        { "NumberOfResolutions", { "4" } },
        { "NumberOfSamplesForExactGradient", { "4096" } },
        { "NumberOfSpatialSamples", { "2048" } },
        { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
        { "Registration", { "MultiResolutionRegistration" } },
        { "ResampleInterpolator", { "FinalBSplineInterpolator" } },
        { "Resampler", { "DefaultResampler" } },
        { "ResultImageFormat", { "nii" } },
        { "Transform", { "SplineKernelTransform" } },
        { "WriteIterationInfo", { "false" } },
        { "WriteResultImage", { "true" } },
      } },
    { "translation",
      {
        { "AutomaticParameterEstimation", { "true" } },
        { "AutomaticTransformInitialization", { "true" } },
        { "CheckNumberOfSamples", { "true" } },
        { "DefaultPixelValue", { "0" } },
        { "FinalBSplineInterpolationOrder", { "3" } },
        { "FixedImagePyramid", { "FixedSmoothingImagePyramid" } },
        { "ImageSampler", { "RandomCoordinate" } },
        { "Interpolator", { "LinearInterpolator" } },
        { "MaximumNumberOfIterations", { "256" } },
        { "MaximumNumberOfSamplingAttempts", { "8" } },
        { "Metric", { "AdvancedMattesMutualInformation" } },
        { "MovingImagePyramid", { "MovingSmoothingImagePyramid" } },
        { "NewSamplesEveryIteration", { "true" } },
        { "NumberOfResolutions", { "4" } },
        { "NumberOfSamplesForExactGradient", { "4096" } },
        { "NumberOfSpatialSamples", { "2048" } },
        { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
        { "Registration", { "MultiResolutionRegistration" } },
        { "ResampleInterpolator", { "FinalBSplineInterpolator" } },
        { "Resampler", { "DefaultResampler" } },
        { "ResultImageFormat", { "nii" } },
        { "Transform", { "TranslationTransform" } },
        { "WriteIterationInfo", { "false" } },
        { "WriteResultImage", { "true" } },
      } },
  };

  for (const auto & [transformName, expectedParameterMap] : expectedParameterMaps)
  {
    SCOPED_TRACE(testing::Message() << "transformName = " << std::quoted(transformName));

    const auto actualParameterMap = elx::ParameterObject::GetDefaultParameterMap(transformName);
    EXPECT_EQ(expectedParameterMap.size(), actualParameterMap.size());

    for (const auto & [parameterName, parameterValues] : expectedParameterMap)
    {
      EXPECT_EQ(actualParameterMap.at(parameterName), parameterValues)
        << "parameterName = " << std::quoted(parameterName);
    }
  }
}
