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

#include "GTesting/elxGTestUtilities.h"

#include "elxCoreMainGTestUtilities.h"
#include "elxDefaultConstruct.h"

// ITK header file:
#include <itkFileTools.h>
#include <itksys/SystemTools.hxx>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <string>

// Type aliases:
using ParameterMapType = elx::ParameterObject::ParameterMapType;
using ParameterMapVectorType = elx::ParameterObject::ParameterMapVectorType;
using ParameterValueVectorType = elx::ParameterObject::ParameterValueVectorType;
using ParameterFileNameVectorType = elx::ParameterObject::ParameterFileNameVectorType;

// Using-declarations:
using elx::CoreMainGTestUtilities::GetCurrentBinaryDirectoryPath;
using elx::CoreMainGTestUtilities::GetNameOfTest;
using elx::CoreMainGTestUtilities::TypeHolder;

// Tests that ParameterObject::WriteParameterFiles writes all the specified files.
GTEST_TEST(ParameterObject, WriteParameterFiles)
{
  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  elx::DefaultConstruct<elx::ParameterObject> parameterObject{};

  for (const std::size_t numberOfMaps : { 0, 1, 2 })
  {
    const std::string outputDirectoryPath = rootOutputDirectoryPath + '/' + std::to_string(numberOfMaps);
    itk::FileTools::CreateDirectory(outputDirectoryPath);

    parameterObject.SetParameterMaps(ParameterMapVectorType(numberOfMaps));

    ParameterFileNameVectorType fileNames{};

    for (std::size_t i{}; i < numberOfMaps; ++i)
    {
      fileNames.push_back(outputDirectoryPath + '/' + "ParameterFile." + std::to_string(i) + ".txt");
    }

    parameterObject.WriteParameterFiles(fileNames);

    // Check that each of the specified files is written to disk.
    for (const auto & fileName : fileNames)
    {
      EXPECT_TRUE(itksys::SystemTools::FileExists(fileName.c_str(), true));
    }
  }
}


// Tests that any numeric parameter value is printed (by `<<` into an output stream) literally as it was originally
// placed into the ParameterMap.
GTEST_TEST(ParameterObject, PrintOriginalNumericValues)
{
  static constexpr auto check = [](const std::string & parameterValue) {
    elx::DefaultConstruct<elx::ParameterObject> parameterObject{};

    const std::string parameterName = "ParameterName";

    parameterObject.SetParameterMap(elx::ParameterObject::ParameterMapType{ { parameterName, { parameterValue } } });
    std::ostringstream outputStringStream;

    outputStringStream << parameterObject;
    const auto        printedString = outputStringStream.str();
    const std::string expectedParameter = '(' + parameterName + ' ' + parameterValue + ")\n";
    ASSERT_GT(printedString.size(), expectedParameter.size());
    EXPECT_EQ(std::string_view(printedString).substr(printedString.size() - expectedParameter.size()),
              expectedParameter);
  };

  for (const std::string parameterValue : { "0", "1", "1.23456789", "0.1", "-0.1", "3e+38", "4e+38" })
  {
    check(parameterValue);
  }

  check(std::to_string(std::numeric_limits<std::int32_t>::min()));
  check(std::to_string(std::numeric_limits<std::int32_t>::max()));
  check(std::to_string(std::numeric_limits<std::uint32_t>::max()));
  check(std::to_string(std::numeric_limits<std::int64_t>::min()));
  check(std::to_string(std::numeric_limits<std::int64_t>::max()));
  check(std::to_string(std::numeric_limits<std::uint64_t>::max()));

  {
    // Test +/- 100000000000000000000
    constexpr auto power_of_ten = 10.0 * static_cast<double>(itk::Math::UnsignedPower(10, 19));
    check(elx::Conversion::ToString(power_of_ten));
    check(elx::Conversion::ToString(-power_of_ten));
  }
  {
    // Test +/- 999999999999999
    constexpr auto power_of_ten = static_cast<double>(itk::Math::UnsignedPower(10, 15));
    check(elx::Conversion::ToString(power_of_ten - 1.0));
    check(elx::Conversion::ToString(1.0 - power_of_ten));
  }
  {
    // Test +/- 0.000001
    constexpr auto power_of_ten = 1.0 / static_cast<double>(itk::Math::UnsignedPower(10, 6));
    check(elx::Conversion::ToString(power_of_ten));
    check(elx::Conversion::ToString(-power_of_ten));
  }

  const auto checkNumericLimits = [](const auto typeHolder) {
    (void)typeHolder;
    using FloatingPointType = typename decltype(typeHolder)::Type;
    using NumericLimits = std::numeric_limits<FloatingPointType>;
    for (const auto value : { NumericLimits::epsilon(),
                              NumericLimits::min(),
                              NumericLimits::lowest(),
                              NumericLimits::max(),
                              NumericLimits::denorm_min(),
                              -NumericLimits::denorm_min() })
    {
      check(elx::Conversion::ToString(value));
    }
  };

  checkNumericLimits(TypeHolder<float>{});
  checkNumericLimits(TypeHolder<double>{});
}


// Tests HasParameter method with index parameter
GTEST_TEST(ParameterObject, HasParameterWithIndex)
{
  elx::DefaultConstruct<elx::ParameterObject> parameterObject{};

  // Test with empty parameter object
  EXPECT_FALSE(parameterObject.HasParameter(0, "NonExistentParameter"));

  // Create parameter maps
  ParameterMapType parameterMap1;
  parameterMap1["ExistingParameter1"] = { "value1" };
  parameterMap1["ExistingParameter2"] = { "value2a", "value2b" };

  ParameterMapType parameterMap2;
  parameterMap2["ExistingParameter1"] = { "differentValue" };
  parameterMap2["ExistingParameter3"] = { "value3" };

  ParameterMapVectorType parameterMapVector = { parameterMap1, parameterMap2 };
  parameterObject.SetParameterMaps(parameterMapVector);

  // Test existing parameters in first map (index 0)
  EXPECT_TRUE(parameterObject.HasParameter(0, "ExistingParameter1"));
  EXPECT_TRUE(parameterObject.HasParameter(0, "ExistingParameter2"));
  EXPECT_FALSE(parameterObject.HasParameter(0, "ExistingParameter3"));
  EXPECT_FALSE(parameterObject.HasParameter(0, "NonExistentParameter"));

  // Test existing parameters in second map (index 1)
  EXPECT_TRUE(parameterObject.HasParameter(1, "ExistingParameter1"));
  EXPECT_FALSE(parameterObject.HasParameter(1, "ExistingParameter2"));
  EXPECT_TRUE(parameterObject.HasParameter(1, "ExistingParameter3"));
  EXPECT_FALSE(parameterObject.HasParameter(1, "NonExistentParameter"));

  // Test with out-of-bounds index
  EXPECT_FALSE(parameterObject.HasParameter(2, "ExistingParameter1"));
  EXPECT_FALSE(parameterObject.HasParameter(10, "ExistingParameter1"));
}


// Tests HasParameter method without index parameter (searches all maps)
GTEST_TEST(ParameterObject, HasParameterWithoutIndex)
{
  elx::DefaultConstruct<elx::ParameterObject> parameterObject{};

  // Test with empty parameter object
  EXPECT_FALSE(parameterObject.HasParameter("NonExistentParameter"));

  // Create parameter maps
  ParameterMapType parameterMap1;
  parameterMap1["OnlyInMap1"] = { "value1" };
  parameterMap1["InBothMaps"] = { "value2" };

  ParameterMapType parameterMap2;
  parameterMap2["OnlyInMap2"] = { "value3" };
  parameterMap2["InBothMaps"] = { "differentValue" };

  ParameterMapVectorType parameterMapVector = { parameterMap1, parameterMap2 };
  parameterObject.SetParameterMaps(parameterMapVector);

  // Test parameters that exist in specific maps
  EXPECT_TRUE(parameterObject.HasParameter("OnlyInMap1")); // Only in first map
  EXPECT_TRUE(parameterObject.HasParameter("OnlyInMap2")); // Only in second map
  EXPECT_TRUE(parameterObject.HasParameter("InBothMaps")); // In both maps

  // Test non-existent parameter
  EXPECT_FALSE(parameterObject.HasParameter("NonExistentParameter"));

  // Test with single parameter map
  ParameterMapVectorType singleMapVector = { parameterMap1 };
  parameterObject.SetParameterMaps(singleMapVector);

  EXPECT_TRUE(parameterObject.HasParameter("OnlyInMap1"));
  EXPECT_TRUE(parameterObject.HasParameter("InBothMaps"));
  EXPECT_FALSE(parameterObject.HasParameter("OnlyInMap2"));
  EXPECT_FALSE(parameterObject.HasParameter("NonExistentParameter"));
}


//  Tests that ParameterObject::GetDefaultParameterMap("nonrigid") throws an exception.
GTEST_TEST(ParameterObject, GetDefaultParameterMapThrowsExceptionOnTransformNameNonrigid)
{
  EXPECT_THROW(elx::ParameterObject::GetDefaultParameterMap("nonrigid"), itk::ExceptionObject);
}


// Tests that the GridSpacingSchedule of a map returned by GetDefaultParameterMap is as expected.
GTEST_TEST(ParameterObject, GridSpacingScheduleOfDefaultParameterMap)
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
GTEST_TEST(ParameterObject, MaximumNumberOfIterationsOfDefaultParameterMap)
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
GTEST_TEST(ParameterObject, CheckDefaultParameterMapForEachTransformName)
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
