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

#define _USE_MATH_DEFINES // For M_PI.

// First include the header file to be tested:
#include <itkElastixRegistrationMethod.h>

#include <itkTransformixFilter.h>

#include "GTesting/elxGTestUtilities.h"

#include "elxCoreMainGTestUtilities.h"
#include "elxDefaultConstruct.h"
#include "elxForEachSupportedImageType.h"
#include "elxTransformIO.h"

// ITK header file:
#include <itkAffineTransform.h>
#include <itkBSplineTransform.h>
#include <itkCompositeTransform.h>
#include <itkEuler2DTransform.h>
#include <itkImage.h>
#include <itkIndexRange.h>
#include <itkFileTools.h>
#include <itkSimilarity2DTransform.h>
#include <itkTranslationTransform.h>
#include <itkTransformFileReader.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For transform
#include <cmath>     // For M_PI
#include <map>
#include <random>
#include <string>
#include <utility> // For pair


// Type aliases:
using ParameterMapType = itk::ParameterFileParser::ParameterMapType;
using ParameterType = ParameterMapType::value_type;
using ParameterValuesType = itk::ParameterFileParser::ParameterValuesType;
using ParameterMapVectorType = elx::ParameterObject::ParameterMapVectorType;

// Using-declarations:
using elx::CoreMainGTestUtilities::CheckNew;
using elx::CoreMainGTestUtilities::ConvertStringsToVectorOfDouble;
using elx::CoreMainGTestUtilities::ConvertToOffset;
using elx::CoreMainGTestUtilities::CreateImage;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::CreateParameterMap;
using elx::CoreMainGTestUtilities::CreateParameterObject;
using elx::CoreMainGTestUtilities::CreateRandomImageDomain;
using elx::CoreMainGTestUtilities::DerefRawPointer;
using elx::CoreMainGTestUtilities::DerefSmartPointer;
using elx::CoreMainGTestUtilities::FillImageRegion;
using elx::CoreMainGTestUtilities::Front;
using elx::CoreMainGTestUtilities::GetCurrentBinaryDirectoryPath;
using elx::CoreMainGTestUtilities::GetDataDirectoryPath;
using elx::CoreMainGTestUtilities::GetNameOfTest;
using elx::CoreMainGTestUtilities::GetTransformParametersFromFilter;
using elx::CoreMainGTestUtilities::ImageDomain;
using elx::CoreMainGTestUtilities::TypeHolder;
using elx::CoreMainGTestUtilities::minimumImageSizeValue;
using elx::GTestUtilities::MakeMergedMap;


template <typename TImage>
using ElastixRegistrationMethodType = itk::ElastixRegistrationMethod<TImage, TImage>;

namespace
{

auto
ParameterToCurlyBracedString(const ParameterMapType::value_type & parameter)
{
  std::string result = " { \"" + parameter.first + "\", {";
  for (const auto & str : parameter.second)
  {
    if (&str != &(parameter.second.front()))
    {
      result += ", ";
    }
    result += " \"" + str + "\" ";
  }

  result += "} }";

  return result;
}


const ParameterMapType defaultRegistrationParameterMap =
  CreateParameterMap({ // Parameters in alphabetic order:
                       ParameterType{ "ASGDParameterEstimationMethod", { "Original" } },
                       ParameterType{ "AutomaticParameterEstimation", { "true" } },
                       ParameterType{ "BSplineInterpolationOrder", { "1" } },
                       ParameterType{ "CheckNumberOfSamples", { "false" } },
                       ParameterType{ "FinalBSplineInterpolationOrder", { "3" } },
                       ParameterType{ "FixedImagePyramid", { "FixedSmoothingImagePyramid" } },
                       ParameterType{ "FixedInternalImagePixelType", { "float" } },
                       ParameterType{ "HowToCombineTransforms", { "Compose" } },
                       ParameterType{ "InitialTransformParametersFileName", { "NoInitialTransform" } },
                       ParameterType{ "Interpolator", { "BSplineInterpolator" } },
                       ParameterType{ "MaxBandCovSize", { "192" } },
                       ParameterType{ "MaximumNumberOfSamplingAttempts", { "0" } },
                       ParameterType{ "MaximumStepLength", { "1" } },
                       ParameterType{ "MaximumStepLengthRatio", { "1" } },
                       ParameterType{ "MovingImagePyramid", { "MovingSmoothingImagePyramid" } },
                       ParameterType{ "MovingInternalImagePixelType", { "float" } },
                       ParameterType{ "NewSamplesEveryIteration", { "false" } },
                       ParameterType{ "NumberOfBandStructureSamples", { "10" } },
                       ParameterType{ "NumberOfGradientMeasurements", { "0" } },
                       ParameterType{ "NumberOfJacobianMeasurements", { "1000" } },
                       ParameterType{ "NumberOfSamplesForExactGradient", { "100000" } },
                       ParameterType{ "Registration", { "MultiResolutionRegistration" } },
                       ParameterType{ "ResampleInterpolator", { "FinalBSplineInterpolator" } },
                       ParameterType{ "Resampler", { "DefaultResampler" } },
                       ParameterType{ "ShowExactMetricValue", { "false" } },
                       ParameterType{ "SigmoidInitialTime", { "0" } },
                       ParameterType{ "SigmoidScaleFactor", { "0.1" } },
                       ParameterType{ "SP_a", { "0.602" } },
                       ParameterType{ "SP_A", { "20" } },
                       ParameterType{ "SP_alpha", { "0.602" } },
                       ParameterType{ "SubtractMean", { "true" } },
                       ParameterType{ "UseAdaptiveStepSizes", { "true" } },
                       ParameterType{ "UseConstantStep", { "false" } },
                       ParameterType{ "UseDirectionCosines", { "true" } },
                       ParameterType{ "UseMultiThreadingForMetrics", { "true" } },
                       ParameterType{ "WriteResultImage", { "true" } } });

const ParameterMapType defaultTransformParameterMap = CreateParameterMap(
  { // Parameters in alphabetic order:
    ParameterType{ "BSplineTransformSplineOrder", { "3" } },
    ParameterType{ "FinalBSplineInterpolationOrder", { "3" } },
    ParameterType{ "FixedImagePyramid", { "FixedSmoothingImagePyramid" } },
    ParameterType{ "FixedInternalImagePixelType", { "float" } },
    ParameterType{ "GridDirection", elx::Conversion::ToVectorOfStrings(itk::Matrix<int, 2, 2>::GetIdentity()) },
    ParameterType{ "GridIndex", ParameterValuesType(2, "0") },
    ParameterType{ "GridOrigin", ParameterValuesType(2, "0") },
    ParameterType{ "GridSize", ParameterValuesType(2, "1") },
    ParameterType{ "GridSpacing", ParameterValuesType(2, "1") },
    ParameterType{ "HowToCombineTransforms", { "Compose" } },
    ParameterType{ "InitialTransformParametersFileName", { "NoInitialTransform" } },
    ParameterType{ "Interpolator", { "BSplineInterpolator" } },
    ParameterType{ "MovingImagePyramid", { "MovingSmoothingImagePyramid" } },
    ParameterType{ "MovingInternalImagePixelType", { "float" } },
    ParameterType{ "NewSamplesEveryIteration", { "false" } },
    ParameterType{ "Registration", { "MultiResolutionRegistration" } },
    ParameterType{ "ResampleInterpolator", { "FinalBSplineInterpolator" } },
    ParameterType{ "Resampler", { "DefaultResampler" } },
    ParameterType{ "UseCyclicTransform", { "false" } },
    ParameterType{ "UseDirectionCosines", { "true" } } });


auto
DefaultTransformParameter(const ParameterMapType::value_type & parameter)
{

  EXPECT_EQ(defaultTransformParameterMap.count(parameter.first), 1)
    << " parameter = " << ParameterToCurlyBracedString(parameter);
  EXPECT_EQ(defaultTransformParameterMap.at(parameter.first), parameter.second);

  return parameter;
}

auto
NonDefaultTransformParameter(const ParameterMapType::value_type & parameter)
{
  const auto end = defaultTransformParameterMap.cend();
  EXPECT_EQ(std::find(defaultTransformParameterMap.cbegin(), end, parameter), end);

  if (const auto found = defaultTransformParameterMap.find(parameter.first); found != end)
  {
    EXPECT_NE(found->second, parameter.second);
  }

  return parameter;
}

auto
DefaultRegistrationParameter(const ParameterMapType::value_type & parameter)
{

  EXPECT_EQ(defaultRegistrationParameterMap.count(parameter.first), 1)
    << " parameter = " << ParameterToCurlyBracedString(parameter);
  EXPECT_EQ(defaultRegistrationParameterMap.at(parameter.first), parameter.second);

  return parameter;
}

auto
NonDefaultRegistrationParameter(const ParameterMapType::value_type & parameter)
{
  const auto end = defaultRegistrationParameterMap.cend();
  EXPECT_EQ(std::find(defaultRegistrationParameterMap.cbegin(), end, parameter), end);

  if (const auto found = defaultRegistrationParameterMap.find(parameter.first); found != end)
  {
    EXPECT_NE(found->second, parameter.second);
  }

  return parameter;
}


template <unsigned int VDimension>
auto
ConvertIndexToOffset(const itk::Index<VDimension> & index)
{
  return index - itk::Index<VDimension>{};
};


template <unsigned int VDimension>
void
Expect_equal_output_SetInitialTransformParameterObject_and_Transformix_SetTransformParameterObject(
  const ParameterMapVectorType &  transformParameterMaps,
  const ImageDomain<VDimension> & fixedImageDomain,
  const ImageDomain<VDimension> & movingImageDomain)
{
  ASSERT_FALSE(transformParameterMaps.empty());

  using PixelType = float;
  using ImageType = itk::Image<PixelType, VDimension>;

  itk::Size<VDimension> movingImageSize;
  std::iota(movingImageSize.begin(), movingImageSize.end(), 5U);

  elx::DefaultConstruct<ImageType> fixedImage{};
  fixedImageDomain.ToImage(fixedImage);
  fixedImage.Allocate(true);

  elx::DefaultConstruct<ImageType> movingImage{};
  movingImageDomain.ToImage(movingImage);
  movingImage.Allocate(true);
  const itk::ImageBufferRange<ImageType> movingImageBufferRange(movingImage);

  std::mt19937 randomNumberEngine{};

  std::generate(movingImageBufferRange.begin(), movingImageBufferRange.end(), [&randomNumberEngine] {
    return std::uniform_real_distribution<PixelType>{ PixelType{ 1 }, PixelType{ 2 } }(randomNumberEngine);
  });

  elx::DefaultConstruct<elx::ParameterObject> registrationParameterObject{};

  // Parameter map of a registration that "does nothing".
  const ParameterMapType registrationParameterMap = CreateParameterMap({
    // Default parameters in alphabetic order:
    DefaultRegistrationParameter({ "ASGDParameterEstimationMethod", { "Original" } }),
    DefaultRegistrationParameter({ "AutomaticParameterEstimation", { "true" } }),
    DefaultRegistrationParameter({ "BSplineInterpolationOrder", { "1" } }),
    DefaultRegistrationParameter({ "CheckNumberOfSamples", { "false" } }),
    DefaultRegistrationParameter({ "FinalBSplineInterpolationOrder", { "3" } }),
    DefaultRegistrationParameter({ "FixedImagePyramid", { "FixedSmoothingImagePyramid" } }),
    DefaultRegistrationParameter({ "FixedInternalImagePixelType", { "float" } }),
    DefaultRegistrationParameter({ "HowToCombineTransforms", { "Compose" } }),
    DefaultRegistrationParameter({ "InitialTransformParametersFileName", { "NoInitialTransform" } }),
    DefaultRegistrationParameter({ "Interpolator", { "BSplineInterpolator" } }),
    DefaultRegistrationParameter({ "MaxBandCovSize", { "192" } }),
    DefaultRegistrationParameter({ "MaximumNumberOfSamplingAttempts", { "0" } }),
    DefaultRegistrationParameter({ "MaximumStepLength", { "1" } }),
    DefaultRegistrationParameter({ "MaximumStepLengthRatio", { "1" } }),
    DefaultRegistrationParameter({ "MovingImagePyramid", { "MovingSmoothingImagePyramid" } }),
    DefaultRegistrationParameter({ "MovingInternalImagePixelType", { "float" } }),
    DefaultRegistrationParameter({ "NewSamplesEveryIteration", { "false" } }),
    DefaultRegistrationParameter({ "NumberOfBandStructureSamples", { "10" } }),
    DefaultRegistrationParameter({ "NumberOfGradientMeasurements", { "0" } }),
    DefaultRegistrationParameter({ "NumberOfJacobianMeasurements", { "1000" } }),
    DefaultRegistrationParameter({ "NumberOfSamplesForExactGradient", { "100000" } }),
    DefaultRegistrationParameter({ "Registration", { "MultiResolutionRegistration" } }),
    DefaultRegistrationParameter({ "ResampleInterpolator", { "FinalBSplineInterpolator" } }),
    DefaultRegistrationParameter({ "Resampler", { "DefaultResampler" } }),
    DefaultRegistrationParameter({ "ShowExactMetricValue", { "false" } }),
    DefaultRegistrationParameter({ "SigmoidInitialTime", { "0" } }),
    DefaultRegistrationParameter({ "SigmoidScaleFactor", { "0.1" } }),
    DefaultRegistrationParameter({ "SP_a", { "0.602" } }),
    DefaultRegistrationParameter({ "SP_A", { "20" } }),
    DefaultRegistrationParameter({ "SP_alpha", { "0.602" } }),
    DefaultRegistrationParameter({ "SubtractMean", { "true" } }),
    DefaultRegistrationParameter({ "UseAdaptiveStepSizes", { "true" } }),
    DefaultRegistrationParameter({ "UseConstantStep", { "false" } }),
    DefaultRegistrationParameter({ "UseDirectionCosines", { "true" } }),
    DefaultRegistrationParameter({ "UseMultiThreadingForMetrics", { "true" } }),
    DefaultRegistrationParameter({ "WriteResultImage", { "true" } }),
    // Non-default parameters in alphabetic order:
    NonDefaultRegistrationParameter({ "AutomaticTransformInitialization", { "false" } }),
    NonDefaultRegistrationParameter({ "ImageSampler", { "Full" } }), // required
    NonDefaultRegistrationParameter({ "MaximumNumberOfIterations", { "0" } }),
    NonDefaultRegistrationParameter({ "Metric", { "AdvancedNormalizedCorrelation" } }), // default ""
    NonDefaultRegistrationParameter({ "NumberOfResolutions", { "1" } }),
    NonDefaultRegistrationParameter({ "Optimizer", { "AdaptiveStochasticGradientDescent" } }), // default ""
    NonDefaultRegistrationParameter({ "Transform", { "TranslationTransform" } }),              // default ""
  });

  registrationParameterObject.SetParameterMaps(
    ParameterMapVectorType(transformParameterMaps.size(), registrationParameterMap));

  elx::DefaultConstruct<elx::ParameterObject> transformParameterObject{};
  elx::DefaultConstruct<elx::ParameterObject> transformixParameterObject{};

  // Add the parameters that specify the fixed image domain to the last transformix parameter map.
  auto transformixParameterMaps = transformParameterMaps;
  transformixParameterMaps.back().merge(fixedImageDomain.AsParameterMap());

  transformParameterObject.SetParameterMaps(transformParameterMaps);
  transformixParameterObject.SetParameterMaps(transformixParameterMaps);
  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  elx::DefaultConstruct<itk::TransformixFilter<ImageType>>        transformix{};

  registration.SetParameterObject(&registrationParameterObject);
  registration.SetInitialTransformParameterObject(&transformParameterObject);
  transformix.SetTransformParameterObject(&transformixParameterObject);

  registration.SetFixedImage(&fixedImage);
  registration.SetMovingImage(&movingImage);
  transformix.SetMovingImage(&movingImage);
  registration.Update();

  transformix.Update();

  const auto & transformixOutput = DerefRawPointer(transformix.GetOutput());

  // Sanity checks, checking that our test is non-trivial.
  EXPECT_NE(transformixOutput, fixedImage);
  EXPECT_NE(transformixOutput, movingImage);

  const auto & actualRegistrationOutput = DerefRawPointer(registration.GetOutput());
  EXPECT_EQ(actualRegistrationOutput, transformixOutput);
}


template <unsigned NDimension, unsigned NSplineOrder>
void
Test_WriteBSplineTransformToItkFileFormat(const std::string & rootOutputDirectoryPath)
{
  using PixelType = float;
  using ImageType = itk::Image<PixelType, NDimension>;
  const auto image = CreateImage<PixelType>(itk::Size<NDimension>::Filled(4));

  using ItkBSplineTransformType = itk::BSplineTransform<double, NDimension, NSplineOrder>;
  const elx::DefaultConstruct<ItkBSplineTransformType> itkBSplineTransform;

  const auto defaultFixedParameters = itkBSplineTransform.GetFixedParameters();

  // FixedParameters store the grid size, origin, spacing, and direction, according to the ITK `BSplineTransform`
  // default-constructor at
  // https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.0/Modules/Core/Transform/include/itkBSplineTransform.hxx#L35-L61.
  constexpr auto expectedNumberOfFixedParameters = NDimension * (NDimension + 3);
  ASSERT_EQ(defaultFixedParameters.size(), expectedNumberOfFixedParameters);

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  registration.SetFixedImage(image);
  registration.SetMovingImage(image);

  for (const std::string fileNameExtension : { "h5", "tfm" })
  {
    const std::string outputDirectoryPath = rootOutputDirectoryPath + "/" + std::to_string(NDimension) + "D_" +
                                            "SplineOrder=" + std::to_string(NSplineOrder) +
                                            "_FileNameExtension=" + fileNameExtension;
    itk::FileTools::CreateDirectory(outputDirectoryPath);

    registration.SetOutputDirectory(outputDirectoryPath);
    registration.SetParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "AutomaticTransformInitialization", "false" },
                              { "ImageSampler", "Full" },
                              { "BSplineTransformSplineOrder", std::to_string(NSplineOrder) },
                              { "ITKTransformOutputFileNameExtension", fileNameExtension },
                              { "MaximumNumberOfIterations", "0" },
                              { "Metric", "AdvancedNormalizedCorrelation" },
                              { "Optimizer", "AdaptiveStochasticGradientDescent" },
                              { "Transform", "BSplineTransform" } }));
    registration.Update();

    const itk::TransformBase::ConstPointer readTransform =
      elx::TransformIO::Read(outputDirectoryPath + "/TransformParameters.0." + fileNameExtension);

    const itk::TransformBase & actualTransform = DerefSmartPointer(readTransform);

    EXPECT_EQ(typeid(actualTransform), typeid(ItkBSplineTransformType));
    EXPECT_EQ(actualTransform.GetParameters(), itkBSplineTransform.GetParameters());

    const auto actualFixedParameters = actualTransform.GetFixedParameters();
    ASSERT_EQ(actualFixedParameters.size(), expectedNumberOfFixedParameters);

    for (unsigned i{}; i < NDimension; ++i)
    {
      EXPECT_EQ(actualFixedParameters[i], defaultFixedParameters[i]);
    }
    for (unsigned i{ NDimension }; i < 3 * NDimension; ++i)
    {
      // The actual values of the FixedParameters for grid origin and spacing differ from the corresponding
      // default-constructed transform! That is expected!
      EXPECT_NE(actualFixedParameters[i], defaultFixedParameters[i]);
    }
    for (unsigned i{ 3 * NDimension }; i < expectedNumberOfFixedParameters; ++i)
    {
      EXPECT_EQ(actualFixedParameters[i], defaultFixedParameters[i]);
    }
  }
}
} // namespace


static_assert(sizeof(itk::ElastixLogLevel) == sizeof(elx::log::level),
              "The log level enum types should have the same size!");

static_assert(sizeof(itk::ElastixLogLevel) == 1, "The log level enum type should have just one byte!");

static_assert(itk::ElastixLogLevel::Info == itk::ElastixLogLevel{}, "The default log level should be `Info`!");

static_assert(static_cast<int>(itk::ElastixLogLevel::Info) == static_cast<int>(elx::log::level::info) &&
                static_cast<int>(itk::ElastixLogLevel::Warning) == static_cast<int>(elx::log::level::warn) &&
                static_cast<int>(itk::ElastixLogLevel::Error) == static_cast<int>(elx::log::level::err) &&
                static_cast<int>(itk::ElastixLogLevel::Off) == static_cast<int>(elx::log::level::off),
              "Corresponding log level enumerators should have the same underlying integer value!");


GTEST_TEST(itkElastixRegistrationMethod, LogLevel)
{
  using ImageType = itk::Image<float>;
  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> elastixRegistrationMethod;

  ASSERT_EQ(elastixRegistrationMethod.GetLogLevel(), itk::ElastixLogLevel{});

  for (const auto logLevel : { itk::ElastixLogLevel::Info,
                               itk::ElastixLogLevel::Warning,
                               itk::ElastixLogLevel::Error,
                               itk::ElastixLogLevel::Off })
  {
    elastixRegistrationMethod.SetLogLevel(logLevel);
    EXPECT_EQ(elastixRegistrationMethod.GetLogLevel(), logLevel);
  }
}


GTEST_TEST(itkElastixRegistrationMethod, IsDefaultInitialized)
{
  constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;

  const elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> elastixRegistrationMethod;

  EXPECT_EQ(elastixRegistrationMethod.GetInitialTransformParameterFileName(), std::string{});
  EXPECT_EQ(elastixRegistrationMethod.GetFixedPointSetFileName(), std::string{});
  EXPECT_EQ(elastixRegistrationMethod.GetMovingPointSetFileName(), std::string{});
  EXPECT_EQ(elastixRegistrationMethod.GetOutputDirectory(), std::string{});
  EXPECT_EQ(elastixRegistrationMethod.GetLogFileName(), std::string{});
  EXPECT_FALSE(elastixRegistrationMethod.GetLogToConsole());
  EXPECT_FALSE(elastixRegistrationMethod.GetLogToFile());
  EXPECT_EQ(elastixRegistrationMethod.GetNumberOfThreads(), 0);
}


// Tests that the value zero is rejected for the "NumberOfResolutions" parameter.
GTEST_TEST(itkElastixRegistrationMethod, RejectZeroValueForNumberOfResolution)
{
  constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  const itk::Size<ImageDimension> imageSize{ { 5, 6 } };

  elx::DefaultConstruct<elx::ParameterObject>                     parameterObject{};
  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  registration.SetFixedImage(CreateImage<PixelType>(imageSize));
  registration.SetMovingImage(CreateImage<PixelType>(imageSize));
  registration.SetParameterObject(&parameterObject);

  elx::ParameterObject::ParameterMapType parameterMap{ // Parameters in alphabetic order:
                                                       { "ImageSampler", { "Full" } },
                                                       { "MaximumNumberOfIterations", { "2" } },
                                                       { "Metric", { "AdvancedNormalizedCorrelation" } },
                                                       { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                                                       { "Transform", { "TranslationTransform" } }
  };

  parameterObject.SetParameterMap(parameterMap);

  // OK: "NumberOfResolutions" is unspecified, so use its default value.
  EXPECT_NO_THROW(registration.Update());

  for (const unsigned int numberOfResolutions : { 1, 2 })
  {
    // OK: Use a value greater than zero.
    parameterMap["NumberOfResolutions"] = { std::to_string(numberOfResolutions) };
    parameterObject.SetParameterMap(parameterMap);
    EXPECT_NO_THROW(registration.Update());
  }

  // Expected to be rejected: the value zero.
  parameterMap["NumberOfResolutions"] = { "0" };
  parameterObject.SetParameterMap(parameterMap);
  EXPECT_THROW(registration.Update(), itk::ExceptionObject);
}


// Tests registering two small (5x6) binary images, which are translated with respect to each other.
GTEST_TEST(itkElastixRegistrationMethod, Translation)
{
  constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  registration.SetFixedImage(fixedImage);
  registration.SetMovingImage(movingImage);
  registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                          { "ImageSampler", "Full" },
                                                          { "MaximumNumberOfIterations", "2" },
                                                          { "Metric", "AdvancedNormalizedCorrelation" },
                                                          { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                          { "Transform", "TranslationTransform" } }));
  registration.Update();

  const auto transformParameters = GetTransformParametersFromFilter(registration);
  EXPECT_EQ(ConvertToOffset<ImageDimension>(transformParameters), translationOffset);
}


// Tests "MaximumNumberOfIterations" value "0"
GTEST_TEST(itkElastixRegistrationMethod, MaximumNumberOfIterationsZero)
{
  constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  for (const auto optimizer :
       { "AdaptiveStochasticGradientDescent", "FiniteDifferenceGradientDescent", "StandardGradientDescent" })
  {
    elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

    registration.SetFixedImage(fixedImage);
    registration.SetMovingImage(movingImage);
    registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                            { "ImageSampler", "Full" },
                                                            { "MaximumNumberOfIterations", "0" },
                                                            { "Metric", "AdvancedNormalizedCorrelation" },
                                                            { "Optimizer", optimizer },
                                                            { "Transform", "TranslationTransform" } }));
    registration.Update();

    const auto transformParameters = GetTransformParametersFromFilter(registration);

    for (const auto & transformParameter : transformParameters)
    {
      EXPECT_EQ(transformParameter, 0.0);
    }
  }
}


// Tests "AutomaticTransformInitializationMethod" "CenterOfGravity".
GTEST_TEST(itkElastixRegistrationMethod, AutomaticTransformInitializationCenterOfGravity)
{
  constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  for (const bool automaticTransformInitialization : { false, true })
  {
    elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

    registration.SetFixedImage(fixedImage);
    registration.SetMovingImage(movingImage);
    registration.SetParameterObject(CreateParameterObject(
      { // Parameters in alphabetic order:
        { "AutomaticTransformInitialization", automaticTransformInitialization ? "true" : "false" },
        { "AutomaticTransformInitializationMethod", "CenterOfGravity" },
        { "ImageSampler", "Full" },
        { "MaximumNumberOfIterations", "0" },
        { "Metric", "AdvancedNormalizedCorrelation" },
        { "Optimizer", "AdaptiveStochasticGradientDescent" },
        { "Transform", "TranslationTransform" } }));
    registration.Update();

    const auto transformParameters = GetTransformParametersFromFilter(registration);
    const auto estimatedOffset = ConvertToOffset<ImageDimension>(transformParameters);
    EXPECT_EQ(estimatedOffset == translationOffset, automaticTransformInitialization);
  }
}


// Tests registering two images, having "WriteResultImage" set.
GTEST_TEST(itkElastixRegistrationMethod, WriteResultImage)
{
  constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  for (const bool writeResultImage : { true, false })
  {
    elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

    registration.SetFixedImage(fixedImage);
    registration.SetMovingImage(movingImage);
    registration.SetParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "ImageSampler", "Full" },
                              { "MaximumNumberOfIterations", "2" },
                              { "Metric", "AdvancedNormalizedCorrelation" },
                              { "Optimizer", "AdaptiveStochasticGradientDescent" },
                              { "Transform", "TranslationTransform" },
                              { "WriteResultImage", (writeResultImage ? "true" : "false") } }));
    registration.Update();

    const auto &       output = DerefRawPointer(registration.GetOutput());
    const auto &       outputImageSize = output.GetBufferedRegion().GetSize();
    const auto * const outputBufferPointer = output.GetBufferPointer();

    if (writeResultImage)
    {
      EXPECT_EQ(outputImageSize, imageSize);
      ASSERT_NE(outputBufferPointer, nullptr);

      // When "WriteResultImage" is true, expect an output image that is very much like the fixed image.
      for (const auto index : itk::ZeroBasedIndexRange<ImageDimension>(imageSize))
      {
        EXPECT_EQ(std::round(output.GetPixel(index)), std::round(fixedImage->GetPixel(index)));
      }
    }
    else
    {
      // When "WriteResultImage" is false, expect an empty output image.
      EXPECT_EQ(outputImageSize, ImageType::SizeType());
      EXPECT_EQ(outputBufferPointer, nullptr);
    }

    const auto transformParameters = GetTransformParametersFromFilter(registration);
    EXPECT_EQ(ConvertToOffset<ImageDimension>(transformParameters), translationOffset);
  }
}


// Tests registering two images, having a custom "ResultImageName" specified.
GTEST_TEST(itkElastixRegistrationMethod, ResultImageName)
{
  constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  const auto        numberOfResolutions = 2u;
  const std::string customResultImageName = "CustomResultImageName";

  const auto getOutputSubdirectoryPath = [rootOutputDirectoryPath](const bool useCustomResultImageName) {
    return rootOutputDirectoryPath + '/' +
           (useCustomResultImageName ? "DefaultResultImageName" : "CustomResultImageName");
  };

  for (const bool useCustomResultImageName : { true, false })
  {
    elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

    const std::string outputSubdirectoryPath = getOutputSubdirectoryPath(useCustomResultImageName);
    itk::FileTools::CreateDirectory(outputSubdirectoryPath);
    registration.SetOutputDirectory(outputSubdirectoryPath);
    registration.SetFixedImage(fixedImage);
    registration.SetMovingImage(movingImage);

    auto parameterMap = CreateParameterMap({ // Parameters in alphabetic order:
                                             { "ImageSampler", "Full" },
                                             { "MaximumNumberOfIterations", "2" },
                                             { "Metric", "AdvancedNormalizedCorrelation" },
                                             { "NumberOfResolutions", std::to_string(numberOfResolutions) },
                                             { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                             { "Transform", "TranslationTransform" },
                                             { "WriteResultImageAfterEachResolution", "true" } });

    if (useCustomResultImageName)
    {
      parameterMap["ResultImageName"] = { customResultImageName };
    }
    const auto parameterObject = elx::ParameterObject::New();
    parameterObject->SetParameterMap(parameterMap);
    registration.SetParameterObject(parameterObject);
    registration.Update();
  }

  for (unsigned int resolutionNumber{ 0 }; resolutionNumber < numberOfResolutions; ++resolutionNumber)
  {
    const auto fileNamePostFix = ".0.R" + std::to_string(resolutionNumber) + ".mhd";
    const auto expectedImage =
      itk::ReadImage<ImageType>(getOutputSubdirectoryPath(false) + "/result" + fileNamePostFix);
    const auto actualImage =
      itk::ReadImage<ImageType>(getOutputSubdirectoryPath(true) + '/' + customResultImageName + fileNamePostFix);

    ASSERT_NE(expectedImage, nullptr);
    ASSERT_NE(actualImage, nullptr);
    EXPECT_EQ(*actualImage, *expectedImage);
  }
}


// Tests that the origin of the output image is equal to the origin of the fixed image (by default).
GTEST_TEST(itkElastixRegistrationMethod, OutputHasSameOriginAsFixedImage)
{
  constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 12, 16 } };
  const IndexType  fixedImageRegionIndex{ { 3, 9 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  for (const auto fixedImageOrigin : { itk::MakePoint(-1.0, -2.0), ImageType::PointType(), itk::MakePoint(0.25, 0.75) })
  {
    fixedImage->SetOrigin(fixedImageOrigin);

    for (const auto movingImageOrigin :
         { itk::MakePoint(-1.0, -2.0), ImageType::PointType(), itk::MakePoint(0.25, 0.75) })
    {
      movingImage->SetOrigin(movingImageOrigin);

      elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

      registration.SetFixedImage(fixedImage);
      registration.SetMovingImage(movingImage);
      registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                              { "ImageSampler", "Full" },
                                                              { "MaximumNumberOfIterations", "2" },
                                                              { "Metric", "AdvancedNormalizedCorrelation" },
                                                              { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                              { "Transform", "TranslationTransform" } }));
      registration.Update();

      const auto & output = DerefRawPointer(registration.GetOutput());

      // The most essential check of this test.
      EXPECT_EQ(output.GetOrigin(), fixedImageOrigin);

      ASSERT_EQ(output.GetBufferedRegion().GetSize(), imageSize);
      ASSERT_NE(output.GetBufferPointer(), nullptr);

      // Expect an output image that is very much like the fixed image.
      for (const auto & index : itk::ZeroBasedIndexRange<ImageDimension>(imageSize))
      {
        EXPECT_EQ(std::round(output.GetPixel(index)), std::round(fixedImage->GetPixel(index)));
      }

      const auto transformParameters = GetTransformParametersFromFilter(registration);

      ASSERT_EQ(transformParameters.size(), ImageDimension);

      for (std::size_t i{}; i < ImageDimension; ++i)
      {
        EXPECT_EQ(std::round(transformParameters[i] + fixedImageOrigin[i] - movingImageOrigin[i]),
                  translationOffset[i]);
      }
    }
  }
}


GTEST_TEST(itkElastixRegistrationMethod, InitialTransformParameterFile)
{
  using PixelType = float;
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType initialTranslation{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = CreateImage<PixelType>(imageSize);

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  registration.SetFixedImage(fixedImage);
  registration.SetInitialTransformParameterFileName(GetDataDirectoryPath() +
                                                    "/Translation(1,-2)/TransformParameters.txt");

  registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                          { "ImageSampler", "Full" },
                                                          { "MaximumNumberOfIterations", "2" },
                                                          { "Metric", "AdvancedNormalizedCorrelation" },
                                                          { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                          { "Transform", "TranslationTransform" } }));

  for (const auto index :
       itk::ImageRegionIndexRange<ImageDimension>(itk::ImageRegion<ImageDimension>({ 0, -2 }, { 2, 3 })))
  {
    const auto actualTranslation = ConvertIndexToOffset(index);
    movingImage->FillBuffer(0);
    FillImageRegion(*movingImage, fixedImageRegionIndex + actualTranslation, regionSize);
    registration.SetMovingImage(movingImage);
    registration.Update();

    const auto transformParameters = GetTransformParametersFromFilter(registration);
    EXPECT_EQ(initialTranslation + ConvertToOffset<ImageDimension>(transformParameters), actualTranslation);
  }
}


GTEST_TEST(itkElastixRegistrationMethod, SetInitialTransformParameterObject)
{
  using PixelType = float;
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType initialTranslation{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = CreateImage<PixelType>(imageSize);

  elx::DefaultConstruct<elx::ParameterObject>                     registrationParameterObject{};
  elx::DefaultConstruct<elx::ParameterObject>                     initialTransformParameterObject{};
  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  registration.SetFixedImage(fixedImage);
  registration.SetInitialTransformParameterObject(&initialTransformParameterObject);
  registration.SetParameterObject(&registrationParameterObject);

  const elx::ParameterObject::ParameterMapType registrationParameterMap{
    // Parameters in alphabetic order:
    { "ImageSampler", { "Full" } },
    { "MaximumNumberOfIterations", { "2" } },
    { "Metric", { "AdvancedNormalizedCorrelation" } },
    { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
    { "Transform", { "TranslationTransform" } }
  };

  for (const unsigned int numberOfRegistrationParameterMaps : { 1, 2, 3 })
  {
    using ParameterMapVectorType = elx::ParameterObject::ParameterMapVectorType;

    // Specify multiple (one or more) registration parameter maps.
    registrationParameterObject.SetParameterMaps(
      ParameterMapVectorType(numberOfRegistrationParameterMaps, registrationParameterMap));

    // Test both one and two transform parameter maps (both specifying a (1, -2) translation in this case).
    for (const auto & initialTransformParameterMaps :
         { ParameterMapVectorType{ { { "NumberOfParameters", { "2" } },
                                     { "Transform", { "TranslationTransform" } },
                                     { "TransformParameters", { "1", "-2" } } } },
           ParameterMapVectorType{ { { "NumberOfParameters", { "2" } },
                                     { "Transform", { "TranslationTransform" } },
                                     { "TransformParameters", { "1", "0" } } },
                                   { { "NumberOfParameters", { "2" } },
                                     { "Transform", { "TranslationTransform" } },
                                     { "TransformParameters", { "0", "-2" } } } } })
    {
      initialTransformParameterObject.SetParameterMaps(initialTransformParameterMaps);

      // Do the test for a few possible translations.
      for (const auto index :
           itk::ImageRegionIndexRange<ImageDimension>(itk::ImageRegion<ImageDimension>({ 0, -2 }, { 2, 3 })))
      {
        const auto actualTranslation = ConvertIndexToOffset(index);
        movingImage->FillBuffer(0);
        FillImageRegion(*movingImage, fixedImageRegionIndex + actualTranslation, regionSize);
        registration.SetMovingImage(movingImage);
        registration.Update();

        const auto & transformParameterMaps =
          DerefRawPointer(registration.GetTransformParameterObject()).GetParameterMaps();

        ASSERT_EQ(transformParameterMaps.size(), numberOfRegistrationParameterMaps);

        // All registration parameter maps, except for the first one, should just have a zero-translation.
        for (unsigned int i{ 1 }; i < numberOfRegistrationParameterMaps; ++i)
        {
          const auto transformParameters =
            ConvertStringsToVectorOfDouble(transformParameterMaps[i].at("TransformParameters"));
          EXPECT_EQ(ConvertToOffset<ImageDimension>(transformParameters), OffsetType{});
        }

        // Together the initial translation and the first registration should have the actual image translation.
        const auto transformParameters =
          ConvertStringsToVectorOfDouble(transformParameterMaps.front().at("TransformParameters"));
        EXPECT_EQ(initialTranslation + ConvertToOffset<ImageDimension>(transformParameters), actualTranslation);
      }
    }
  }
}


GTEST_TEST(itkElastixRegistrationMethod, SetInitialTransformParameterObjectVersusTransformix)
{
  {
    std::mt19937 randomNumberEngine{};

    const auto ImageDimension = 2U;

    const auto randomSign = [&randomNumberEngine] { return (randomNumberEngine() % 2 == 0) ? -1.0 : 1.0; };

    const std::array translationTransformParameters = {
      randomSign() * (1.0 + std::uniform_real_distribution<>{}(randomNumberEngine)),
      randomSign() * (1.0 + std::uniform_real_distribution<>{}(randomNumberEngine))
    };

    const elx::ParameterObject::ParameterMapType translationTransformParameterMap = CreateParameterMap(
      { // Default parameters in alphabetic order:
        DefaultTransformParameter({ "FinalBSplineInterpolationOrder", { "3" } }),
        DefaultTransformParameter({ "FixedInternalImagePixelType", { "float" } }),
        DefaultTransformParameter({ "HowToCombineTransforms", { "Compose" } }),
        DefaultTransformParameter({ "InitialTransformParametersFileName", { "NoInitialTransform" } }),
        DefaultTransformParameter({ "MovingInternalImagePixelType", { "float" } }),
        DefaultTransformParameter({ "ResampleInterpolator", { "FinalBSplineInterpolator" } }),
        DefaultTransformParameter({ "Resampler", { "DefaultResampler" } }),
        // Non-default parameters in alphabetic order:
        NonDefaultTransformParameter(
          { "NumberOfParameters", { std::to_string(translationTransformParameters.size()) } }),
        NonDefaultTransformParameter({ "Transform", { "TranslationTransform" } }),
        NonDefaultTransformParameter(
          { "TransformParameters", elx::Conversion::ToVectorOfStrings(translationTransformParameters) }) });

    constexpr auto gridValueSize = 4U;

    std::array<double, ImageDimension * itk::Math::UnsignedPower(gridValueSize, ImageDimension)>
      bsplineTransformParameters;

    std::generate(bsplineTransformParameters.begin(), bsplineTransformParameters.end(), [&randomNumberEngine] {
      return std::uniform_real_distribution<>{ -1.0, 1.0 }(randomNumberEngine);
    });

    const elx::ParameterObject::ParameterMapType bsplineTransformParameterMap = CreateParameterMap(
      { // Default parameters in alphabetic order:
        DefaultTransformParameter({ "FinalBSplineInterpolationOrder", { "3" } }),
        DefaultTransformParameter({ "FixedInternalImagePixelType", { "float" } }),
        DefaultTransformParameter({ "HowToCombineTransforms", { "Compose" } }),
        DefaultTransformParameter({ "InitialTransformParametersFileName", { "NoInitialTransform" } }),
        DefaultTransformParameter({ "MovingInternalImagePixelType", { "float" } }),
        DefaultTransformParameter({ "ResampleInterpolator", { "FinalBSplineInterpolator" } }),
        DefaultTransformParameter({ "Resampler", { "DefaultResampler" } }),
        DefaultTransformParameter({ "BSplineTransformSplineOrder", { "3" } }),
        DefaultTransformParameter({ "UseCyclicTransform", { "false" } }),
        DefaultTransformParameter({ "GridIndex", ParameterValuesType(ImageDimension, "0") }),
        DefaultTransformParameter({ "GridSpacing", ParameterValuesType(ImageDimension, "1") }),
        DefaultTransformParameter({ "GridOrigin", ParameterValuesType(ImageDimension, "0") }),
        DefaultTransformParameter(
          { "GridDirection",
            elx::Conversion::ToVectorOfStrings(itk::Matrix<int, ImageDimension, ImageDimension>::GetIdentity()) }),
        // Non-default parameters in alphabetic order:
        NonDefaultTransformParameter({ "GridSize", ParameterValuesType(2, std::to_string(gridValueSize)) }),
        NonDefaultTransformParameter({ "NumberOfParameters", { std::to_string(bsplineTransformParameters.size()) } }),
        NonDefaultTransformParameter({ "Transform", { "BSplineTransform" } }),
        NonDefaultTransformParameter(
          { "TransformParameters", elx::Conversion::ToVectorOfStrings(bsplineTransformParameters) }) });

    using ImageDomainType = ImageDomain<ImageDimension>;

    // ITK's RecursiveSeparableImageFilter "requires a minimum of four pixels along the dimension to be processed", at
    // https://github.com/InsightSoftwareConsortium/ITK/blob/v5.3.0/Modules/Filtering/ImageFilterBase/include/itkRecursiveSeparableImageFilter.hxx#L226
    enum
    {
      smallImageSizeValue = 8
    };

    const ImageDomainType simpleImageDomain{
      ImageDomainType::SizeType::Filled(smallImageSizeValue),
    };

    const auto createRandomImageDomain = [&randomNumberEngine] {
      const auto createRandomDirection = [&randomNumberEngine] {
        const auto randomRotation = std::uniform_real_distribution<>{ -M_PI / 8, M_PI / 8 }(randomNumberEngine);
        const auto cosRandomRotation = std::cos(randomRotation);
        const auto sinRandomRotation = std::sin(randomRotation);
        const itk::SpacePrecisionType randomDirectionMatrix[][2] = { { cosRandomRotation, sinRandomRotation },
                                                                     { -sinRandomRotation, cosRandomRotation } };
        return ImageDomainType::DirectionType{ randomDirectionMatrix };
      };
      const auto createRandomIndex = [&randomNumberEngine] {
        ImageDomainType::IndexType randomIndex{};
        std::generate(randomIndex.begin(), randomIndex.end(), [&randomNumberEngine] {
          return std::uniform_int_distribution<itk::IndexValueType>{ -1, 2 }(randomNumberEngine);
        });
        return randomIndex;
      };
      const auto createRandomImageSize = [&randomNumberEngine] {
        ImageDomainType::SizeType randomImageSize{};
        std::generate(randomImageSize.begin(), randomImageSize.end(), [&randomNumberEngine] {
          return std::uniform_int_distribution<itk::SizeValueType>{ smallImageSizeValue,
                                                                    (3 * smallImageSizeValue) / 2 }(randomNumberEngine);
        });
        return randomImageSize;
      };
      const auto createRandomSpacing = [&randomNumberEngine] {
        ImageDomainType::SpacingType randomSpacing{};
        std::generate(randomSpacing.begin(), randomSpacing.end(), [&randomNumberEngine] {
          return std::uniform_real_distribution<itk::SpacePrecisionType>{ 0.75, 1.5 }(randomNumberEngine);
        });
        return randomSpacing;
      };
      const auto createRandomPoint = [&randomNumberEngine] {
        ImageDomainType::PointType randomPoint{};
        std::generate(randomPoint.begin(), randomPoint.end(), [&randomNumberEngine] {
          return std::uniform_real_distribution<itk::SpacePrecisionType>{ -2, 2 }(randomNumberEngine);
        });
        return randomPoint;
      };

      return ImageDomainType{ createRandomDirection(),
                              createRandomIndex(),
                              createRandomImageSize(),
                              createRandomSpacing(),
                              createRandomPoint() };
    };

    Expect_equal_output_SetInitialTransformParameterObject_and_Transformix_SetTransformParameterObject(
      { translationTransformParameterMap }, simpleImageDomain, simpleImageDomain);
    Expect_equal_output_SetInitialTransformParameterObject_and_Transformix_SetTransformParameterObject(
      { translationTransformParameterMap }, createRandomImageDomain(), createRandomImageDomain());
    Expect_equal_output_SetInitialTransformParameterObject_and_Transformix_SetTransformParameterObject(
      { translationTransformParameterMap, bsplineTransformParameterMap },
      createRandomImageDomain(),
      createRandomImageDomain());
  }
}


GTEST_TEST(itkElastixRegistrationMethod, InitialTransformParameterFileLinkToTransformFile)
{
  using PixelType = float;
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  using RegistrationMethodType = ElastixRegistrationMethodType<ImageType>;

  const OffsetType initialTranslation{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = CreateImage<PixelType>(imageSize);

  const auto createRegistration = [fixedImage](const std::string & initialTransformParameterFileName) {
    const auto registration = CheckNew<RegistrationMethodType>();
    registration->SetFixedImage(fixedImage);
    registration->SetInitialTransformParameterFileName(GetDataDirectoryPath() + "/Translation(1,-2)/" +
                                                       initialTransformParameterFileName);
    registration->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                             { "ImageSampler", "Full" },
                                                             { "MaximumNumberOfIterations", "2" },
                                                             { "Metric", "AdvancedNormalizedCorrelation" },
                                                             { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                             { "Transform", "TranslationTransform" } }));
    return registration;
  };

  const auto registration1 = createRegistration("TransformParameters.txt");

  for (const auto transformParameterFileName :
       { "TransformParameters-link-to-ITK-tfm-file.txt",
         "TransformParameters-link-to-ITK-HDF5-file.txt",
         "TransformParameters-link-to-file-with-special-chars-in-path-name.txt" })
  {
    const auto registration2 = createRegistration(transformParameterFileName);

    for (const auto index :
         itk::ImageRegionIndexRange<ImageDimension>(itk::ImageRegion<ImageDimension>({ 0, -2 }, { 2, 3 })))
    {
      movingImage->FillBuffer(0);
      FillImageRegion(*movingImage, fixedImageRegionIndex + ConvertIndexToOffset(index), regionSize);

      const auto updateAndRetrieveTransformParameterMap = [movingImage](RegistrationMethodType & registration) {
        registration.SetMovingImage(movingImage);
        registration.Update();
        const elx::ParameterObject & transformParameterObject =
          DerefRawPointer(registration.GetTransformParameterObject());
        const auto & transformParameterMaps = transformParameterObject.GetParameterMaps();
        EXPECT_EQ(transformParameterMaps.size(), 1);
        return Front(transformParameterMaps);
      };

      const auto transformParameterMap1 = updateAndRetrieveTransformParameterMap(*registration1);
      const auto transformParameterMap2 = updateAndRetrieveTransformParameterMap(*registration2);

      ASSERT_EQ(transformParameterMap1.size(), transformParameterMap2.size());
      for (const auto & transformParameter : transformParameterMap1)
      {
        const auto found = transformParameterMap2.find(transformParameter.first);
        ASSERT_NE(found, transformParameterMap2.end());

        if (transformParameter.first == "InitialTransformParametersFileName")
        {
          ASSERT_NE(*found, transformParameter);
        }
        else
        {
          ASSERT_EQ(*found, transformParameter);
        }
      }
    }
  }
}


GTEST_TEST(itkElastixRegistrationMethod, GetCombinationTransform)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<ImageType::PixelType>(itk::Size<ImageDimension>{ 5, 6 });

  struct NameAndItkTransform
  {
    const char *                                                    name;
    itk::Transform<double, ImageDimension, ImageDimension>::Pointer itkTransform;
  };

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  registration.SetFixedImage(image);
  registration.SetMovingImage(image);

  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  for (const bool useInitialTransform : { false, true })
  {
    registration.SetInitialTransformParameterFileName(
      useInitialTransform ? (GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt") : "");

    const std::string outputSubdirectoryPath =
      rootOutputDirectoryPath + "/" + (useInitialTransform ? "InitialTranslation(1,-2)" : "NoInitialTransform");
    itk::FileTools::CreateDirectory(outputSubdirectoryPath);

    for (const auto nameAndItkTransform :
         { NameAndItkTransform{ "AffineTransform", itk::AffineTransform<double, ImageDimension>::New() },
           NameAndItkTransform{ "BSplineTransform", itk::BSplineTransform<double, ImageDimension>::New() },
           NameAndItkTransform{ "EulerTransform", itk::Euler2DTransform<>::New() },
           NameAndItkTransform{ "RecursiveBSplineTransform", itk::BSplineTransform<double, ImageDimension>::New() },
           NameAndItkTransform{ "SimilarityTransform", itk::Similarity2DTransform<>::New() },
           NameAndItkTransform{ "TranslationTransform", itk::TranslationTransform<double, ImageDimension>::New() } })
    {
      const auto & expectedItkTransform = *(nameAndItkTransform.itkTransform);
      const auto   expectedNumberOfFixedParameters = expectedItkTransform.GetFixedParameters().size();

      registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                              { "AutomaticTransformInitialization", "false" },
                                                              { "ImageSampler", "Full" },
                                                              { "MaximumNumberOfIterations", "0" },
                                                              { "Metric", "AdvancedNormalizedCorrelation" },
                                                              { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                              { "Transform", nameAndItkTransform.name } }));
      registration.Update();

      using CompositeTransformType = itk::CompositeTransform<double, ImageDimension>;
      const auto combinationTransform = registration.GetCombinationTransform();

      EXPECT_NE(combinationTransform, nullptr);
    }
  }
}


GTEST_TEST(itkElastixRegistrationMethod, GetNumberOfTransforms)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<ImageType::PixelType>(itk::Size<ImageDimension>{ 5, 6 });

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  registration.SetFixedImage(image);
  registration.SetMovingImage(image);

  for (const bool useInitialTransform : { false, true })
  {
    registration.SetInitialTransformParameterFileName(
      useInitialTransform ? (GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt") : "");

    registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                            { "AutomaticTransformInitialization", "false" },
                                                            { "ImageSampler", "Full" },
                                                            { "MaximumNumberOfIterations", "0" },
                                                            { "Metric", "AdvancedNormalizedCorrelation" },
                                                            { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                            { "Transform", "BSplineTransform" } }));
    registration.Update();
    EXPECT_EQ(registration.GetNumberOfTransforms(), useInitialTransform ? 2 : 1);
  }
}


GTEST_TEST(itkElastixRegistrationMethod, GetNthTransform)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<ImageType::PixelType>(itk::Size<ImageDimension>{ 5, 6 });

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  registration.SetFixedImage(image);
  registration.SetMovingImage(image);

  for (const bool useInitialTransform : { false, true })
  {
    registration.SetInitialTransformParameterFileName(
      useInitialTransform ? (GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt") : "");

    const std::string nameOfLastTransform = "BSplineTransform";
    registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                            { "AutomaticTransformInitialization", "false" },
                                                            { "ImageSampler", "Full" },
                                                            { "MaximumNumberOfIterations", "0" },
                                                            { "Metric", "AdvancedNormalizedCorrelation" },
                                                            { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                            { "Transform", nameOfLastTransform } }));
    registration.Update();

    const unsigned int numberOfTransforms{ useInitialTransform ? 2U : 1U };

    for (unsigned int n{ 0 }; n < numberOfTransforms; ++n)
    {
      EXPECT_NE(registration.GetNthTransform(n), nullptr);
    }
  }
}


GTEST_TEST(itkElastixRegistrationMethod, ConvertToItkTransform)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<ImageType::PixelType>(itk::Size<ImageDimension>{ 5, 6 });

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  registration.SetFixedImage(image);
  registration.SetMovingImage(image);

  for (const bool useInitialTransform : { false, true })
  {
    registration.SetInitialTransformParameterFileName(
      useInitialTransform ? (GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt") : "");

    const std::string nameOfLastTransform = "BSplineTransform";
    registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                            { "AutomaticTransformInitialization", "false" },
                                                            { "ImageSampler", "Full" },
                                                            { "MaximumNumberOfIterations", "0" },
                                                            { "Metric", "AdvancedNormalizedCorrelation" },
                                                            { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                            { "Transform", nameOfLastTransform } }));
    registration.Update();

    const unsigned int numberOfTransforms{ useInitialTransform ? 2U : 1U };

    for (unsigned int n{ 0 }; n < numberOfTransforms; ++n)
    {
      // TODO Check result
      ElastixRegistrationMethodType<ImageType>::ConvertToItkTransform(*registration.GetNthTransform(n));
    }
  }
}

GTEST_TEST(itkElastixRegistrationMethod, WriteCompositeTransform)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<ImageType::PixelType>(itk::Size<ImageDimension>{ 5, 6 });

  struct NameAndItkTransform
  {
    const char *                                                    name;
    itk::Transform<double, ImageDimension, ImageDimension>::Pointer itkTransform;
  };

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  registration.SetFixedImage(image);
  registration.SetMovingImage(image);

  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  for (const bool useInitialTransform : { false, true })
  {
    registration.SetInitialTransformParameterFileName(
      useInitialTransform ? (GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt") : "");

    const std::string outputSubdirectoryPath =
      rootOutputDirectoryPath + "/" + (useInitialTransform ? "InitialTranslation(1,-2)" : "NoInitialTransform");
    itk::FileTools::CreateDirectory(outputSubdirectoryPath);

    for (const auto nameAndItkTransform :
         { NameAndItkTransform{ "AffineTransform", itk::AffineTransform<double, ImageDimension>::New() },
           NameAndItkTransform{ "BSplineTransform", itk::BSplineTransform<double, ImageDimension>::New() },
           NameAndItkTransform{ "EulerTransform", itk::Euler2DTransform<>::New() },
           NameAndItkTransform{ "RecursiveBSplineTransform", itk::BSplineTransform<double, ImageDimension>::New() },
           NameAndItkTransform{ "SimilarityTransform", itk::Similarity2DTransform<>::New() },
           NameAndItkTransform{ "TranslationTransform", itk::TranslationTransform<double, ImageDimension>::New() } })
    {
      for (const std::string fileNameExtension : { "", "h5", "tfm" })
      {
        const std::string outputDirectoryPath =
          outputSubdirectoryPath + "/" + nameAndItkTransform.name + fileNameExtension;
        itk::FileTools::CreateDirectory(outputDirectoryPath);

        registration.SetOutputDirectory(outputDirectoryPath);

        registration.SetParameterObject(
          CreateParameterObject({ // Parameters in alphabetic order:
                                  { "AutomaticTransformInitialization", "false" },
                                  { "ImageSampler", "Full" },
                                  { "ITKTransformOutputFileNameExtension", fileNameExtension },
                                  { "MaximumNumberOfIterations", "0" },
                                  { "Metric", "AdvancedNormalizedCorrelation" },
                                  { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                  { "Transform", nameAndItkTransform.name },
                                  { "WriteITKCompositeTransform", "true" } }));
        registration.Update();

        if (!fileNameExtension.empty())
        {
          const auto & expectedItkTransform = *(nameAndItkTransform.itkTransform);
          const auto   expectedNumberOfFixedParameters = expectedItkTransform.GetFixedParameters().size();

          const itk::TransformBase::ConstPointer singleTransform =
            elx::TransformIO::Read(outputDirectoryPath + "/TransformParameters.0." + fileNameExtension);

          using CompositeTransformType = itk::CompositeTransform<double, ImageDimension>;

          const itk::TransformBase::Pointer compositeTransform =
            elx::TransformIO::Read(outputDirectoryPath + "/TransformParameters.0-Composite." + fileNameExtension);
          const auto & transformQueue =
            DerefRawPointer(dynamic_cast<const CompositeTransformType *>(compositeTransform.GetPointer()))
              .GetTransformQueue();

          ASSERT_EQ(transformQueue.size(), useInitialTransform ? 2 : 1);

          const itk::TransformBase * const frontTransform = transformQueue.front();

          for (const auto actualTransformPtr : { singleTransform.GetPointer(), frontTransform })
          {
            const itk::TransformBase & actualTransform = DerefRawPointer(actualTransformPtr);

            EXPECT_EQ(typeid(actualTransform), typeid(expectedItkTransform));
            EXPECT_EQ(actualTransform.GetParameters(), expectedItkTransform.GetParameters());

            // Note that the actual values of the FixedParameters may not be exactly like the expected
            // default-constructed transform.
            EXPECT_EQ(actualTransform.GetFixedParameters().size(), expectedNumberOfFixedParameters);
          }
          EXPECT_EQ(singleTransform->GetFixedParameters(), frontTransform->GetFixedParameters());

          if (useInitialTransform)
          {
            // Expect that the back of the transformQueue has a translation according to the
            // InitialTransformParameterFileName.
            const auto & backTransform = DerefSmartPointer(transformQueue.back());
            const auto & translationTransform =
              DerefRawPointer(dynamic_cast<const itk::TranslationTransform<double, ImageDimension> *>(&backTransform));
            EXPECT_EQ(translationTransform.GetOffset(), itk::MakeVector(1.0, -2.0));
          }
        }
      }
    }
  }
}


GTEST_TEST(itkElastixRegistrationMethod, WriteBSplineTransformToItkFileFormat)
{
  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  Test_WriteBSplineTransformToItkFileFormat<2, 1>(rootOutputDirectoryPath);
  Test_WriteBSplineTransformToItkFileFormat<2, 2>(rootOutputDirectoryPath);
  Test_WriteBSplineTransformToItkFileFormat<2, 3>(rootOutputDirectoryPath);
  Test_WriteBSplineTransformToItkFileFormat<3, 1>(rootOutputDirectoryPath);
  Test_WriteBSplineTransformToItkFileFormat<3, 2>(rootOutputDirectoryPath);
  Test_WriteBSplineTransformToItkFileFormat<3, 3>(rootOutputDirectoryPath);
}


// Tests registering two small (8x8) binary images, which are translated with respect to each other.
GTEST_TEST(itkElastixRegistrationMethod, EulerTranslation2D)
{
  using PixelType = float;
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const auto imageSizeValue = 8;
  const auto imageSize = SizeType::Filled(imageSizeValue);
  const auto fixedImageRegionIndex = IndexType::Filled(imageSizeValue / 2 - 1);

  const auto setPixelsOfSquareRegion = [](ImageType & image, const IndexType & regionIndex) {
    // Set a different value to each of the pixels of a little square region, to ensure that no rotation is assumed.
    const itk::ImageRegionRange<ImageType> imageRegionRange{ image, { regionIndex, SizeType::Filled(2) } };
    std::iota(std::begin(imageRegionRange), std::end(imageRegionRange), 1);
  };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  setPixelsOfSquareRegion(*fixedImage, fixedImageRegionIndex);

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  registration.SetFixedImage(fixedImage);
  registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                          { "AutomaticTransformInitialization", "false" },
                                                          { "ImageSampler", "Full" },
                                                          { "MaximumNumberOfIterations", "2" },
                                                          { "Metric", "AdvancedNormalizedCorrelation" },
                                                          { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                          { "Transform", "EulerTransform" } }));

  const auto movingImage = CreateImage<PixelType>(imageSize);

  // Test translation for each direction from (-1, -1) to (1, 1).
  for (const auto & index : itk::ZeroBasedIndexRange<ImageDimension>(SizeType::Filled(3)))
  {
    movingImage->FillBuffer(0);
    const OffsetType translation = index - IndexType::Filled(1);
    setPixelsOfSquareRegion(*movingImage, fixedImageRegionIndex + translation);

    registration.SetMovingImage(movingImage);
    registration.Update();

    const auto transformParameters = GetTransformParametersFromFilter(registration);
    ASSERT_EQ(transformParameters.size(), 3);

    // The detected rotation angle is expected to be close to zero.
    // (Absolute angle values of up to 3.77027e-06 were encountered, which seems acceptable.)
    const auto rotationAngle = transformParameters[0];
    EXPECT_LT(std::abs(rotationAngle), 1e-5);

    for (unsigned i{}; i <= 1; ++i)
    {
      EXPECT_EQ(std::round(transformParameters[i + 1]), translation[i]);
    }
  }
}


// Tests registering two images which are rotated with respect to each other.
GTEST_TEST(itkElastixRegistrationMethod, EulerDiscRotation2D)
{
  using PixelType = float;
  enum
  {
    ImageDimension = 2,
    imageSizeValue = 128
  };

  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using RegionType = ImageType::RegionType;

  const auto imageSize = SizeType::Filled(imageSizeValue);
  const auto setPixelsOfDisc = [imageSize](ImageType & image, const double rotationAngle) {
    for (const auto & index : itk::ZeroBasedIndexRange<ImageDimension>{ imageSize })
    {
      std::array<double, ImageDimension> offset;

      for (int i{}; i < ImageDimension; ++i)
      {
        offset[i] = index[i] - ((imageSizeValue - 1) / 2.0);
      }

      constexpr auto radius = (imageSizeValue / 2.0) - 2.0;

      if (std::inner_product(offset.begin(), offset.end(), offset.begin(), 0.0) < (radius * radius))
      {
        const auto directionAngle = std::atan2(offset[1], offset[0]);

        // Estimate the turn (between 0 and 1), rotated according to the specified rotation angle.
        const auto rotatedDirectionTurn =
          std::fmod(std::fmod((directionAngle + rotationAngle) / (2.0 * M_PI), 1.0) + 1.0, 1.0);

        // Multiplication by 64 may be useful for integer pixel types.
        image.SetPixel(index, static_cast<PixelType>(64.0 * rotatedDirectionTurn));
      }
    }
  };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  setPixelsOfDisc(*fixedImage, 0.0);

  const auto movingImage = CreateImage<PixelType>(imageSize);

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  registration.SetFixedImage(fixedImage);
  registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                          { "AutomaticTransformInitialization", "false" },
                                                          { "AutomaticScalesEstimation", "true" },
                                                          { "ImageSampler", "Full" },
                                                          { "MaximumNumberOfIterations", "16" },
                                                          { "Metric", "AdvancedNormalizedCorrelation" },
                                                          { "Optimizer", "RegularStepGradientDescent" },
                                                          { "Transform", "EulerTransform" },
                                                          { "WriteResultImage", "false" } }));

  for (const auto degree : { -2, 0, 1, 30 })
  {
    constexpr auto radiansPerDegree = M_PI / 180.0;

    setPixelsOfDisc(*movingImage, degree * radiansPerDegree);
    registration.SetMovingImage(movingImage);
    registration.Update();

    const auto transformParameters = GetTransformParametersFromFilter(registration);
    ASSERT_EQ(transformParameters.size(), 3);

    EXPECT_EQ(std::round(transformParameters[0] / radiansPerDegree), -degree); // rotation angle
    EXPECT_EQ(std::round(transformParameters[1]), 0.0);                        // translation X
    EXPECT_EQ(std::round(transformParameters[2]), 0.0);                        // translation Y
  }
}


// Checks a minimum size moving image having the same pixel type as any of the supported internal pixel types.
GTEST_TEST(itkElastixRegistrationMethod, CheckMinimumMovingImageHavingInternalPixelType)
{
  elx::ForEachSupportedImageType([](const auto elxTypedef) {
    using ElxTypedef = decltype(elxTypedef);
    using ImageType = typename ElxTypedef::MovingImageType;
    constexpr auto ImageDimension = ElxTypedef::MovingDimension;

    using PixelType = typename ImageType::PixelType;

    const auto imageSize = itk::Size<ElxTypedef::MovingDimension>::Filled(minimumImageSizeValue);
    const ImageDomain<ElxTypedef::MovingDimension> imageDomain(imageSize);

    elx::DefaultConstruct<ImageType> fixedImage{};
    imageDomain.ToImage(fixedImage);
    fixedImage.Allocate(true);

    elx::DefaultConstruct<ImageType> movingImage{};
    imageDomain.ToImage(movingImage);
    movingImage.Allocate(true);

    // Some "extreme" values to test if each of them is preserved during the transformation.
    const std::array pixelValues{ PixelType{},
                                  PixelType{ 1 },
                                  std::numeric_limits<PixelType>::lowest(),
                                  std::numeric_limits<PixelType>::min(),
                                  PixelType{ std::numeric_limits<PixelType>::max() - 1 },
                                  std::numeric_limits<PixelType>::max() };
    std::copy(pixelValues.cbegin(), pixelValues.cend(), itk::ImageBufferRange<ImageType>(movingImage).begin());

    // A dummy registration (that does not do any optimization).
    elx::DefaultConstruct<itk::ElastixRegistrationMethod<ImageType, ImageType>> registration{};

    registration.SetParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "AutomaticParameterEstimation", { "false" } },
                              { "FixedInternalImagePixelType", { ElxTypedef::FixedPixelTypeString } },
                              { "MovingInternalImagePixelType", { ElxTypedef::MovingPixelTypeString } },
                              { "ImageSampler", "Full" },
                              { "MaximumNumberOfIterations", "0" },
                              { "Metric", "AdvancedNormalizedCorrelation" },
                              { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                              { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                              { "Transform", "TranslationTransform" } }));
    registration.SetFixedImage(&fixedImage);
    registration.SetMovingImage(&movingImage);
    registration.Update();

    EXPECT_EQ(DerefRawPointer(registration.GetOutput()), movingImage);
  });
}


// Checks a zero-filled moving image with a random domain, having the same pixel type as any of the supported internal
// pixel types.
GTEST_TEST(itkElastixRegistrationMethod, CheckZeroFilledMovingImageWithRandomDomainHavingInternalPixelType)
{
  std::mt19937 randomNumberEngine{};

  elx::ForEachSupportedImageType([&randomNumberEngine](const auto elxTypedef) {
    using ElxTypedef = decltype(elxTypedef);
    using ImageType = typename ElxTypedef::MovingImageType;
    constexpr auto ImageDimension = ElxTypedef::MovingDimension;

    using PixelType = typename ImageType::PixelType;

    auto imageDomain = CreateRandomImageDomain<ElxTypedef::MovingDimension>(randomNumberEngine);

    // Reset index to avoid "FixedImageRegion does not overlap the fixed image buffered region" exceptions from
    // itk::ImageToImageMetric::Initialize()
    imageDomain.index = {};

    // Create an image with values 1, 2, 3, ... N. We could have used arbitrary pixel values instead.
    const auto fixedImage = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);

    elx::DefaultConstruct<ImageType> movingImage{};
    imageDomain.ToImage(movingImage);
    movingImage.Allocate(true);

    // A dummy registration (that does not do any optimization).
    elx::DefaultConstruct<itk::ElastixRegistrationMethod<ImageType, ImageType>> registration{};

    registration.SetParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "AutomaticParameterEstimation", { "false" } },
                              { "FixedInternalImagePixelType", { ElxTypedef::FixedPixelTypeString } },
                              { "MovingInternalImagePixelType", { ElxTypedef::MovingPixelTypeString } },
                              { "ImageSampler", "Full" },
                              { "MaximumNumberOfIterations", "0" },
                              { "Metric", "AdvancedNormalizedCorrelation" },
                              { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                              { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                              { "Transform", "TranslationTransform" } }));
    registration.SetFixedImage(fixedImage);
    registration.SetMovingImage(&movingImage);
    registration.Update();

    EXPECT_EQ(DerefRawPointer(registration.GetOutput()), movingImage);
  });
}


// Checks a minimum size moving image using any supported internal pixel type (which may be different from the input
// pixel type).
GTEST_TEST(itkElastixRegistrationMethod, CheckMinimumMovingImageUsingAnyInternalPixelType)
{
  const auto check = [](const auto inputPixelTypeHolder) {
    elx::ForEachSupportedImageType([](const auto elxTypedef) {
      using ElxTypedef = decltype(elxTypedef);
      using InputPixelType = typename decltype(inputPixelTypeHolder)::Type;
      using InputImageType = itk::Image<InputPixelType, ElxTypedef::MovingDimension>;

      const ImageDomain<ElxTypedef::MovingDimension> imageDomain(
        itk::Size<ElxTypedef::MovingDimension>::Filled(minimumImageSizeValue));

      elx::DefaultConstruct<InputImageType> fixedImage{};
      imageDomain.ToImage(fixedImage);
      fixedImage.Allocate(true);

      const auto movingImage = CreateImageFilledWithSequenceOfNaturalNumbers<InputPixelType>(imageDomain);

      // A dummy registration (that does not do any optimization).
      elx::DefaultConstruct<itk::ElastixRegistrationMethod<InputImageType, InputImageType>> registration{};

      registration.SetParameterObject(
        CreateParameterObject({ // Parameters in alphabetic order:
                                { "AutomaticParameterEstimation", { "false" } },
                                { "FixedInternalImagePixelType", { ElxTypedef::FixedPixelTypeString } },
                                { "MovingInternalImagePixelType", { ElxTypedef::MovingPixelTypeString } },
                                { "ImageSampler", "Full" },
                                { "MaximumNumberOfIterations", "0" },
                                { "Metric", "AdvancedNormalizedCorrelation" },
                                { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                                { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                                { "Transform", "TranslationTransform" } }));
      registration.SetFixedImage(&fixedImage);
      registration.SetMovingImage(movingImage);
      registration.Update();

      EXPECT_EQ(DerefRawPointer(registration.GetOutput()), DerefSmartPointer(movingImage));
    });
  };

  check(TypeHolder<char>{});
  check(TypeHolder<short>{});
  check(TypeHolder<float>{});
  check(TypeHolder<double>{});
}


// Checks a zero-filled moving image with a random domain, using any supported internal pixel type (which may be
// different from the input pixel type).
GTEST_TEST(itkElastixRegistrationMethod, CheckZeroFilledMovingImageWithRandomDomainUsingAnyInternalPixelType)
{
  std::mt19937 randomNumberEngine{};

  const auto check = [&randomNumberEngine](const auto inputPixelTypeHolder) {
    elx::ForEachSupportedImageType([&randomNumberEngine](const auto elxTypedef) {
      using ElxTypedef = decltype(elxTypedef);
      using InputPixelType = typename decltype(inputPixelTypeHolder)::Type;
      using InputImageType = itk::Image<InputPixelType, ElxTypedef::MovingDimension>;

      auto imageDomain = CreateRandomImageDomain<ElxTypedef::MovingDimension>(randomNumberEngine);

      // Reset index to avoid "FixedImageRegion does not overlap the fixed image buffered region" exceptions from
      // itk::ImageToImageMetric::Initialize()
      imageDomain.index = {};

      // Create an image with values 1, 2, 3, ... N. We could have used arbitrary pixel values instead.
      const auto fixedImage = CreateImageFilledWithSequenceOfNaturalNumbers<InputPixelType>(imageDomain);

      elx::DefaultConstruct<InputImageType> movingImage{};
      imageDomain.ToImage(movingImage);
      movingImage.Allocate(true);

      // A dummy registration (that does not do any optimization).
      elx::DefaultConstruct<itk::ElastixRegistrationMethod<InputImageType, InputImageType>> registration{};

      registration.SetParameterObject(
        CreateParameterObject({ // Parameters in alphabetic order:
                                { "AutomaticParameterEstimation", { "false" } },
                                { "FixedInternalImagePixelType", { ElxTypedef::FixedPixelTypeString } },
                                { "MovingInternalImagePixelType", { ElxTypedef::MovingPixelTypeString } },
                                { "ImageSampler", "Full" },
                                { "MaximumNumberOfIterations", "0" },
                                { "Metric", "AdvancedNormalizedCorrelation" },
                                { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                                { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                                { "Transform", "TranslationTransform" } }));
      registration.SetFixedImage(fixedImage);
      registration.SetMovingImage(&movingImage);
      registration.Update();

      EXPECT_EQ(DerefRawPointer(registration.GetOutput()), movingImage);
    });
  };

  check(TypeHolder<char>{});
  check(TypeHolder<short>{});
  check(TypeHolder<float>{});
  check(TypeHolder<double>{});
}
