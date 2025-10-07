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
#include <itkDisplacementFieldTransform.h>
#include <itkEuler2DTransform.h>
#include <itkEuler3DTransform.h>
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
using elx::CoreMainGTestUtilities::DerefSmartPointer;
using elx::CoreMainGTestUtilities::FillImageRegion;
using elx::CoreMainGTestUtilities::FillImageRegionWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::Front;
using elx::CoreMainGTestUtilities::GetCurrentBinaryDirectoryPath;
using elx::CoreMainGTestUtilities::GetDataDirectoryPath;
using elx::CoreMainGTestUtilities::GetNameOfTest;
using elx::CoreMainGTestUtilities::GetTransformParametersFromFilter;
using elx::CoreMainGTestUtilities::ImageDomain;
using elx::CoreMainGTestUtilities::TypeHolder;
using elx::CoreMainGTestUtilities::minimumImageSizeValue;
using elx::GTestUtilities::MakeMergedMap;

using itk::Deref;
using itk::Statistics::MersenneTwisterRandomVariateGenerator;


template <typename TImage>
using ElastixRegistrationMethodType = itk::ElastixRegistrationMethod<TImage, TImage>;

namespace
{
double
ConvertDegreesToRadians(double degrees)
{
  return degrees * M_PI / 180.0;
};

double
ConvertRadiansToDegrees(double radians)
{
  return radians * 180.0 / M_PI;
};


auto
CreateRotationMatrix(const double radians)
{
  const double cosAngle = std::cos(radians);
  const double sinAngle = std::sin(radians);
  return itk::Matrix<double, 2, 2>({ { cosAngle, -sinAngle }, { sinAngle, cosAngle } });
};


// Adjusts the origin of the image in order to make the image center coincide with the world center.
template <unsigned int VDimension>
void
AdjustOriginToMakeImageCenterCoincideWithWorldCenter(itk::ImageBase<VDimension> & image)
{
  const auto [imageIndex, imageSize] = image.GetBufferedRegion();

  itk::ContinuousIndex<double, VDimension> centerIndex = imageIndex;

  for (unsigned int i = 0; i < VDimension; ++i)
  {
    centerIndex[i] += (imageSize[i] - 1) / 2.0;
  }

  const auto centerPoint = image.template TransformContinuousIndexToPhysicalPoint<double>(centerIndex);
  image.SetOrigin(image.GetOrigin() - centerPoint.GetVectorFromOrigin());
}

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
                       ParameterType{ "InitialTransformParameterFileName", { "NoInitialTransform" } },
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
    ParameterType{ "InitialTransformParameterFileName", { "NoInitialTransform" } },
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
  fixedImage.AllocateInitialized();

  elx::DefaultConstruct<ImageType> movingImage{};
  movingImageDomain.ToImage(movingImage);
  movingImage.AllocateInitialized();
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
    DefaultRegistrationParameter({ "InitialTransformParameterFileName", { "NoInitialTransform" } }),
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

  const auto & transformixOutput = Deref(transformix.GetOutput());

  // Sanity checks, checking that our test is non-trivial.
  EXPECT_NE(transformixOutput, fixedImage);
  EXPECT_NE(transformixOutput, movingImage);

  const auto & actualRegistrationOutput = Deref(registration.GetOutput());
  EXPECT_EQ(actualRegistrationOutput, transformixOutput);
}


// Expects transformix to yield the same output image as an elastix registration, when transformix has the
// "TransformParameterObject" taken from the elastix registration result.
template <unsigned int VDimension>
void
Expect_equal_output_Transformix_SetTransformParameterObject_GetTransformParameterObject(
  const ParameterMapVectorType &  initialTransformParameterMaps,
  const ImageDomain<VDimension> & fixedImageDomain,
  const ImageDomain<VDimension> & movingImageDomain)
{
  using PixelType = float;
  using ImageType = itk::Image<PixelType, VDimension>;

  std::mt19937 randomNumberEngine{};

  const auto fillImageBufferRandomly = [&randomNumberEngine](ImageType & image) {
    const itk::ImageBufferRange<ImageType> imageBufferRange(image);

    std::generate(imageBufferRange.begin(), imageBufferRange.end(), [&randomNumberEngine] {
      return std::uniform_real_distribution<PixelType>{ PixelType{ 1 }, PixelType{ 2 } }(randomNumberEngine);
    });
  };

  itk::Size<VDimension> movingImageSize;
  std::iota(movingImageSize.begin(), movingImageSize.end(), 5U);

  elx::DefaultConstruct<ImageType> fixedImage{};
  fixedImageDomain.ToImage(fixedImage);
  fixedImage.AllocateInitialized();
  fillImageBufferRandomly(fixedImage);

  elx::DefaultConstruct<ImageType> movingImage{};
  movingImageDomain.ToImage(movingImage);
  movingImage.AllocateInitialized();
  fillImageBufferRandomly(movingImage);

  elx::DefaultConstruct<elx::ParameterObject> registrationParameterObject{};

  const ParameterMapType registrationParameterMap = CreateParameterMap({
    // Non-default parameters in alphabetic order:
    NonDefaultRegistrationParameter({ "ImageSampler", { "Full" } }), // required
    NonDefaultRegistrationParameter({ "MaximumNumberOfIterations", { "2" } }),
    NonDefaultRegistrationParameter({ "Metric", { "AdvancedNormalizedCorrelation" } }), // default ""
    NonDefaultRegistrationParameter({ "NumberOfResolutions", { "1" } }),
    NonDefaultRegistrationParameter({ "Optimizer", { "AdaptiveStochasticGradientDescent" } }), // default ""
    // RequiredRatioOfValidSamples as in the example in the elxMetricBase.h documentation. The FAQ even suggests 0.05:
    // https://github.com/SuperElastix/elastix/wiki/FAQ/702a35cf0f5e0cf797b531fcbe3297ff9a9f3a18#i-am-getting-the-error-message-too-many-samples-map-outside-moving-image-buffer-what-does-that-mean
    NonDefaultRegistrationParameter({ "RequiredRatioOfValidSamples", { "0.1" } }),
    NonDefaultRegistrationParameter({ "Transform", { "BSplineTransform" } }), // default ""
  });

  registrationParameterObject.SetParameterMap(registrationParameterMap);

  elx::DefaultConstruct<elx::ParameterObject> initialTransformParameterObject{};
  initialTransformParameterObject.SetParameterMaps(initialTransformParameterMaps);

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  registration.SetParameterObject(&registrationParameterObject);
  registration.SetInitialTransformParameterObject(&initialTransformParameterObject);
  registration.SetFixedImage(&fixedImage);
  registration.SetMovingImage(&movingImage);
  registration.Update();

  elx::DefaultConstruct<itk::TransformixFilter<ImageType>> transformix{};
  transformix.SetTransformParameterObject(registration.GetTransformParameterObject());
  transformix.SetMovingImage(&movingImage);
  transformix.Update();

  const auto & transformixOutput = Deref(transformix.GetOutput());

  // Sanity checks, checking that our test is non-trivial.
  EXPECT_NE(movingImage, fixedImage);
  EXPECT_NE(transformixOutput, fixedImage);
  EXPECT_NE(transformixOutput, movingImage);

  const auto & actualRegistrationOutput = Deref(registration.GetOutput());
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
  static constexpr auto expectedNumberOfFixedParameters = NDimension * (NDimension + 3);
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


template <typename TParametersValueType, unsigned int VInputDimension, unsigned int VOutputDimension>
itk::SizeValueType
GetNumberOfTransforms(const itk::Transform<TParametersValueType, VInputDimension, VOutputDimension> & transform)
{
  if (const auto multiTransform =
        dynamic_cast<const itk::MultiTransform<TParametersValueType, VInputDimension, VOutputDimension> *>(&transform))
  {
    return multiTransform->GetNumberOfTransforms();
  }
  return 1;
};


template <typename T>
auto
MakeVectorContainer(std::vector<T> stdVector)
{
  const auto vectorContainer = itk::VectorContainer<itk::SizeValueType, T>::New();
  vectorContainer->CastToSTLContainer() = std::move(stdVector);
  return vectorContainer;
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
  static constexpr auto ImageDimension = 2U;
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
  static constexpr auto ImageDimension = 2U;
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
  static constexpr auto ImageDimension = 2U;
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


// Tests registering two images, which have the same pixel data, but different origin.
GTEST_TEST(itkElastixRegistrationMethod, TranslationOrigin)
{
  static constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using PointType = ImageType::PointType;
  using SizeType = itk::Size<ImageDimension>;

  const auto testImage = CreateImage<PixelType>(SizeType{ 5, 6 });
  FillImageRegion(*testImage, { 1, 3 }, SizeType::Filled(2));

  const PointType points[] = { PointType(),
                               itk::MakeFilled<PointType>(-0.5),
                               itk::MakePoint(0.0, 1.0),
                               itk::MakePoint(1.0, 0.0),
                               itk::MakePoint(0.25, 0.75) };

  for (const auto originOfFixedImage : points)
  {
    elx::DefaultConstruct<ImageType> fixedImage{};
    fixedImage.Graft(testImage);
    fixedImage.SetOrigin(originOfFixedImage);

    for (const auto originOfMovingImage : points)
    {
      elx::DefaultConstruct<ImageType> movingImage{};
      movingImage.Graft(testImage);
      movingImage.SetOrigin(originOfMovingImage);

      elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
      registration.SetFixedImage(&fixedImage);
      registration.SetMovingImage(&movingImage);
      registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                              { "ImageSampler", "Full" },
                                                              { "MaximumNumberOfIterations", "50" },
                                                              { "Metric", "AdvancedNormalizedCorrelation" },
                                                              { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                              { "Transform", "TranslationTransform" } }));
      registration.Update();
      const auto transformParameters = GetTransformParametersFromFilter(registration);

      ASSERT_EQ(transformParameters.size(), ImageDimension);

      const auto expectedTransformParameters = originOfMovingImage - originOfFixedImage;

      SCOPED_TRACE(testing::Message() << "Origins: " << originOfMovingImage << " and " << originOfFixedImage);

      for (unsigned int i = 0; i < ImageDimension; ++i)
      {
        // The argument for `abs_error` is chosen just large enough to pass the tests.
        EXPECT_NEAR(transformParameters[i], expectedTransformParameters[i], 2.0 * DBL_EPSILON) << " with i = " << i;
      }
    }
  }
}


// Tests "MaximumNumberOfIterations" value "0"
GTEST_TEST(itkElastixRegistrationMethod, MaximumNumberOfIterationsZero)
{
  static constexpr auto ImageDimension = 2U;
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
  static constexpr auto ImageDimension = 2U;
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
  static constexpr auto ImageDimension = 2U;
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

    const auto &       output = Deref(registration.GetOutput());
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
  static constexpr auto ImageDimension = 2U;
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
           (useCustomResultImageName ? "CustomResultImageName" : "DefaultResultImageName");
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

  const auto fileNamePostFix = ".0.mhd";
  const auto expectedImage = itk::ReadImage<ImageType>(getOutputSubdirectoryPath(false) + "/result" + fileNamePostFix);
  const auto actualImage =
    itk::ReadImage<ImageType>(getOutputSubdirectoryPath(true) + '/' + customResultImageName + fileNamePostFix);

  ASSERT_NE(expectedImage, nullptr);
  ASSERT_NE(actualImage, nullptr);
  EXPECT_EQ(*actualImage, *expectedImage);
}


// Tests that the origin of the output image is equal to the origin of the fixed image (by default).
GTEST_TEST(itkElastixRegistrationMethod, OutputHasSameOriginAsFixedImage)
{
  static constexpr auto ImageDimension = 2U;
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

      const auto & output = Deref(registration.GetOutput());

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
  static constexpr auto ImageDimension = 2U;
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


GTEST_TEST(itkElastixRegistrationMethod, InitialTransformParameterFileWithInitialTransformParameterFile)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;

  const auto doDummyRegistration =
    [](const std::string & initialTransformParameterFileName) -> itk::SmartPointer<ImageType> {
    const ImageType::SizeType imageSize{ { 5, 6 } };

    std::mt19937 randomNumberEngine{};

    const auto fillImageBufferRandomly = [&randomNumberEngine](ImageType & image) {
      const itk::ImageBufferRange<ImageType> imageBufferRange(image);

      std::generate(imageBufferRange.begin(), imageBufferRange.end(), [&randomNumberEngine] {
        return std::uniform_real_distribution<PixelType>{ PixelType{ 1 }, PixelType{ 2 } }(randomNumberEngine);
      });
    };

    elx::DefaultConstruct<ImageType> fixedImage{};
    fixedImage.SetRegions(imageSize);
    fixedImage.AllocateInitialized();
    fillImageBufferRandomly(fixedImage);

    elx::DefaultConstruct<ImageType> movingImage{};
    movingImage.SetRegions(imageSize);
    movingImage.AllocateInitialized();
    fillImageBufferRandomly(movingImage);

    elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
    registration.SetFixedImage(&fixedImage);
    registration.SetMovingImage(&movingImage);
    registration.SetInitialTransformParameterFileName(initialTransformParameterFileName);

    registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                            { "AutomaticTransformInitialization", "false" },
                                                            { "ImageSampler", "Full" },
                                                            { "MaximumNumberOfIterations", "0" },
                                                            { "Metric", "AdvancedNormalizedCorrelation" },
                                                            { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                            { "Transform", "TranslationTransform" } }));
    registration.Update();
    return registration.GetOutput();
  };

  EXPECT_EQ(
    DerefSmartPointer(doDummyRegistration(
      GetDataDirectoryPath() + "/Translation(1,-2)/TransformParametersWithInitialTransformParameterFile.txt")),
    DerefSmartPointer(doDummyRegistration(GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt")));
}


GTEST_TEST(itkElastixRegistrationMethod, SetInitialTransform)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType initialTranslation{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  using TransformType = ElastixRegistrationMethodType<ImageType>::TransformType;

  const TransformType::ConstPointer singleInitialTransform = [] {
    const auto singleTransform = itk::TranslationTransform<double, ImageDimension>::New();
    singleTransform->SetOffset(itk::MakeVector(1.0, -2.0));
    return singleTransform;
  }();

  const TransformType::ConstPointer compositeInitialTransform = [] {
    const auto translationTransformX = itk::TranslationTransform<double, ImageDimension>::New();
    translationTransformX->SetOffset(itk::MakeVector(1.0, 0.0));
    const auto translationTransformY = itk::TranslationTransform<double, ImageDimension>::New();
    translationTransformY->SetOffset(itk::MakeVector(0.0, -2.0));

    const auto compositeTransform = itk::CompositeTransform<double, ImageDimension>::New();
    compositeTransform->AddTransform(translationTransformX);
    compositeTransform->AddTransform(translationTransformY);
    return compositeTransform;
  }();

  // Test both a single and a composite transform as initial transform.
  for (const TransformType * const initialTransform : { singleInitialTransform, compositeInitialTransform })
  {
    const auto fixedImage = CreateImage<PixelType>(imageSize);
    FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

    const auto movingImage = CreateImage<PixelType>(imageSize);

    elx::DefaultConstruct<elx::ParameterObject>                     registrationParameterObject{};
    elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
    registration.SetFixedImage(fixedImage);
    registration.SetInitialTransform(initialTransform);
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
      // Specify multiple (one or more) registration parameter maps.
      registrationParameterObject.SetParameterMaps(
        ParameterMapVectorType(numberOfRegistrationParameterMaps, registrationParameterMap));

      const auto numberOfInitialTransformParameterMaps = GetNumberOfTransforms(*initialTransform);

      // Do the test for a few possible translations.
      for (const auto index :
           itk::ImageRegionIndexRange<ImageDimension>(itk::ImageRegion<ImageDimension>({ 0, -2 }, { 2, 3 })))
      {
        const auto actualTranslation = ConvertIndexToOffset(index);
        movingImage->FillBuffer(0);
        FillImageRegion(*movingImage, fixedImageRegionIndex + actualTranslation, regionSize);
        registration.SetMovingImage(movingImage);
        registration.Update();

        const auto & transformParameterMaps = Deref(registration.GetTransformParameterObject()).GetParameterMaps();

        ASSERT_EQ(transformParameterMaps.size(),
                  numberOfInitialTransformParameterMaps + numberOfRegistrationParameterMaps);

        // All transform parameter maps, except for the initial transformations and the transform parameter map of the
        // first registration should just have a zero-translation.
        for (auto i = numberOfInitialTransformParameterMaps + 1; i < numberOfRegistrationParameterMaps; ++i)
        {
          const auto transformParameters =
            ConvertStringsToVectorOfDouble(transformParameterMaps[i].at("TransformParameters"));
          EXPECT_EQ(ConvertToOffset<ImageDimension>(transformParameters), OffsetType{});
        }

        // Together the initial translation and the first registration should yield the actual image translation.
        const auto transformParameters = ConvertStringsToVectorOfDouble(
          transformParameterMaps[numberOfInitialTransformParameterMaps].at("TransformParameters"));
        EXPECT_EQ(initialTranslation + ConvertToOffset<ImageDimension>(transformParameters), actualTranslation);
      }
    }
  }
}


GTEST_TEST(itkElastixRegistrationMethod, SetInitialTransformParameterObject)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 2U;
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
      const auto numberOfInitialTransformParameterMaps = initialTransformParameterMaps.size();
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

        const auto & transformParameterMaps = Deref(registration.GetTransformParameterObject()).GetParameterMaps();

        ASSERT_EQ(transformParameterMaps.size(),
                  numberOfInitialTransformParameterMaps + numberOfRegistrationParameterMaps);

        for (std::size_t i{}; i < numberOfInitialTransformParameterMaps; ++i)
        {
          EXPECT_EQ(transformParameterMaps[i], initialTransformParameterMaps[i]);
        }

        // All registration parameter maps, except for the first one, should just have a zero-translation.
        for (auto i = numberOfInitialTransformParameterMaps + 1; i < numberOfRegistrationParameterMaps; ++i)
        {
          const auto transformParameters =
            ConvertStringsToVectorOfDouble(transformParameterMaps[i].at("TransformParameters"));
          EXPECT_EQ(ConvertToOffset<ImageDimension>(transformParameters), OffsetType{});
        }

        // Together the initial translation and the first registration should have the actual image translation.
        const auto transformParameters = ConvertStringsToVectorOfDouble(
          transformParameterMaps[numberOfInitialTransformParameterMaps].at("TransformParameters"));
        EXPECT_EQ(initialTranslation + ConvertToOffset<ImageDimension>(transformParameters), actualTranslation);
      }
    }
  }
}


GTEST_TEST(itkElastixRegistrationMethod, SetExternalTransformAsInitialTransform)
{
  static constexpr unsigned int ImageDimension{ 2 };

  using PixelType = float;
  using SizeType = itk::Size<ImageDimension>;
  const SizeType imageSize{ { 5, 6 } };

  using ImageType = itk::Image<PixelType, ImageDimension>;
  using TransformixFilterType = itk::TransformixFilter<ImageType>;

  const ImageDomain<ImageDimension> imageDomain(imageSize);

  elx::DefaultConstruct<itk::TranslationTransform<double, ImageDimension>> itkTransform;
  itkTransform.SetOffset(itk::MakeVector(1.0, -2.0));

  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType initialTranslation{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
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

  registrationParameterObject.SetParameterMap(registrationParameterMap);

  const ParameterMapType initialTransformParameterMap{ { "NumberOfParameters", { "0" } },
                                                       { "Transform", { "ExternalTransform" } },
                                                       { "TransformAddress",
                                                         { elx::Conversion::ObjectPtrToString(&itkTransform) } } };
  initialTransformParameterObject.SetParameterMap(initialTransformParameterMap);

  // Do the test for a few possible translations.
  for (const auto index :
       itk::ImageRegionIndexRange<ImageDimension>(itk::ImageRegion<ImageDimension>({ 0, -2 }, { 2, 3 })))
  {
    const auto actualTranslation = ConvertIndexToOffset(index);
    movingImage->FillBuffer(0);
    FillImageRegion(*movingImage, fixedImageRegionIndex + actualTranslation, regionSize);
    registration.SetMovingImage(movingImage);
    registration.Update();

    const auto & transformParameterMaps = Deref(registration.GetTransformParameterObject()).GetParameterMaps();

    ASSERT_EQ(transformParameterMaps.size(), 2);

    EXPECT_EQ(transformParameterMaps.front(), initialTransformParameterMap);

    // Together the initial translation and the first registration should have the actual image translation.
    const auto transformParameters =
      ConvertStringsToVectorOfDouble(transformParameterMaps.back().at("TransformParameters"));
    EXPECT_EQ(initialTranslation + ConvertToOffset<ImageDimension>(transformParameters), actualTranslation);
  }
}


GTEST_TEST(itkElastixRegistrationMethod, SetExternalInitialTransform)
{
  static constexpr unsigned int ImageDimension{ 2 };

  using PixelType = float;
  using SizeType = itk::Size<ImageDimension>;
  const SizeType imageSize{ { 5, 6 } };

  using ImageType = itk::Image<PixelType, ImageDimension>;
  using TransformixFilterType = itk::TransformixFilter<ImageType>;

  const ImageDomain<ImageDimension> imageDomain(imageSize);

  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType initialTranslation{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  elx::DefaultConstruct<itk::DisplacementFieldTransform<double, ImageDimension>> itkTransform{};
  const auto displacementField = itk::Image<itk::Vector<double, ImageDimension>, ImageDimension>::New();

  displacementField->SetRegions(imageSize);
  displacementField->AllocateInitialized();
  itkTransform.SetDisplacementField(displacementField);

  const itk::ImageBufferRange displacementFieldImageBufferRange{ *displacementField };
  std::fill_n(displacementFieldImageBufferRange.begin(),
              displacementFieldImageBufferRange.size(),
              // C++17 note: for itk::Vector (ITK 5.3.0) template argument deduction (CTAD) cannot be used here! The
              // template arguments of `itk::Vector{ std::array{ 1.0, -2.0 } }` are deduced to `std::array<double, 2>,
              // 3` by both GNU 9.4.0 and MacOS11/Xcode_13.2.1/MacOSX12.1 Clang.
              itk::Vector<double, ImageDimension>{ std::array{ 1.0, -2.0 } });

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = CreateImage<PixelType>(imageSize);

  elx::DefaultConstruct<elx::ParameterObject>                     registrationParameterObject{};
  elx::DefaultConstruct<elx::ParameterObject>                     initialTransformParameterObject{};
  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  registration.SetFixedImage(fixedImage);
  registration.SetExternalInitialTransform(&itkTransform);
  registration.SetParameterObject(&registrationParameterObject);

  const elx::ParameterObject::ParameterMapType registrationParameterMap{
    // Parameters in alphabetic order:
    { "ImageSampler", { "Full" } },
    { "MaximumNumberOfIterations", { "2" } },
    { "Metric", { "AdvancedNormalizedCorrelation" } },
    { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
    { "Transform", { "TranslationTransform" } }
  };

  registrationParameterObject.SetParameterMap(registrationParameterMap);

  const ParameterMapType initialTransformParameterMap{ { "NumberOfParameters", { "0" } },
                                                       { "Transform", { "ExternalTransform" } },
                                                       { "TransformAddress",
                                                         { elx::Conversion::ObjectPtrToString(&itkTransform) } } };

  // Do the test for a few possible translations.
  for (const auto index :
       itk::ImageRegionIndexRange<ImageDimension>(itk::ImageRegion<ImageDimension>({ 0, -2 }, { 2, 3 })))
  {
    const auto actualTranslation = ConvertIndexToOffset(index);
    movingImage->FillBuffer(0);
    FillImageRegion(*movingImage, fixedImageRegionIndex + actualTranslation, regionSize);
    registration.SetMovingImage(movingImage);
    registration.Update();

    const auto & transformParameterMaps = Deref(registration.GetTransformParameterObject()).GetParameterMaps();

    ASSERT_EQ(transformParameterMaps.size(), 2);

    EXPECT_EQ(transformParameterMaps.front(), initialTransformParameterMap);

    // Together the initial translation and the first registration should have the actual image translation.
    const auto transformParameters =
      ConvertStringsToVectorOfDouble(transformParameterMaps.back().at("TransformParameters"));
    EXPECT_EQ(initialTranslation + ConvertToOffset<ImageDimension>(transformParameters), actualTranslation);
  }
}


GTEST_TEST(itkElastixRegistrationMethod, SetExternalInitialTransformAndOutputDirectory)
{
  const std::string outputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(outputDirectoryPath);

  static constexpr auto ImageDimension = 2U;

  using PixelType = float;
  using SizeType = itk::Size<ImageDimension>;
  const SizeType imageSize{ { 5, 6 } };

  using ImageType = itk::Image<PixelType, ImageDimension>;

  const auto displacementField = itk::Image<itk::Vector<double, ImageDimension>, ImageDimension>::New();

  displacementField->SetRegions(imageSize);
  displacementField->AllocateInitialized();

  std::mt19937 randomNumberEngine{};

  // Generate a rather arbitrary displacement field.
  const itk::ImageBufferRange displacementFieldImageBufferRange{ *displacementField };
  std::generate_n(
    displacementFieldImageBufferRange.begin(), displacementFieldImageBufferRange.size(), [&randomNumberEngine] {
      itk::Vector<double, ImageDimension> displacementVector{};

      std::generate_n(displacementVector.begin(), ImageDimension, [&randomNumberEngine] {
        return std::uniform_int_distribution<>{ -1, 1 }(randomNumberEngine);
      });
      return displacementVector;
    });

  elx::DefaultConstruct<itk::DisplacementFieldTransform<double, ImageDimension>> itkTransform{};
  itkTransform.SetDisplacementField(displacementField);

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);

  const elx::ParameterObject::ParameterMapType registrationParameterMap{
    // Parameters in alphabetic order:
    { "ImageSampler", { "Full" } },
    { "MaximumNumberOfIterations", { "2" } },
    { "Metric", { "AdvancedNormalizedCorrelation" } },
    { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
    { "Transform", { "TranslationTransform" } }
  };

  elx::DefaultConstruct<elx::ParameterObject> registrationParameterObject{};
  registrationParameterObject.SetParameterMap(registrationParameterMap);

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  registration.SetFixedImage(fixedImage);
  registration.SetMovingImage(movingImage);
  registration.SetExternalInitialTransform(&itkTransform);
  registration.SetOutputDirectory(outputDirectoryPath);
  registration.SetParameterObject(&registrationParameterObject);
  registration.Update();

  // Read back the initial transform that should have been written by registration.Update().
  const auto reader = itk::TransformFileReader::New();
  reader->SetFileName(outputDirectoryPath + "/InitialTransform.0.tfm");
  reader->Update();

  // Check that the read transform is equal to the initially specified ITK transform.
  const auto & readTransformList = Deref(reader->GetTransformList());
  ASSERT_EQ(readTransformList.size(), 1);
  const auto & readTransform = DerefSmartPointer(readTransformList.front());
  EXPECT_EQ(readTransform.GetParameters(), itkTransform.GetParameters());
  EXPECT_EQ(readTransform.GetFixedParameters(), itkTransform.GetFixedParameters());
  EXPECT_EQ(readTransform.GetTransformTypeAsString(), itkTransform.GetTransformTypeAsString());
}


// Tests that the CombinationTransform produced by a registration using an external initial transform can be converted
// to an ITK CompositeTransform. Tests that this CompositeTransform has a pointer to the initial transform as its "back
// transform".
GTEST_TEST(itkElastixRegistrationMethod, SetExternalInitialTransformAndConvertToItkTransform)
{
  static constexpr auto ImageDimension = 2u;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  const itk::Size<ImageDimension> imageSize{ { 5, 6 } };

  elx::DefaultConstruct<itk::DisplacementFieldTransform<double, ImageDimension>> externalTransform{};
  externalTransform.SetDisplacementField(CreateImage<itk::Vector<double, ImageDimension>, ImageDimension>(imageSize));

  elx::DefaultConstruct<elx::ParameterObject> registrationParameterObject{};
  registrationParameterObject.SetParameterMap(
    ParameterMapType{ // Parameters in alphabetic order:
                      { "ImageSampler", { "Full" } },
                      { "MaximumNumberOfIterations", { "2" } },
                      { "Metric", { "AdvancedNormalizedCorrelation" } },
                      { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                      { "Transform", { "TranslationTransform" } } });

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  registration.SetFixedImage(CreateImage<PixelType>(imageSize));
  registration.SetMovingImage(CreateImage<PixelType>(imageSize));
  registration.SetParameterObject(&registrationParameterObject);
  registration.SetExternalInitialTransform(&externalTransform);
  registration.Update();

  const auto combinationTransform = registration.GetCombinationTransform();
  const auto convertedTransform =
    ElastixRegistrationMethodType<ImageType>::ConvertToItkTransform(Deref(combinationTransform));

  const auto & compositeTransform =
    Deref(dynamic_cast<itk::CompositeTransform<double, ImageDimension> *>(convertedTransform.GetPointer()));
  ASSERT_EQ(compositeTransform.GetNumberOfTransforms(), 2);
  EXPECT_NE(compositeTransform.GetFrontTransform(), &externalTransform);
  EXPECT_EQ(compositeTransform.GetBackTransform(), &externalTransform);
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
        DefaultTransformParameter({ "InitialTransformParameterFileName", { "NoInitialTransform" } }),
        DefaultTransformParameter({ "MovingInternalImagePixelType", { "float" } }),
        DefaultTransformParameter({ "ResampleInterpolator", { "FinalBSplineInterpolator" } }),
        DefaultTransformParameter({ "Resampler", { "DefaultResampler" } }),
        // Non-default parameters in alphabetic order:
        NonDefaultTransformParameter(
          { "NumberOfParameters", { std::to_string(translationTransformParameters.size()) } }),
        NonDefaultTransformParameter({ "Transform", { "TranslationTransform" } }),
        NonDefaultTransformParameter(
          { "TransformParameters", elx::Conversion::ToVectorOfStrings(translationTransformParameters) }) });

    static constexpr auto gridValueSize = 4U;

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
        DefaultTransformParameter({ "InitialTransformParameterFileName", { "NoInitialTransform" } }),
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
    static constexpr auto smallImageSizeValue = 8U;
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
    Expect_equal_output_Transformix_SetTransformParameterObject_GetTransformParameterObject(
      { translationTransformParameterMap }, createRandomImageDomain(), createRandomImageDomain());
  }
}


GTEST_TEST(itkElastixRegistrationMethod, InitialTransformParameterFileLinkToTransformFile)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 2U;
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
        const elx::ParameterObject & transformParameterObject = Deref(registration.GetTransformParameterObject());
        const auto &                 transformParameterMaps = transformParameterObject.GetParameterMaps();
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

        if (transformParameter.first == "InitialTransformParameterFileName")
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
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<ImageType::PixelType>(itk::Size<ImageDimension>{ 5, 6 });

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

    for (const char * const transformName : { "AffineTransform",
                                              "BSplineTransform",
                                              "EulerTransform",
                                              "RecursiveBSplineTransform",
                                              "SimilarityTransform",
                                              "TranslationTransform" })
    {
      registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                              { "AutomaticTransformInitialization", "false" },
                                                              { "ImageSampler", "Full" },
                                                              { "MaximumNumberOfIterations", "0" },
                                                              { "Metric", "AdvancedNormalizedCorrelation" },
                                                              { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                              { "Transform", transformName } }));
      registration.Update();

      const auto combinationTransform = registration.GetCombinationTransform();
      EXPECT_NE(combinationTransform, nullptr);
    }
  }
}


GTEST_TEST(itkElastixRegistrationMethod, GetNumberOfTransforms)
{
  static constexpr auto ImageDimension = 2U;
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
  static constexpr auto ImageDimension = 2U;
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
  static constexpr auto ImageDimension = 2U;
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
      const auto result =
        ElastixRegistrationMethodType<ImageType>::ConvertToItkTransform(*registration.GetNthTransform(n));

      ASSERT_NE(result, nullptr);
      EXPECT_EQ(result->GetReferenceCount(), 1);
    }
  }
}

GTEST_TEST(itkElastixRegistrationMethod, WriteCompositeTransform)
{
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<ImageType::PixelType>(itk::Size<ImageDimension>{ 5, 6 });

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

    using PairType = std::pair<const char *, itk::Transform<double, ImageDimension, ImageDimension>::Pointer>;

    for (const auto [transformName, itkTransform] :
         { PairType{ "AffineTransform", itk::AffineTransform<double, ImageDimension>::New() },
           PairType{ "BSplineTransform", itk::BSplineTransform<double, ImageDimension>::New() },
           PairType{ "EulerTransform", itk::Euler2DTransform<>::New() },
           PairType{ "RecursiveBSplineTransform", itk::BSplineTransform<double, ImageDimension>::New() },
           PairType{ "SimilarityTransform", itk::Similarity2DTransform<>::New() },
           PairType{ "TranslationTransform", itk::TranslationTransform<double, ImageDimension>::New() } })
    {
      for (const std::string fileNameExtension : { "", "h5", "tfm" })
      {
        const std::string outputDirectoryPath = outputSubdirectoryPath + "/" + transformName + fileNameExtension;
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
                                  { "Transform", transformName },
                                  { "WriteITKCompositeTransform", "true" } }));
        registration.Update();

        if (!fileNameExtension.empty())
        {
          const auto & expectedItkTransform = *itkTransform;
          const auto   expectedNumberOfFixedParameters = expectedItkTransform.GetFixedParameters().size();

          const itk::TransformBase::ConstPointer singleTransform =
            elx::TransformIO::Read(outputDirectoryPath + "/TransformParameters.0." + fileNameExtension);

          using CompositeTransformType = itk::CompositeTransform<double, ImageDimension>;

          const itk::TransformBase::Pointer compositeTransform =
            elx::TransformIO::Read(outputDirectoryPath + "/TransformParameters.0-Composite." + fileNameExtension);
          const auto & transformQueue =
            Deref(dynamic_cast<const CompositeTransformType *>(compositeTransform.GetPointer())).GetTransformQueue();

          ASSERT_EQ(transformQueue.size(), useInitialTransform ? 2 : 1);

          const itk::TransformBase * const frontTransform = transformQueue.front();

          for (const auto actualTransformPtr : { singleTransform.GetPointer(), frontTransform })
          {
            const itk::TransformBase & actualTransform = Deref(actualTransformPtr);

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
              Deref(dynamic_cast<const itk::TranslationTransform<double, ImageDimension> *>(&backTransform));
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


// Tests registering two small (8x8) images, which are translated with respect to each other.
GTEST_TEST(itkElastixRegistrationMethod, SimilarityTranslation2D)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const auto imageSizeValue = 8;
  const auto imageSize = SizeType::Filled(imageSizeValue);
  const auto fixedImageRegionIndex = IndexType::Filled(imageSizeValue / 2 - 1);
  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegionWithSequenceOfNaturalNumbers(*fixedImage, fixedImageRegionIndex, SizeType::Filled(2));

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  registration.SetFixedImage(fixedImage);
  registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                          { "AutomaticTransformInitialization", "false" },
                                                          { "ImageSampler", "Full" },
                                                          { "MaximumNumberOfIterations", "2" },
                                                          { "Metric", "AdvancedNormalizedCorrelation" },
                                                          { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                          { "Transform", "SimilarityTransform" } }));

  const auto movingImage = CreateImage<PixelType>(imageSize);

  // Test translation for each direction from (-1, -1) to (1, 1).
  for (const auto & index : itk::ZeroBasedIndexRange<ImageDimension>(SizeType::Filled(3)))
  {
    movingImage->FillBuffer(0);
    const OffsetType translation = index - IndexType::Filled(1);
    FillImageRegionWithSequenceOfNaturalNumbers(*movingImage, fixedImageRegionIndex + translation, SizeType::Filled(2));

    registration.SetMovingImage(movingImage);
    registration.Update();

    const auto transformParameters = GetTransformParametersFromFilter(registration);
    ASSERT_EQ(transformParameters.size(), 4);

    // An absolute error of ~0.00022 was encountered, which seems acceptable.
    EXPECT_NEAR(transformParameters[0], 1.0, 0.001) << "The scale should be near one";

    // Absolute errors of ~3.8e-06 were encountered, which seems acceptable.
    EXPECT_LT(std::abs(transformParameters[1]), 1e-5) << "The estimated rotation angle should be near zero radians";

    for (unsigned i{}; i <= 1; ++i)
    {
      EXPECT_EQ(std::round(transformParameters[i + 2]), translation[i]);
    }
  }
}


// Tests registering two small images, which are rescaled with respect to each other.
GTEST_TEST(itkElastixRegistrationMethod, SimilarityScaling2D)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;

  // Sets the pixels of a square object at the specified index, having the specified size.
  const auto setPixelsOfObject =
    [](ImageType & image, const itk::IndexValueType & indexValue, const itk::SizeValueType sizeValueType) {
      const itk::ImageRegionRange<ImageType> imageRegionRange{
        image, { IndexType::Filled(indexValue), SizeType::Filled(sizeValueType) }
      };
      std::fill(std::begin(imageRegionRange), std::end(imageRegionRange), 1);
    };

  for (const itk::SizeValueType fixedObjectSizeValue : { 32, 40 })
  {
    for (const itk::SizeValueType movingObjectSizeValue : { 32, 40 })
    {
      const itk::IndexValueType fixedObjectIndexValue = fixedObjectSizeValue;
      const itk::SizeValueType  fixedImageSizeValue = 4 * fixedObjectSizeValue;

      const itk::IndexValueType movingObjectIndexValue = movingObjectSizeValue;
      const itk::SizeValueType  movingImageSizeValue = 4 * movingObjectSizeValue;

      const auto fixedImage = CreateImage<PixelType>(SizeType::Filled(fixedImageSizeValue));
      setPixelsOfObject(*fixedImage, fixedObjectIndexValue, fixedObjectSizeValue);

      elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
      registration.SetFixedImage(fixedImage);
      registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                              { "ImageSampler", "Full" },
                                                              { "MaximumNumberOfIterations", "50" },
                                                              { "Metric", "AdvancedNormalizedCorrelation" },
                                                              { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                              { "Transform", "SimilarityTransform" } }));

      const auto movingImage = CreateImage<PixelType>(SizeType::Filled(movingImageSizeValue));
      setPixelsOfObject(*movingImage, movingObjectIndexValue, movingObjectSizeValue);
      registration.SetMovingImage(movingImage);
      registration.Update();
      const auto & resultImage = itk::Deref(registration.GetOutput());

      ASSERT_EQ(resultImage.GetBufferedRegion(), fixedImage->GetBufferedRegion());

      const itk::ImageBufferRange resultImageBufferRange(resultImage);
      const itk::ImageBufferRange fixedImageBufferRange(*fixedImage);
      const auto                  numberOfPixels = fixedImageBufferRange.size();

      for (std::size_t i{}; i < numberOfPixels; ++i)
      {
        EXPECT_EQ(std::round(resultImageBufferRange[i]), fixedImageBufferRange[i]);
      }

      const auto transformParameters = GetTransformParametersFromFilter(registration);
      ASSERT_EQ(transformParameters.size(), 4);

      // An error of ~0.013 was observed.
      EXPECT_NEAR(transformParameters[0],
                  static_cast<double>(movingObjectSizeValue) / static_cast<double>(fixedObjectSizeValue),
                  0.05)
        << "The estimated scaling factor";

      // Absolute errors of ~3.8e-06 were encountered, which seems acceptable.
      EXPECT_LT(std::abs(transformParameters[1]), 1e-5) << "The estimated rotation angle should be near zero radians";
    }
  }
}


// Tests registering two small (8x8) images, which are translated with respect to each other.
GTEST_TEST(itkElastixRegistrationMethod, EulerTranslation2D)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const auto imageSizeValue = 8;
  const auto imageSize = SizeType::Filled(imageSizeValue);
  const auto fixedImageRegionIndex = IndexType::Filled(imageSizeValue / 2 - 1);

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegionWithSequenceOfNaturalNumbers(*fixedImage, fixedImageRegionIndex, SizeType::Filled(2));

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
    FillImageRegionWithSequenceOfNaturalNumbers(*movingImage, fixedImageRegionIndex + translation, SizeType::Filled(2));

    registration.SetMovingImage(movingImage);
    registration.Update();

    const auto transformParameters = GetTransformParametersFromFilter(registration);
    ASSERT_EQ(transformParameters.size(), 3);

    // Absolute errors of ~3.8e-06 were encountered, which seems acceptable.
    EXPECT_LT(std::abs(transformParameters[0]), 1e-5) << "The estimated rotation angle should be near zero radians";

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

      static constexpr auto radius = (imageSizeValue / 2.0) - 2.0;

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
    static constexpr auto radiansPerDegree = M_PI / 180.0;

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


// Tests registering two images which just have a different direction matrix.
GTEST_TEST(itkElastixRegistrationMethod, EulerDirection2D)
{
  using PixelType = float;
  static constexpr unsigned int ImageDimension{ 2 };
  using ImageType = itk::Image<PixelType, ImageDimension>;

  // Sets the pixels of a disc object in the center of the image, with each pixel value corresponding to its direction
  // angle with respect to the image center.
  const auto testImage = [] {
    static constexpr itk::SizeValueType imageSizeValue{ 32 };
    static constexpr auto               imageSize = itk::Size<ImageDimension>::Filled(imageSizeValue);

    const auto image = CreateImage<PixelType>(imageSize);
    for (const auto & index : itk::ZeroBasedIndexRange<ImageDimension>(imageSize))
    {
      itk::Vector<double, ImageDimension> vectorFromImageCenter;

      for (int i{}; i < ImageDimension; ++i)
      {
        vectorFromImageCenter[i] = static_cast<double>(index[i]) - (double{ imageSizeValue - 1 } / 2.0);
      }

      static constexpr auto radius = (double{ imageSizeValue } / 2.0) - 1.0;

      if (vectorFromImageCenter.GetSquaredNorm() < itk::Math::sqr(radius))
      {
        // Set pixel value according to its direction angle, in radians between -PI and PI.
        image->SetPixel(index, std::atan2(vectorFromImageCenter[1], vectorFromImageCenter[0]));
      }
    }
    return image;
  }();

  elx::DefaultConstruct<ImageType> fixedImage{};
  fixedImage.Graft(testImage);
  AdjustOriginToMakeImageCenterCoincideWithWorldCenter(fixedImage);

  // Note: the test may fail when getting closer to 180 degrees.
  for (const auto degrees : { -90, -1, 0, 1, 2, 30, 90 })
  {
    SCOPED_TRACE(testing::Message() << "degrees = " << degrees);

    elx::DefaultConstruct<ImageType> movingImage{};
    movingImage.Graft(testImage);
    movingImage.SetDirection(CreateRotationMatrix(ConvertDegreesToRadians(degrees)));

    // For this test, it's essential to adjust the origin _after_ setting the direction.
    AdjustOriginToMakeImageCenterCoincideWithWorldCenter(movingImage);

    elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

    registration.SetFixedImage(&fixedImage);
    registration.SetMovingImage(&movingImage);
    registration.SetParameterObject(
      CreateParameterObject(ParameterMapType{ // Parameters in alphabetic order:
                                              { "AutomaticScalesEstimation", { "true" } },
                                              { "ImageSampler", { "Full" } },
                                              { "MaximumNumberOfIterations", { "50" } },
                                              { "Metric", { "AdvancedNormalizedCorrelation" } },
                                              { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                                              { "Transform", { "EulerTransform" } },
                                              { "WriteResultImage", { "false" } } }));
    registration.Update();

    const auto transformParameters = GetTransformParametersFromFilter(registration);
    ASSERT_EQ(transformParameters.size(), 3);

    EXPECT_NEAR(ConvertRadiansToDegrees(transformParameters[0]), degrees, 0.5);
    EXPECT_NEAR(transformParameters[1], 0.0, 0.5); // translation Y
    EXPECT_NEAR(transformParameters[2], 0.0, 0.5); // translation Y
  }
}


// Tests registering two small (8x8) images, which are translated with respect to each other. Using bspline.
GTEST_TEST(itkElastixRegistrationMethod, BSplineTranslation2D)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const auto imageSizeValue = 8;
  const auto imageSize = SizeType::Filled(imageSizeValue);
  const auto fixedImageRegionIndex = IndexType::Filled(imageSizeValue / 2 - 1);

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegionWithSequenceOfNaturalNumbers(*fixedImage, fixedImageRegionIndex, SizeType::Filled(2));

  for (const char * const transformName : { "BSplineTransform", "RecursiveBSplineTransform" })
  {
    SCOPED_TRACE(testing::Message() << "transformName = " << std::quoted(transformName));
    elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
    registration.SetFixedImage(fixedImage);
    registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                            { "AutomaticTransformInitialization", "false" },
                                                            { "ImageSampler", "Full" },
                                                            // Five iterations appears sufficient for this simple test.
                                                            { "MaximumNumberOfIterations", "8" },
                                                            { "Metric", "AdvancedNormalizedCorrelation" },
                                                            { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                            { "Transform", transformName } }));

    // Test translation for each direction from (-1, -1) to (1, 1).
    for (const auto & index : itk::ZeroBasedIndexRange<ImageDimension>(SizeType::Filled(3)))
    {
      const auto       movingImage = CreateImage<PixelType>(imageSize);
      const OffsetType translation = index - IndexType::Filled(1);
      FillImageRegionWithSequenceOfNaturalNumbers(
        *movingImage, fixedImageRegionIndex + translation, SizeType::Filled(2));

      registration.SetMovingImage(movingImage);
      registration.Update();

      const auto transformParameters = GetTransformParametersFromFilter(registration);

      static constexpr std::size_t expectedNumberOfParametersPerDimension{ 16 };
      static constexpr std::size_t expectedNumberOfParameters{ expectedNumberOfParametersPerDimension *
                                                               ImageDimension };
      ASSERT_EQ(transformParameters.size(), expectedNumberOfParameters);

      for (std::size_t i{}; i < expectedNumberOfParameters; ++i)
      {
        EXPECT_DOUBLE_EQ(std::round(transformParameters[i]),
                         translation.at(i / expectedNumberOfParametersPerDimension));
      }
    }
  }
}


// Checks a minimum size moving image having the same pixel type as any of the supported internal pixel types.
GTEST_TEST(itkElastixRegistrationMethod, CheckMinimumMovingImageHavingInternalPixelType)
{
  elx::ForEachSupportedImageType([](const auto elxTypedef) {
    using ElxTypedef = decltype(elxTypedef);
    using ImageType = typename ElxTypedef::MovingImageType;
    static constexpr auto ImageDimension = ElxTypedef::MovingDimension;

    using PixelType = typename ImageType::PixelType;

    const auto imageSize = itk::Size<ElxTypedef::MovingDimension>::Filled(minimumImageSizeValue);
    const ImageDomain<ElxTypedef::MovingDimension> imageDomain(imageSize);

    elx::DefaultConstruct<ImageType> fixedImage{};
    imageDomain.ToImage(fixedImage);
    fixedImage.AllocateInitialized();

    elx::DefaultConstruct<ImageType> movingImage{};
    imageDomain.ToImage(movingImage);
    movingImage.AllocateInitialized();

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

    EXPECT_EQ(Deref(registration.GetOutput()), movingImage);
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
    static constexpr auto ImageDimension = ElxTypedef::MovingDimension;

    using PixelType = typename ImageType::PixelType;

    auto imageDomain = CreateRandomImageDomain<ElxTypedef::MovingDimension>(randomNumberEngine);

    // Reset index to avoid "FixedImageRegion does not overlap the fixed image buffered region" exceptions from
    // itk::ImageToImageMetric::Initialize()
    imageDomain.index = {};

    // Create an image with values 1, 2, 3, ... N. We could have used arbitrary pixel values instead.
    const auto fixedImage = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);

    elx::DefaultConstruct<ImageType> movingImage{};
    imageDomain.ToImage(movingImage);
    movingImage.AllocateInitialized();

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

    EXPECT_EQ(Deref(registration.GetOutput()), movingImage);
  });
}


// Checks a minimum size moving image using any supported internal pixel type (which may be different from the input
// pixel type).
GTEST_TEST(itkElastixRegistrationMethod, CheckMinimumMovingImageUsingAnyInternalPixelType)
{
  const auto check = [](const auto inputPixelTypeHolder) {
    (void)inputPixelTypeHolder;
    elx::ForEachSupportedImageType([](const auto elxTypedef) {
      using ElxTypedef = decltype(elxTypedef);
      using InputPixelType = typename decltype(inputPixelTypeHolder)::Type;
      using InputImageType = itk::Image<InputPixelType, ElxTypedef::MovingDimension>;

      const ImageDomain<ElxTypedef::MovingDimension> imageDomain(
        itk::Size<ElxTypedef::MovingDimension>::Filled(minimumImageSizeValue));

      elx::DefaultConstruct<InputImageType> fixedImage{};
      imageDomain.ToImage(fixedImage);
      fixedImage.AllocateInitialized();

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

      EXPECT_EQ(Deref(registration.GetOutput()), DerefSmartPointer(movingImage));
      EXPECT_EQ(registration.GetNumberOfTransforms(), 1);
      EXPECT_NE(registration.GetNthTransform(0), nullptr);
      EXPECT_NE(registration.GetCombinationTransform(), nullptr);
    });
  };

  check(TypeHolder<char>{});
  check(TypeHolder<short>{});
  check(TypeHolder<std::int64_t>{});
  check(TypeHolder<std::uint64_t>{});
  check(TypeHolder<float>{});
  check(TypeHolder<double>{});
}


// Checks a zero-filled moving image with a random domain, using any supported internal pixel type (which may be
// different from the input pixel type).
GTEST_TEST(itkElastixRegistrationMethod, CheckZeroFilledMovingImageWithRandomDomainUsingAnyInternalPixelType)
{
  std::mt19937 randomNumberEngine{};

  const auto check = [&randomNumberEngine](const auto inputPixelTypeHolder) {
    (void)inputPixelTypeHolder;
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
      movingImage.AllocateInitialized();

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

      EXPECT_EQ(Deref(registration.GetOutput()), movingImage);
    });
  };

  check(TypeHolder<char>{});
  check(TypeHolder<short>{});
  check(TypeHolder<float>{});
  check(TypeHolder<double>{});
}


// Checks that InitialTransform and ExternalInitialTransform are mutually exclusive.
GTEST_TEST(itkElastixRegistrationMethod, SetAndGetInitialTransform)
{
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;

  elx::DefaultConstruct<itk::DisplacementFieldTransform<double, ImageDimension>> displacementFieldTransform{};
  const elx::DefaultConstruct<itk::TranslationTransform<double, ImageDimension>> translationTransform{};

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
  EXPECT_EQ(registration.GetInitialTransform(), nullptr);
  EXPECT_EQ(registration.GetExternalInitialTransform(), nullptr);

  registration.SetInitialTransform(&translationTransform);
  EXPECT_EQ(registration.GetInitialTransform(), &translationTransform);
  EXPECT_EQ(registration.GetExternalInitialTransform(), nullptr);

  registration.SetExternalInitialTransform(&displacementFieldTransform);
  EXPECT_EQ(registration.GetInitialTransform(), nullptr);
  EXPECT_EQ(registration.GetExternalInitialTransform(), &displacementFieldTransform);

  registration.SetInitialTransform(nullptr);
  EXPECT_EQ(registration.GetInitialTransform(), nullptr);
  EXPECT_EQ(registration.GetExternalInitialTransform(), nullptr);

  registration.SetExternalInitialTransform(nullptr);
  EXPECT_EQ(registration.GetInitialTransform(), nullptr);
  EXPECT_EQ(registration.GetExternalInitialTransform(), nullptr);
}


// Tests that ComputeZYX = true yields a different result than ComputeZYX = false.
GTEST_TEST(itkElastixRegistrationMethod, EulerStackTransformComputeZYX)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 4U;
  using ImageType = itk::Image<PixelType, ImageDimension>;

  const auto                      numberOfImages = 4U;
  const auto                      imageSizeValue = 6U;
  const itk::Size<ImageDimension> imageStackSize{ imageSizeValue, imageSizeValue, imageSizeValue, numberOfImages };
  const auto                      imageStack = CreateImage<PixelType>(imageStackSize);

  const itk::ImageBufferRange<ImageType> imageBufferRange(*imageStack);

  std::mt19937 randomNumberEngine{};

  std::generate(imageBufferRange.begin(), imageBufferRange.end(), [&randomNumberEngine] {
    return std::uniform_real_distribution<PixelType>{ PixelType{ 0 }, PixelType{ 2 } }(randomNumberEngine);
  });

  const auto doRegistration = [imageStack](const bool computeZYX) {
    elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};
    registration.SetFixedImage(imageStack);
    registration.SetMovingImage(imageStack);
    registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                            { "AutomaticTransformInitialization", "false" },
                                                            { "ComputeZYX", elx::Conversion::BoolToString(computeZYX) },
                                                            { "ImageSampler", "RandomCoordinate" },
                                                            { "Interpolator", "ReducedDimensionBSplineInterpolator" },
                                                            { "MaximumNumberOfIterations", "3" },
                                                            { "Metric", "AdvancedNormalizedCorrelation" },
                                                            { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                            { "Transform", "EulerStackTransform" } }));
    registration.Update();
    return GetTransformParametersFromFilter(registration);
  };

  const auto transformParameters = doRegistration(true);

  const auto expectedNumberofParametersPerImage = itk::Euler3DTransform<>::New()->GetParameters().size();
  EXPECT_EQ(transformParameters.size(), numberOfImages * expectedNumberofParametersPerImage);

  // Sanity check: Expect at least one non-zero parameter value, otherwise the test is probably trivial.
  EXPECT_TRUE(std::any_of(
    transformParameters.cbegin(), transformParameters.cend(), [](const double parameter) { return parameter != 0.0; }));

  // Sanity check: Expect the same result twice, when running with computeZYX = true twice.
  EXPECT_EQ(transformParameters, doRegistration(true));

  // Expect that the result is different that, when running with computeZYX = false.
  EXPECT_NE(transformParameters, doRegistration(false));
}


GTEST_TEST(itkElastixRegistrationMethod, EuclideanDistancePointMetric)
{
  static constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const SizeType   imageSize{ { 5, 6 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);

  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  using PointType = itk::Point<float, ImageDimension>;

  const PointType fixedPoint{};
  PointType       movingPoint = fixedPoint;

  for (unsigned int i{}; i < ImageDimension; ++i)
  {
    movingPoint[i] += translationOffset[i];
  }

  registration.SetFixedImage(fixedImage);
  registration.SetMovingImage(movingImage);
  registration.SetFixedPoints(::MakeVectorContainer(std::vector<PointType>{ fixedPoint }));
  registration.SetMovingPoints(::MakeVectorContainer(std::vector<PointType>{ movingPoint }));
  registration.SetParameterObject(CreateParameterObject(
    ParameterMapType{ // Parameters in alphabetic order:
                      { "ImageSampler", { "Full" } },
                      { "MaximumNumberOfIterations", { "2" } },
                      { "Metric", { "AdvancedNormalizedCorrelation", "CorrespondingPointsEuclideanDistanceMetric" } },
                      { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                      { "Registration", { "MultiMetricMultiResolutionRegistration" } },
                      { "Transform", { "TranslationTransform" } } }));
  registration.Update();

  const auto transformParameters = GetTransformParametersFromFilter(registration);
  EXPECT_EQ(ConvertToOffset<ImageDimension>(transformParameters), translationOffset);
}


// Tests the use of the OutputTransformParameterFileFormat parameter.
GTEST_TEST(itkElastixRegistrationMethod, OutputTransformParameterFileFormat)
{
  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  using ImageType = itk::Image<float>;
  elx::DefaultConstruct<ImageType> image{};
  image.SetRegions(itk::MakeSize(5, 6));
  image.AllocateInitialized();

  elx::DefaultConstruct<elx::ParameterObject>                     parameterObject{};
  elx::DefaultConstruct<ElastixRegistrationMethodType<ImageType>> registration{};

  registration.SetFixedImage(&image);
  registration.SetMovingImage(&image);
  registration.SetParameterObject(&parameterObject);

  const auto check = [&parameterObject, &registration, rootOutputDirectoryPath](
                       const std::string & outputTransformParameterFileFormat,
                       const std::string & expectedTransformParameterFileExtension) {
    const std::string outputDirectoryPath = rootOutputDirectoryPath + "/" + outputTransformParameterFileFormat;
    itk::FileTools::CreateDirectory(outputDirectoryPath);

    registration.SetOutputDirectory(outputDirectoryPath);

    elx::ParameterObject::ParameterMapType parameterMap{
      // Parameters in alphabetic order:
      { "ImageSampler", { "Full" } },
      { "MaximumNumberOfIterations", { "2" } },
      { "Metric", { "AdvancedNormalizedCorrelation" } },
      { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
      { "OutputTransformParameterFileFormat", { outputTransformParameterFileFormat } },
      { "Transform", { "TranslationTransform" } }
    };

    parameterObject.SetParameterMap(parameterMap);
    registration.Update();

    const std::string transformParameterFileName =
      outputDirectoryPath + "/TransformParameters.0" + expectedTransformParameterFileExtension;

    EXPECT_TRUE(itksys::SystemTools::FileExists(outputDirectoryPath + "/TransformParameters.0" +
                                                expectedTransformParameterFileExtension));

    elx::DefaultConstruct<itk::TransformixFilter<ImageType>> transformix{};
    transformix.SetTransformParameterFileName(transformParameterFileName);
    return itk::Deref(transformix.GetTransformParameterObject()).GetParameterMaps();
  };

  const ParameterMapVectorType parameterMapsFromToml = check("TOML", ".toml");
  const ParameterMapVectorType parameterMapsFromText = check("txt", ".txt");

  EXPECT_EQ(parameterMapsFromToml, parameterMapsFromText);
}
