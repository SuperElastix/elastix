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
#include "elastixlib.h"

// ITK header files:
#include <itkImage.h>
#include <itkImageRegionIterator.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For transform.
#include <array>


// Tests registering two small (5x6) binary images, using the example code from
// Elastix manual paragraph "Running elastix". 
GTEST_TEST(ElastixLib, ExampleFromManualRunningElastix)
{
  using elastix::ELASTIX;
  using RegistrationParametersContainerType = ELASTIX::ParameterMapListType;
  using ITKImageType = itk::Image<float>;
  constexpr auto ImageDimension = ITKImageType::ImageDimension;
  using RegionType = itk::ImageRegion<ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;
  using RegionIteratorType = itk::ImageRegionIterator<ITKImageType>;

  const std::pair<std::string, std::string> parameterArray[] =
  {
    // Parameters with non-default values (A-Z):
    { "FixedImageDimension", std::to_string(ImageDimension)},
    { "ImageSampler", "Full" },
    { "MaximumNumberOfIterations", "2" }, // Default value: 500
    { "Metric", "AdvancedNormalizedCorrelation" },
    { "MovingImageDimension", std::to_string(ImageDimension)},
    { "NumberOfResolutions", "2" }, // Default value: 3
    { "Optimizer", "AdaptiveStochasticGradientDescent" },
    { "Transform", "TranslationTransform" },
    // Parameters with default values (A-Z):
    { "ASGDParameterEstimationMethod", "Original" },
    { "AutomaticParameterEstimation", "true" },
    { "AutomaticTransformInitialization", "false" },
    { "BSplineInterpolationOrder", "1" },
    { "CheckNumberOfSamples", "true" },
    { "FinalBSplineInterpolationOrder", "3" },
    { "FixedImagePyramid", "FixedSmoothingImagePyramid" },
    { "FixedInternalImagePixelType", "float" },
    { "Interpolator", "BSplineInterpolator" },
    { "MaxBandCovSize", "192" },
    { "MaximumNumberOfSamplingAttempts", "0" },
    { "MaximumStepLength", "1" },
    { "MaximumStepLengthRatio", "1" },
    { "MovingImagePyramid", "MovingSmoothingImagePyramid" },
    { "MovingInternalImagePixelType", "float" },
    { "NewSamplesEveryIteration", "false" },
    { "NumberOfBandStructureSamples", "10" },
    { "NumberOfGradientMeasurements", "0" },
    { "NumberOfJacobianMeasurements", "1000" },
    { "NumberOfSamplesForExactGradient", "100000" },
    { "Registration", "MultiResolutionRegistration" },
    { "ResampleInterpolator", "FinalBSplineInterpolator" },
    { "Resampler", "DefaultResampler" },
    { "ShowExactMetricValue", "false" },
    { "SigmoidInitialTime", "0" },
    { "SigmoidScaleFactor", "0.1" },
    { "SP_A", "20" },
    { "SubtractMean", "true" },
    { "UseAdaptiveStepSizes", "true" },
    { "UseConstantStep", "false" },
    { "UseDirectionCosines", "true" },
    { "UseMultiThreadingForMetrics", "true" },
    { "WriteResultImage", "true" },
  };

  std::map < std::string, std::vector< std::string > > parameters;

  for (const auto& pair: parameterArray)
  {
    const auto result = parameters.insert({ pair.first, {pair.second} } );

    ASSERT_EQ(std::make_pair(pair, result.second), std::make_pair(pair, true));
  }

  const OffsetType translationOffset{ { 1, -2 } };
  const auto regionSize = SizeType::Filled(2);
  const SizeType imageSize{ { 5, 6 } };

  const auto fixed_image = ITKImageType::New();
  fixed_image->SetRegions(imageSize);
  fixed_image->Allocate(true);

  const IndexType fixedImageRegionIndex{ { 1, 3 } };

  for (RegionIteratorType it(fixed_image, RegionType{ fixedImageRegionIndex, regionSize }); !it.IsAtEnd(); ++it)
  {
    it.Set(1);
  }

  const auto moving_image = ITKImageType::New();
  moving_image->SetRegions(imageSize);
  moving_image->Allocate(true);

  for (RegionIteratorType it(moving_image, RegionType{ fixedImageRegionIndex + translationOffset, regionSize }); !it.IsAtEnd(); ++it)
  {
    it.Set(1);
  }

  const std::string output_directory(".");
  const bool write_log_file{ false };
  const bool output_to_console{ false };

  //////////////////////////////////////////////////////////////////////////
  // Code snippet from Manual paragraph "Running elastix" starts here >>>
  ELASTIX elastix;
  int error = 0;
  try
  {
    error = elastix.RegisterImages(
      static_cast<typename itk::DataObject::Pointer>(fixed_image.GetPointer()),
      static_cast<typename itk::DataObject::Pointer>(moving_image.GetPointer()),
      parameters,        // Parameter map read in previous code
      output_directory,  // Directory where output is written, if enabled
      write_log_file,    // Enable/disable writing of elastix.log
      output_to_console, // Enable/disable output to console
      nullptr,           // Provide fixed image mask (optional, nullptr = no mask)
      nullptr            // Provide moving image mask (optional, nullptr = no mask)
    );
  }
  catch (itk::ExceptionObject &err)
  {
    // Do some error handling.
    std::cerr << err.what() << '\n';
  }

  if (error == 0)
  {
    if (elastix.GetResultImage().IsNotNull())
    {
      // Typedef the ITKImageType first...
      ITKImageType * output_image = static_cast<ITKImageType *>(
        elastix.GetResultImage().GetPointer());
      EXPECT_NE(output_image, nullptr);
    }
  }
  else
  {
    // Registration failure. Do some error handling.
  }

  // Get transform parameters of all registration steps.
  RegistrationParametersContainerType transform_parameters
    = elastix.GetTransformParameterMapList();

  // <<< Code snippet from Manual paragraph "Running elastix" ends here
  //////////////////////////////////////////////////////////////////////////

  ASSERT_TRUE(!transform_parameters.empty());
  EXPECT_EQ(transform_parameters.size(), 1);

  const auto& first = transform_parameters.front();
  const auto found = first.find("TransformParameters");
  ASSERT_NE(found, first.cend());

  const auto& transformParameters = found->second;
  ASSERT_EQ(transformParameters.size(), ImageDimension);

  std::array<double, ImageDimension> estimatedTranslationOffset;

  std::transform(transformParameters.cbegin(), transformParameters.cend(), estimatedTranslationOffset.begin(), [](const std::string& arg)
  {
    return std::stod(arg);
  });

  OffsetType roundedTranslationOffset;

  std::transform(estimatedTranslationOffset.cbegin(), estimatedTranslationOffset.cend(), roundedTranslationOffset.begin(), [](const double arg)
  {
    return static_cast<itk::OffsetValueType>( std::round(arg) );
  });

  EXPECT_EQ(roundedTranslationOffset, translationOffset);
}
