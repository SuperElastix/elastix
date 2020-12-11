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
#include <itkImageRegionRange.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For transform.
#include <array>
#include <initializer_list>
#include <limits>
#include <vector>

namespace
{

template <typename>
constexpr const char *
GetPixelTypeName() = delete;

template <>
constexpr const char *
GetPixelTypeName<short>()
{
  return "short";
}

template <>
constexpr const char *
GetPixelTypeName<float>()
{
  return "float";
}

// Converts the specified strings to an array of double.
// Assumes that each string represents a floating point number.
template <unsigned VDimension>
std::array<double, VDimension>
ConvertStringsToArrayOfDouble(const std::vector<std::string> & strings)
{
  // Wrap ASSERT_EQ in a lambda, because it returns void!
  [&strings] { ASSERT_EQ(strings.size(), VDimension); }();

  std::array<double, VDimension> result;

  for (std::size_t i{}; i < VDimension; ++i)
  {
    const auto & str = strings[i];
    std::size_t  index{};
    result[i] = std::stod(str, &index);

    // Test that all characters have been processed, by std::stod.
    [&str, index] { ASSERT_EQ(str.size(), index); }();
  }

  return result;
}


// Converts the specified array of double to itk::Offset, by rounding each element.
template <std::size_t VDimension>
itk::Offset<VDimension>
ConvertArrayOfDoubleToOffset(const std::array<double, VDimension> & doubles)
{
  itk::Offset<VDimension> result;

  for (std::size_t i{}; i < VDimension; ++i)
  {
    const auto roundedValue = std::round(doubles[i]);

    // Wrap ASSERT calls in a lambda, because they return void!
    [roundedValue] {
      ASSERT_GE(roundedValue, std::numeric_limits<itk::OffsetValueType>::min());
      ASSERT_LE(roundedValue, std::numeric_limits<itk::OffsetValueType>::max());
    }();

    result[i] = static_cast<itk::OffsetValueType>(roundedValue);
  }

  return result;
}


template <unsigned VDimension>
void
ExpectRoundedTransformParametersEqualOffset(const elastix::ELASTIX &        elastixObject,
                                            const itk::Offset<VDimension> & offset)
{
  const auto transformParameterMaps = elastixObject.GetTransformParameterMapList();

  ASSERT_TRUE(!transformParameterMaps.empty());
  EXPECT_EQ(transformParameterMaps.size(), 1);

  const auto & transformParameterMap = transformParameterMaps.front();
  const auto   found = transformParameterMap.find("TransformParameters");
  ASSERT_NE(found, transformParameterMap.cend());

  const auto transformParameters = ConvertStringsToArrayOfDouble<VDimension>(found->second);
  EXPECT_EQ(ConvertArrayOfDoubleToOffset(transformParameters), offset);
}


template <typename TPixel, unsigned int VImageDimension>
void
FillImageRegion(itk::Image<TPixel, VImageDimension> & image,
                const itk::Index<VImageDimension> &   regionIndex,
                const itk::Size<VImageDimension> &    regionSize)
{
  using ImageRegionRangeType = itk::Experimental::ImageRegionRange<itk::Image<TPixel, VImageDimension>>;
  const ImageRegionRangeType imageRegionRange{ image, itk::ImageRegion<VImageDimension>{ regionIndex, regionSize } };
  std::fill(std::begin(imageRegionRange), std::end(imageRegionRange), 1);
}

template <unsigned VImageDimension>
std::map<std::string, std::vector<std::string>>
CreateParameterMap(std::initializer_list<std::pair<std::string, std::string>> initializerList)
{
  const std::vector<std::string> imageDimensionVector = { std::to_string(VImageDimension) };

  std::map<std::string, std::vector<std::string>> result{ { "FixedImageDimension", imageDimensionVector },
                                                          { "MovingImageDimension", imageDimensionVector } };

  for (const auto & pair : initializerList)
  {
    [&pair, &result] { ASSERT_TRUE(result.insert({ pair.first, { pair.second } }).second); }();
  }
  return result;
}

template <unsigned VImageDimension, typename TPixel = float>
void
Expect_TransformParameters_are_zero_when_fixed_image_is_moving_image()
{
  using ImageType = itk::Image<TPixel, VImageDimension>;
  using SizeType = itk::Size<VImageDimension>;

  const auto parameterMap =
    CreateParameterMap<VImageDimension>({ { "ImageSampler", "Full" },
                                          { "FixedInternalImagePixelType", GetPixelTypeName<TPixel>() },
                                          { "Metric", "AdvancedNormalizedCorrelation" },
                                          { "MovingInternalImagePixelType", GetPixelTypeName<TPixel>() },
                                          { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                          { "Transform", "TranslationTransform" } });

  const auto regionSizeValue = 2;
  const auto imageSizeValue = 4;

  const auto image = ImageType::New();
  image->SetRegions(SizeType::Filled(imageSizeValue));
  image->Allocate(true);
  FillImageRegion(*image, itk::Index<VImageDimension>::Filled(1), SizeType::Filled(regionSizeValue));

  elastix::ELASTIX elastixObject;
  ASSERT_EQ(elastixObject.RegisterImages(image, image, parameterMap, ".", false, false), 0);

  const auto transformParameterMaps = elastixObject.GetTransformParameterMapList();

  ASSERT_TRUE(!transformParameterMaps.empty());
  EXPECT_EQ(transformParameterMaps.size(), 1);

  const auto & transformParameterMap = transformParameterMaps.front();
  const auto   found = transformParameterMap.find("TransformParameters");
  ASSERT_NE(found, transformParameterMap.cend());

  const auto & transformParameters = found->second;
  ASSERT_EQ(transformParameters.size(), VImageDimension);

  for (const auto & transformParameter : transformParameters)
  {
    EXPECT_EQ(transformParameter, "0");
  }
}


} // namespace


// Tests registering two small (5x6) binary images, using the example code from
// Elastix manual paragraph "Running elastix".
GTEST_TEST(ElastixLib, ExampleFromManualRunningElastix)
{
  using elastix::ELASTIX;
  using RegistrationParametersContainerType = ELASTIX::ParameterMapListType;
  using ITKImageType = itk::Image<float>;
  constexpr auto ImageDimension = ITKImageType::ImageDimension;

  const auto parameters = CreateParameterMap<ImageDimension>({
    // Parameters with non-default values (A-Z):
    { "ImageSampler", "Full" },
    { "MaximumNumberOfIterations", "2" }, // Default value: 500
    { "Metric", "AdvancedNormalizedCorrelation" },
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
  });

  const itk::Size<ImageDimension>   imageSize{ { 5, 6 } };
  const itk::Size<ImageDimension>   regionSize = itk::Size<ImageDimension>::Filled(2);
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 3 } };
  const itk::Offset<ImageDimension> translationOffset{ { 1, -2 } };

  const auto fixed_image = ITKImageType::New();
  fixed_image->SetRegions(imageSize);
  fixed_image->Allocate(true);
  FillImageRegion(*fixed_image, fixedImageRegionIndex, regionSize);

  const auto moving_image = ITKImageType::New();
  moving_image->SetRegions(imageSize);
  moving_image->Allocate(true);
  FillImageRegion(*moving_image, fixedImageRegionIndex + translationOffset, regionSize);

  const std::string output_directory(".");
  const bool        write_log_file{ false };
  const bool        output_to_console{ false };

  //////////////////////////////////////////////////////////////////////////
  // Code snippet from Manual paragraph "Running elastix" starts here >>>
  ELASTIX elastix;
  int     error = 0;
  try
  {
    error = elastix.RegisterImages(static_cast<typename itk::DataObject::Pointer>(fixed_image.GetPointer()),
                                   static_cast<typename itk::DataObject::Pointer>(moving_image.GetPointer()),
                                   parameters,        // Parameter map read in previous code
                                   output_directory,  // Directory where output is written, if enabled
                                   write_log_file,    // Enable/disable writing of elastix.log
                                   output_to_console, // Enable/disable output to console
                                   nullptr,           // Provide fixed image mask (optional, nullptr = no mask)
                                   nullptr            // Provide moving image mask (optional, nullptr = no mask)
    );
  }
  catch (itk::ExceptionObject & err)
  {
    // Do some error handling.
    std::cerr << err.what() << '\n';
  }

  if (error == 0)
  {
    if (elastix.GetResultImage().IsNotNull())
    {
      // Typedef the ITKImageType first...
      ITKImageType * output_image = static_cast<ITKImageType *>(elastix.GetResultImage().GetPointer());
      EXPECT_NE(output_image, nullptr);
    }
  }
  else
  {
    // Registration failure. Do some error handling.
  }

  // Get transform parameters of all registration steps.
  RegistrationParametersContainerType transform_parameters = elastix.GetTransformParameterMapList();

  // <<< Code snippet from Manual paragraph "Running elastix" ends here
  //////////////////////////////////////////////////////////////////////////

  ExpectRoundedTransformParametersEqualOffset(elastix, translationOffset);
}


// Tests that the TransformParameters of a translation are all zero when the
// fixed and the moving image are the same.
GTEST_TEST(ElastixLib, TransformParametersAreZeroWhenFixedImageIsMovingImage)
{
  Expect_TransformParameters_are_zero_when_fixed_image_is_moving_image<2>();
  Expect_TransformParameters_are_zero_when_fixed_image_is_moving_image<3>();
}


// Tests specifically for pixel type short that the TransformParameters of a
// translation are all zero when the fixed and the moving image are the same.
GTEST_TEST(ElastixLib, ForThreeDimensionalShortPixelsTransformParametersAreZeroWhenFixedImageIsMovingImage)
{
  Expect_TransformParameters_are_zero_when_fixed_image_is_moving_image<3, short>();
}


// Tests specifically for 4-D short that the TransformParameters of a
// translation are all zero when the fixed and the moving image are the same.
GTEST_TEST(ElastixLib, ForFourDimensionalShortPixelsTransformParametersAreZeroWhenFixedImageIsMovingImage)
{
  Expect_TransformParameters_are_zero_when_fixed_image_is_moving_image<4, short>();
}


// Tests registering two small binary images.
GTEST_TEST(ElastixLib, Translation3D)
{
  constexpr auto ImageDimension = 3;
  using ImageType = itk::Image<float, ImageDimension>;

  const auto parameterMap = CreateParameterMap<ImageDimension>({ { "ImageSampler", "Full" },
                                                                 { "MaximumNumberOfIterations", "3" },
                                                                 { "Metric", "AdvancedNormalizedCorrelation" },
                                                                 { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                                 { "Transform", "TranslationTransform" } });

  const itk::Size<ImageDimension>   imageSize{ { 5, 7, 9 } };
  const itk::Size<ImageDimension>   regionSize = itk::Size<ImageDimension>::Filled(2);
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 2, 3 } };
  const itk::Offset<ImageDimension> translationOffset{ { 1, 2, 3 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->Allocate(true);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->Allocate(true);
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  elastix::ELASTIX elastixObject;

  ASSERT_EQ(elastixObject.RegisterImages(fixedImage, movingImage, parameterMap, ".", false, false), 0);
  ExpectRoundedTransformParametersEqualOffset(elastixObject, translationOffset);
}
