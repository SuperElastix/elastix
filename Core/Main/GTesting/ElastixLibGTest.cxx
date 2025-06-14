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
#include "transformixlib.h"
#include "elxForEachSupportedImageType.h"

#include "elxCoreMainGTestUtilities.h"

// ITK header files:
#include <itkImage.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For transform.
#include <array>
#include <limits>
#include <tuple>
#include <utility> // For pair.
#include <vector>


// Using-declarations:
using elx::CoreMainGTestUtilities::ConvertToOffset;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::CreateParameterMap;
using elx::CoreMainGTestUtilities::FillImageRegion;
using elx::CoreMainGTestUtilities::GetTransformParametersFromMaps;
using elx::CoreMainGTestUtilities::minimumImageSizeValue;
using elx::CoreMainGTestUtilities::TypeHolder;


namespace
{

template <typename TPixel>
constexpr const char *
GetPixelTypeName()
{
  constexpr auto pixelTypeNames = std::make_tuple(std::make_pair<char>(0, "char"),
                                                  std::make_pair<unsigned char>(0, "unsigned char"),
                                                  std::make_pair<short>(0, "short"),
                                                  std::make_pair<unsigned short>(0, "unsigned short"),
                                                  std::make_pair<int>(0, "int"),
                                                  std::make_pair<unsigned int>(0, "unsigned int"),
                                                  std::make_pair<long>(0, "long"),
                                                  std::make_pair<unsigned long>(0, "unsigned long"),
                                                  std::make_pair<float>(0, "float"),
                                                  std::make_pair<double>(0, "double"));

  return std::get<std::pair<TPixel, const char *>>(pixelTypeNames).second;
}


template <unsigned VDimension>
void
ExpectRoundedTransformParametersEqualOffset(const elastix::ELASTIX &        elastixObject,
                                            const itk::Offset<VDimension> & offset)
{
  const auto transformParameters = GetTransformParametersFromMaps(elastixObject.GetTransformParameterMapList());
  EXPECT_EQ(ConvertToOffset<VDimension>(transformParameters), offset);
}


} // namespace


// Tests registering two small (5x6) binary images, using the example code from
// Elastix manual paragraph "Running elastix".
GTEST_TEST(ElastixLib, ExampleFromManualRunningElastix)
{
  using elastix::ELASTIX;
  using RegistrationParametersContainerType = ELASTIX::ParameterMapListType;
  using ITKImageType = itk::Image<float>;
  static constexpr auto ImageDimension = ITKImageType::ImageDimension;

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
  fixed_image->AllocateInitialized();
  FillImageRegion(*fixed_image, fixedImageRegionIndex, regionSize);

  const auto moving_image = ITKImageType::New();
  moving_image->SetRegions(imageSize);
  moving_image->AllocateInitialized();
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
  elx::ForEachSupportedImageType([](const auto elxTypedef) {
    using ElxTypedef = decltype(elxTypedef);
    using ImageType = typename ElxTypedef::FixedImageType;
    static constexpr auto Dimension = ImageType::ImageDimension;
    using PixelType = typename ImageType::PixelType;
    using SizeType = itk::Size<Dimension>;

    const auto parameterMap =
      CreateParameterMap<Dimension>({ { "ImageSampler", "Full" },
                                      { "FixedInternalImagePixelType", GetPixelTypeName<PixelType>() },
                                      { "MaximumNumberOfIterations", "2" },
                                      { "Metric", "AdvancedNormalizedCorrelation" },
                                      { "MovingInternalImagePixelType", GetPixelTypeName<PixelType>() },
                                      { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                      { "Transform", "TranslationTransform" } });

    const auto regionSizeValue = 2;
    const auto imageSizeValue = 4;

    const auto image = ImageType::New();
    image->SetRegions(SizeType::Filled(imageSizeValue));
    image->AllocateInitialized();
    FillImageRegion(*image, itk::Index<Dimension>::Filled(1), SizeType::Filled(regionSizeValue));

    elastix::ELASTIX elastixObject;
    ASSERT_EQ(elastixObject.RegisterImages(image, image, parameterMap, ".", false, false), 0);

    const auto transformParameters = GetTransformParametersFromMaps(elastixObject.GetTransformParameterMapList());

    for (const auto & transformParameter : transformParameters)
    {
      EXPECT_EQ(transformParameter, 0.0);
    }
  });
}


// Tests registering two small binary images.
GTEST_TEST(ElastixLib, Translation3D)
{
  static constexpr auto ImageDimension = 3;
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
  fixedImage->AllocateInitialized();
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->AllocateInitialized();
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  elastix::ELASTIX elastixObject;

  ASSERT_EQ(elastixObject.RegisterImages(fixedImage, movingImage, parameterMap, ".", false, false), 0);
  ExpectRoundedTransformParametersEqualOffset(elastixObject, translationOffset);
}


// Tests registering two small 3-D binary images, including only the pixels
// of a single slice, by specifying a mask for the fixed image.
GTEST_TEST(ElastixLib, SingleSliceMaskedTranslation3D)
{
  static constexpr auto ImageDimension = 3;
  using ImageType = itk::Image<float, ImageDimension>;

  const auto parameterMap = CreateParameterMap<ImageDimension>({ { "ErodeMask", "false" },
                                                                 { "ImageSampler", "Full" },
                                                                 { "MaximumNumberOfIterations", "3" },
                                                                 { "Metric", "AdvancedNormalizedCorrelation" },
                                                                 { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                                 { "Transform", "TranslationTransform" } });

  const itk::Size<ImageDimension>   imageSize{ { 5, 6, 8 } };
  const itk::IndexValueType         z = imageSize[2] / 2;
  const itk::Size<ImageDimension>   regionSize{ 2, 2, 1 };
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 3, z } };
  const itk::Offset<ImageDimension> translationOffset{ { 1, -2, 0 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->AllocateInitialized();
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto maskImage = itk::Image<unsigned char, ImageDimension>::New();
  maskImage->SetRegions(imageSize);
  maskImage->AllocateInitialized();
  FillImageRegion(*maskImage, { 0, 0, z }, { imageSize[0], imageSize[1], 1 });

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->AllocateInitialized();
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  elastix::ELASTIX elastixObject;

  ASSERT_EQ(elastixObject.RegisterImages(fixedImage, movingImage, parameterMap, ".", false, false, maskImage), 0);
  ExpectRoundedTransformParametersEqualOffset(elastixObject, translationOffset);
}


// Tests the ResultImagePixelType parameter.
GTEST_TEST(ElastixLib, ResultImagePixelType)
{
  static constexpr auto ImageDimension = 2;
  using InputPixelType = float;

  static constexpr auto imageSize = itk::Size<ImageDimension>::Filled(minimumImageSizeValue);
  const auto            inputImage = CreateImageFilledWithSequenceOfNaturalNumbers<InputPixelType>(imageSize);

  const auto expectSpecifiedResultImagePixelType = [&inputImage](const char * const resultImagePixelType,
                                                                 const auto expectedResultImagePixelTypeHolder) {
    (void)expectedResultImagePixelTypeHolder;

    elastix::ELASTIX elastixObject{};

    const auto parameterMap = CreateParameterMap<ImageDimension>({ { "ImageSampler", "Full" },
                                                                   { "MaximumNumberOfIterations", "1" },
                                                                   { "Metric", "AdvancedNormalizedCorrelation" },
                                                                   { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                                   { "ResultImagePixelType", resultImagePixelType },
                                                                   { "Transform", "TranslationTransform" } });

    // Run a very simple registration, that should certainly succeed.
    ASSERT_EQ(elastixObject.RegisterImages(inputImage, inputImage, parameterMap, ".", false, false), 0);

    const auto resultImage = elastixObject.GetResultImage();
    ASSERT_NE(resultImage, nullptr);

    using ExpectedResultImagePixelType = typename decltype(expectedResultImagePixelTypeHolder)::Type;
    using ExpectedResultImageType = itk::Image<ExpectedResultImagePixelType, ImageDimension>;

    EXPECT_NE(dynamic_cast<const ExpectedResultImageType *>(resultImage.get()), nullptr);
  };

  // Check various legacy values for ResultImagePixelType:
  expectSpecifiedResultImagePixelType("char", TypeHolder<char>{});
  expectSpecifiedResultImagePixelType("short", TypeHolder<short>{});
  expectSpecifiedResultImagePixelType("int", TypeHolder<int>{});
  expectSpecifiedResultImagePixelType("long", TypeHolder<long>{});
  expectSpecifiedResultImagePixelType("unsigned long", TypeHolder<unsigned long>{});
  expectSpecifiedResultImagePixelType("float", TypeHolder<float>{});
  expectSpecifiedResultImagePixelType("double", TypeHolder<double>{});

  // Check various possible fixed width values for ResultImagePixelType:
  expectSpecifiedResultImagePixelType("int8", TypeHolder<std::int8_t>{});
  expectSpecifiedResultImagePixelType("uint8", TypeHolder<std::uint8_t>{});
  expectSpecifiedResultImagePixelType("int64", TypeHolder<std::int64_t>{});
  expectSpecifiedResultImagePixelType("uint64", TypeHolder<std::uint64_t>{});
  expectSpecifiedResultImagePixelType("float32", TypeHolder<float>{});
  expectSpecifiedResultImagePixelType("float64", TypeHolder<double>{});

  for (const char * const invalidResultImagePixelType : { "", "void0", "not-a-valid-type" })
  {
    elastix::ELASTIX elastixObject{};

    const auto parameterMap =
      CreateParameterMap<ImageDimension>({ { "ImageSampler", "Full" },
                                           { "MaximumNumberOfIterations", "1" },
                                           { "Metric", "AdvancedNormalizedCorrelation" },
                                           { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                           { "ResultImagePixelType", invalidResultImagePixelType },
                                           { "Transform", "TranslationTransform" } });

    // Run a very simple registration and expect it to fail (returning error code 1), for an invalid
    // ResultImagePixelType.
    EXPECT_EQ(elastixObject.RegisterImages(inputImage, inputImage, parameterMap, ".", false, false), 1);
    EXPECT_EQ(elastixObject.GetResultImage(), nullptr);
  }
}


// Tests the ResultImagePixelType parameter.
GTEST_TEST(TransformixLib, ResultImagePixelType)
{
  static constexpr auto ImageDimension = 2;
  using InputPixelType = float;

  static constexpr auto imageSize = itk::Size<ImageDimension>::Filled(minimumImageSizeValue);
  const auto            inputImage = CreateImageFilledWithSequenceOfNaturalNumbers<InputPixelType>(imageSize);

  const auto expectSpecifiedResultImagePixelType = [&inputImage](const char * const resultImagePixelType,
                                                                 const auto expectedResultImagePixelTypeHolder) {
    (void)expectedResultImagePixelTypeHolder;

    transformix::TRANSFORMIX transformixObject{};

    transformix::TRANSFORMIX::ParameterMapType transformParameterMap{
      { "FixedImageDimension", { std::to_string(ImageDimension) } },
      { "MovingImageDimension", { std::to_string(ImageDimension) } },
      { "NumberOfParameters", { "2" } },
      { "ResultImagePixelType", { resultImagePixelType } },
      { "Transform", { "TranslationTransform" } },
      { "TransformParameters", { "0", "0" } }
    };

    // Do a dummy identity transformation, that should certainly succeed.
    ASSERT_EQ(transformixObject.TransformImage(inputImage, transformParameterMap, ".", false, false), 0);

    const auto resultImage = transformixObject.GetResultImage();
    ASSERT_NE(resultImage, nullptr);

    using ExpectedResultImagePixelType = typename decltype(expectedResultImagePixelTypeHolder)::Type;
    using ExpectedResultImageType = itk::Image<ExpectedResultImagePixelType, ImageDimension>;

    EXPECT_NE(dynamic_cast<const ExpectedResultImageType *>(resultImage.get()), nullptr);
  };

  // Check various legacy values for ResultImagePixelType:
  expectSpecifiedResultImagePixelType("char", TypeHolder<char>{});
  expectSpecifiedResultImagePixelType("short", TypeHolder<short>{});
  expectSpecifiedResultImagePixelType("int", TypeHolder<int>{});
  expectSpecifiedResultImagePixelType("long", TypeHolder<long>{});
  expectSpecifiedResultImagePixelType("unsigned long", TypeHolder<unsigned long>{});
  expectSpecifiedResultImagePixelType("float", TypeHolder<float>{});
  expectSpecifiedResultImagePixelType("double", TypeHolder<double>{});

  // Check various possible fixed width values for ResultImagePixelType:
  expectSpecifiedResultImagePixelType("int8", TypeHolder<std::int8_t>{});
  expectSpecifiedResultImagePixelType("uint8", TypeHolder<std::uint8_t>{});
  expectSpecifiedResultImagePixelType("int64", TypeHolder<std::int64_t>{});
  expectSpecifiedResultImagePixelType("uint64", TypeHolder<std::uint64_t>{});
  expectSpecifiedResultImagePixelType("float32", TypeHolder<float>{});
  expectSpecifiedResultImagePixelType("float64", TypeHolder<double>{});

  for (const char * const invalidResultImagePixelType : { "", "void0", "not-a-valid-type" })
  {
    transformix::TRANSFORMIX transformixObject{};

    transformix::TRANSFORMIX::ParameterMapType transformParameterMap{
      { "FixedImageDimension", { std::to_string(ImageDimension) } },
      { "MovingImageDimension", { std::to_string(ImageDimension) } },
      { "NumberOfParameters", { "2" } },
      { "ResultImagePixelType", { invalidResultImagePixelType } },
      { "Transform", { "TranslationTransform" } },
      { "TransformParameters", { "0", "0" } }
    };

    // Do a dummy identity transformation and expect it to fail (returning error code 1), for an invalid
    // ResultImagePixelType.
    EXPECT_EQ(transformixObject.TransformImage(inputImage, transformParameterMap, ".", false, false), 1);
    EXPECT_EQ(transformixObject.GetResultImage(), nullptr);
  }
}
