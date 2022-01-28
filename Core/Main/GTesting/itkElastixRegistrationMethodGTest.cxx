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

#define _USE_MATH_DEFINES

// First include the header file to be tested:
#include <itkElastixRegistrationMethod.h>

#include "elxCoreMainGTestUtilities.h"
#include "elxDefaultConstructibleSubclass.h"
#include "elxTransformIO.h"
#include "GTesting/elxGTestUtilities.h"

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
#include <string>
#include <utility> // For pair


// Using-declarations:
using elx::CoreMainGTestUtilities::CheckNew;
using elx::CoreMainGTestUtilities::ConvertToOffset;
using elx::CoreMainGTestUtilities::CreateImage;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::CreateParameterObject;
using elx::CoreMainGTestUtilities::Deref;
using elx::CoreMainGTestUtilities::DerefSmartPointer;
using elx::CoreMainGTestUtilities::FillImageRegion;
using elx::CoreMainGTestUtilities::Front;
using elx::CoreMainGTestUtilities::GetCurrentBinaryDirectoryPath;
using elx::CoreMainGTestUtilities::GetDataDirectoryPath;
using elx::CoreMainGTestUtilities::GetNameOfTest;
using elx::CoreMainGTestUtilities::GetTransformParametersFromFilter;
using elx::GTestUtilities::MakePoint;
using elx::GTestUtilities::MakeVector;


namespace
{
template <unsigned NDimension, unsigned NSplineOrder>
void
Test_WriteBSplineTransformToItkFileFormat(const std::string & rootOutputDirectoryPath)
{
  using PixelType = float;
  using ImageType = itk::Image<PixelType, NDimension>;
  const auto image = CreateImage<PixelType>(itk::Size<NDimension>::Filled(4));

  using ItkBSplineTransformType = itk::BSplineTransform<double, NDimension, NSplineOrder>;
  const elx::DefaultConstructibleSubclass<ItkBSplineTransformType> itkBSplineTransform;

  const auto defaultFixedParameters = itkBSplineTransform.GetFixedParameters();

  // FixedParameters store the grid size, origin, spacing, and direction, according to the ITK `BSplineTransform`
  // default-constructor at
  // https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.0/Modules/Core/Transform/include/itkBSplineTransform.hxx#L35-L61.
  constexpr auto expectedNumberOfFixedParameters = NDimension * (NDimension + 3);
  ASSERT_EQ(defaultFixedParameters.size(), expectedNumberOfFixedParameters);

  const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();

  filter->SetFixedImage(image);
  filter->SetMovingImage(image);

  for (const std::string fileNameExtension : { "h5", "tfm" })
  {
    const std::string outputDirectoryPath = rootOutputDirectoryPath + "/" + std::to_string(NDimension) + "D_" +
                                            "SplineOrder=" + std::to_string(NSplineOrder) +
                                            "_FileNameExtension=" + fileNameExtension;
    itk::FileTools::CreateDirectory(outputDirectoryPath);

    filter->SetOutputDirectory(outputDirectoryPath);
    filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                       { "AutomaticTransformInitialization", "false" },
                                                       { "ImageSampler", "Full" },
                                                       { "BSplineTransformSplineOrder", std::to_string(NSplineOrder) },
                                                       { "ITKTransformOutputFileNameExtension", fileNameExtension },
                                                       { "MaximumNumberOfIterations", "0" },
                                                       { "Metric", "AdvancedNormalizedCorrelation" },
                                                       { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                       { "Transform", "BSplineTransform" } }));
    filter->Update();

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

  const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();

  filter->SetFixedImage(fixedImage);
  filter->SetMovingImage(movingImage);
  filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                     { "ImageSampler", "Full" },
                                                     { "MaximumNumberOfIterations", "2" },
                                                     { "Metric", "AdvancedNormalizedCorrelation" },
                                                     { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                     { "Transform", "TranslationTransform" } }));
  filter->Update();

  const auto transformParameters = GetTransformParametersFromFilter(*filter);
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
    const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();

    filter->SetFixedImage(fixedImage);
    filter->SetMovingImage(movingImage);
    filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                       { "ImageSampler", "Full" },
                                                       { "MaximumNumberOfIterations", "0" },
                                                       { "Metric", "AdvancedNormalizedCorrelation" },
                                                       { "Optimizer", optimizer },
                                                       { "Transform", "TranslationTransform" } }));
    filter->Update();

    const auto transformParameters = GetTransformParametersFromFilter(*filter);

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
    const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();

    filter->SetFixedImage(fixedImage);
    filter->SetMovingImage(movingImage);
    filter->SetParameterObject(CreateParameterObject(
      { // Parameters in alphabetic order:
        { "AutomaticTransformInitialization", automaticTransformInitialization ? "true" : "false" },
        { "AutomaticTransformInitializationMethod", "CenterOfGravity" },
        { "ImageSampler", "Full" },
        { "MaximumNumberOfIterations", "0" },
        { "Metric", "AdvancedNormalizedCorrelation" },
        { "Optimizer", "AdaptiveStochasticGradientDescent" },
        { "Transform", "TranslationTransform" } }));
    filter->Update();

    const auto transformParameters = GetTransformParametersFromFilter(*filter);
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
    const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();

    filter->SetFixedImage(fixedImage);
    filter->SetMovingImage(movingImage);
    filter->SetParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "ImageSampler", "Full" },
                              { "MaximumNumberOfIterations", "2" },
                              { "Metric", "AdvancedNormalizedCorrelation" },
                              { "Optimizer", "AdaptiveStochasticGradientDescent" },
                              { "Transform", "TranslationTransform" },
                              { "WriteResultImage", (writeResultImage ? "true" : "false") } }));
    filter->Update();

    const auto &       output = Deref(filter->GetOutput());
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

    const auto transformParameters = GetTransformParametersFromFilter(*filter);
    EXPECT_EQ(ConvertToOffset<ImageDimension>(transformParameters), translationOffset);
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

  for (const auto fixedImageOrigin : { MakePoint(-1.0, -2.0), ImageType::PointType(), MakePoint(0.25, 0.75) })
  {
    fixedImage->SetOrigin(fixedImageOrigin);

    for (const auto movingImageOrigin : { MakePoint(-1.0, -2.0), ImageType::PointType(), MakePoint(0.25, 0.75) })
    {
      movingImage->SetOrigin(movingImageOrigin);

      const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();

      filter->SetFixedImage(fixedImage);
      filter->SetMovingImage(movingImage);
      filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                         { "ImageSampler", "Full" },
                                                         { "MaximumNumberOfIterations", "2" },
                                                         { "Metric", "AdvancedNormalizedCorrelation" },
                                                         { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                         { "Transform", "TranslationTransform" } }));
      filter->Update();

      const auto & output = Deref(filter->GetOutput());

      // The most essential check of this test.
      EXPECT_EQ(output.GetOrigin(), fixedImageOrigin);

      ASSERT_EQ(output.GetBufferedRegion().GetSize(), imageSize);
      ASSERT_NE(output.GetBufferPointer(), nullptr);

      // Expect an output image that is very much like the fixed image.
      for (const auto & index : itk::ZeroBasedIndexRange<ImageDimension>(imageSize))
      {
        EXPECT_EQ(std::round(output.GetPixel(index)), std::round(fixedImage->GetPixel(index)));
      }

      const auto transformParameters = GetTransformParametersFromFilter(*filter);

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

  const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();

  filter->SetFixedImage(fixedImage);
  filter->SetInitialTransformParameterFileName(GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt");

  filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                     { "ImageSampler", "Full" },
                                                     { "MaximumNumberOfIterations", "2" },
                                                     { "Metric", "AdvancedNormalizedCorrelation" },
                                                     { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                     { "Transform", "TranslationTransform" } }));

  const auto toOffset = [](const IndexType & index) { return index - IndexType(); };

  for (const auto index :
       itk::ImageRegionIndexRange<ImageDimension>(itk::ImageRegion<ImageDimension>({ 0, -2 }, { 2, 3 })))
  {
    movingImage->FillBuffer(0);
    FillImageRegion(*movingImage, fixedImageRegionIndex + toOffset(index), regionSize);
    filter->SetMovingImage(movingImage);
    filter->Update();

    const auto transformParameters = GetTransformParametersFromFilter(*filter);
    ASSERT_EQ(transformParameters.size(), ImageDimension);

    for (unsigned i{}; i < ImageDimension; ++i)
    {
      EXPECT_EQ(std::round(transformParameters[i]), index[i] - initialTranslation[i]);
    }
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

  using RegistrationMethodType = itk::ElastixRegistrationMethod<ImageType, ImageType>;

  const OffsetType initialTranslation{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = CreateImage<PixelType>(imageSize);

  const auto toOffset = [](const IndexType & index) { return index - IndexType(); };

  const auto createFilter = [fixedImage](const std::string & initialTransformParameterFileName) {
    const auto filter = CheckNew<RegistrationMethodType>();
    filter->SetFixedImage(fixedImage);
    filter->SetInitialTransformParameterFileName(GetDataDirectoryPath() + "/Translation(1,-2)/" +
                                                 initialTransformParameterFileName);
    filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                       { "ImageSampler", "Full" },
                                                       { "MaximumNumberOfIterations", "2" },
                                                       { "Metric", "AdvancedNormalizedCorrelation" },
                                                       { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                       { "Transform", "TranslationTransform" } }));
    return filter;
  };

  const auto filter1 = createFilter("TransformParameters.txt");

  for (const auto transformParameterFileName :
       { "TransformParameters-link-to-ITK-tfm-file.txt",
         "TransformParameters-link-to-ITK-HDF5-file.txt",
         "TransformParameters-link-to-file-with-special-chars-in-path-name.txt" })
  {
    const auto filter2 = createFilter(transformParameterFileName);

    for (const auto index :
         itk::ImageRegionIndexRange<ImageDimension>(itk::ImageRegion<ImageDimension>({ 0, -2 }, { 2, 3 })))
    {
      movingImage->FillBuffer(0);
      FillImageRegion(*movingImage, fixedImageRegionIndex + toOffset(index), regionSize);

      const auto updateAndRetrieveTransformParameterMap = [movingImage](RegistrationMethodType & filter) {
        filter.SetMovingImage(movingImage);
        filter.Update();
        const elx::ParameterObject & transformParameterObject = Deref(filter.GetTransformParameterObject());
        const auto &                 transformParameterMaps = transformParameterObject.GetParameterMap();
        EXPECT_EQ(transformParameterMaps.size(), 1);
        return Front(transformParameterMaps);
      };

      const auto transformParameterMap1 = updateAndRetrieveTransformParameterMap(*filter1);
      const auto transformParameterMap2 = updateAndRetrieveTransformParameterMap(*filter2);

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

  const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();
  filter->SetFixedImage(image);
  filter->SetMovingImage(image);

  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  for (const bool useInitialTransform : { false, true })
  {
    filter->SetInitialTransformParameterFileName(
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

        filter->SetOutputDirectory(outputDirectoryPath);

        filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                           { "AutomaticTransformInitialization", "false" },
                                                           { "ImageSampler", "Full" },
                                                           { "ITKTransformOutputFileNameExtension", fileNameExtension },
                                                           { "MaximumNumberOfIterations", "0" },
                                                           { "Metric", "AdvancedNormalizedCorrelation" },
                                                           { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                           { "Transform", nameAndItkTransform.name },
                                                           { "WriteITKCompositeTransform", "true" } }));
        filter->Update();

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
            EXPECT_EQ(translationTransform.GetOffset(), MakeVector(1.0, -2.0));
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

  const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();
  filter->SetFixedImage(fixedImage);
  filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
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

    filter->SetMovingImage(movingImage);
    filter->Update();

    const auto transformParameters = GetTransformParametersFromFilter(*filter);
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

  const auto filter = CheckNew<itk::ElastixRegistrationMethod<ImageType, ImageType>>();

  filter->SetFixedImage(fixedImage);
  filter->SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                     { "AutomaticTransformInitialization", "false" },
                                                     { "ImageSampler", "Full" },
                                                     { "MaximumNumberOfIterations", "50" },
                                                     { "Metric", "AdvancedNormalizedCorrelation" },
                                                     { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                     { "Transform", "EulerTransform" } }));

  for (const auto degree :
       { -1
#ifdef NDEBUG
         // Test three degrees only in Release mode, as it takes too much time for a Debug configuration.
         ,
         0,
         1
#endif
       })
  {
    constexpr auto radiansPerDegree = M_PI / 180.0;

    setPixelsOfDisc(*movingImage, degree * radiansPerDegree);
    filter->SetMovingImage(movingImage);
    filter->Update();

    const auto transformParameters = GetTransformParametersFromFilter(*filter);
    ASSERT_EQ(transformParameters.size(), 3);

    EXPECT_EQ(std::round(transformParameters[0] / radiansPerDegree), -degree); // rotation angle
    EXPECT_EQ(std::round(transformParameters[1]), 0.0);                        // translation X
    EXPECT_EQ(std::round(transformParameters[2]), 0.0);                        // translation Y
  }
}
