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
#include <itkTransformixFilter.h>
#include <itkElastixRegistrationMethod.h>
#include <itkParameterFileParser.h>

#include "elxCoreMainGTestUtilities.h"
#include "elxDefaultConstruct.h"
#include "elxTransformIO.h"
#include "GTesting/elxGTestUtilities.h"
#include "elxForEachSupportedImageType.h"

#include <itkStackTransform.h>


// ITK header files:
#include <itkAffineTransform.h>
#include <itkBSplineTransform.h>
#include <itkCompositeTransform.h>
#include <itkDeref.h>
#include <itkDisplacementFieldTransform.h>
#include <itkEuler2DTransform.h>
#include <itkEuler3DTransform.h>
#include <itkFileTools.h>
#include <itkImage.h>
#include <itkImageBufferRange.h>
#include <itkNumberToString.h>
#include <itkResampleImageFilter.h>
#include <itkSimilarity2DTransform.h>
#include <itkSimilarity3DTransform.h>
#include <itkTranslationTransform.h>


// GoogleTest header file:
#include <gtest/gtest.h>

#include <algorithm> // For equal and transform.
#include <cmath>
#include <map>
#include <random>
#include <regex>
#include <string>


// Type aliases:
using ParameterMapType = itk::ParameterFileParser::ParameterMapType;
using ParameterValuesType = itk::ParameterFileParser::ParameterValuesType;
using ParameterMapVectorType = elx::ParameterObject::ParameterMapVectorType;

// Using-declarations:
using elx::CoreMainGTestUtilities::CheckNew;
using elx::CoreMainGTestUtilities::CreateImage;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;
using elx::CoreMainGTestUtilities::CreateParameterObject;
using elx::CoreMainGTestUtilities::CreateRandomImageDomain;
using elx::CoreMainGTestUtilities::DerefSmartPointer;
using elx::CoreMainGTestUtilities::FillImageRegion;
using elx::CoreMainGTestUtilities::GetCurrentBinaryDirectoryPath;
using elx::CoreMainGTestUtilities::GetDataDirectoryPath;
using elx::CoreMainGTestUtilities::GetNameOfTest;
using elx::CoreMainGTestUtilities::GenerateRandomSign;
using elx::CoreMainGTestUtilities::ImageDomain;
using elx::CoreMainGTestUtilities::TypeHolder;
using elx::CoreMainGTestUtilities::minimumImageSizeValue;
using elx::GTestUtilities::GeneratePseudoRandomParameters;
using elx::GTestUtilities::MakeMergedMap;
using itk::Deref;

template <typename TMovingImage>
using DefaultConstructibleTransformixFilter = elx::DefaultConstruct<itk::TransformixFilter<TMovingImage>>;


namespace
{
template <typename T>
auto
ConvertToItkVector(const T & arg)
{
  itk::Vector<double, T::Dimension> result;
  std::copy_n(arg.begin(), T::Dimension, result.begin());
  return result;
}


template <unsigned ImageDimension>
auto
CreateDefaultDirectionParameterValues()
{
  static constexpr auto numberOfValues = ImageDimension * ImageDimension;
  ParameterValuesType   values(numberOfValues, "0");

  for (std::size_t i{}; i < numberOfValues; i += (ImageDimension + 1))
  {
    values[i] = "1";
  }
  return values;
}


template <typename T>
auto
ConvertToParameterValues(const T & container)
{
  ParameterValuesType parameterValues(container.size());
  std::transform(std::begin(container),
                 std::end(container),
                 parameterValues.begin(),
                 [](decltype(*std::begin(container)) inputValue) { return itk::NumberToString<double>{}(inputValue); });
  return parameterValues;
}


// Translates an image by the specified offset, using itk::TransformixFilter,
// specifying "TranslationTransform" as Transform.
template <typename TImage>
itk::SmartPointer<TImage>
TranslateImage(TImage & image, const typename TImage::OffsetType & translationOffset)
{
  static constexpr auto ImageDimension = TImage::ImageDimension;

  DefaultConstructibleTransformixFilter<TImage> filter;

  filter.SetMovingImage(&image);
  filter.SetTransformParameterObject(
    CreateParameterObject({ // Parameters in alphabetic order:
                            { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                            { "Index", ParameterValuesType(ImageDimension, "0") },
                            { "NumberOfParameters", { std::to_string(ImageDimension) } },
                            { "Origin", ParameterValuesType(ImageDimension, "0") },
                            { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                            { "Size", ConvertToParameterValues(image.GetRequestedRegion().GetSize()) },
                            { "Transform", ParameterValuesType{ "TranslationTransform" } },
                            { "TransformParameters", ConvertToParameterValues(translationOffset) },
                            { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
  filter.Update();

  return &Deref(filter.GetOutput());
}


template <typename TPixel, unsigned int VImageDimension>
void
ExpectEqualImages(const itk::Image<TPixel, VImageDimension> & actualImage,
                  const itk::Image<TPixel, VImageDimension> & expectedImage)
{
  EXPECT_EQ(actualImage, expectedImage);
}


template <typename TImage>
bool
ImageBuffer_has_nonzero_pixel_values(const TImage & image)
{
  const itk::ImageBufferRange<const TImage> imageBufferRange(image);
  return std::any_of(imageBufferRange.cbegin(),
                     imageBufferRange.cend(),
                     [](const typename TImage::PixelType pixelValue) { return pixelValue != 0; });
}


template <typename TPixel, unsigned VImageDimension>
auto
CreateTransformixFilter(itk::Image<TPixel, VImageDimension> &                            image,
                        const itk::Transform<double, VImageDimension, VImageDimension> & itkTransform,
                        const std::string & initialTransformParameterFileName = "NoInitialTransform",
                        const std::string & howToCombineTransforms = "Compose")
{
  const auto filter = CheckNew<itk::TransformixFilter<itk::Image<TPixel, VImageDimension>>>();
  filter->SetMovingImage(&image);
  filter->SetTransform(&itkTransform);
  filter->SetTransformParameterObject(
    CreateParameterObject({ // Parameters in alphabetic order:
                            { "Direction", CreateDefaultDirectionParameterValues<VImageDimension>() },
                            { "HowToCombineTransforms", { howToCombineTransforms } },
                            { "Index", ParameterValuesType(VImageDimension, "0") },
                            { "InitialTransformParameterFileName", { initialTransformParameterFileName } },
                            { "Origin", ParameterValuesType(VImageDimension, "0") },
                            { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                            { "Size", ConvertToParameterValues(image.GetBufferedRegion().GetSize()) },
                            { "Spacing", ParameterValuesType(VImageDimension, "1") } }));
  filter->Update();
  return filter;
}


template <typename TPixel, unsigned VImageDimension>
itk::SmartPointer<itk::Image<TPixel, VImageDimension>>
RetrieveOutputFromTransformixFilter(itk::Image<TPixel, VImageDimension> &                            image,
                                    const itk::Transform<double, VImageDimension, VImageDimension> & itkTransform,
                                    const std::string & initialTransformParameterFileName = "NoInitialTransform",
                                    const std::string & howToCombineTransforms = "Compose")
{
  const auto transformixFilter =
    CreateTransformixFilter(image, itkTransform, initialTransformParameterFileName, howToCombineTransforms);
  const auto output = transformixFilter->GetOutput();
  EXPECT_NE(output, nullptr);
  return output;
}


template <typename TPixel, unsigned VImageDimension>
auto
CreateResampleImageFilter(const itk::Image<TPixel, VImageDimension> &                      image,
                          const itk::Transform<double, VImageDimension, VImageDimension> & itkTransform)
{
  const auto filter =
    itk::ResampleImageFilter<itk::Image<TPixel, VImageDimension>, itk::Image<TPixel, VImageDimension>>::New();
  filter->SetInput(&image);
  filter->SetTransform(&itkTransform);
  filter->SetSize(image.GetBufferedRegion().GetSize());
  filter->Update();
  return filter;
}


template <typename TPixel, unsigned VImageDimension>
void
Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
  itk::Image<TPixel, VImageDimension> &                            inputImage,
  const itk::Transform<double, VImageDimension, VImageDimension> & itkTransform)
{
  const auto resampleImageFilter = CreateResampleImageFilter(inputImage, itkTransform);
  const auto transformixFilter = CreateTransformixFilter(inputImage, itkTransform);

  const auto & resampleImageFilterOutput = Deref(DerefSmartPointer(resampleImageFilter).GetOutput());
  const auto & transformixFilterOutput = Deref(DerefSmartPointer(transformixFilter).GetOutput());

  // Check that the ResampleImageFilter output isn't equal to the input image,
  // otherwise the test itself would be less interesting.
  EXPECT_NE(resampleImageFilterOutput, inputImage);

  // Check that the output is not simply a black image, otherwise the test
  // itself would be less interesting.
  EXPECT_TRUE(ImageBuffer_has_nonzero_pixel_values(transformixFilterOutput));

  ExpectEqualImages(transformixFilterOutput, resampleImageFilterOutput);
}


// Creates a transform of the specified (typically derived) type, and implicitly converts to an itk::Transform pointer.
template <typename TTransform>
itk::SmartPointer<itk::Transform<typename TTransform::ParametersValueType,
                                 TTransform::InputSpaceDimension,
                                 TTransform::OutputSpaceDimension>>
CreateTransform()
{
  return TTransform::New();
}


// Creates a matrix-and-offset-transform of the specified (typically derived) type, and implicitly converts to
// an itk::MatrixOffsetTransformBase pointer.
template <typename TTransform>
itk::SmartPointer<itk::MatrixOffsetTransformBase<typename TTransform::ParametersValueType,
                                                 TTransform::InputSpaceDimension,
                                                 TTransform::OutputSpaceDimension>>
CreateMatrixOffsetTransform()
{
  return TTransform::New();
}

template <typename TImage>
void
ExpectAlmostEqualPixelValues(const TImage & actualImage, const TImage & expectedImage, const double tolerance)
{
  // Expect the specified tolerance value to be greater than zero, otherwise
  // `ExpectEqualImages` should have been called instead.
  EXPECT_GT(tolerance, 0.0);

  using ImageBufferRangeType = itk::ImageBufferRange<const TImage>;

  const ImageBufferRangeType actualImageBufferRange(actualImage);
  const ImageBufferRangeType expectedImageBufferRange(expectedImage);

  ASSERT_EQ(actualImageBufferRange.size(), expectedImageBufferRange.size());

  const auto beginOfExpectedImageBuffer = expectedImageBufferRange.cbegin();

  // First expect that _not_ all pixel values are not _exactly_ equal,
  // otherwise `ExpectEqualImages` should probably have been called instead!
  EXPECT_FALSE(std::equal(actualImageBufferRange.cbegin(), actualImageBufferRange.cend(), beginOfExpectedImageBuffer));

  auto expectedImageIterator = beginOfExpectedImageBuffer;

  const itk::ZeroBasedIndexRange<TImage::ImageDimension> indexRange(actualImage.GetBufferedRegion().GetSize());
  auto                                                   indexIterator = indexRange.cbegin();
  for (const typename TImage::PixelType actualPixelValue : actualImageBufferRange)
  {
    EXPECT_LE(std::abs(actualPixelValue - *expectedImageIterator), tolerance)
      << " actualPixelValue = " << actualPixelValue << "; expectedPixelValue = " << *expectedImageIterator
      << " index = " << *indexIterator;
    ++expectedImageIterator;
    ++indexIterator;
  }
}


template <typename TImage>
void
Expect_Transformix_output_equals_registration_output_from_file(const testing::Test &    test,
                                                               const std::string &      subdirectoryName,
                                                               TImage &                 fixedImage,
                                                               TImage &                 movingImage,
                                                               const ParameterMapType & parameterMap)
{
  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(test);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  const std::string outputDirectoryPath = rootOutputDirectoryPath + '/' + subdirectoryName;
  itk::FileTools::CreateDirectory(outputDirectoryPath);


  elx::DefaultConstruct<itk::ElastixRegistrationMethod<TImage, TImage>> registration;

  registration.SetFixedImage(&fixedImage);
  registration.SetMovingImage(&movingImage);
  registration.SetParameterObject(CreateParameterObject(parameterMap));
  registration.SetOutputDirectory(outputDirectoryPath);
  registration.Update();

  const auto & registrationOutputImage = Deref(registration.GetOutput());

  const itk::ImageBufferRange<const TImage> registrationOutputImageBufferRange(registrationOutputImage);
  const auto beginOfRegistrationOutputImageBuffer = registrationOutputImageBufferRange.cbegin();
  const auto endOfRegistrationOutputImageBuffer = registrationOutputImageBufferRange.cend();
  const itk::ImageBufferRange<const TImage> movingImageBufferRange(movingImage);

  ASSERT_NE(beginOfRegistrationOutputImageBuffer, endOfRegistrationOutputImageBuffer);

  const auto firstRegistrationOutputPixel = *beginOfRegistrationOutputImageBuffer;

  // Check that the registrationOutputImage is not a uniform image, otherwise the test
  // probably just does does not make much sense.
  EXPECT_FALSE(std::all_of(
    beginOfRegistrationOutputImageBuffer,
    endOfRegistrationOutputImageBuffer,
    [firstRegistrationOutputPixel](const auto pixelValue) { return pixelValue == firstRegistrationOutputPixel; }));

  // Check that the registrationOutputImage has different pixel values than the moving image, otherwise the test
  // probably just does does not make much sense either.
  EXPECT_FALSE(std::equal(beginOfRegistrationOutputImageBuffer,
                          endOfRegistrationOutputImageBuffer,
                          movingImageBufferRange.cbegin(),
                          movingImageBufferRange.cend()));

  DefaultConstructibleTransformixFilter<TImage> transformixFilter;

  transformixFilter.SetMovingImage(&movingImage);

  const auto parameterObject = CheckNew<elx::ParameterObject>();

  parameterObject->SetParameterMap(
    itk::ParameterFileParser::ReadParameterMap(outputDirectoryPath + "/TransformParameters.0.txt"));

  transformixFilter.SetTransformParameterObject(parameterObject);

  transformixFilter.Update();

  EXPECT_EQ(Deref(transformixFilter.GetOutput()), registrationOutputImage);
}


template <unsigned NDimension, unsigned NSplineOrder>
void
Test_BSplineViaExternalTransformFile(const std::string & rootOutputDirectoryPath)
{
  const auto imageSize = itk::Size<NDimension>::Filled(4);
  using PixelType = float;

  elx::DefaultConstruct<itk::BSplineTransform<double, NDimension, NSplineOrder>> bsplineTransform;
  const auto inputImage = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageSize);
  bsplineTransform.SetTransformDomainPhysicalDimensions(ConvertToItkVector(imageSize));
  bsplineTransform.SetParameters(GeneratePseudoRandomParameters(bsplineTransform.GetParameters().size(), -1.0));

  DefaultConstructibleTransformixFilter<itk::Image<PixelType, NDimension>> transformixFilter;
  transformixFilter.SetMovingImage(inputImage);

  for (const std::string fileNameExtension : { "h5", "tfm" })
  {
    const std::string transformFilePathName = rootOutputDirectoryPath + '/' + std::to_string(NDimension) +
                                              "D_SplineOrder=" + std::to_string(NSplineOrder) + '.' + fileNameExtension;
    elx::TransformIO::Write(bsplineTransform, transformFilePathName);

    transformixFilter.SetTransformParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "Direction", CreateDefaultDirectionParameterValues<NDimension>() },
                              { "Index", ParameterValuesType(NDimension, "0") },
                              { "Origin", ParameterValuesType(NDimension, "0") },
                              { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                              { "Size", ConvertToParameterValues(imageSize) },
                              { "Transform", ParameterValuesType{ "File" } },
                              { "TransformFileName", { transformFilePathName } },
                              { "Spacing", ParameterValuesType(NDimension, "1") } }));
    transformixFilter.Update();

    const auto resampleImageFilter = CreateResampleImageFilter(*inputImage, bsplineTransform);

    ExpectEqualImages(Deref(transformixFilter.GetOutput()), Deref(resampleImageFilter->GetOutput()));
  }
}


using Lines = std::deque<std::string>;

auto
ReadFile(const std::string & filePath)
{
  Lines result;

  std::ifstream inputFileStream{ filePath };
  std::string   line;
  while (std::getline(inputFileStream, line))
  {
    result.push_back(line);
  }
  return result;
}


void
WriteFile(const std::string & filePath, const Lines & lines)
{
  std::ofstream outputFileStream{ filePath };
  for (const auto & line : lines)
  {
    outputFileStream << line << '\n';
  }
}


// Converts the lines from legacy elastix parameter file format to TOML.
void
ConvertTxtToToml(Lines & lines)
{
  unsigned numberOfParameters{};

  for (auto & line : lines)
  {
    if (!line.empty())
    {
      // TOML uses a hash symbol for comment.
      if (const auto pos = line.find("//"); pos != std::string::npos)
      {
        line.replace(pos, 2, "#");
      }

      std::match_results<std::string::const_iterator> results;

      // Try to match a line, having an elastix parameter, of the form "(Name Value ... Value)".
      if (std::regex_match(line, results, std::regex{ R"delimiter(\s*\(\s*(\w+)\s+([^\)]+)\s*\)(.*))delimiter" }) &&
          results.size() == 4)
      {
        ++numberOfParameters;

        auto values = results[2].str();

        assert(!values.empty());

        const auto removeDoubleQuotes = [&values](const std::string_view value) {
          for (;;)
          {
            if (const auto foundPos = values.find(value); foundPos == std::string::npos)
            {
              break;
            }
            else
            {
              values.replace(foundPos, value.size(), std::string_view(value.data() + 1, value.size() - 2));
            }
          }
        };

        // TOML uses lowercase for Booleans, just like the elastix legacy format, but without quotes.
        removeDoubleQuotes("\"false\"");
        removeDoubleQuotes("\"true\"");

        // Replace tabs with spaces.
        std::replace(values.begin(), values.end(), '\t', ' ');

        // Replace double spaces with single spaces.
        for (;;)
        {
          if (const auto foundPos = values.find("  "); foundPos == std::string::npos)
          {
            break;
          }
          else
          {
            values.replace(foundPos, 2, " ");
          }
        }

        unsigned numberOfAddedCommas{};
        if (const auto numberOfChars = values.size(); numberOfChars > 1)
        {
          bool evenNumberOfDoubleQuotes = values.back() != '"';

          // Prepend a comma to each space. TOML uses a comma as separator between two array elements.
          for (auto i = numberOfChars - 2; i > 0; --i)
          {
            if (values[i] == '"')
            {
              evenNumberOfDoubleQuotes = !evenNumberOfDoubleQuotes;
            }
            else
            {
              if (evenNumberOfDoubleQuotes && (values[i] == ' '))
              {
                values.replace(i, 1, ", ");
                ++numberOfAddedCommas;
              }
            }
          }
        }

        if (numberOfAddedCommas > 0)
        {
          // TOML uses square brackets for arrays.
          values = '[' + values + ']';
        }
        line = results[1].str() + " = " + values + results[3].str();
      }
    }
  }
}


void
ConvertTxtFileToToml(const std::string & inputFilePath, const std::string & outputFilePath)
{
  auto lines = ReadFile(inputFilePath);
  ConvertTxtToToml(lines);
  WriteFile(outputFilePath, lines);
}

} // namespace


GTEST_TEST(itkTransformixFilter, IsDefaultInitialized)
{
  static constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using TransformixFilterType = itk::TransformixFilter<itk::Image<PixelType, ImageDimension>>;

  const elx::DefaultConstruct<TransformixFilterType> transformixFilter;

  EXPECT_EQ(transformixFilter.GetFixedPointSetFileName(), std::string{});
  EXPECT_EQ(transformixFilter.GetOutputDirectory(), std::string{});
  EXPECT_FALSE(transformixFilter.GetComputeSpatialJacobian());
  EXPECT_FALSE(transformixFilter.GetComputeDeterminantOfSpatialJacobian());
  EXPECT_FALSE(transformixFilter.GetComputeDeformationField());
  EXPECT_EQ(transformixFilter.GetLogFileName(), std::string{});
  EXPECT_FALSE(transformixFilter.GetLogToConsole());
  EXPECT_FALSE(transformixFilter.GetLogToFile());
  EXPECT_EQ(transformixFilter.GetOutputMesh(), nullptr);
}


// Tests translating a small (5x6) binary image, having a 2x2 white square.
GTEST_TEST(itkTransformixFilter, Translation2D)
{
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;

  const itk::Offset<ImageDimension> translationOffset{ { 1, -2 } };
  const auto                        regionSize = SizeType::Filled(2);
  const SizeType                    imageSize{ { 5, 6 } };
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->AllocateInitialized();
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->AllocateInitialized();
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const auto transformedImage = TranslateImage(*movingImage, translationOffset);

  ExpectEqualImages(*transformedImage, *fixedImage);
}


// Tests translating a small (5x6) binary image, using a TransformParameterFileName.
GTEST_TEST(itkTransformixFilter, Translation2DTransformParameterFileName)
{
  static constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;

  const itk::Offset<ImageDimension> translationOffset{ { 1, -2 } };
  const auto                        regionSize = SizeType::Filled(2);
  const SizeType                    imageSize{ { 5, 6 } };
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->AllocateInitialized();
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->AllocateInitialized();
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  DefaultConstructibleTransformixFilter<ImageType> filter;

  filter.SetMovingImage(movingImage);
  filter.SetTransformParameterFileName(GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters-Size-5x6.txt");
  filter.Update();


  const auto transformedImage = filter.GetOutput();
  ExpectEqualImages(*transformedImage, *fixedImage);
}


// Tests translating a mesh of two points.
GTEST_TEST(itkTransformixFilter, MeshTranslation2D)
{
  static constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using TransformixFilterType = itk::TransformixFilter<itk::Image<PixelType, ImageDimension>>;
  using VectorType = itk::Vector<float, ImageDimension>;

  for (const auto & translationVector : { VectorType{}, VectorType(0.5f), itk::MakeVector(1.0f, -2.0f) })
  {
    elx::DefaultConstruct<TransformixFilterType> transformixFilter;
    const auto                                   inputMesh = TransformixFilterType::MeshType::New();
    inputMesh->SetPoint(0, {});
    inputMesh->SetPoint(1, itk::MakePoint(1.0f, 2.0f));

    transformixFilter.SetInputMesh(inputMesh);
    const auto movingImage =
      CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(itk::Size<ImageDimension>::Filled(1));

    transformixFilter.SetMovingImage(movingImage);

    const ParameterValuesType imageDimensionParameterValue = { std::to_string(ImageDimension) };
    const ParameterValuesType zeroParameterValues(ImageDimension, "0");
    const ParameterValuesType oneParameterValues(ImageDimension, "1");

    transformixFilter.SetTransformParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                              { "FixedImageDimension", imageDimensionParameterValue },
                              { "Index", zeroParameterValues },
                              { "MovingImageDimension", imageDimensionParameterValue },
                              { "NumberOfParameters", imageDimensionParameterValue },
                              { "Origin", zeroParameterValues },
                              { "Size", oneParameterValues },
                              { "Spacing", oneParameterValues },
                              { "Transform", ParameterValuesType{ "TranslationTransform" } },
                              { "TransformParameters", ConvertToParameterValues(translationVector) } }));
    transformixFilter.Update();

    const auto outputMesh = transformixFilter.GetOutputMesh();
    const auto expectedNumberOfPoints = inputMesh->GetNumberOfPoints();

    const auto & inputPoints = Deref(DerefSmartPointer(inputMesh).GetPoints());
    const auto & outputPoints = Deref(Deref(outputMesh).GetPoints());

    ASSERT_EQ(outputPoints.size(), expectedNumberOfPoints);

    for (size_t i = 0; i < expectedNumberOfPoints; ++i)
    {
      EXPECT_EQ(outputPoints[i], inputPoints[i] + translationVector);
    }
  }
}


// Tests translating a small (5x7x9) binary 3D image, having a 2x2x2 white cube.
GTEST_TEST(itkTransformixFilter, Translation3D)
{
  static constexpr auto ImageDimension = 3U;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;

  const itk::Offset<ImageDimension> translationOffset{ { 1, 2, 3 } };
  const auto                        regionSize = SizeType::Filled(2);
  const SizeType                    imageSize{ { 5, 7, 9 } };
  const itk::Index<ImageDimension>  fixedImageRegionIndex{ { 1, 2, 3 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->AllocateInitialized();
  FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->AllocateInitialized();
  FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const auto transformedImage = TranslateImage(*movingImage, translationOffset);

  ExpectEqualImages(*transformedImage, *fixedImage);
}


GTEST_TEST(itkTransformixFilter, TranslationViaExternalTransformFile)
{
  static constexpr auto ImageDimension = 2U;
  using PixelType = float;

  const itk::Offset<ImageDimension> translationOffset{ { 1, -2 } };
  const itk::Size<ImageDimension>   imageSize{ { 5, 6 } };

  const auto movingImage = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageSize);
  const auto expectedOutputImage = TranslateImage(*movingImage, translationOffset);

  for (const std::string transformFileName :
       { "ITK-Transform.tfm", "ITK-HDF5-Transform.h5", "Special characters [(0-9,;!@#$%&)]/ITK-Transform.tfm" })
  {
    const auto transformFilePathName = GetDataDirectoryPath() + "/Translation(1,-2)/" + transformFileName;
    DefaultConstructibleTransformixFilter<itk::Image<PixelType, ImageDimension>> filter;

    filter.SetMovingImage(movingImage);
    filter.SetTransformParameterObject(
      CreateParameterObject({ // Parameters in alphabetic order:
                              { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                              { "Index", ParameterValuesType(ImageDimension, "0") },
                              { "Origin", ParameterValuesType(ImageDimension, "0") },
                              { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                              { "Size", ConvertToParameterValues(imageSize) },
                              { "Transform", ParameterValuesType{ "File" } },
                              { "TransformFileName", { transformFilePathName } },
                              { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
    filter.Update();
    const auto * const outputImage = filter.GetOutput();
    ExpectEqualImages(Deref(outputImage), *expectedOutputImage);
  }
}


GTEST_TEST(itkTransformixFilter, BSplineViaExternalTransformFile)
{
  const std::string rootOutputDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
  itk::FileTools::CreateDirectory(rootOutputDirectoryPath);

  Test_BSplineViaExternalTransformFile<2, 1>(rootOutputDirectoryPath);
  Test_BSplineViaExternalTransformFile<3, 1>(rootOutputDirectoryPath);
  Test_BSplineViaExternalTransformFile<2, 2>(rootOutputDirectoryPath);
  Test_BSplineViaExternalTransformFile<3, 2>(rootOutputDirectoryPath);
  Test_BSplineViaExternalTransformFile<2, 3>(rootOutputDirectoryPath);
  Test_BSplineViaExternalTransformFile<3, 3>(rootOutputDirectoryPath);
}


GTEST_TEST(itkTransformixFilter, ITKTranslationTransform2D)
{
  static constexpr auto ImageDimension = 2U;

  elx::DefaultConstruct<itk::TranslationTransform<double, ImageDimension>> itkTransform;
  itkTransform.SetOffset(itk::MakeVector(1.0, -2.0));

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(itk::MakeSize(5, 6)), itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKTranslationTransform3D)
{
  static constexpr auto ImageDimension = 3U;

  elx::DefaultConstruct<itk::TranslationTransform<double, ImageDimension>> itkTransform;
  itkTransform.SetOffset(itk::MakeVector(1.0, -2.0, 3.0));

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(itk::MakeSize(5, 6, 7)), itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKAffineTransform2D)
{
  static constexpr auto ImageDimension = 2U;

  elx::DefaultConstruct<itk::AffineTransform<double, ImageDimension>> itkTransform;
  itkTransform.SetTranslation(itk::MakeVector(1.0, -2.0));
  itkTransform.SetCenter(itk::MakePoint(2.5, 3.0));
  itkTransform.Rotate2D(M_PI_4);

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(itk::MakeSize(5, 6)), itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKAffineTransform3D)
{
  static constexpr auto ImageDimension = 3U;

  elx::DefaultConstruct<itk::AffineTransform<double, ImageDimension>> itkTransform;
  itkTransform.SetTranslation(itk::MakeVector(1.0, 2.0, 3.0));
  itkTransform.SetCenter(itk::MakePoint(3.0, 2.0, 1.0));
  itkTransform.Rotate3D(itk::Vector<double, ImageDimension>(1.0), M_PI_4);

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(itk::MakeSize(5, 6, 7)), itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKEulerTransform2D)
{
  elx::DefaultConstruct<itk::Euler2DTransform<double>> itkTransform;
  itkTransform.SetTranslation(itk::MakeVector(1.0, -2.0));
  itkTransform.SetCenter(itk::MakePoint(2.5, 3.0));
  itkTransform.SetAngle(M_PI_4);

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(itk::MakeSize(5, 6)), itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKEulerTransform3D)
{
  elx::DefaultConstruct<itk::Euler3DTransform<double>> itkTransform;
  itkTransform.SetTranslation(itk::MakeVector(1.0, -2.0, 3.0));
  itkTransform.SetCenter(itk::MakePoint(3.0, 2.0, 1.0));
  itkTransform.SetRotation(M_PI_2, M_PI_4, M_PI_4 / 2.0);

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(itk::MakeSize(5, 6, 7)), itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKSimilarityTransform2D)
{
  elx::DefaultConstruct<itk::Similarity2DTransform<double>> itkTransform;
  itkTransform.SetScale(0.75);
  itkTransform.SetTranslation(itk::MakeVector(1.0, -2.0));
  itkTransform.SetCenter(itk::MakePoint(2.5, 3.0));
  itkTransform.SetAngle(M_PI_4);

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(itk::MakeSize(5, 6)), itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKSimilarityTransform3D)
{
  elx::DefaultConstruct<itk::Similarity3DTransform<double>> itkTransform;
  itkTransform.SetScale(0.75);
  itkTransform.SetTranslation(itk::MakeVector(1.0, -2.0, 3.0));
  itkTransform.SetCenter(itk::MakePoint(3.0, 2.0, 1.0));
  itkTransform.SetRotation(itk::Vector<double, 3>(1.0), M_PI_4);

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(itk::MakeSize(5, 6, 7)), itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKBSplineTransform2D)
{
  elx::DefaultConstruct<itk::BSplineTransform<double, 2>> itkTransform;

  const auto imageSize = itk::MakeSize(5, 6);

  // Note that this unit test would fail if TransformDomainPhysicalDimensions would not be set.
  itkTransform.SetTransformDomainPhysicalDimensions(ConvertToItkVector(imageSize));
  itkTransform.SetParameters(GeneratePseudoRandomParameters(itkTransform.GetParameters().size(), -1.0));

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(imageSize), itkTransform);
}


GTEST_TEST(itkTransformixFilter, ITKBSplineTransform3D)
{
  elx::DefaultConstruct<itk::BSplineTransform<double, 3>> itkTransform;

  const auto imageSize = itk::MakeSize(5, 6, 7);

  // Note that this unit test would fail if TransformDomainPhysicalDimensions would not be set.
  itkTransform.SetTransformDomainPhysicalDimensions(ConvertToItkVector(imageSize));
  itkTransform.SetParameters(GeneratePseudoRandomParameters(itkTransform.GetParameters().size(), -1.0));

  Expect_TransformixFilter_output_equals_ResampleImageFilter_output(
    *CreateImageFilledWithSequenceOfNaturalNumbers<float>(imageSize), itkTransform);
}


GTEST_TEST(itkTransformixFilter, CombineTranslationAndDefaultTransform)
{
  const auto            imageSize = itk::MakeSize(5, 6);
  const auto            inputImage = CreateImageFilledWithSequenceOfNaturalNumbers<float>(imageSize);
  static constexpr auto dimension = decltype(imageSize)::Dimension;

  using ParametersValueType = double;

  // Create a translated image, which is the expected output image.
  elx::DefaultConstruct<itk::TranslationTransform<ParametersValueType, dimension>> translationTransform;
  translationTransform.SetOffset(itk::MakeVector(1, -2));
  const auto   resampleImageFilter = CreateResampleImageFilter(*inputImage, translationTransform);
  const auto & expectedOutputImage = Deref(resampleImageFilter->GetOutput());

  const std::string initialTransformParameterFileName =
    GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt";

  for (const auto defaultTransform : { CreateTransform<itk::AffineTransform<ParametersValueType, dimension>>(),
                                       CreateTransform<itk::BSplineTransform<ParametersValueType, dimension>>(),
                                       CreateTransform<itk::Euler2DTransform<ParametersValueType>>(),
                                       CreateTransform<itk::Similarity2DTransform<ParametersValueType>>(),
                                       CreateTransform<itk::TranslationTransform<ParametersValueType, dimension>>() })
  {
    const auto actualOutputImage =
      RetrieveOutputFromTransformixFilter(*inputImage, *defaultTransform, initialTransformParameterFileName);
    EXPECT_EQ(*actualOutputImage, expectedOutputImage);
  }

  const elx::DefaultConstruct<itk::TranslationTransform<ParametersValueType, dimension>> defaultTransform;

  for (const std::string transformParameterFileName :
       { "TransformParameters-link-to-ITK-tfm-file.txt",
         "TransformParameters-link-to-ITK-HDF5-file.txt",
         "TransformParameters-link-to-file-with-special-chars-in-path-name.txt" })
  {
    const auto actualOutputImage = RetrieveOutputFromTransformixFilter(
      *inputImage, defaultTransform, GetDataDirectoryPath() + "/Translation(1,-2)/" + transformParameterFileName);
    EXPECT_EQ(*actualOutputImage, expectedOutputImage);
  }
}


GTEST_TEST(itkTransformixFilter, CombineTranslationAndInverseTranslation)
{
  const auto            imageSize = itk::MakeSize(5, 6);
  static constexpr auto dimension = decltype(imageSize)::Dimension;

  const auto inputImage = itk::Image<float, dimension>::New();
  inputImage->SetRegions(imageSize);
  inputImage->AllocateInitialized();
  FillImageRegion(*inputImage, { 2, 1 }, itk::Size<dimension>::Filled(2));

  using ParametersValueType = double;

  const std::string initialTransformParameterFileName =
    GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt";

  const auto offset = itk::MakeVector(1.0, -2.0);
  const auto inverseOffset = -offset;

  // Sanity check: when only an identity transform is applied, the transform from the TransformParameters.txt file
  // makes the output image unequal to the input image.
  const elx::DefaultConstruct<itk::TranslationTransform<ParametersValueType, dimension>> identityTransform;

  EXPECT_NE(*(RetrieveOutputFromTransformixFilter(*inputImage, identityTransform, initialTransformParameterFileName)),
            *inputImage);

  // The inverse of the transform from the TransformParameters.txt file.
  const auto inverseTranslationTransform = [inverseOffset] {
    const auto transform = itk::TranslationTransform<ParametersValueType, dimension>::New();
    transform->SetOffset(inverseOffset);
    return transform;
  }();

  EXPECT_EQ(*(RetrieveOutputFromTransformixFilter(
              *inputImage, *inverseTranslationTransform, initialTransformParameterFileName)),
            *inputImage);

  for (const auto matrixOffsetTransform :
       { CreateMatrixOffsetTransform<itk::AffineTransform<ParametersValueType, dimension>>(),
         CreateMatrixOffsetTransform<itk::Euler2DTransform<ParametersValueType>>(),
         CreateMatrixOffsetTransform<itk::Similarity2DTransform<ParametersValueType>>() })
  {
    matrixOffsetTransform->SetOffset(inverseOffset);
    EXPECT_EQ(
      *(RetrieveOutputFromTransformixFilter(*inputImage, *matrixOffsetTransform, initialTransformParameterFileName)),
      *inputImage);
  }

  const auto inverseBSplineTransform = [imageSize, inverseOffset] {
    const auto transform = itk::BSplineTransform<ParametersValueType, dimension>::New();
    transform->SetTransformDomainPhysicalDimensions(ConvertToItkVector(imageSize));

    const auto                       numberOfParameters = transform->GetParameters().size();
    itk::OptimizerParameters<double> parameters(numberOfParameters, inverseOffset[1]);
    std::fill_n(parameters.begin(), numberOfParameters / 2, inverseOffset[0]);
    transform->SetParameters(parameters);
    return transform;
  }();

  const auto inverseBSplineOutputImage = RetrieveOutputFromTransformixFilter(
    *inputImage, *inverseBSplineTransform, initialTransformParameterFileName, "Add");
  ExpectAlmostEqualPixelValues(*inverseBSplineOutputImage, *inputImage, 1e-15);
}


GTEST_TEST(itkTransformixFilter, CombineTranslationAndScale)
{
  const auto            imageSize = itk::MakeSize(5, 6);
  static constexpr auto dimension = decltype(imageSize)::Dimension;
  const auto            inputImage = CreateImageFilledWithSequenceOfNaturalNumbers<float>(imageSize);

  using ParametersValueType = double;

  const std::string initialTransformParameterFileName =
    GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt";

  elx::DefaultConstruct<itk::AffineTransform<ParametersValueType, dimension>> scaleTransform;
  scaleTransform.Scale(2.0);

  elx::DefaultConstruct<itk::TranslationTransform<ParametersValueType, dimension>> translationTransform;
  translationTransform.SetOffset(itk::MakeVector(1.0, -2.0));

  const auto transformixOutput =
    RetrieveOutputFromTransformixFilter(*inputImage, scaleTransform, initialTransformParameterFileName);

  elx::DefaultConstruct<itk::CompositeTransform<double, 2>> translationAndScaleTransform;
  translationAndScaleTransform.AddTransform(&translationTransform);
  translationAndScaleTransform.AddTransform(&scaleTransform);

  elx::DefaultConstruct<itk::CompositeTransform<double, 2>> scaleAndTranslationTransform;
  scaleAndTranslationTransform.AddTransform(&scaleTransform);
  scaleAndTranslationTransform.AddTransform(&translationTransform);

  // Expect that Transformix output is unequal (!) to the output of the corresponding ITK translation + scale composite
  // transform.
  EXPECT_NE(DerefSmartPointer(transformixOutput),
            *(CreateResampleImageFilter(*inputImage, translationAndScaleTransform)->GetOutput()));

  // Expect that Transformix output is equal to the output of the corresponding ITK scale + translation composite
  // transform (in that order). Note that itk::CompositeTransform processed the transforms in reverse order (compared to
  // elastix).
  EXPECT_EQ(DerefSmartPointer(transformixOutput),
            *(CreateResampleImageFilter(*inputImage, scaleAndTranslationTransform)->GetOutput()));
}


GTEST_TEST(itkTransformixFilter, OutputEqualsRegistrationOutputForStackTransform)
{
  using PixelType = float;
  static constexpr auto ImageDimension = 3U;

  enum
  {
    imageSizeX = 5,
    imageSizeY = 6,
    imageSizeZ = 4
  };

  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<PixelType, ImageDimension>({ imageSizeX, imageSizeY, imageSizeZ });

  for (const std::string transformName :
       { "AffineLogStackTransform", "BSplineStackTransform", "TranslationStackTransform", "EulerStackTransform" })
  {
    for (const std::string fileNameExtension : { "", "h5", "tfm" })
    {
      Expect_Transformix_output_equals_registration_output_from_file(
        *this,
        transformName + fileNameExtension,
        *image,
        *image,
        ParameterMapType{ // Parameters in alphabetic order:
                          { "AutomaticTransformInitialization", { "false" } },
                          { "ImageSampler", { "Full" } },
                          { "MaximumNumberOfIterations", { "2" } },
                          { "Metric", { "VarianceOverLastDimensionMetric" } },
                          { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                          { "ITKTransformOutputFileNameExtension", { fileNameExtension } },
                          { "Transform", { transformName } } });
    }
  }
}


GTEST_TEST(itkTransformixFilter, OutputEqualsRegistrationOutputForBSplineStackTransform)
{
  using PixelType = float;
  enum
  {
    ImageDimension = 3,
    imageSizeX = 5,
    imageSizeY = 6,
    imageSizeZ = 4
  };

  const auto image =
    CreateImageFilledWithSequenceOfNaturalNumbers<PixelType, ImageDimension>({ imageSizeX, imageSizeY, imageSizeZ });
  const std::string transformName = "BSplineStackTransform";
  for (const unsigned splineOrder : { 1, 2, 3 })
  {
    for (const std::string fileNameExtension : { "", "h5", "tfm" })
    {
      Expect_Transformix_output_equals_registration_output_from_file(
        *this,
        "SplineOrder=" + std::to_string(splineOrder) + (fileNameExtension.empty() ? "" : ('_' + fileNameExtension)),
        *image,
        *image,
        ParameterMapType{ // Parameters in alphabetic order:
                          { "AutomaticTransformInitialization", { "false" } },
                          { "BSplineTransformSplineOrder", { std::to_string(splineOrder) } },
                          { "ImageSampler", { "Full" } },
                          { "MaximumNumberOfIterations", { "2" } },
                          { "Metric", { "VarianceOverLastDimensionMetric" } },
                          { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                          { "ITKTransformOutputFileNameExtension", { fileNameExtension } },
                          { "Transform", { transformName } } });
    }
  }
}


// Tests setting an `itk::TranslationTransform`, to transform a simple image and a small mesh.
GTEST_TEST(itkTransformixFilter, SetTranslationTransform)
{
  using PixelType = float;
  static constexpr unsigned int ImageDimension{ 2 };

  using SizeType = itk::Size<ImageDimension>;
  const itk::Offset<ImageDimension> translationOffset{ { 1, -2 } };
  const auto                        translationVector = ConvertToItkVector(translationOffset);

  const auto                       regionSize = SizeType::Filled(2);
  const SizeType                   imageSize{ { 5, 6 } };
  const itk::Index<ImageDimension> fixedImageRegionIndex{ { 1, 3 } };

  using ImageType = itk::Image<PixelType, ImageDimension>;
  using TransformixFilterType = itk::TransformixFilter<ImageType>;

  elx::DefaultConstruct<ImageType> fixedImage{};
  fixedImage.SetRegions(imageSize);
  fixedImage.AllocateInitialized();
  FillImageRegion(fixedImage, fixedImageRegionIndex, regionSize);

  elx::DefaultConstruct<ImageType> movingImage{};
  movingImage.SetRegions(imageSize);
  movingImage.AllocateInitialized();
  FillImageRegion(movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  elx::DefaultConstruct<itk::TranslationTransform<double, ImageDimension>> transform{};
  transform.SetOffset(translationVector);

  elx::DefaultConstruct<TransformixFilterType::MeshType> inputMesh{};
  inputMesh.SetPoint(0, {});
  inputMesh.SetPoint(1, itk::MakePoint(1.0f, 2.0f));

  elx::DefaultConstruct<TransformixFilterType> transformixFilter{};
  transformixFilter.SetInputMesh(&inputMesh);
  transformixFilter.SetMovingImage(&movingImage);
  transformixFilter.SetTransform(&transform);
  transformixFilter.SetTransformParameterObject(
    CreateParameterObject({ // Parameters in alphabetic order:
                            { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                            { "Index", ParameterValuesType(ImageDimension, "0") },
                            { "Origin", ParameterValuesType(ImageDimension, "0") },
                            { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                            { "Size", ConvertToParameterValues(imageSize) },
                            { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
  transformixFilter.Update();

  ExpectEqualImages(Deref(transformixFilter.GetOutput()), fixedImage);

  const auto outputMesh = transformixFilter.GetOutputMesh();
  const auto expectedNumberOfPoints = inputMesh.GetNumberOfPoints();

  const auto & inputPoints = Deref(inputMesh.GetPoints());
  const auto & outputPoints = Deref(Deref(outputMesh).GetPoints());

  ASSERT_EQ(outputPoints.size(), expectedNumberOfPoints);

  for (size_t i = 0; i < expectedNumberOfPoints; ++i)
  {
    EXPECT_EQ(outputPoints[i], inputPoints[i] + translationVector);
  }
}


GTEST_TEST(itkTransformixFilter, SetCombinationTransform)
{
  static constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using ImageType = itk::Image<PixelType, ImageDimension>;
  const itk::Size<ImageDimension> imageSize{ { 5, 6 } };

  const auto numberOfPixels = imageSize.CalculateProductOfElements();

  const auto fixedImage = CreateImage<PixelType>(imageSize);
  const auto movingImage = CreateImage<PixelType>(imageSize);

  std::mt19937 randomNumberEngine;

  std::generate_n(fixedImage->GetBufferPointer(), numberOfPixels, [&randomNumberEngine] {
    return std::uniform_real_distribution<>{ 1, 32 }(randomNumberEngine);
  });
  std::generate_n(movingImage->GetBufferPointer(), numberOfPixels, [&randomNumberEngine] {
    return std::uniform_real_distribution<>{ 32, 64 }(randomNumberEngine);
  });

  EXPECT_NE(*movingImage, *fixedImage);

  for (const bool useInitialTransform : { false, true })
  {
    const std::string initialTransformParameterFileName =
      useInitialTransform ? (GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters.txt") : "";

    for (const char * const transformName : { "AffineTransform",
                                              "BSplineTransform",
                                              "EulerTransform",
                                              "RecursiveBSplineTransform",
                                              "SimilarityTransform",
                                              "TranslationTransform" })
    {
      elx::DefaultConstruct<itk::ElastixRegistrationMethod<ImageType, ImageType>> registration;
      registration.SetFixedImage(fixedImage);
      registration.SetMovingImage(movingImage);
      registration.SetInitialTransformParameterFileName(initialTransformParameterFileName);
      registration.SetParameterObject(CreateParameterObject({ // Parameters in alphabetic order:
                                                              { "AutomaticTransformInitialization", { "false" } },
                                                              { "ImageSampler", { "Full" } },
                                                              { "MaximumNumberOfIterations", { "2" } },
                                                              { "Metric", { "AdvancedNormalizedCorrelation" } },
                                                              { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                                                              { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                                                              { "Transform", { transformName } } }));
      registration.Update();

      const ImageType & registrationOutputImage = Deref(registration.GetOutput());

      EXPECT_NE(registrationOutputImage, *fixedImage);
      EXPECT_NE(registrationOutputImage, *movingImage);

      const auto combinationTransform = registration.GetCombinationTransform();

      EXPECT_NE(combinationTransform, nullptr);

      elx::DefaultConstruct<itk::TransformixFilter<ImageType>> transformixFilter{};
      transformixFilter.SetMovingImage(movingImage);
      transformixFilter.SetCombinationTransform(combinationTransform);
      transformixFilter.SetTransformParameterObject(
        CreateParameterObject({ // Parameters in alphabetic order:
                                { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                                { "Index", ParameterValuesType(ImageDimension, "0") },
                                { "Origin", ParameterValuesType(ImageDimension, "0") },
                                { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                                { "Size", ConvertToParameterValues(imageSize) },
                                { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
      transformixFilter.Update();

      ExpectEqualImages(Deref(transformixFilter.GetOutput()), registrationOutputImage);
    }
  }
}


// Tests that Update() throws an exception when the transform parameter object has zero parameter maps.
GTEST_TEST(itkTransformixFilter, UpdateThrowsExceptionOnZeroParameterMaps)
{
  using PixelType = float;
  static constexpr unsigned int ImageDimension{ 2 };
  using ImageType = itk::Image<PixelType, ImageDimension>;
  const auto imageSize = ImageType::SizeType::Filled(2);

  for (const bool useZeroParameterMaps : { false, true })
  {
    elx::DefaultConstruct<ImageType> image{};
    image.SetRegions(imageSize);
    image.AllocateInitialized();

    elx::DefaultConstruct<itk::TranslationTransform<double, ImageDimension>> transform{};

    elx::DefaultConstruct<elx::ParameterObject> transformParameterObject{};

    const auto parameterMaps = useZeroParameterMaps
                                 ? ParameterMapVectorType{}
                                 : ParameterMapVectorType{ ParameterMapType{
                                     { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                                     { "Index", ParameterValuesType(ImageDimension, "0") },
                                     { "Origin", ParameterValuesType(ImageDimension, "0") },
                                     { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                                     { "Size", ConvertToParameterValues(imageSize) },
                                     { "Spacing", ParameterValuesType(ImageDimension, "1") } } };

    transformParameterObject.SetParameterMaps(parameterMaps);

    elx::DefaultConstruct<itk::TransformixFilter<ImageType>> transformixFilter{};
    transformixFilter.SetMovingImage(&image);
    transformixFilter.SetTransform(&transform);
    transformixFilter.SetTransformParameterObject(&transformParameterObject);

    if (useZeroParameterMaps)
    {
      EXPECT_THROW(transformixFilter.Update(), itk::ExceptionObject);
    }
    else
    {
      // A valid parameter map was specified, do not expect an exception when calling Update(). (This is just a sanity
      // check. The essential check is in the `if (useZeroParameterMaps)` clause.)
      transformixFilter.Update();
    }
  }
}


// Tests that Update() throws an exception when the transform is a CompositeTransform that has zero subtransforms.
GTEST_TEST(itkTransformixFilter, UpdateThrowsExceptionOnEmptyCompositeTransform)
{
  using PixelType = float;
  static constexpr unsigned int ImageDimension{ 2 };
  using ImageType = itk::Image<PixelType, ImageDimension>;
  const itk::Size<ImageDimension> imageSize{ { 5, 6 } };

  elx::DefaultConstruct<ImageType> movingImage{};
  movingImage.SetRegions(imageSize);
  movingImage.AllocateInitialized();

  elx::DefaultConstruct<itk::TranslationTransform<double, ImageDimension>> translationTransform{};
  elx::DefaultConstruct<itk::CompositeTransform<double, ImageDimension>>   compositeTransform{};
  compositeTransform.AddTransform(&translationTransform);

  const elx::DefaultConstruct<itk::CompositeTransform<double, ImageDimension>> emptyCompositeTransform{};

  elx::DefaultConstruct<itk::TransformixFilter<ImageType>> transformixFilter{};
  transformixFilter.SetMovingImage(&movingImage);
  transformixFilter.SetTransformParameterObject(
    CreateParameterObject({ // Parameters in alphabetic order:
                            { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                            { "Index", ParameterValuesType(ImageDimension, "0") },
                            { "Origin", ParameterValuesType(ImageDimension, "0") },
                            { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                            { "Size", ConvertToParameterValues(imageSize) },
                            { "Spacing", ParameterValuesType(ImageDimension, "1") } }));

  for ([[maybe_unused]] const bool isSecondIteration : { false, true })
  {
    transformixFilter.SetTransform(&emptyCompositeTransform);
    EXPECT_THROW(transformixFilter.Update(), itk::ExceptionObject);

    // compositeTransform is non-empty.
    transformixFilter.SetTransform(&compositeTransform);
    transformixFilter.Update();
  }
}


// Tests setting an `itk::CompositeTransform` which consists of a translation and a scaling.
GTEST_TEST(itkTransformixFilter, SetCompositeTransformOfTranslationAndScale)
{
  using PixelType = float;
  const auto                    imageSize = itk::MakeSize(5, 6);
  static constexpr unsigned int ImageDimension{ decltype(imageSize)::Dimension };
  using ImageType = itk::Image<PixelType, ImageDimension>;

  const auto inputImage = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageSize);

  using ParametersValueType = double;

  elx::DefaultConstruct<itk::AffineTransform<ParametersValueType, ImageDimension>> scaleTransform{};
  scaleTransform.Scale(2.0);

  elx::DefaultConstruct<itk::TranslationTransform<ParametersValueType, ImageDimension>> translationTransform{};
  translationTransform.SetOffset(itk::MakeVector(1.0, -2.0));

  elx::DefaultConstruct<itk::CompositeTransform<double, 2>> compositeTransform{};
  compositeTransform.AddTransform(&scaleTransform);
  compositeTransform.AddTransform(&translationTransform);

  const ParameterMapType transformParameterMap = {
    // Parameters in alphabetic order:
    { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
    { "Index", ParameterValuesType(ImageDimension, "0") },
    { "Origin", ParameterValuesType(ImageDimension, "0") },
    { "ResampleInterpolator", { "FinalLinearInterpolator" } },
    { "Size", ConvertToParameterValues(imageSize) },
    { "Spacing", ParameterValuesType(ImageDimension, "1") }
  };

  for (size_t numberOfParameterMaps{ 1 }; numberOfParameterMaps <= 3; ++numberOfParameterMaps)
  {
    elx::DefaultConstruct<elx::ParameterObject> transformParameterObject{};
    transformParameterObject.SetParameterMaps(ParameterMapVectorType(numberOfParameterMaps, transformParameterMap));

    elx::DefaultConstruct<itk::TransformixFilter<ImageType>> transformixFilter{};
    transformixFilter.SetMovingImage(inputImage);
    transformixFilter.SetTransform(&compositeTransform);
    transformixFilter.SetTransformParameterObject(&transformParameterObject);
    transformixFilter.Update();

    EXPECT_EQ(Deref(transformixFilter.GetOutput()),
              *(CreateResampleImageFilter(*inputImage, compositeTransform)->GetOutput()));
  }
}


// Tests ComputeSpatialJacobianDeterminantImage and ComputeSpatialJacobianMatrixImage on a simple translation.
GTEST_TEST(itkTransformixFilter, ComputeSpatialJacobianDeterminantImage)
{
  using PixelType = float;
  static constexpr unsigned int ImageDimension{ 2 };

  using SizeType = itk::Size<ImageDimension>;
  const SizeType imageSize{ { 5, 6 } };

  using ImageType = itk::Image<PixelType, ImageDimension>;
  using TransformixFilterType = itk::TransformixFilter<ImageType>;

  elx::DefaultConstruct<ImageType> movingImage{};
  movingImage.SetRegions(imageSize);
  movingImage.AllocateInitialized();

  elx::DefaultConstruct<itk::TranslationTransform<double, ImageDimension>> transform{};
  transform.SetOffset(itk::MakeVector(1.0, -2.0));

  elx::DefaultConstruct<TransformixFilterType> transformixFilter{};
  transformixFilter.SetMovingImage(&movingImage);
  transformixFilter.SetTransform(&transform);
  transformixFilter.SetTransformParameterObject(
    CreateParameterObject({ // Parameters in alphabetic order:
                            { "Direction", CreateDefaultDirectionParameterValues<ImageDimension>() },
                            { "Index", ParameterValuesType(ImageDimension, "0") },
                            { "Origin", ParameterValuesType(ImageDimension, "0") },
                            { "ResampleInterpolator", { "FinalLinearInterpolator" } },
                            { "Size", ConvertToParameterValues(imageSize) },
                            { "Spacing", ParameterValuesType(ImageDimension, "1") } }));
  transformixFilter.Update();

  const auto determinantImage = transformixFilter.ComputeSpatialJacobianDeterminantImage();
  const auto matrixImage = transformixFilter.ComputeSpatialJacobianMatrixImage();

  const itk::ImageRegion<ImageDimension> expectedBufferedRegion({}, imageSize);
  EXPECT_EQ(DerefSmartPointer(determinantImage).GetBufferedRegion(), expectedBufferedRegion);
  EXPECT_EQ(DerefSmartPointer(matrixImage).GetBufferedRegion(), expectedBufferedRegion);

  for (const auto determinant : itk::MakeImageBufferRange(determinantImage.GetPointer()))
  {
    EXPECT_EQ(determinant, 1.0f);
  }

  const auto expectedMatrix = TransformixFilterType::SpatialJacobianMatrixImageType::PixelType::GetIdentity();

  for (const auto & matrix : itk::MakeImageBufferRange(matrixImage.GetPointer()))
  {
    EXPECT_EQ(matrix, expectedMatrix);
  }
}


// Checks a minimum size moving image having the same pixel type as any of the supported internal pixel types.
GTEST_TEST(itkTransformixFilter, CheckMinimumMovingImageHavingInternalPixelType)
{
  elx::ForEachSupportedImageType([](const auto elxTypedef) {
    using ElxTypedef = decltype(elxTypedef);
    using ImageType = typename ElxTypedef::MovingImageType;
    static constexpr auto ImageDimension = ElxTypedef::MovingDimension;

    using PixelType = typename ImageType::PixelType;
    using SizeType = itk::Size<ImageDimension>;

    const ImageDomain<ElxTypedef::MovingDimension> imageDomain(
      itk::Size<ElxTypedef::MovingDimension>::Filled(minimumImageSizeValue));

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

    const ParameterMapType parameterMap = MakeMergedMap(
      { // Parameters in alphabetic order:
        { "FixedInternalImagePixelType", { ElxTypedef::FixedPixelTypeString } },
        { "MovingInternalImagePixelType", { ElxTypedef::MovingPixelTypeString } },
        { "ResampleInterpolator", { "FinalLinearInterpolator" } } },
      imageDomain.AsParameterMap());

    const elx::DefaultConstruct<itk::TranslationTransform<double, ImageType::ImageDimension>> identityTransform{};
    elx::DefaultConstruct<itk::TransformixFilter<ImageType>>                                  transformixFilter{};

    transformixFilter.SetMovingImage(&movingImage);
    transformixFilter.SetTransform(&identityTransform);
    transformixFilter.SetTransformParameterObject(CreateParameterObject(parameterMap));
    transformixFilter.Update();

    EXPECT_EQ(Deref(transformixFilter.GetOutput()), movingImage);
  });
}


// Checks a zero-filled moving image with a random domain, having the same pixel type as any of the supported internal
// pixel types.
GTEST_TEST(itkTransformixFilter, CheckZeroFilledMovingImageWithRandomDomainHavingInternalPixelType)
{
  std::mt19937 randomNumberEngine{};

  elx::ForEachSupportedImageType([&randomNumberEngine](const auto elxTypedef) {
    using ElxTypedef = decltype(elxTypedef);
    using ImageType = typename ElxTypedef::MovingImageType;
    static constexpr auto ImageDimension = ElxTypedef::MovingDimension;

    using PixelType = typename ImageType::PixelType;
    using SizeType = itk::Size<ImageDimension>;

    const auto imageDomain = CreateRandomImageDomain<ElxTypedef::MovingDimension>(randomNumberEngine);

    elx::DefaultConstruct<ImageType> movingImage{};
    imageDomain.ToImage(movingImage);
    movingImage.AllocateInitialized();

    const ParameterMapType parameterMap = MakeMergedMap(
      { // Parameters in alphabetic order:
        { "FixedInternalImagePixelType", { ElxTypedef::FixedPixelTypeString } },
        { "MovingInternalImagePixelType", { ElxTypedef::MovingPixelTypeString } },
        { "ResampleInterpolator", { "FinalLinearInterpolator" } } },
      imageDomain.AsParameterMap());

    const elx::DefaultConstruct<itk::TranslationTransform<double, ImageType::ImageDimension>> identityTransform{};
    elx::DefaultConstruct<itk::TransformixFilter<ImageType>>                                  transformixFilter{};

    transformixFilter.SetMovingImage(&movingImage);
    transformixFilter.SetTransform(&identityTransform);
    transformixFilter.SetTransformParameterObject(CreateParameterObject(parameterMap));
    transformixFilter.Update();

    EXPECT_EQ(Deref(transformixFilter.GetOutput()), movingImage);
  });
}


// Checks a minimum size moving image using any supported internal pixel type (which may be different from the input
// pixel type).
GTEST_TEST(itkTransformixFilter, CheckMinimumMovingImageUsingAnyInternalPixelType)
{
  const auto check = [](const auto inputPixelTypeHolder) {
    (void)inputPixelTypeHolder;
    elx::ForEachSupportedImageType([](const auto elxTypedef) {
      using ElxTypedef = decltype(elxTypedef);
      using InputPixelType = typename decltype(inputPixelTypeHolder)::Type;
      using InputImageType = itk::Image<InputPixelType, ElxTypedef::MovingDimension>;

      const ImageDomain<ElxTypedef::MovingDimension> imageDomain(
        itk::Size<ElxTypedef::MovingDimension>::Filled(minimumImageSizeValue));
      const auto movingImage = CreateImageFilledWithSequenceOfNaturalNumbers<InputPixelType>(imageDomain);

      const ParameterMapType parameterMap = MakeMergedMap(
        { // Parameters in alphabetic order:
          { "FixedInternalImagePixelType", { ElxTypedef::FixedPixelTypeString } },
          { "MovingInternalImagePixelType", { ElxTypedef::MovingPixelTypeString } },
          { "ResampleInterpolator", { "FinalLinearInterpolator" } } },
        imageDomain.AsParameterMap());

      const elx::DefaultConstruct<itk::TranslationTransform<double, ElxTypedef::MovingDimension>> identityTransform{};
      elx::DefaultConstruct<itk::TransformixFilter<InputImageType>>                               transformixFilter{};

      transformixFilter.SetMovingImage(movingImage);
      transformixFilter.SetTransform(&identityTransform);
      transformixFilter.SetTransformParameterObject(CreateParameterObject(parameterMap));
      transformixFilter.Update();

      EXPECT_EQ(Deref(transformixFilter.GetOutput()), DerefSmartPointer(movingImage));
    });
  };

  check(TypeHolder<char>{});
  check(TypeHolder<unsigned int>{});
  check(TypeHolder<std::int64_t>{});
  check(TypeHolder<std::uint64_t>{});
  check(TypeHolder<double>{});
}

// Checks a zero-filled moving image with a random domain, using any supported internal pixel type (which may be
// different from the input pixel type).
GTEST_TEST(itkTransformixFilter, CheckZeroFilledMovingImageWithRandomDomainUsingAnyInternalPixelType)
{
  std::mt19937 randomNumberEngine{};

  const auto check = [&randomNumberEngine](const auto inputPixelTypeHolder) {
    (void)inputPixelTypeHolder;
    elx::ForEachSupportedImageType([&randomNumberEngine](const auto elxTypedef) {
      using ElxTypedef = decltype(elxTypedef);
      using InputPixelType = typename decltype(inputPixelTypeHolder)::Type;
      using InputImageType = itk::Image<InputPixelType, ElxTypedef::MovingDimension>;

      const auto imageDomain = CreateRandomImageDomain<ElxTypedef::MovingDimension>(randomNumberEngine);
      elx::DefaultConstruct<InputImageType> movingImage{};
      imageDomain.ToImage(movingImage);
      movingImage.AllocateInitialized();

      const ParameterMapType parameterMap = MakeMergedMap(
        { // Parameters in alphabetic order:
          { "FixedInternalImagePixelType", { ElxTypedef::FixedPixelTypeString } },
          { "MovingInternalImagePixelType", { ElxTypedef::MovingPixelTypeString } },
          { "ResampleInterpolator", { "FinalLinearInterpolator" } } },
        imageDomain.AsParameterMap());

      const elx::DefaultConstruct<itk::TranslationTransform<double, ElxTypedef::MovingDimension>> identityTransform{};
      elx::DefaultConstruct<itk::TransformixFilter<InputImageType>>                               transformixFilter{};

      transformixFilter.SetMovingImage(&movingImage);
      transformixFilter.SetTransform(&identityTransform);
      transformixFilter.SetTransformParameterObject(CreateParameterObject(parameterMap));
      transformixFilter.Update();

      EXPECT_EQ(Deref(transformixFilter.GetOutput()), movingImage);
    });
  };

  check(TypeHolder<char>{});
  check(TypeHolder<unsigned int>{});
  check(TypeHolder<double>{});
}


GTEST_TEST(itkTransformixFilter, ExternalTransform)
{
  static constexpr unsigned int ImageDimension{ 2 };

  using PixelType = float;
  using SizeType = itk::Size<ImageDimension>;
  const SizeType imageSize{ { 5, 6 } };

  using ImageType = itk::Image<PixelType, ImageDimension>;
  using TransformixFilterType = itk::TransformixFilter<ImageType>;

  const ImageDomain<ImageDimension> imageDomain(imageSize);
  const auto                        movingImage = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);

  elx::DefaultConstruct<itk::TranslationTransform<double, ImageDimension>> itkTransform;
  itkTransform.SetOffset(itk::MakeVector(1.0, -2.0));

  elx::DefaultConstruct<TransformixFilterType> transformixFilter{};
  transformixFilter.SetMovingImage(movingImage);
  transformixFilter.SetTransformParameterObject(CreateParameterObject(MakeMergedMap(
    { // Parameters in alphabetic order:
      { "ResampleInterpolator", { "FinalLinearInterpolator" } },
      { "Transform", ParameterValuesType{ "ExternalTransform" } },
      { "TransformAddress", { elx::Conversion::ObjectPtrToString(&itkTransform) } } },
    imageDomain.AsParameterMap())));
  transformixFilter.Update();

  const auto resampleImageFilter = CreateResampleImageFilter(*movingImage, itkTransform);

  EXPECT_EQ(Deref(transformixFilter.GetOutput()), Deref(resampleImageFilter->GetOutput()));
}

GTEST_TEST(itkTransformixFilter, SetExternalTransform)
{
  static constexpr auto ImageDimension = 2U;
  using PixelType = float;
  using SizeType = itk::Size<ImageDimension>;
  const SizeType imageSize{ { 5, 6 } };

  using ImageType = itk::Image<PixelType, ImageDimension>;
  using TransformixFilterType = itk::TransformixFilter<ImageType>;

  const ImageDomain<ImageDimension> imageDomain(imageSize);
  const auto                        movingImage = CreateImageFilledWithSequenceOfNaturalNumbers<PixelType>(imageDomain);

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

  elx::DefaultConstruct<TransformixFilterType> transformixFilter{};
  transformixFilter.SetMovingImage(movingImage);
  transformixFilter.SetExternalTransform(&itkTransform);
  transformixFilter.SetTransformParameterObject(CreateParameterObject(
    MakeMergedMap({ { "ResampleInterpolator", { "FinalLinearInterpolator" } } }, imageDomain.AsParameterMap())));
  transformixFilter.Update();

  const auto resampleImageFilter = CreateResampleImageFilter(*movingImage, itkTransform);

  EXPECT_EQ(Deref(transformixFilter.GetOutput()), Deref(resampleImageFilter->GetOutput()));
}


// Tests support of the TOML file format for transform parameter files.
GTEST_TEST(itkTransformixFilter, SupportTomlAsTransformParameterFileFormat)
{
  static constexpr auto ImageDimension = 2U;
  using PixelType = float;
  elx::DefaultConstruct<itk::TransformixFilter<itk::Image<PixelType, ImageDimension>>> transformixFilter{};

  // Test two arbitrary transform parameter files.
  for (const std::string txtParameterFileName :
       { GetDataDirectoryPath() + "/Translation(1,-2)/TransformParameters-Size-5x6.txt",
         GetDataDirectoryPath() + "/../Baselines/TransformParameters_3DCT_lung.affine.inverse.txt" })
  {
    transformixFilter.SetTransformParameterFileName(txtParameterFileName);

    const ParameterMapVectorType parameterMapsFromTxt =
      itk::Deref(transformixFilter.GetTransformParameterObject()).GetParameterMaps();

    const std::string testDirectoryPath = GetCurrentBinaryDirectoryPath() + '/' + GetNameOfTest(*this);
    itk::FileTools::CreateDirectory(testDirectoryPath);

    const std::string tomlParameterFileName = testDirectoryPath + "/TransformParameters.toml";
    ConvertTxtFileToToml(txtParameterFileName, tomlParameterFileName);

    transformixFilter.SetTransformParameterFileName(tomlParameterFileName);

    const ParameterMapVectorType parameterMapsFromToml =
      itk::Deref(transformixFilter.GetTransformParameterObject()).GetParameterMaps();

    EXPECT_EQ(parameterMapsFromToml, parameterMapsFromTxt);
  }
}
