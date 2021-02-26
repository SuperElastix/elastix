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
#include <elxElastixFilter.h>

#include "elxCoreMainGTestUtilities.h"

// ITK header file:
#include <itkImage.h>

// GoogleTest header file:
#include <gtest/gtest.h>

#include <array>
#include <algorithm> // For transform
#include <map>
#include <string>
#include <utility> // For pair
#include <vector>


namespace
{
template <typename TImage>
std::vector<std::string>
GetTransformParameters(const elx::ElastixFilter<TImage, TImage> & filter)
{
  const auto transformParameterObject = filter.GetTransformParameterObject();

  if (transformParameterObject != nullptr)
  {
    const auto & transformParameterMaps = transformParameterObject->GetParameterMap();

    if (transformParameterMaps.size() == 1)
    {
      const auto & transformParameterMap = transformParameterMaps.front();
      const auto   found = transformParameterMap.find("TransformParameters");

      if (found != transformParameterMap.cend())
      {
        return found->second;
      }
    }
  }
  return {};
}


void
TestUpdatingMultipleFilters(const bool inParallel)
{
  // Create an array of black images (pixel value zero) with a little "white"
  // region (pixel value 1) inside.
  constexpr auto ImageDimension = 2;
  using ImageType = itk::Image<float, ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;

  const auto imageSizeValue = 8;

  const itk::ImageRegion<ImageDimension> imageRegion{ itk::Index<ImageDimension>::Filled(imageSizeValue / 2),
                                                      SizeType::Filled(2) };

  const auto                                     numberOfImages = 10;
  std::array<ImageType::Pointer, numberOfImages> images;

  std::generate(images.begin(), images.end(), ImageType::New);

  for (const auto & image : images)
  {
    image->SetRegions(SizeType::Filled(imageSizeValue));
    image->Allocate(true);
    const itk::Experimental::ImageRegionRange<ImageType> imageRegionRange{ *image, imageRegion };
    std::fill(std::begin(imageRegionRange), std::end(imageRegionRange), 1.0f);
  }

  const auto parameterObject = elx::ParameterObject::New();
  parameterObject->SetParameterMap(
    elx::ParameterObject::ParameterMapType{ // Parameters in alphabetic order:
                                            { "ImageSampler", { "Full" } },
                                            { "MaximumNumberOfIterations", { "2" } },
                                            { "Metric", { "AdvancedNormalizedCorrelation" } },
                                            { "Optimizer", { "AdaptiveStochasticGradientDescent" } },
                                            { "Transform", { "TranslationTransform" } },
                                            { "WriteFinalTransformParameters", { "false" } } });

  // Create a filter for each image, to register an image with the next one.
  std::array<itk::SmartPointer<elx::ElastixFilter<ImageType, ImageType>>, numberOfImages> filters;
  std::generate(filters.begin(), filters.end(), elx::ElastixFilter<ImageType, ImageType>::New);

  for (auto i = 0; i < numberOfImages; ++i)
  {
    auto & filter = *(filters[i]);

    filter.LogToConsoleOff();
    filter.LogToFileOff();
    filter.SetFixedImage(images[i]);
    // Choose the next image (from the array of images) as moving image.
    filter.SetMovingImage(images[(i + std::size_t{ 1 }) % numberOfImages]);
    filter.SetParameterObject(parameterObject);
  }

  if (inParallel)
  {
    // Note: The OpenMP implementation of GCC (including GCC 5.5 and GCC 10.2)
    // does not seem to support value-initialization of a for-loop index by
    // empty pair of braces, as in `for (int i{}; i < n; ++i)`.
#pragma omp parallel for
    for (int i = 0; i < numberOfImages; ++i)
    {
      filters[i]->Update();
    }
  }
  else
  {
    for (int i{}; i < numberOfImages; ++i)
    {
      filters[i]->Update();
    }
  }

  // Check if the TransformParameters of each filter are as expected.
  const std::vector<std::string> expectedTransformParameters(ImageDimension, "0");

  for (const auto & filter : filters)
  {
    EXPECT_EQ(GetTransformParameters(*filter), expectedTransformParameters);
  }
}

} // namespace


// Tests registering two small (5x6) binary images, which are translated with respect to each other.
GTEST_TEST(ElastixFilter, Translation)
{
  constexpr auto ImageDimension = 2U;
  using ImageType = itk::Image<float, ImageDimension>;
  using RegionType = itk::ImageRegion<ImageDimension>;
  using SizeType = itk::Size<ImageDimension>;
  using IndexType = itk::Index<ImageDimension>;
  using OffsetType = itk::Offset<ImageDimension>;

  const OffsetType translationOffset{ { 1, -2 } };
  const auto       regionSize = SizeType::Filled(2);
  const SizeType   imageSize{ { 5, 6 } };
  const IndexType  fixedImageRegionIndex{ { 1, 3 } };

  const auto fixedImage = ImageType::New();
  fixedImage->SetRegions(imageSize);
  fixedImage->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*fixedImage, fixedImageRegionIndex, regionSize);

  const auto movingImage = ImageType::New();
  movingImage->SetRegions(imageSize);
  movingImage->Allocate(true);
  elx::CoreMainGTestUtilities::FillImageRegion(*movingImage, fixedImageRegionIndex + translationOffset, regionSize);

  const auto parameterObject = elastix::ParameterObject::New();

  const std::pair<std::string, std::string> parameterArray[] = { // Parameters in alphabetic order:
                                                                 { "ImageSampler", "Full" },
                                                                 { "MaximumNumberOfIterations", "2" },
                                                                 { "Metric", "AdvancedNormalizedCorrelation" },
                                                                 { "Optimizer", "AdaptiveStochasticGradientDescent" },
                                                                 { "Transform", "TranslationTransform" }
  };

  std::map<std::string, std::vector<std::string>> parameterMap;

  for (const auto & pair : parameterArray)
  {
    const auto result = parameterMap.insert({ pair.first, { pair.second } });

    ASSERT_EQ(std::make_pair(pair, result.second), std::make_pair(pair, true));
  }

  parameterObject->SetParameterMap(parameterMap);

  const auto filter = elastix::ElastixFilter<ImageType, ImageType>::New();
  ASSERT_NE(filter, nullptr);

  filter->SetFixedImage(fixedImage);
  filter->SetMovingImage(movingImage);
  filter->SetParameterObject(parameterObject);
  filter->Update();

  const auto   transformParameterObject = filter->GetTransformParameterObject();
  const auto & transformParameterMaps = transformParameterObject->GetParameterMap();

  ASSERT_TRUE(!transformParameterMaps.empty());
  EXPECT_EQ(transformParameterMaps.size(), 1);

  const auto & transformParameterMap = transformParameterMaps.front();
  const auto   found = transformParameterMap.find("TransformParameters");
  ASSERT_NE(found, transformParameterMap.cend());

  const auto & transformParameters = found->second;
  ASSERT_EQ(transformParameters.size(), ImageDimension);

  for (unsigned i{}; i < ImageDimension; ++i)
  {
    EXPECT_EQ(std::round(std::stod(transformParameters[i])), translationOffset[i]);
  }
}


GTEST_TEST(ElastixFilter, UpdateSerially)
{
  TestUpdatingMultipleFilters(false);
}


GTEST_TEST(ElastixFilter, UpdateInParallel)
{
  TestUpdatingMultipleFilters(true);
}
