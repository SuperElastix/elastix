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
#include "itkParzenWindowHistogramImageToImageMetric.h"
#include <itkImageBufferRange.h>
#include "GTesting/elxCoreMainGTestUtilities.h"
#include <gtest/gtest.h>

// The template to be tested.
using itk::ParzenWindowHistogramImageToImageMetric;
using elx::CoreMainGTestUtilities::CreateImageFilledWithSequenceOfNaturalNumbers;


// Checks the protected member function NormalizeJointPDF.
GTEST_TEST(ParzenWindowHistogramImageToImageMetric, NormalizeJointPDF)
{
  using ImageType = itk::Image<int>;
  using ParzenWindowHistogramImageToImageMetricType = ParzenWindowHistogramImageToImageMetric<ImageType, ImageType>;

  class DerivedMetric : ParzenWindowHistogramImageToImageMetricType
  {
  public:
    static void
    TestNormalizeJointPDF(const double factor)
    {
      const auto pdf = CreateImageFilledWithSequenceOfNaturalNumbers<PDFValueType>(itk::Size<>{ 4, 5 });

      const itk::ImageBufferRange imageBufferRange(*pdf);

      const std::vector<PDFValueType> originalPDFValues(imageBufferRange.cbegin(), imageBufferRange.cend());

      ParzenWindowHistogramImageToImageMetricType::NormalizeJointPDF(pdf, factor);

      auto originalPDFValueIterator = originalPDFValues.cbegin();

      for (const PDFValueType pixelValue : imageBufferRange)
      {
        EXPECT_EQ(pixelValue, *originalPDFValueIterator * factor);
        ++originalPDFValueIterator;
      }
    }
  };

  for (const auto factor : { 0.0, 0.5, 1.0 })
  {
    DerivedMetric::TestNormalizeJointPDF(factor);
  }
}
