/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkComputeImageExtremaFilter_hxx
#define itkComputeImageExtremaFilter_hxx
#include "itkComputeImageExtremaFilter.h"

#include <itkImageRegionConstIterator.h>
#include "elxMaskHasSameImageDomain.h"

namespace itk
{

template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::BeforeStreamedGenerateData()
{
  if (m_ImageSpatialMask == nullptr)
  {
    Superclass::BeforeStreamedGenerateData();
  }
  else
  {
    // Resize the thread temporaries
    m_Count = SizeValueType{};
    m_SumOfSquares = RealType{};
    m_ThreadSum = RealType{};
    m_ThreadMin = NumericTraits<PixelType>::max();
    m_ThreadMax = NumericTraits<PixelType>::NonpositiveMin();

    if (this->GetImageSpatialMask())
    {
      this->m_SameGeometry = elastix::MaskHasSameImageDomain(*m_ImageSpatialMask, *(this->GetInput()));
    }
    else
    {
      this->m_SameGeometry = false;
    }
  }
}

template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::AfterStreamedGenerateData()
{
  if (m_ImageSpatialMask == nullptr)
  {
    Superclass::AfterStreamedGenerateData();
  }
  else
  {
    const SizeValueType count = m_Count;
    const RealType      sumOfSquares(m_SumOfSquares);
    const PixelType     minimum = m_ThreadMin;
    const PixelType     maximum = m_ThreadMax;
    const RealType      sum(m_ThreadSum);

    const RealType mean = sum / static_cast<RealType>(count);
    const RealType variance =
      (sumOfSquares - (sum * sum / static_cast<RealType>(count))) / (static_cast<RealType>(count) - 1);
    const RealType sigma = std::sqrt(variance);

    // Set the outputs
    this->SetMinimum(minimum);
    this->SetMaximum(maximum);
    this->SetMean(mean);
    this->SetSigma(sigma);
    this->SetVariance(variance);
    this->SetSum(sum);
    this->SetSumOfSquares(sumOfSquares);
  }
}

template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::ThreadedStreamedGenerateData(const RegionType & regionForThread)
{
  if (m_ImageSpatialMask == nullptr)
  {
    Superclass::ThreadedStreamedGenerateData(regionForThread);
  }
  else
  {
    this->ThreadedGenerateDataImageSpatialMask(regionForThread);
  }
} // end ThreadedGenerateData()

template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::ThreadedGenerateDataImageSpatialMask(const RegionType & regionForThread)
{
  if (regionForThread.GetSize(0) == 0)
  {
    return;
  }
  RealType      sum{};
  RealType      sumOfSquares{};
  SizeValueType count{};
  PixelType     min = NumericTraits<PixelType>::max();
  PixelType     max = NumericTraits<PixelType>::NonpositiveMin();

  const auto & inputImage = *(this->GetInput());

  if (this->m_SameGeometry)
  {
    const auto & maskImage = *(this->m_ImageSpatialMask->GetImage());

    for (ImageRegionConstIterator<TInputImage> it(&inputImage, regionForThread); !it.IsAtEnd(); ++it)
    {
      if (maskImage.GetPixel(it.GetIndex()) != PixelType{})
      {
        const PixelType value = it.Get();
        const auto      realValue = static_cast<RealType>(value);

        min = std::min(min, value);
        max = std::max(max, value);

        sum += realValue;
        sumOfSquares += (realValue * realValue);
        ++count;
      }
    } // end for
  }
  else
  {
    for (ImageRegionConstIterator<TInputImage> it(&inputImage, regionForThread); !it.IsAtEnd(); ++it)
    {
      PointType point;
      inputImage.TransformIndexToPhysicalPoint(it.GetIndex(), point);
      if (this->m_ImageSpatialMask->IsInsideInWorldSpace(point))
      {
        const PixelType value = it.Get();
        const auto      realValue = static_cast<RealType>(value);

        min = std::min(min, value);
        max = std::max(max, value);

        sum += realValue;
        sumOfSquares += (realValue * realValue);
        ++count;
      }
    } // end for
  }   // end if

  const std::lock_guard<std::mutex> lockGuard(m_Mutex);
  m_ThreadSum += sum;
  m_SumOfSquares += sumOfSquares;
  m_Count += count;
  m_ThreadMin = std::min(min, m_ThreadMin);
  m_ThreadMax = std::max(max, m_ThreadMax);

} // end ThreadedGenerateDataImageSpatialMask()

} // end namespace itk
#endif
