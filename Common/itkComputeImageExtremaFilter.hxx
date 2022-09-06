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

namespace itk
{

template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::BeforeStreamedGenerateData()
{
  if (!this->m_UseMask)
  {
    Superclass::BeforeStreamedGenerateData();
  }
  else
  {
    // Resize the thread temporaries
    m_Count = NumericTraits<SizeValueType>::ZeroValue();
    m_SumOfSquares = NumericTraits<RealType>::ZeroValue();
    m_ThreadSum = NumericTraits<RealType>::ZeroValue();
    m_ThreadMin = NumericTraits<PixelType>::max();
    m_ThreadMax = NumericTraits<PixelType>::NonpositiveMin();

    if (this->GetImageSpatialMask())
    {
      this->SameGeometry();
    }
    else
    {
      this->m_SameGeometry = false;
    }
  }
}

template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::SameGeometry()
{
  if (this->GetInput()->GetLargestPossibleRegion().GetSize() ==
        this->m_ImageSpatialMask->GetImage()->GetLargestPossibleRegion().GetSize() &&
      this->GetInput()->GetOrigin() == this->m_ImageSpatialMask->GetImage()->GetOrigin())
  {
    this->m_SameGeometry = true;
  }
}

template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::AfterStreamedGenerateData()
{
  if (!this->m_UseMask)
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
  if (!this->m_UseMask)
  {
    Superclass::ThreadedStreamedGenerateData(regionForThread);
  }
  else
  {
    if (this->GetImageSpatialMask())
    {
      this->ThreadedGenerateDataImageSpatialMask(regionForThread);
    }
    if (this->GetImageMask())
    {
      this->ThreadedGenerateDataImageMask(regionForThread);
    }
  }
} // end ThreadedGenerateData()

template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::ThreadedGenerateDataImageSpatialMask(const RegionType & regionForThread)
{
  const SizeValueType size0 = regionForThread.GetSize(0);
  if (size0 == 0)
  {
    return;
  }
  RealType  realValue;
  PixelType value;

  RealType      sum = NumericTraits<RealType>::ZeroValue();
  RealType      sumOfSquares = NumericTraits<RealType>::ZeroValue();
  SizeValueType count = NumericTraits<SizeValueType>::ZeroValue();
  PixelType     min = NumericTraits<PixelType>::max();
  PixelType     max = NumericTraits<PixelType>::NonpositiveMin();

  if (this->m_SameGeometry)
  {
    ImageRegionConstIterator<TInputImage> it(this->GetInput(), regionForThread);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      if (this->m_ImageSpatialMask->GetImage()->GetPixel(it.GetIndex()) != NumericTraits<PixelType>::ZeroValue())
      {
        value = it.Get();
        realValue = static_cast<RealType>(value);

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
    ImageRegionConstIterator<TInputImage> it(this->GetInput(), regionForThread);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      PointType point;
      this->GetInput()->TransformIndexToPhysicalPoint(it.GetIndex(), point);
      if (this->m_ImageSpatialMask->IsInsideInWorldSpace(point))
      {
        value = it.Get();
        realValue = static_cast<RealType>(value);

        min = std::min(min, value);
        max = std::max(max, value);

        sum += realValue;
        sumOfSquares += (realValue * realValue);
        ++count;
      }
    } // end for
  }   // end if

  std::lock_guard<std::mutex> mutexHolder(m_Mutex);
  m_ThreadSum += sum;
  m_SumOfSquares += sumOfSquares;
  m_Count += count;
  m_ThreadMin = std::min(min, m_ThreadMin);
  m_ThreadMax = std::max(max, m_ThreadMax);

} // end ThreadedGenerateDataImageSpatialMask()


template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::ThreadedGenerateDataImageMask(const RegionType & regionForThread)
{
  const SizeValueType size0 = regionForThread.GetSize(0);
  if (size0 == 0)
  {
    return;
  }
  RealType  realValue;
  PixelType value;

  RealType      sum = NumericTraits<RealType>::ZeroValue();
  RealType      sumOfSquares = NumericTraits<RealType>::ZeroValue();
  SizeValueType count = NumericTraits<SizeValueType>::ZeroValue();
  PixelType     min = NumericTraits<PixelType>::max();
  PixelType     max = NumericTraits<PixelType>::NonpositiveMin();

  ImageRegionConstIterator<TInputImage> it(this->GetInput(), regionForThread);
  it.GoToBegin();

  // do the work
  while (!it.IsAtEnd())
  {
    PointType point;
    this->GetInput()->TransformIndexToPhysicalPoint(it.GetIndex(), point);
    if (this->m_ImageMask->IsInsideInWorldSpace(point))
    {
      value = it.Get();
      realValue = static_cast<RealType>(value);

      min = std::min(min, value);
      max = std::max(max, value);

      sum += realValue;
      sumOfSquares += (realValue * realValue);
      ++count;
    }
    ++it;
  } // end while

  std::lock_guard<std::mutex> mutexHolder(m_Mutex);
  m_ThreadSum += sum;
  m_SumOfSquares += sumOfSquares;
  m_Count += count;
  m_ThreadMin = std::min(min, m_ThreadMin);
  m_ThreadMax = std::max(max, m_ThreadMax);

} // end ThreadedGenerateDataImageMask()

} // end namespace itk
#endif
