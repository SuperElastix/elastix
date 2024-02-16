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
#include <itkImageScanlineIterator.h>

#include "elxMaskHasSameImageDomain.h"

namespace itk
{

template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::BeforeStreamedGenerateData()
{
  m_ThreadMin = NumericTraits<PixelType>::max();
  m_ThreadMax = NumericTraits<PixelType>::NonpositiveMin();
  m_SameGeometry =
    (m_ImageSpatialMask != nullptr) && elastix::MaskHasSameImageDomain(*m_ImageSpatialMask, *(this->GetInput()));
}


template <typename TInputImage>
auto
ComputeImageExtremaFilter<TInputImage>::RetrieveMinMax(const TInputImage &                inputImage,
                                                       const InputImageRegionType &       regionForThread,
                                                       const ImageSpatialMaskType * const imageSpatialMask,
                                                       const bool                         sameGeometry) -> MinMaxResult
{
  PixelType min = NumericTraits<PixelType>::max();
  PixelType max = NumericTraits<PixelType>::NonpositiveMin();

  if (imageSpatialMask)
  {
    if (sameGeometry)
    {
      const auto & maskImage = *(imageSpatialMask->GetImage());

      for (ImageRegionConstIterator<TInputImage> it(&inputImage, regionForThread); !it.IsAtEnd(); ++it)
      {
        if (maskImage.GetPixel(it.GetIndex()) != PixelType{})
        {
          const PixelType value = it.Get();
          min = std::min(min, value);
          max = std::max(max, value);
        }
      }
    }
    else
    {
      for (ImageRegionConstIterator<TInputImage> it(&inputImage, regionForThread); !it.IsAtEnd(); ++it)
      {
        typename ImageSpatialMaskType::PointType point;
        inputImage.TransformIndexToPhysicalPoint(it.GetIndex(), point);
        if (imageSpatialMask->IsInsideInWorldSpace(point))
        {
          const PixelType value = it.Get();
          min = std::min(min, value);
          max = std::max(max, value);
        }
      }
    }
  }
  else
  {
    for (ImageScanlineConstIterator<TInputImage> it(&inputImage, regionForThread); !it.IsAtEnd(); it.NextLine())
    {
      while (!it.IsAtEndOfLine())
      {
        const PixelType value = it.Get();
        min = std::min(min, value);
        max = std::max(max, value);
        ++it;
      }
    }
  }
  return { min, max };
}


template <typename TInputImage>
void
ComputeImageExtremaFilter<TInputImage>::ThreadedStreamedGenerateData(const InputImageRegionType & regionForThread)
{
  if (regionForThread.GetSize(0) > 0)
  {
    const MinMaxResult minMaxResult =
      RetrieveMinMax(*(this->GetInput()), regionForThread, m_ImageSpatialMask, m_SameGeometry);

    // Lock after calling RetrieveMinMax.
    const std::lock_guard<std::mutex> lockGuard(m_Mutex);
    m_ThreadMin = std::min(minMaxResult.Min, m_ThreadMin);
    m_ThreadMax = std::max(minMaxResult.Max, m_ThreadMax);
  }
}

} // end namespace itk
#endif
