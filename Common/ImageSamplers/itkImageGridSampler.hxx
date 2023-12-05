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
#ifndef itkImageGridSampler_hxx
#define itkImageGridSampler_hxx

#include "itkImageGridSampler.h"

#include "itkImageRegionConstIteratorWithIndex.h"
#include "elxDeref.h"

#include <algorithm> // For accumulate.
#include <cassert>

namespace itk
{

/**
 * ******************* SetSampleGridSpacing *******************
 */

template <class TInputImage>
void
ImageGridSampler<TInputImage>::SetSampleGridSpacing(const SampleGridSpacingType & arg)
{
  this->SetNumberOfSamples(0);
  if (m_SampleGridSpacing != arg)
  {
    m_SampleGridSpacing = arg;
    this->Modified();
  }
} // end SetSampleGridSpacing()


/**
 * ******************* GenerateData *******************
 */

template <class TInputImage>
void
ImageGridSampler<TInputImage>::GenerateData()
{
  /** Get handles to the input image, output sample container, and the mask. */
  const InputImageType &          inputImage = elastix::Deref(this->GetInput());
  ImageSampleContainerType &      sampleContainer = elastix::Deref(this->GetOutput());
  typename MaskType::ConstPointer mask = this->GetMask();

  // Take capacity from the output container, and clear it.
  std::vector<ImageSampleType> sampleVector;
  sampleContainer.swap(sampleVector);
  sampleVector.clear();

  /** Take into account the possibility of a smaller bounding box around the mask */
  this->SetNumberOfSamples(m_RequestedNumberOfSamples);

  const auto croppedInputImageRegion = this->GetCroppedInputImageRegion();

  /** Determine the grid. */
  SampleGridSizeType         sampleGridSize;
  SampleGridIndexType        sampleGridIndex = croppedInputImageRegion.GetIndex();
  const InputImageSizeType & inputImageSize = croppedInputImageRegion.GetSize();
  for (unsigned int dim = 0; dim < InputImageDimension; ++dim)
  {
    /** The number of sample point along one dimension. */
    sampleGridSize[dim] = 1 + ((inputImageSize[dim] - 1) / m_SampleGridSpacing[dim]);

    /** The position of the first sample along this dimension is
     * chosen to center the grid nicely on the input image region.
     */
    sampleGridIndex[dim] += (inputImageSize[dim] - ((sampleGridSize[dim] - 1) * m_SampleGridSpacing[dim] + 1)) / 2;
  }

  /** Prepare for looping over the grid. */
  SampleGridIndexType index = sampleGridIndex;

  if (mask.IsNull())
  {
    /** Calculate the number of samples on the grid. */
    const std::size_t numberOfSamplesOnGrid =
      std::accumulate(sampleGridSize.cbegin(), sampleGridSize.cend(), std::size_t{ 1 }, std::multiplies<>{});

    sampleVector.reserve(numberOfSamplesOnGrid);

    /** Ugly loop over the grid. */
    for (unsigned int t = 0; t < GetGridSizeValue<3>(sampleGridSize); ++t)
    {
      for (unsigned int z = 0; z < GetGridSizeValue<2>(sampleGridSize); ++z)
      {
        for (unsigned int y = 0; y < sampleGridSize[1]; ++y)
        {
          for (unsigned int x = 0; x < sampleGridSize[0]; ++x)
          {
            ImageSampleType tempSample;

            // Get sampled fixed image value.
            tempSample.m_ImageValue = inputImage.GetPixel(index);

            // Translate index to point.
            inputImage.TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

            // Store sample in container.
            sampleVector.push_back(tempSample);

            // Jump to next position on grid.
            index[0] += m_SampleGridSpacing[0];
          }
          JumpToNextGridPosition<1>(index, sampleGridIndex);
        }
        JumpToNextGridPosition<2>(index, sampleGridIndex);
      }
      JumpToNextGridPosition<3>(index, sampleGridIndex);
    }

    assert(sampleVector.size() == numberOfSamplesOnGrid);

  } // end if no mask
  else
  {
    mask->UpdateSource();

    /* Ugly loop over the grid; checks also if a sample falls within the mask. */
    for (unsigned int t = 0; t < GetGridSizeValue<3>(sampleGridSize); ++t)
    {
      for (unsigned int z = 0; z < GetGridSizeValue<2>(sampleGridSize); ++z)
      {
        for (unsigned int y = 0; y < sampleGridSize[1]; ++y)
        {
          for (unsigned int x = 0; x < sampleGridSize[0]; ++x)
          {
            ImageSampleType tempSample;

            // Translate index to point.
            inputImage.TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

            if (mask->IsInsideInWorldSpace(tempSample.m_ImageCoordinates))
            {
              // Get sampled fixed image value.
              tempSample.m_ImageValue = inputImage.GetPixel(index);

              // Store sample in container.
              sampleVector.push_back(tempSample);

            } // end if in mask

            // Jump to next position on grid
            index[0] += m_SampleGridSpacing[0];
          }
          JumpToNextGridPosition<1>(index, sampleGridIndex);
        }
        JumpToNextGridPosition<2>(index, sampleGridIndex);
      }
      JumpToNextGridPosition<3>(index, sampleGridIndex);
    }
  } // else (if mask exists)

  // Move the samples from the vector into the output container.
  sampleContainer.swap(sampleVector);

} // end GenerateData()


/**
 * ******************* SetNumberOfSamples *******************
 */

template <class TInputImage>
void
ImageGridSampler<TInputImage>::SetNumberOfSamples(unsigned long nrofsamples)
{
  /** Store what the user wanted. */
  if (m_RequestedNumberOfSamples != nrofsamples)
  {
    m_RequestedNumberOfSamples = nrofsamples;
    this->Modified();
  }

  /** Do nothing if nothing is needed. */
  if (nrofsamples == 0)
  {
    return;
  }

  /** This function assumes that the input has been set. */
  if (!this->GetInput())
  {
    itkExceptionMacro("ERROR: only call the function SetNumberOfSamples() after the input has been set.");
  }

  /** Compute an isotropic grid spacing (in voxels),
   * which realises the nrofsamples approximately.
   * This is realized by evenly distributing the samples over
   * the volume of the bounding box of the mask.
   */

  /** Get the cropped image region volume in voxels. */
  this->CropInputImageRegion();
  const double allvoxels = static_cast<double>(this->GetCroppedInputImageRegion().GetNumberOfPixels());

  /** Compute the fraction in voxels. */
  const double fraction = allvoxels / static_cast<double>(nrofsamples);

  /** Compute the grid spacing. */
  const double indimd = static_cast<double>(InputImageDimension);
  int          gridSpacing = static_cast<int>( // no unsigned int version of rnd, max
    Math::Round<double>(std::pow(fraction, 1.0 / indimd)));
  gridSpacing = std::max(1, gridSpacing);

  /** Set gridSpacings for all dimensions
   * Do not use the SetSampleGridSpacing function because it calls
   * SetNumberOfSamples(0) internally.
   */
  SampleGridSpacingType gridSpacings;
  gridSpacings.Fill(gridSpacing);
  if (m_SampleGridSpacing != gridSpacings)
  {
    m_SampleGridSpacing = gridSpacings;
    this->Modified();
  }

} // end SetNumberOfSamples()


/**
 * ******************* PrintSelf *******************
 */

template <class TInputImage>
void
ImageGridSampler<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << "SampleGridSpacing: " << m_SampleGridSpacing << std::endl;
  os << "RequestedNumberOfSamples: " << m_RequestedNumberOfSamples << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkImageGridSampler_hxx
