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
  if (this->m_SampleGridSpacing != arg)
  {
    this->m_SampleGridSpacing = arg;
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
  InputImageConstPointer                     inputImage = this->GetInput();
  typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
  typename MaskType::ConstPointer            mask = this->GetMask();

  /** Clear the container. */
  sampleContainer->Initialize();

  /** Set up a region iterator within the user specified image region. */
  using InputImageIterator = ImageRegionConstIteratorWithIndex<InputImageType>;
  InputImageIterator iter(inputImage, this->GetCroppedInputImageRegion());

  /** Take into account the possibility of a smaller bounding box around the mask */
  this->SetNumberOfSamples(this->m_RequestedNumberOfSamples);

  /** Determine the grid. */
  SampleGridIndexType        index;
  SampleGridSizeType         sampleGridSize;
  SampleGridIndexType        sampleGridIndex = this->GetCroppedInputImageRegion().GetIndex();
  const InputImageSizeType & inputImageSize = this->GetCroppedInputImageRegion().GetSize();
  unsigned long              numberOfSamplesOnGrid = 1;
  for (unsigned int dim = 0; dim < InputImageDimension; ++dim)
  {
    /** The number of sample point along one dimension. */
    sampleGridSize[dim] = 1 + ((inputImageSize[dim] - 1) / this->GetSampleGridSpacing()[dim]);

    /** The position of the first sample along this dimension is
     * chosen to center the grid nicely on the input image region.
     */
    sampleGridIndex[dim] +=
      (inputImageSize[dim] - ((sampleGridSize[dim] - 1) * this->GetSampleGridSpacing()[dim] + 1)) / 2;

    /** Update the number of samples on the grid. */
    numberOfSamplesOnGrid *= sampleGridSize[dim];
  }

  /** Prepare for looping over the grid. */
  unsigned int dim_z = 1;
  unsigned int dim_t = 1;
  if (InputImageDimension > 2)
  {
    dim_z = sampleGridSize[2];
    if (InputImageDimension > 3)
    {
      dim_t = sampleGridSize[3];
    }
  }
  index = sampleGridIndex;

  if (mask.IsNull())
  {
    /** Ugly loop over the grid. */
    for (unsigned int t = 0; t < dim_t; ++t)
    {
      for (unsigned int z = 0; z < dim_z; ++z)
      {
        for (unsigned int y = 0; y < sampleGridSize[1]; ++y)
        {
          for (unsigned int x = 0; x < sampleGridSize[0]; ++x)
          {
            ImageSampleType tempsample;

            // Get sampled fixed image value.
            tempsample.m_ImageValue = inputImage->GetPixel(index);

            // Translate index to point.
            inputImage->TransformIndexToPhysicalPoint(index, tempsample.m_ImageCoordinates);

            // Jump to next position on grid.
            index[0] += this->m_SampleGridSpacing[0];

            // Store sample in container.
            sampleContainer->push_back(tempsample);

          } // end x
          index[0] = sampleGridIndex[0];
          index[1] += this->m_SampleGridSpacing[1];

        } // end y
        if (InputImageDimension > 2)
        {
          index[1] = sampleGridIndex[1];
          index[2] += this->m_SampleGridSpacing[2];
        }
      } // end z
      if (InputImageDimension > 3)
      {
        index[2] = sampleGridIndex[2];
        index[3] += this->m_SampleGridSpacing[3];
      }
    } // end t

  } // end if no mask
  else
  {
    if (mask->GetSource())
    {
      mask->GetSource()->Update();
    }
    /* Ugly loop over the grid; checks also if a sample falls within the mask. */
    for (unsigned int t = 0; t < dim_t; ++t)
    {
      for (unsigned int z = 0; z < dim_z; ++z)
      {
        for (unsigned int y = 0; y < sampleGridSize[1]; ++y)
        {
          for (unsigned int x = 0; x < sampleGridSize[0]; ++x)
          {
            ImageSampleType tempsample;

            // Translate index to point.
            inputImage->TransformIndexToPhysicalPoint(index, tempsample.m_ImageCoordinates);

            if (mask->IsInsideInWorldSpace(tempsample.m_ImageCoordinates))
            {
              // Get sampled fixed image value.
              tempsample.m_ImageValue = inputImage->GetPixel(index);

              // Store sample in container.
              sampleContainer->push_back(tempsample);

            } // end if in mask
              // Jump to next position on grid
            index[0] += this->m_SampleGridSpacing[0];

          } // end x
          index[0] = sampleGridIndex[0];
          index[1] += this->m_SampleGridSpacing[1];

        } // end y
        if (InputImageDimension > 2)
        {
          index[1] = sampleGridIndex[1];
          index[2] += this->m_SampleGridSpacing[2];
        }
      } // end z
      if (InputImageDimension > 3)
      {
        index[2] = sampleGridIndex[2];
        index[3] += this->m_SampleGridSpacing[3];
      }
    } // end t
  }   // else (if mask exists)

} // end GenerateData()


/**
 * ******************* SetNumberOfSamples *******************
 */

template <class TInputImage>
void
ImageGridSampler<TInputImage>::SetNumberOfSamples(unsigned long nrofsamples)
{
  /** Store what the user wanted. */
  if (this->m_RequestedNumberOfSamples != nrofsamples)
  {
    this->m_RequestedNumberOfSamples = nrofsamples;
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
    itkExceptionMacro(<< "ERROR: only call the function SetNumberOfSamples() after the input has been set.");
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
  int          gridspacing = static_cast<int>( // no unsigned int version of rnd, max
    Math::Round<double>(std::pow(fraction, 1.0 / indimd)));
  gridspacing = std::max(1, gridspacing);

  /** Set gridspacings for all dimensions
   * Do not use the SetSampleGridSpacing function because it calls
   * SetNumberOfSamples(0) internally.
   */
  SampleGridSpacingType gridspacings;
  gridspacings.Fill(gridspacing);
  if (this->GetSampleGridSpacing() != gridspacings)
  {
    this->m_SampleGridSpacing = gridspacings;
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

  os << "SampleGridSpacing: " << this->m_SampleGridSpacing << std::endl;
  os << "RequestedNumberOfSamples: " << this->m_RequestedNumberOfSamples << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkImageGridSampler_hxx
