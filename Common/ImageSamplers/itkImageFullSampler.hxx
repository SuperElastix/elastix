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
#ifndef itkImageFullSampler_hxx
#define itkImageFullSampler_hxx

#include "itkImageFullSampler.h"

#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template <class TInputImage>
void
ImageFullSampler<TInputImage>::GenerateData()
{
  /** If desired we exercise a multi-threaded version. */
  if (this->m_UseMultiThread)
  {
    /** Calls ThreadedGenerateData(). */
    return Superclass::GenerateData();
  }

  /** Get handles to the input image, output sample container, and the mask. */
  InputImageConstPointer                     inputImage = this->GetInput();
  typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
  typename MaskType::ConstPointer            mask = this->GetMask();

  // Take capacity from the output container, and clear it.
  std::vector<ImageSampleType> sampleVector;
  sampleContainer->swap(sampleVector);
  sampleVector.clear();

  const auto croppedInputImageRegion = this->GetCroppedInputImageRegion();

  /** Set up a region iterator within the user specified image region. */
  using InputImageIterator = ImageRegionConstIteratorWithIndex<InputImageType>;

  /** Fill the sample container. */
  if (mask.IsNull())
  {
    /** Try to reserve memory. If no mask is used this can raise std
     * exceptions when the input image is large.
     */
    try
    {
      sampleVector.reserve(croppedInputImageRegion.GetNumberOfPixels());
    }
    catch (const std::exception & excp)
    {
      std::string message = "std: ";
      message += excp.what();
      message += "\nERROR: failed to allocate memory for the sample container.";
      itkExceptionMacro(<< message);
    }
    catch (...)
    {
      itkExceptionMacro("ERROR: failed to allocate memory for the sample container.");
    }

    /** Simply loop over the image and store all samples in the container. */
    for (InputImageIterator iter(inputImage, croppedInputImageRegion); !iter.IsAtEnd(); ++iter)
    {
      ImageSampleType tempSample;

      /** Get sampled index */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point */
      inputImage->TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

      /** Get sampled image value */
      tempSample.m_ImageValue = iter.Get();

      /** Store in container */
      sampleVector.push_back(tempSample);

    } // end for
  }   // end if no mask
  else
  {
    mask->UpdateSource();

    /** Loop over the image and check if the points falls within the mask. */
    for (InputImageIterator iter(inputImage, croppedInputImageRegion); !iter.IsAtEnd(); ++iter)
    {
      ImageSampleType tempSample;

      /** Get sampled index. */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point. */
      inputImage->TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

      if (mask->IsInsideInWorldSpace(tempSample.m_ImageCoordinates))
      {
        /** Get sampled image value. */
        tempSample.m_ImageValue = iter.Get();

        /** Store in container. */
        sampleVector.push_back(tempSample);

      } // end if
    }   // end for
  }     // end else (if mask exists)

  // Move the samples from the vector into the output container.
  sampleContainer->swap(sampleVector);

} // end GenerateData()


/**
 * ******************* ThreadedGenerateData *******************
 */

template <class TInputImage>
void
ImageFullSampler<TInputImage>::ThreadedGenerateData(const InputImageRegionType & inputRegionForThread,
                                                    ThreadIdType                 threadId)
{
  /** Get handles to the input image, mask and the output. */
  InputImageConstPointer          inputImage = this->GetInput();
  typename MaskType::ConstPointer mask = this->GetMask();
  ImageSampleContainerPointer &   sampleContainerThisThread // & ???
    = this->m_ThreaderSampleContainer[threadId];

  // Take capacity from the container of this thread, and clear it.
  std::vector<ImageSampleType> sampleVector;
  sampleContainerThisThread->swap(sampleVector);
  sampleVector.clear();

  /** Set up a region iterator within the user specified image region. */
  using InputImageIterator = ImageRegionConstIteratorWithIndex<InputImageType>;
  // InputImageIterator iter( inputImage, this->GetCroppedInputImageRegion() );

  /** Fill the sample container. */
  const unsigned long chunkSize = inputRegionForThread.GetNumberOfPixels();
  if (mask.IsNull())
  {
    /** Try to reserve memory. If no mask is used this can raise std
     * exceptions when the input image is large.
     */
    try
    {
      sampleVector.reserve(chunkSize);
    }
    catch (const std::exception & excp)
    {
      std::string message = "std: ";
      message += excp.what();
      message += "\nERROR: failed to allocate memory for the sample container.";
      itkExceptionMacro(<< message);
    }
    catch (...)
    {
      itkExceptionMacro("ERROR: failed to allocate memory for the sample container.");
    }

    /** Simply loop over the image and store all samples in the container. */
    for (InputImageIterator iter(inputImage, inputRegionForThread); !iter.IsAtEnd(); ++iter)
    {
      ImageSampleType tempSample;

      /** Get sampled index */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point */
      inputImage->TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

      /** Get sampled image value */
      tempSample.m_ImageValue = iter.Get();

      /** Store in container. */
      sampleVector.push_back(tempSample);

    } // end for
  }   // end if no mask
  else
  {
    mask->UpdateSource();

    /** Loop over the image and check if the points falls within the mask. */
    for (InputImageIterator iter(inputImage, inputRegionForThread); !iter.IsAtEnd(); ++iter)
    {
      ImageSampleType tempSample;

      /** Get sampled index. */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point. */
      inputImage->TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

      if (mask->IsInsideInWorldSpace(tempSample.m_ImageCoordinates))
      {
        /** Get sampled image value. */
        tempSample.m_ImageValue = iter.Get();

        /**  Store in container. */
        sampleVector.push_back(tempSample);

      } // end if
    }   // end for
  }     // end else (if mask exists)

  // Move the samples from the vector into the container for this thread.
  sampleContainerThisThread->swap(sampleVector);

} // end ThreadedGenerateData()


/**
 * ******************* PrintSelf *******************
 */

template <class TInputImage>
void
ImageFullSampler<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkImageFullSampler_hxx
