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

  /** Clear the container. */
  sampleContainer->Initialize();

  /** Set up a region iterator within the user specified image region. */
  using InputImageIterator = ImageRegionConstIteratorWithIndex<InputImageType>;
  InputImageIterator iter(inputImage, this->GetCroppedInputImageRegion());

  /** Fill the sample container. */
  if (mask.IsNull())
  {
    /** Try to reserve memory. If no mask is used this can raise std
     * exceptions when the input image is large.
     */
    try
    {
      sampleContainer->Reserve(this->GetCroppedInputImageRegion().GetNumberOfPixels());
    }
    catch (const std::exception & excp)
    {
      std::string message = "std: ";
      message += excp.what();
      message += "\nERROR: failed to allocate memory for the sample container.";
      const char * message2 = message.c_str();
      itkExceptionMacro(<< message2);
    }
    catch (...)
    {
      itkExceptionMacro(<< "ERROR: failed to allocate memory for the sample container.");
    }

    /** Simply loop over the image and store all samples in the container. */
    ImageSampleType tempSample;
    unsigned long   ind = 0;
    for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter, ++ind)
    {
      /** Get sampled index */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point */
      inputImage->TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

      /** Get sampled image value */
      tempSample.m_ImageValue = iter.Get();

      /** Store in container */
      sampleContainer->SetElement(ind, tempSample);

    } // end for
  }   // end if no mask
  else
  {
    if (mask->GetSource())
    {
      mask->GetSource()->Update();
    }

    /** Loop over the image and check if the points falls within the mask. */
    ImageSampleType tempSample;
    for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
    {
      /** Get sampled index. */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point. */
      inputImage->TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

      if (mask->IsInsideInWorldSpace(tempSample.m_ImageCoordinates))
      {
        /** Get sampled image value. */
        tempSample.m_ImageValue = iter.Get();

        /** Store in container. */
        sampleContainer->push_back(tempSample);

      } // end if
    }   // end for
  }     // end else (if mask exists)

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

  /** Set up a region iterator within the user specified image region. */
  using InputImageIterator = ImageRegionConstIteratorWithIndex<InputImageType>;
  // InputImageIterator iter( inputImage, this->GetCroppedInputImageRegion() );
  InputImageIterator iter(inputImage, inputRegionForThread);

  /** Fill the sample container. */
  const unsigned long chunkSize = inputRegionForThread.GetNumberOfPixels();
  if (mask.IsNull())
  {
    /** Try to reserve memory. If no mask is used this can raise std
     * exceptions when the input image is large.
     */
    try
    {
      sampleContainerThisThread->Reserve(chunkSize);
    }
    catch (const std::exception & excp)
    {
      std::string message = "std: ";
      message += excp.what();
      message += "\nERROR: failed to allocate memory for the sample container.";
      const char * message2 = message.c_str();
      itkExceptionMacro(<< message2);
    }
    catch (...)
    {
      itkExceptionMacro(<< "ERROR: failed to allocate memory for the sample container.");
    }

    /** Simply loop over the image and store all samples in the container. */
    ImageSampleType tempSample;
    unsigned long   ind = 0;
    for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter, ++ind)
    {
      /** Get sampled index */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point */
      inputImage->TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

      /** Get sampled image value */
      tempSample.m_ImageValue = iter.Get();

      /** Store in container. */
      sampleContainerThisThread->SetElement(ind, tempSample);

    } // end for
  }   // end if no mask
  else
  {
    if (mask->GetSource())
    {
      mask->GetSource()->Update();
    }

    /** Loop over the image and check if the points falls within the mask. */
    ImageSampleType tempSample;
    for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
    {
      /** Get sampled index. */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point. */
      inputImage->TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

      if (mask->IsInsideInWorldSpace(tempSample.m_ImageCoordinates))
      {
        /** Get sampled image value. */
        tempSample.m_ImageValue = iter.Get();

        /**  Store in container. */
        sampleContainerThisThread->push_back(tempSample);

      } // end if
    }   // end for
  }     // end else (if mask exists)

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
