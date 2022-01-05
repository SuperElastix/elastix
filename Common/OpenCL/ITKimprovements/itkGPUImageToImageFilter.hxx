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
#ifndef itkGPUImageToImageFilter_hxx
#define itkGPUImageToImageFilter_hxx

#include "itkGPUImageToImageFilter.h"
#include "itkGPUImage.h"

namespace itk
{
template <typename TInputImage, typename TOutputImage, typename TParentImageFilter>
GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GPUImageToImageFilter()
  : m_GPUEnabled(true)
{
  m_GPUKernelManager = OpenCLKernelManager::New();
  Superclass::SetNumberOfWorkUnits(1);
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TOutputImage, typename TParentImageFilter>
void
GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::SetNumberOfWorkUnits(ThreadIdType _arg)
{
  Superclass::SetNumberOfWorkUnits(1);
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TOutputImage, typename TParentImageFilter>
void
GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "GPU: " << (m_GPUEnabled ? "Enabled" : "Disabled") << std::endl;
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TOutputImage, typename TParentImageFilter>
void
GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GenerateData()
{
  if (!m_GPUEnabled) // call CPU update function
  {
    Superclass::GenerateData();
  }
  else // call GPU update function
  {
    // Call a method that can be overridden by a subclass to allocate
    // memory for the filter's outputs
    this->AllocateOutputs();

    // Call a method that can be overridden by a subclass to perform
    // some calculations prior to splitting the main computations into
    // separate threads
    this->BeforeThreadedGenerateData();

    this->GPUGenerateData();

    // Update CPU buffer for all outputs
    using GPUOutputImageType = GPUImage<OutputImagePixelType, OutputImageDimension>;
    for (OutputDataObjectIterator it(this); !it.IsAtEnd(); ++it)
    {
      GPUOutputImageType * GPUOutput = dynamic_cast<GPUOutputImageType *>(it.GetOutput());
      if (GPUOutput)
      {
        GPUOutput->UpdateCPUBuffer();
      }
    }

    // Call a method that can be overridden by a subclass to perform
    // some calculations after all the threads have completed
    this->AfterThreadedGenerateData();
  }
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TOutputImage, typename TParentImageFilter>
void
GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(DataObject * graft)
{
  if (!graft)
  {
    itkExceptionMacro(<< "Requested to graft output that is a NULL pointer");
  }

  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
  typename GPUOutputImage::Pointer outputPtr;

  try
  {
    outputPtr = dynamic_cast<GPUOutputImage *>(this->GetOutput());
  }
  catch (...)
  {
    return;
  }

  if (outputPtr.IsNotNull())
  {
    outputPtr->Graft(graft);
  }
  else
  {
    // pointer could not be cast back down
    itkExceptionMacro(<< "itk::GPUImageToImageFilter::GraftOutput() cannot cast " << typeid(graft).name() << " to "
                      << typeid(GPUOutputImage *).name());
  }
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TOutputImage, typename TParentImageFilter>
void
GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(const DataObjectIdentifierType & key,
                                                                                  DataObject * graft)
{
  if (!graft)
  {
    itkExceptionMacro(<< "Requested to graft output that is a NULL pointer");
  }

  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
  typename GPUOutputImage::Pointer outputPtr;

  try
  {
    outputPtr = dynamic_cast<GPUOutputImage *>(this->ProcessObject::GetOutput(key));
  }
  catch (...)
  {
    return;
  }

  if (outputPtr.IsNotNull())
  {
    outputPtr->Graft(graft);
  }
  else
  {
    // pointer could not be cast back down
    itkExceptionMacro(<< "itk::GPUImageToImageFilter::GraftOutput() cannot cast " << typeid(graft).name() << " to "
                      << typeid(GPUOutputImage *).name());
  }
}


} // end namespace itk

#endif
