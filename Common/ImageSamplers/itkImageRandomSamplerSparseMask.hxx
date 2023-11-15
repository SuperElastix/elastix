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
#ifndef itkImageRandomSamplerSparseMask_hxx
#define itkImageRandomSamplerSparseMask_hxx

#include "itkImageRandomSamplerSparseMask.h"
#include "elxDeref.h"

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template <class TInputImage>
void
ImageRandomSamplerSparseMask<TInputImage>::GenerateData()
{
  /** Get a handle to the mask. */
  typename MaskType::ConstPointer mask = this->GetMask();

  /** Sanity check. */
  if (mask.IsNull())
  {
    itkExceptionMacro("ERROR: do not call this function when no mask is supplied.");
  }

  /** Get handles to the input image and output sample container. */
  InputImageConstPointer      inputImage = this->GetInput();
  ImageSampleContainerPointer sampleContainer = this->GetOutput();

  // Take capacity from the output container, and clear it.
  std::vector<ImageSampleType> sampleVector;
  sampleContainer->swap(sampleVector);
  sampleVector.clear();

  /** Make sure the internal full sampler is up-to-date. */
  this->m_InternalFullSampler->SetInput(inputImage);
  this->m_InternalFullSampler->SetMask(mask);
  this->m_InternalFullSampler->SetInputImageRegion(this->GetCroppedInputImageRegion());
  this->m_InternalFullSampler->SetUseMultiThread(Superclass::m_UseMultiThread);

  /** Use try/catch, since the full sampler may crash, due to insufficient memory. */
  try
  {
    this->m_InternalFullSampler->Update();
  }
  catch (ExceptionObject & err)
  {
    std::string message = "ERROR: This ImageSampler internally uses the "
                          "ImageFullSampler. Updating of this internal sampler raised the "
                          "exception:\n";
    message += err.GetDescription();

    std::string            fullSamplerMessage = err.GetDescription();
    std::string::size_type loc =
      fullSamplerMessage.find("ERROR: failed to allocate memory for the sample container", 0);
    if (loc != std::string::npos && this->GetMask() == nullptr)
    {
      message += "\nYou are using the ImageRandomSamplerSparseMask sampler, "
                 "but you did not set a mask. The internal ImageFullSampler therefore "
                 "requires a lot of memory. Consider using the ImageRandomSampler "
                 "instead.";
    }
    itkExceptionMacro(<< message);
  }

  /** Get a handle to the full sampler output. */
  typename ImageSampleContainerType::Pointer allValidSamples = this->m_InternalFullSampler->GetOutput();
  unsigned long                              numberOfValidSamples = allValidSamples->Size();


  /** If desired we exercise a multi-threaded version. */
  if (Superclass::m_UseMultiThread)
  {
    m_RandomIndices.clear();
    m_RandomIndices.reserve(Superclass::m_NumberOfSamples);

    for (unsigned int i = 0; i < Superclass::m_NumberOfSamples; ++i)
    {
      m_RandomIndices.push_back(m_RandomGenerator->GetIntegerVariate(numberOfValidSamples - 1));
    }

    auto & samples = elastix::Deref(sampleContainer).CastToSTLContainer();
    samples.resize(m_RandomIndices.size());

    m_OptionalUserData.emplace(elastix::Deref(allValidSamples).CastToSTLConstContainer(), m_RandomIndices, samples);

    MultiThreaderBase & multiThreader = elastix::Deref(this->ProcessObject::GetMultiThreader());
    multiThreader.SetSingleMethod(&Self::ThreaderCallback, &*m_OptionalUserData);
    multiThreader.SingleMethodExecute();
    return;
  }

  /** Take random samples from the allValidSamples-container. */
  for (unsigned int i = 0; i < this->GetNumberOfSamples(); ++i)
  {
    unsigned long randomIndex = this->m_RandomGenerator->GetIntegerVariate(numberOfValidSamples - 1);
    sampleVector.push_back(allValidSamples->ElementAt(randomIndex));
  }

  // Move the samples from the vector into the output container.
  sampleContainer->swap(sampleVector);

} // end GenerateData()


template <class TInputImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
ImageRandomSamplerSparseMask<TInputImage>::ThreaderCallback(void * const arg)
{
  assert(arg);
  const auto & info = *static_cast<const MultiThreaderBase::WorkUnitInfo *>(arg);

  assert(info.UserData);
  auto & userData = *static_cast<UserData *>(info.UserData);

  const auto & randomIndices = userData.m_RandomIndices;
  auto &       samples = userData.m_Samples;

  const auto totalNumberOfSamples = samples.size();
  assert(totalNumberOfSamples == randomIndices.size());

  const auto numberOfSamplesPerWorkUnit = totalNumberOfSamples / info.NumberOfWorkUnits;
  const auto remainderNumberOfSamples = totalNumberOfSamples % info.NumberOfWorkUnits;

  const auto offset =
    info.WorkUnitID * numberOfSamplesPerWorkUnit + std::min<size_t>(info.WorkUnitID, remainderNumberOfSamples);
  const auto   beginOfRandomIndices = randomIndices.data() + offset;
  const auto   beginOfSamples = samples.data() + offset;
  const auto & allValidSamples = userData.m_AllValidSamples;

  const size_t n{ numberOfSamplesPerWorkUnit + (info.WorkUnitID < remainderNumberOfSamples ? 1 : 0) };

  for (size_t i = 0; i < n; ++i)
  {
    beginOfSamples[i] = allValidSamples[beginOfRandomIndices[i]];
  }
  return ITK_THREAD_RETURN_DEFAULT_VALUE;
}


/**
 * ******************* PrintSelf *******************
 */

template <class TInputImage>
void
ImageRandomSamplerSparseMask<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "InternalFullSampler: " << this->m_InternalFullSampler.GetPointer() << std::endl;
  os << indent << "RandomGenerator: " << this->m_RandomGenerator.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkImageRandomSamplerSparseMask_hxx
