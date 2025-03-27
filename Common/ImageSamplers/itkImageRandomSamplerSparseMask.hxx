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
#include <itkDeref.h>
#include <cassert>

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template <typename TInputImage>
void
ImageRandomSamplerSparseMask<TInputImage>::GenerateData()
{
  /** Get a handle to the mask. */
  const MaskType * const mask = this->Superclass::GetMask();

  /** Sanity check. */
  if (mask == nullptr)
  {
    itkExceptionMacro("ERROR: do not call this function when no mask is supplied. When using the "
                      "ImageRandomSamplerSparseMask sampler, a mask is required. Otherwise you may consider using a "
                      "sampler that does not require a mask, for example, ImageRandomSampler.");
  }

  /** Get handles to the input image and output sample container. */
  const InputImageType &     inputImage = Deref(this->GetInput());
  ImageSampleContainerType & sampleContainer = Deref(this->GetOutput());

  // Take capacity from the output container, and clear it.
  std::vector<ImageSampleType> sampleVector;
  sampleContainer.swap(sampleVector);
  sampleVector.clear();

  /** Make sure the internal full sampler is up-to-date. */
  this->m_InternalFullSampler->SetInput(&inputImage);
  this->m_InternalFullSampler->SetMask(mask);
  this->m_InternalFullSampler->SetInputImageRegion(this->GetCroppedInputImageRegion());
  this->m_InternalFullSampler->SetUseMultiThread(Superclass::m_UseMultiThread);

  /** Use try/catch, since the full sampler may crash, due to insufficient memory. */
  try
  {
    this->m_InternalFullSampler->Update();
  }
  catch (const ExceptionObject & err)
  {
    itkExceptionMacro("ERROR: This ImageSampler internally uses the ImageFullSampler. Updating of this internal "
                      "sampler raised the exception:\n"
                      << err.GetDescription());
  }

  /** Get a handle to the full sampler output. */
  const ImageSampleContainerType & allValidSamples = Deref(this->m_InternalFullSampler->GetOutput());
  unsigned long                    numberOfValidSamples = allValidSamples.Size();

  Statistics::MersenneTwisterRandomVariateGenerator & randomVariateGenerator = Superclass::GetRandomVariateGenerator();

  /** If desired we exercise a multi-threaded version. */
  if (Superclass::m_UseMultiThread)
  {
    m_RandomIndices.clear();
    m_RandomIndices.reserve(Superclass::m_NumberOfSamples);

    for (unsigned int i = 0; i < Superclass::m_NumberOfSamples; ++i)
    {
      m_RandomIndices.push_back(randomVariateGenerator.GetIntegerVariate(numberOfValidSamples - 1));
    }

    auto & samples = sampleContainer.CastToSTLContainer();
    samples.resize(m_RandomIndices.size());

    UserData userData{ allValidSamples.CastToSTLConstContainer(), m_RandomIndices, samples };

    Deref(this->ProcessObject::GetMultiThreader()).SetSingleMethodAndExecute(&Self::ThreaderCallback, &userData);
    return;
  }

  /** Take random samples from the allValidSamples-container. */
  sampleVector.reserve(Superclass::m_NumberOfSamples);

  for (unsigned int i = 0; i < Superclass::m_NumberOfSamples; ++i)
  {
    unsigned long randomIndex = randomVariateGenerator.GetIntegerVariate(numberOfValidSamples - 1);
    sampleVector.push_back(allValidSamples.ElementAt(randomIndex));
  }

  // Move the samples from the vector into the output container.
  sampleContainer.swap(sampleVector);

} // end GenerateData()


template <typename TInputImage>
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

template <typename TInputImage>
void
ImageRandomSamplerSparseMask<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "InternalFullSampler: " << this->m_InternalFullSampler.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkImageRandomSamplerSparseMask_hxx
