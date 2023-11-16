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
#ifndef itkImageRandomSampler_hxx
#define itkImageRandomSampler_hxx

#include "itkImageRandomSampler.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkImageRandomConstIteratorWithIndex.h"
#include "elxDeref.h"
#include <cassert>

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template <class TInputImage>
void
ImageRandomSampler<TInputImage>::GenerateData()
{
  /** Get handles to the input image, output sample container. */
  InputImageConstPointer                     inputImage = this->GetInput();
  typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();

  /** Get a handle to the mask. If there was no mask supplied we exercise a multi-threaded version. */
  typename MaskType::ConstPointer mask = this->GetMask();
  if (mask.IsNull() && Superclass::m_UseMultiThread)
  {
    Superclass::GenerateRandomNumberList();
    const auto & randomNumberList = Superclass::m_RandomNumberList;
    auto &       samples = elastix::Deref(sampleContainer).CastToSTLContainer();
    samples.resize(randomNumberList.size());

    m_OptionalUserData.emplace(
      randomNumberList, elastix::Deref(inputImage), this->GetCroppedInputImageRegion(), samples);

    MultiThreaderBase & multiThreader = elastix::Deref(this->ProcessObject::GetMultiThreader());
    multiThreader.SetSingleMethod(&Self::ThreaderCallback, &*m_OptionalUserData);
    multiThreader.SingleMethodExecute();
    return;
  }

  /** Reserve memory for the output. */
  sampleContainer->Reserve(this->GetNumberOfSamples());

  /** Setup a random iterator over the input image. */
  using RandomIteratorType = ImageRandomConstIteratorWithIndex<InputImageType>;
  RandomIteratorType randIter(inputImage, this->GetCroppedInputImageRegion());

  if (const auto optionalSeed = Superclass::GetOptionalSeed())
  {
    randIter.ReinitializeSeed(*optionalSeed);
  }
  randIter.GoToBegin();

  /** Setup an iterator over the output, which is of ImageSampleContainerType. */
  typename ImageSampleContainerType::Iterator      iter;
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  if (mask.IsNull())
  {
    /** number of samples + 1, because of the initial ++randIter. */
    randIter.SetNumberOfSamples(this->GetNumberOfSamples() + 1);
    /** Advance one, in order to generate the same sequence as when using a mask */
    ++randIter;
    for (iter = sampleContainer->Begin(); iter != end; ++iter)
    {
      /** Get the index, transform it to the physical coordinates and put it in the sample. */
      InputImageIndexType index = randIter.GetIndex();
      inputImage->TransformIndexToPhysicalPoint(index, iter->Value().m_ImageCoordinates);
      /** Get the value and put it in the sample. */
      iter->Value().m_ImageValue = randIter.Get();
      /** Jump to a random position. */
      ++randIter;

    } // end for loop
  }   // end if no mask
  else
  {
    /** Update the mask. */
    mask->UpdateSource();

    /** Make sure we are not eternally trying to find samples: */
    randIter.SetNumberOfSamples(10 * this->GetNumberOfSamples());

    /** Loop over the sample container. */
    InputImagePointType inputPoint;
    bool                insideMask = false;
    for (iter = sampleContainer->Begin(); iter != end; ++iter)
    {
      /** Loop until a valid sample is found. */
      do
      {
        /** Jump to a random position. */
        ++randIter;
        /** Check if we are not trying eternally to find a valid point. */
        if (randIter.IsAtEnd())
        {
          /** Squeeze the sample container to the size that is still valid. */
          typename ImageSampleContainerType::iterator stlnow = sampleContainer->begin();
          typename ImageSampleContainerType::iterator stlend = sampleContainer->end();
          stlnow += iter.Index();
          sampleContainer->erase(stlnow, stlend);
          itkExceptionMacro(
            << "Could not find enough image samples within reasonable time. Probably the mask is too small");
        }
        /** Get the index, and transform it to the physical coordinates. */
        InputImageIndexType index = randIter.GetIndex();
        inputImage->TransformIndexToPhysicalPoint(index, inputPoint);
        /** Check if it's inside the mask. */
        insideMask = mask->IsInsideInWorldSpace(inputPoint);
      } while (!insideMask);

      /** Put the coordinates and the value in the sample. */
      iter->Value().m_ImageCoordinates = inputPoint;
      iter->Value().m_ImageValue = randIter.Get();

    } // end for loop

    /** Extra random sample to make sure the same sequence is generated
     * with and without mask.
     */
    ++randIter;
  }

} // end GenerateData()


template <class TInputImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
ImageRandomSampler<TInputImage>::ThreaderCallback(void * const arg)
{
  assert(arg);
  const auto & info = *static_cast<const MultiThreaderBase::WorkUnitInfo *>(arg);

  assert(info.UserData);
  auto & userData = *static_cast<UserData *>(info.UserData);

  const auto & randomNumberList = userData.m_RandomNumberList;
  auto &       samples = userData.m_Samples;

  const auto totalNumberOfSamples = samples.size();
  assert(totalNumberOfSamples == randomNumberList.size());

  const auto numberOfSamplesPerWorkUnit = totalNumberOfSamples / info.NumberOfWorkUnits;
  const auto remainderNumberOfSamples = totalNumberOfSamples % info.NumberOfWorkUnits;

  const auto offset =
    info.WorkUnitID * numberOfSamplesPerWorkUnit + std::min<size_t>(info.WorkUnitID, remainderNumberOfSamples);
  const auto   beginOfRandomNumbers = randomNumberList.data() + offset;
  const auto   beginOfSamples = samples.data() + offset;
  const auto & inputImage = userData.m_InputImage;

  const InputImageSizeType  regionSize = userData.m_RegionSize;
  const InputImageIndexType regionIndex = userData.m_RegionIndex;

  const size_t n{ numberOfSamplesPerWorkUnit + (info.WorkUnitID < remainderNumberOfSamples ? 1 : 0) };

  for (size_t i = 0; i < n; ++i)
  {
    auto   randomPosition = static_cast<size_t>(beginOfRandomNumbers[i]);
    auto & sample = beginOfSamples[i];

    /** Translate randomPosition to an index, copied from ImageRandomConstIteratorWithIndex. */
    InputImageIndexType positionIndex;

    for (unsigned int dim = 0; dim < InputImageDimension; ++dim)
    {
      const auto sizeInThisDimension = regionSize[dim];
      const auto residual = randomPosition % sizeInThisDimension;
      positionIndex[dim] = static_cast<IndexValueType>(residual) + regionIndex[dim];
      randomPosition -= residual;
      randomPosition /= sizeInThisDimension;
    }

    /** Transform index to the physical coordinates and put it in the sample. */
    inputImage.TransformIndexToPhysicalPoint(positionIndex, sample.m_ImageCoordinates);

    /** Get the value and put it in the sample. */
    sample.m_ImageValue = static_cast<ImageSampleValueType>(inputImage.GetPixel(positionIndex));
  }
  return ITK_THREAD_RETURN_DEFAULT_VALUE;
}


} // end namespace itk

#endif // end #ifndef itkImageRandomSampler_hxx
