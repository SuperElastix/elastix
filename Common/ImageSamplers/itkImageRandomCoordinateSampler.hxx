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
#ifndef itkImageRandomCoordinateSampler_hxx
#define itkImageRandomCoordinateSampler_hxx

#include "itkImageRandomCoordinateSampler.h"
#include "elxDeref.h"
#include <vnl/vnl_math.h>

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template <class TInputImage>
void
ImageRandomCoordinateSampler<TInputImage>::GenerateData()
{
  /** Get handles to the input image, output sample container, and interpolator. */
  InputImageConstPointer                     inputImage = this->GetInput();
  typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
  typename InterpolatorType::Pointer         interpolator = this->GetModifiableInterpolator();

  /** Set up the interpolator. */
  interpolator->SetInputImage(inputImage); // only once?

  const auto croppedInputImageRegion = this->GetCroppedInputImageRegion();

  /** Convert inputImageRegion to bounding box in physical space. */
  const auto                    unitSize = InputImageSizeType::Filled(1);
  InputImageIndexType           smallestIndex = croppedInputImageRegion.GetIndex();
  InputImageIndexType           largestIndex = smallestIndex + croppedInputImageRegion.GetSize() - unitSize;
  InputImageContinuousIndexType smallestImageContIndex(smallestIndex);
  InputImageContinuousIndexType largestImageContIndex(largestIndex);
  InputImageContinuousIndexType smallestContIndex;
  InputImageContinuousIndexType largestContIndex;
  this->GenerateSampleRegion(smallestImageContIndex, largestImageContIndex, smallestContIndex, largestContIndex);

  /** Get a handle to the mask. If there was no mask supplied we exercise a multi-threaded version. */
  typename MaskType::ConstPointer mask = this->GetMask();
  if (mask.IsNull() && this->m_UseMultiThread)
  {
    auto & samples = elastix::Deref(sampleContainer).CastToSTLContainer();
    samples.resize(this->Superclass::m_NumberOfSamples);

    /** Clear the random number list. */
    this->m_RandomNumberList.clear();
    this->m_RandomNumberList.reserve(this->m_NumberOfSamples * InputImageDimension);

    /** Fill the list with random numbers. */
    for (unsigned long i = 0; i < this->m_NumberOfSamples; ++i)
    {
      InputImageContinuousIndexType randomCIndex;

      this->GenerateRandomCoordinate(smallestContIndex, largestContIndex, randomCIndex);
      for (unsigned int j = 0; j < InputImageDimension; ++j)
      {
        this->m_RandomNumberList.push_back(randomCIndex[j]);
      }
    }

    m_OptionalUserData.emplace(this->Superclass::m_RandomNumberList, samples, *inputImage, *interpolator);

    MultiThreaderBase & multiThreader = elastix::Deref(this->ProcessObject::GetMultiThreader());
    multiThreader.SetSingleMethod(&Self::ThreaderCallback, &*m_OptionalUserData);
    multiThreader.SingleMethodExecute();
    return;
  }

  /** Reserve memory for the output. */
  sampleContainer->Reserve(this->GetNumberOfSamples());

  /** Setup an iterator over the output, which is of ImageSampleContainerType. */
  typename ImageSampleContainerType::Iterator      iter;
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  InputImageContinuousIndexType sampleContIndex;
  /** Fill the sample container. */
  if (mask.IsNull())
  {
    /** Start looping over the sample container. */
    for (iter = sampleContainer->Begin(); iter != end; ++iter)
    {
      /** Make a reference to the current sample in the container. */
      InputImagePointType &  samplePoint = iter->Value().m_ImageCoordinates;
      ImageSampleValueType & sampleValue = iter->Value().m_ImageValue;

      /** Walk over the image until we find a valid point. */
      this->GenerateRandomCoordinate(smallestContIndex, largestContIndex, sampleContIndex);

      /** Convert to point */
      inputImage->TransformContinuousIndexToPhysicalPoint(sampleContIndex, samplePoint);

      /** Compute the value at the continuous index. */
      sampleValue = static_cast<ImageSampleValueType>(this->m_Interpolator->EvaluateAtContinuousIndex(sampleContIndex));

    } // end for loop
  }   // end if no mask
  else
  {
    /** Update the mask. */
    mask->UpdateSource();

    /** Set up some variable that are used to make sure we are not forever
     * walking around on this image, trying to look for valid samples. */
    unsigned long numberOfSamplesTried = 0;
    unsigned long maximumNumberOfSamplesToTry = 10 * this->GetNumberOfSamples();

    /** Start looping over the sample container */
    for (iter = sampleContainer->Begin(); iter != end; ++iter)
    {
      /** Make a reference to the current sample in the container. */
      InputImagePointType &  samplePoint = iter->Value().m_ImageCoordinates;
      ImageSampleValueType & sampleValue = iter->Value().m_ImageValue;

      /** Walk over the image until we find a valid point */
      do
      {
        /** Check if we are not trying eternally to find a valid point. */
        ++numberOfSamplesTried;
        if (numberOfSamplesTried > maximumNumberOfSamplesToTry)
        {
          /** Squeeze the sample container to the size that is still valid. */
          typename ImageSampleContainerType::iterator stlnow = sampleContainer->begin();
          typename ImageSampleContainerType::iterator stlend = sampleContainer->end();
          stlnow += iter.Index();
          sampleContainer->erase(stlnow, stlend);
          itkExceptionMacro(
            << "Could not find enough image samples within reasonable time. Probably the mask is too small");
        }

        /** Generate a point in the input image region. */
        this->GenerateRandomCoordinate(smallestContIndex, largestContIndex, sampleContIndex);
        inputImage->TransformContinuousIndexToPhysicalPoint(sampleContIndex, samplePoint);

      } while (!interpolator->IsInsideBuffer(sampleContIndex) || !mask->IsInsideInWorldSpace(samplePoint));

      /** Compute the value at the point. */
      sampleValue = static_cast<ImageSampleValueType>(this->m_Interpolator->EvaluateAtContinuousIndex(sampleContIndex));

    } // end for loop
  }   // end if mask

} // end GenerateData()


/**
 * ******************* ThreaderCallback *******************
 */

template <class TInputImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
ImageRandomCoordinateSampler<TInputImage>::ThreaderCallback(void * const arg)
{
  assert(arg);
  const auto & info = *static_cast<const MultiThreaderBase::WorkUnitInfo *>(arg);

  assert(info.UserData);
  auto & userData = *static_cast<UserData *>(info.UserData);

  const auto & randomNumberList = userData.m_RandomNumberList;
  auto &       samples = userData.m_Samples;
  const auto & interpolator = userData.m_Interpolator;

  const auto totalNumberOfSamples = samples.size();
  assert((totalNumberOfSamples * InputImageDimension) == randomNumberList.size());

  const auto numberOfSamplesPerWorkUnit = totalNumberOfSamples / info.NumberOfWorkUnits;
  const auto remainderNumberOfSamples = totalNumberOfSamples % info.NumberOfWorkUnits;

  const auto offset =
    info.WorkUnitID * numberOfSamplesPerWorkUnit + std::min<size_t>(info.WorkUnitID, remainderNumberOfSamples);
  const auto beginOfRandomNumbers = randomNumberList.data() + InputImageDimension * offset;
  const auto beginOfSamples = samples.data() + offset;

  const auto & inputImage = userData.m_InputImage;

  const size_t n{ numberOfSamplesPerWorkUnit + (info.WorkUnitID < remainderNumberOfSamples ? 1 : 0) };

  for (size_t i = 0; i < n; ++i)
  {
    auto &                        sample = beginOfSamples[i];
    InputImageContinuousIndexType sampleCIndex;

    /** Create a random point out of InputImageDimension random numbers. */
    for (unsigned int j = 0; j < InputImageDimension; ++j)
    {
      sampleCIndex[j] = beginOfRandomNumbers[InputImageDimension * i + j];
    }

    /** Convert to point */
    inputImage.TransformContinuousIndexToPhysicalPoint(sampleCIndex, sample.m_ImageCoordinates);

    /** Compute the value at the continuous index. */
    sample.m_ImageValue = static_cast<ImageSampleValueType>(interpolator.EvaluateAtContinuousIndex(sampleCIndex));

  } // end for loop

  return ITK_THREAD_RETURN_DEFAULT_VALUE;
}


/**
 * ******************* GenerateRandomCoordinate *******************
 */

template <class TInputImage>
void
ImageRandomCoordinateSampler<TInputImage>::GenerateRandomCoordinate(
  const InputImageContinuousIndexType & smallestContIndex,
  const InputImageContinuousIndexType & largestContIndex,
  InputImageContinuousIndexType &       randomContIndex)
{
  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    randomContIndex[i] = static_cast<InputImagePointValueType>(
      this->m_RandomGenerator->GetUniformVariate(smallestContIndex[i], largestContIndex[i]));
  }
} // end GenerateRandomCoordinate()


/**
 * ******************* GenerateSampleRegion *******************
 */

template <class TInputImage>
void
ImageRandomCoordinateSampler<TInputImage>::GenerateSampleRegion(
  const InputImageContinuousIndexType & smallestImageContIndex,
  const InputImageContinuousIndexType & largestImageContIndex,
  InputImageContinuousIndexType &       smallestContIndex,
  InputImageContinuousIndexType &       largestContIndex)
{
  if (!this->GetUseRandomSampleRegion())
  {
    smallestContIndex = smallestImageContIndex;
    largestContIndex = largestImageContIndex;
    return;
  }

  /** Convert sampleRegionSize to continuous index space and
   * compute the maximum allowed value for the smallestContIndex,
   * such that a sample region of size SampleRegionSize still fits.
   */
  using CIndexVectorType = typename InputImageContinuousIndexType::VectorType;
  CIndexVectorType              sampleRegionSize;
  InputImageContinuousIndexType maxSmallestContIndex;
  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    sampleRegionSize[i] = this->GetSampleRegionSize()[i] / this->GetInput()->GetSpacing()[i];
    maxSmallestContIndex[i] = largestImageContIndex[i] - sampleRegionSize[i];

    /** Make sure it is larger than the lower bound. */
    maxSmallestContIndex[i] = std::max(maxSmallestContIndex[i], smallestImageContIndex[i]);
  }

  this->GenerateRandomCoordinate(smallestImageContIndex, maxSmallestContIndex, smallestContIndex);
  largestContIndex = smallestContIndex;
  largestContIndex += sampleRegionSize;

} // end GenerateSampleRegion()


/**
 * ******************* PrintSelf *******************
 */

template <class TInputImage>
void
ImageRandomCoordinateSampler<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Interpolator: " << this->m_Interpolator.GetPointer() << std::endl;
  os << indent << "RandomGenerator: " << this->m_RandomGenerator.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkImageRandomCoordinateSampler_hxx
