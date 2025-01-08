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
#ifndef itkMultiInputImageRandomCoordinateSampler_hxx
#define itkMultiInputImageRandomCoordinateSampler_hxx

#include "itkMultiInputImageRandomCoordinateSampler.h"
#include <vnl/vnl_inverse.h>
#include "itkConfigure.h"
#include <itkDeref.h>

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template <typename TInputImage>
void
MultiInputImageRandomCoordinateSampler<TInputImage>::GenerateData()
{
  /** Check. */
  if (!this->CheckInputImageRegions())
  {
    itkExceptionMacro("ERROR: at least one of the InputImageRegions is not a subregion of the LargestPossibleRegion");
  }

  /** Get handles to the input image, output sample container, and mask. */
  const InputImageType &             inputImage = Deref(this->GetInput());
  auto &                             samples = Deref(this->GetOutput()).CastToSTLContainer();
  const MaskType * const             mask = this->Superclass::GetMask();
  typename InterpolatorType::Pointer interpolator = this->GetModifiableInterpolator();

  /** Set up the interpolator. */
  interpolator->SetInputImage(&inputImage);

  /** Get the intersection of all sample regions. */
  InputImageContinuousIndexType smallestContIndex;
  InputImageContinuousIndexType largestContIndex;
  this->GenerateSampleRegion(smallestContIndex, largestContIndex);

  /** Reserve memory for the output. */
  samples.resize(this->GetNumberOfSamples());

  InputImageContinuousIndexType sampleContIndex;
  /** Fill the sample container. */
  if (mask == nullptr)
  {
    /** Start looping over the sample container. */
    for (auto & sample : samples)
    {
      /** Make a reference to the current sample in the container. */
      InputImagePointType &  samplePoint = sample.m_ImageCoordinates;
      ImageSampleValueType & sampleValue = sample.m_ImageValue;

      /** Generate a point in the input image region. */
      this->GenerateRandomCoordinate(smallestContIndex, largestContIndex, sampleContIndex);

      /** Convert to point */
      inputImage.TransformContinuousIndexToPhysicalPoint(sampleContIndex, samplePoint);

      /** Compute the value at the contindex. */
      sampleValue = static_cast<ImageSampleValueType>(this->m_Interpolator->EvaluateAtContinuousIndex(sampleContIndex));

    } // end for loop
  }   // end if no mask
  else
  {
    /** Update all masks. */
    this->UpdateAllMasks();

    /** Set up some variable that are used to make sure we are not forever
     * walking around on this image, trying to look for valid samples.
     */
    unsigned long numberOfSamplesTried = 0;
    unsigned long maximumNumberOfSamplesToTry = 10 * this->GetNumberOfSamples();

    /** Start looping over the sample container. */
    for (auto & sample : samples)
    {
      /** Make a reference to the current sample in the container. */
      InputImagePointType &  samplePoint = sample.m_ImageCoordinates;
      ImageSampleValueType & sampleValue = sample.m_ImageValue;

      /** Walk over the image until we find a valid point. */
      do
      {
        /** Check if we are not trying eternally to find a valid point. */
        ++numberOfSamplesTried;
        if (numberOfSamplesTried > maximumNumberOfSamplesToTry)
        {
          /** Squeeze the sample container to the size that is still valid. */
          samples.resize(&sample - samples.data());
          itkExceptionMacro(
            "Could not find enough image samples within reasonable time. Probably the mask is too small");
        }

        /** Generate a point in the input image region. */
        this->GenerateRandomCoordinate(smallestContIndex, largestContIndex, sampleContIndex);
        inputImage.TransformContinuousIndexToPhysicalPoint(sampleContIndex, samplePoint);
      } while (!this->IsInsideAllMasks(samplePoint));

      /** Compute the value at the contindex. */
      sampleValue = static_cast<ImageSampleValueType>(this->m_Interpolator->EvaluateAtContinuousIndex(sampleContIndex));

    } // end for loop
  }   // end if mask

} // end GenerateData()


/**
 * ******************* GenerateSampleRegion *******************
 */

template <typename TInputImage>
void
MultiInputImageRandomCoordinateSampler<TInputImage>::GenerateSampleRegion(
  InputImageContinuousIndexType & smallestContIndex,
  InputImageContinuousIndexType & largestContIndex)
{
  /** Get handles to the number of inputs and regions. */
  const unsigned int numberOfInputs = this->GetNumberOfInputs();
  const unsigned int numberOfRegions = this->GetNumberOfInputImageRegions();

  /** Check. */
  if (numberOfRegions != numberOfInputs && numberOfRegions != 1)
  {
    itkExceptionMacro("ERROR: The number of regions should be 1 or the number of inputs.");
  }

  using DirectionType = typename InputImageType::DirectionType;
  DirectionType                              dir0 = this->GetInput(0)->GetDirection();
  typename DirectionType::InternalMatrixType dir0invtemp = vnl_inverse(dir0.GetVnlMatrix());
  DirectionType                              dir0inv(dir0invtemp);
  for (unsigned int i = 1; i < numberOfInputs; ++i)
  {
    DirectionType diri = this->GetInput(i)->GetDirection();
    if (diri != dir0)
    {
      itkExceptionMacro("ERROR: All input images should have the same direction cosines matrix.");
    }
  }

  /** Initialize the smallest and largest point. */
  InputImagePointType smallestPoint;
  InputImagePointType largestPoint;
  smallestPoint.Fill(NumericTraits<InputImagePointValueType>::NonpositiveMin());
  largestPoint.Fill(NumericTraits<InputImagePointValueType>::max());

  /** Determine the intersection of all regions, assuming identical direction cosines,
   * but possibly different origin/spacing.
   * \todo: test this really carefully!
   */
  const auto unitSize = InputImageSizeType::Filled(1);
  for (unsigned int i = 0; i < numberOfRegions; ++i)
  {
    /** Get the outer indices. */
    const InputImageIndexType smallestIndex = this->GetInputImageRegion(i).GetIndex();
    const InputImageIndexType largestIndex = smallestIndex + this->GetInputImageRegion(i).GetSize() - unitSize;

    /** Convert to points */
    InputImagePointType smallestImagePoint;
    InputImagePointType largestImagePoint;
    this->GetInput(i)->TransformIndexToPhysicalPoint(smallestIndex, smallestImagePoint);
    this->GetInput(i)->TransformIndexToPhysicalPoint(largestIndex, largestImagePoint);

    /** apply inverse direction, so that next max-operation makes sense. */
    smallestImagePoint = dir0inv * smallestImagePoint;
    largestImagePoint = dir0inv * largestImagePoint;

    /** Determine intersection. */
    for (unsigned int j = 0; j < InputImageDimension; ++j)
    {
      /** Get the largest smallest point. */
      smallestPoint[j] = std::max(smallestPoint[j], smallestImagePoint[j]);

      /** Get the smallest largest point. */
      largestPoint[j] = std::min(largestPoint[j], largestImagePoint[j]);
    }
  }

  /** Convert to continuous index in input image 0. */
  smallestPoint = dir0 * smallestPoint;
  largestPoint = dir0 * largestPoint;
  smallestContIndex =
    this->GetInput(0)->template TransformPhysicalPointToContinuousIndex<CoordinateType>(smallestPoint);
  largestContIndex = this->GetInput(0)->template TransformPhysicalPointToContinuousIndex<CoordinateType>(largestPoint);

  /** Support for localised mutual information. */
  if (this->GetUseRandomSampleRegion())
  {
    /** Convert sampleRegionSize to continuous index space */
    using CIndexVectorType = typename InputImageContinuousIndexType::VectorType;
    CIndexVectorType sampleRegionSize;
    for (unsigned int i = 0; i < InputImageDimension; ++i)
    {
      sampleRegionSize[i] = this->GetSampleRegionSize()[i] / this->GetInput()->GetSpacing()[i];
    }
    InputImageContinuousIndexType maxSmallestContIndex = largestContIndex;
    maxSmallestContIndex -= sampleRegionSize;
    this->GenerateRandomCoordinate(smallestContIndex, maxSmallestContIndex, smallestContIndex);
    largestContIndex = smallestContIndex;
    largestContIndex += sampleRegionSize;
  }

} // end GenerateSampleRegion()


/**
 * ******************* GenerateRandomCoordinate *******************
 */

template <typename TInputImage>
void
MultiInputImageRandomCoordinateSampler<TInputImage>::GenerateRandomCoordinate(
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
 * ******************* PrintSelf *******************
 */

template <typename TInputImage>
void
MultiInputImageRandomCoordinateSampler<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Interpolator: " << this->m_Interpolator.GetPointer() << std::endl;
  os << indent << "RandomGenerator: " << this->m_RandomGenerator.GetPointer() << std::endl;

} // end PrintSelf


} // end namespace itk

#endif // end #ifndef itkMultiInputImageRandomCoordinateSampler_hxx
