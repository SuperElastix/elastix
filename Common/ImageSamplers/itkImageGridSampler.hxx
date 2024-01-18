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
 * ******************* DetermineGridIndexAndSize *******************
 */

template <class TInputImage>
auto
ImageGridSampler<TInputImage>::DetermineGridIndexAndSize(const InputImageRegionType &  croppedInputImageRegion,
                                                         const SampleGridSpacingType & gridSpacing)
  -> std::pair<SampleGridIndexType, SampleGridSizeType>
{
  SampleGridSizeType         gridSize;
  SampleGridIndexType        gridIndex = croppedInputImageRegion.GetIndex();
  const InputImageSizeType & inputImageSize = croppedInputImageRegion.GetSize();
  for (unsigned int dim = 0; dim < InputImageDimension; ++dim)
  {
    /** The number of sample point along one dimension. */
    gridSize[dim] = 1 + ((inputImageSize[dim] - 1) / gridSpacing[dim]);

    /** The position of the first sample along this dimension is
     * chosen to center the grid nicely on the input image region.
     */
    gridIndex[dim] += (inputImageSize[dim] - ((gridSize[dim] - 1) * gridSpacing[dim] + 1)) / 2;
  }
  return { gridIndex, gridSize };
}


/**
 * ******************* SplitRegion *******************
 */

template <class TInputImage>
auto
ImageGridSampler<TInputImage>::SplitRegion(const InputImageRegionType & inputRegion,
                                           const size_t                 requestedNumberOfSubregions)
  -> std::vector<InputImageRegionType>
{
  if (requestedNumberOfSubregions == 0)
  {
    assert(!"The requested number of subregions must be greater than zero!");
    return {};
  }

  constexpr unsigned int ImageDimension{ TInputImage::ImageDimension };

  const Index<ImageDimension> & inputRegionIndex = inputRegion.GetIndex();
  const Size<ImageDimension> &  inputRegionSize = inputRegion.GetSize();

  static_assert(TInputImage::ImageDimension > 0);

  // split on the outermost dimension available
  unsigned int splitAxis{ ImageDimension - 1 };
  while (inputRegionSize[splitAxis] <= 1)
  {
    if (splitAxis == 0)
    {
      // cannot split
      assert(!"The region size must be greater than 1!");
      return { inputRegion };
    }
    --splitAxis;
  }

  // determine the actual number of pieces that will be generated
  const SizeValueType inputSizeValue = inputRegionSize[splitAxis];
  const auto numberOfValues = static_cast<unsigned int>(((inputSizeValue - 1) / requestedNumberOfSubregions) + 1);
  const auto n = static_cast<unsigned int>((inputSizeValue - 1) / numberOfValues);

  std::vector<InputImageRegionType> subregions{};
  subregions.reserve(n + 1);

  for (size_t i{}; i < n; ++i)
  {
    auto index = inputRegionIndex;
    auto size = inputRegionSize;

    index[splitAxis] += i * numberOfValues;
    size[splitAxis] = numberOfValues;

    subregions.push_back({ index, size });
  }

  auto index = inputRegionIndex;
  auto size = inputRegionSize;

  index[splitAxis] += n * numberOfValues;
  // last thread needs to process the "rest" dimension being split
  size[splitAxis] -= n * numberOfValues;

  subregions.push_back(InputImageRegionType{ index, size });

  assert(subregions.size() == n + 1);
  return subregions;
}


/**
 * ******************* GenerateWorkUnits *******************
 */

template <class TInputImage>
auto
ImageGridSampler<TInputImage>::GenerateWorkUnits(const ThreadIdType             numberOfWorkUnits,
                                                 const InputImageRegionType &   croppedInputImageRegion,
                                                 const SampleGridIndexType      gridIndex,
                                                 const SampleGridSpacingType    gridSpacing,
                                                 std::vector<ImageSampleType> & samples) -> std::vector<WorkUnit>
{
  auto * sampleData = samples.data();

  const auto subregions = SplitRegion(
    croppedInputImageRegion, std::min(numberOfWorkUnits, MultiThreaderBase::GetGlobalMaximumNumberOfThreads()));

  const auto            numberOfSubregions = subregions.size();
  std::vector<WorkUnit> workUnits{};
  workUnits.reserve(numberOfSubregions);

  // Add a work unit for each subregion.
  for (const auto & subregion : subregions)
  {
    [&subregion, gridIndex, gridSpacing, &sampleData, &workUnits] {
      const auto inputIndexForThread = subregion.GetIndex();
      const auto inputSizeForThread = subregion.GetSize();

      SampleGridSizeType gridSizeForThread;
      auto               gridIndexForThread = gridIndex;

      for (unsigned int i{}; i < InputImageDimension; ++i)
      {
        const auto inputSizeValueForThreadAsOffset = static_cast<OffsetValueType>(inputSizeForThread[i]);

        if (inputSizeValueForThreadAsOffset <= 0)
        {
          assert(!"The splitted input region size for any thread should always be greater than zero!");
          return;
        }

        const OffsetValueType gridSpacingValue{ gridSpacing[i] };
        assert(gridSpacingValue > 0);

        const IndexValueType inputIndexValueForThread{ inputIndexForThread[i] };
        const IndexValueType gridIndexValueForAll{ gridIndex[i] };

        IndexValueType & gridIndexValueForThread = gridIndexForThread[i];

        if (inputIndexValueForThread > gridIndexValueForAll)
        {
          const auto difference = inputIndexValueForThread - gridIndexValueForAll;

          gridIndexValueForThread =
            (difference % gridSpacingValue == 0)
              ? inputIndexValueForThread
              : (gridIndexValueForAll + ((1 + (difference / gridSpacingValue)) * gridSpacingValue));
        }
        const IndexValueType endPositionForThread{ inputIndexValueForThread + inputSizeValueForThreadAsOffset };

        if (gridIndexValueForThread >= endPositionForThread)
        {
          return;
        }
        gridSizeForThread[i] =
          1 + static_cast<SizeValueType>((endPositionForThread - gridIndexValueForThread - 1) / gridSpacingValue);
      }
      workUnits.push_back({ gridIndexForThread, gridSizeForThread, sampleData });

      // TODO Use ITK 5.4: sampleData += gridSizeForThread.CalculateProductOfElements()
      sampleData +=
        std::accumulate(gridSizeForThread.cbegin(), gridSizeForThread.cend(), size_t{ 1 }, std::multiplies<>{});
    }();
  }
  assert(workUnits.size() <= numberOfSubregions);
  return workUnits;
}


/**
 * ******************* SingleThreadedGenerateData *******************
 */

template <class TInputImage>
void
ImageGridSampler<TInputImage>::SingleThreadedGenerateData(const TInputImage &            inputImage,
                                                          const MaskType * const         mask,
                                                          const InputImageRegionType &   croppedInputImageRegion,
                                                          const SampleGridSpacingType &  gridSpacing,
                                                          std::vector<ImageSampleType> & samples)
{
  /** Determine the grid. */
  const auto [gridIndex, gridSize] = DetermineGridIndexAndSize(croppedInputImageRegion, gridSpacing);

  /** Prepare for looping over the grid. */
  SampleGridIndexType index = gridIndex;

  if (mask)
  {
    /* Ugly loop over the grid; checks also if a sample falls within the mask. */
    for (unsigned int t = 0; t < GetGridSizeValue<3>(gridSize); ++t)
    {
      for (unsigned int z = 0; z < GetGridSizeValue<2>(gridSize); ++z)
      {
        for (unsigned int y = 0; y < gridSize[1]; ++y)
        {
          for (unsigned int x = 0; x < gridSize[0]; ++x)
          {
            ImageSampleType tempSample;

            // Translate index to point.
            inputImage.TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

            if (mask->IsInsideInWorldSpace(tempSample.m_ImageCoordinates))
            {
              // Get sampled fixed image value.
              tempSample.m_ImageValue = inputImage.GetPixel(index);

              // Store sample in container.
              samples.push_back(tempSample);

            } // end if in mask

            // Jump to next position on grid.
            index[0] += gridSpacing[0];
          }
          JumpToNextGridPosition<1>(index, gridIndex, gridSpacing);
        }
        JumpToNextGridPosition<2>(index, gridIndex, gridSpacing);
      }
      JumpToNextGridPosition<3>(index, gridIndex, gridSpacing);
    }
  } // end (if mask exists)
  else
  {
    const std::size_t numberOfSamplesOnGrid =
      std::accumulate(gridSize.cbegin(), gridSize.cend(), std::size_t{ 1 }, std::multiplies<>{});

    samples.reserve(numberOfSamplesOnGrid);

    /** Ugly loop over the grid. */
    for (unsigned int t = 0; t < GetGridSizeValue<3>(gridSize); ++t)
    {
      for (unsigned int z = 0; z < GetGridSizeValue<2>(gridSize); ++z)
      {
        for (unsigned int y = 0; y < gridSize[1]; ++y)
        {
          for (unsigned int x = 0; x < gridSize[0]; ++x)
          {
            ImageSampleType tempSample;

            // Get sampled fixed image value.
            tempSample.m_ImageValue = inputImage.GetPixel(index);

            // Translate index to point.
            inputImage.TransformIndexToPhysicalPoint(index, tempSample.m_ImageCoordinates);

            // Store sample in container.
            samples.push_back(tempSample);

            // Jump to next position on grid.
            index[0] += gridSpacing[0];
          }
          JumpToNextGridPosition<1>(index, gridIndex, gridSpacing);
        }
        JumpToNextGridPosition<2>(index, gridIndex, gridSpacing);
      }
      JumpToNextGridPosition<3>(index, gridIndex, gridSpacing);
    }

    assert(samples.size() == numberOfSamplesOnGrid);

  } // end (else)
}

/**
 * ******************* MultiThreadedGenerateData *******************
 */

template <class TInputImage>
void
ImageGridSampler<TInputImage>::MultiThreadedGenerateData(MultiThreaderBase &            multiThreader,
                                                         const ThreadIdType             numberOfWorkUnits,
                                                         const TInputImage &            inputImage,
                                                         const MaskType * const         mask,
                                                         const InputImageRegionType &   croppedInputImageRegion,
                                                         const SampleGridSpacingType &  gridSpacing,
                                                         std::vector<ImageSampleType> & samples)
{
  /** Determine the grid. */
  const auto [gridIndex, gridSize] = DetermineGridIndexAndSize(croppedInputImageRegion, gridSpacing);

  // TODO Use ITK 5.4: numberOfSamplesOnGrid = gridSize.CalculateProductOfElements()
  const std::size_t numberOfSamplesOnGrid =
    std::accumulate(gridSize.cbegin(), gridSize.cend(), std::size_t{ 1 }, std::multiplies<>{});

  samples.clear();
  samples.resize(numberOfSamplesOnGrid);

  UserData userData{ inputImage,
                     mask,
                     gridSpacing,
                     GenerateWorkUnits(numberOfWorkUnits, croppedInputImageRegion, gridIndex, gridSpacing, samples) };

  multiThreader.SetSingleMethod(mask ? &Self::ThreaderCallback<true> : &Self::ThreaderCallback<false>, &userData);
  multiThreader.SingleMethodExecute();

  if (mask)
  {
    if (auto & workUnits = userData.WorkUnits; !workUnits.empty())
    {
      auto * sampleData = samples.data() + workUnits.front().NumberOfSamples;

      for (size_t i{ 1 }; i < workUnits.size(); ++i)
      {
        const WorkUnit & workUnit = workUnits[i];

        sampleData = std::copy_n(workUnit.Samples, workUnit.NumberOfSamples, sampleData);
      }

      samples.resize(sampleData - samples.data());
    }
  }
}
/**
 * ******************* GenerateData *******************
 */

template <class TInputImage>
void
ImageGridSampler<TInputImage>::GenerateData()
{
  /** Get handles to the input image, output sample container, and the mask. */
  const InputImageType &     inputImage = elastix::Deref(this->GetInput());
  ImageSampleContainerType & sampleContainer = elastix::Deref(this->GetOutput());
  const MaskType * const     mask = this->Superclass::GetMask();

  if (mask)
  {
    mask->UpdateSource();
  }

  // Take capacity from the output container, and clear it.
  std::vector<ImageSampleType> sampleVector;
  sampleContainer.swap(sampleVector);
  sampleVector.clear();

  /** Take into account the possibility of a smaller bounding box around the mask */
  this->SetNumberOfSamples(m_RequestedNumberOfSamples);

  const auto croppedInputImageRegion = this->GetCroppedInputImageRegion();

  if (Superclass::m_UseMultiThread)
  {
    MultiThreadedGenerateData(elastix::Deref(this->ProcessObject::GetMultiThreader()),
                              ProcessObject::GetNumberOfWorkUnits(),
                              inputImage,
                              mask,
                              croppedInputImageRegion,
                              m_SampleGridSpacing,
                              sampleVector);
  }
  else
  {
    SingleThreadedGenerateData(inputImage, mask, croppedInputImageRegion, m_SampleGridSpacing, sampleVector);
  }
  // Move the samples from the vector into the output container.
  sampleContainer.swap(sampleVector);


} // end GenerateData()


template <class TInputImage>
template <bool VUseMask>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
ImageGridSampler<TInputImage>::ThreaderCallback(void * const arg)
{
  assert(arg);
  const auto & info = *static_cast<const MultiThreaderBase::WorkUnitInfo *>(arg);

  assert(info.UserData);
  auto & userData = *static_cast<UserData *>(info.UserData);

  const auto workUnitID = info.WorkUnitID;

  if (workUnitID >= userData.WorkUnits.size())
  {
    return ITK_THREAD_RETURN_DEFAULT_VALUE;
  }

  auto & workUnit = userData.WorkUnits[workUnitID];

  auto *             samples = workUnit.Samples;
  const auto &       inputImage = userData.InputImage;
  const auto * const mask = userData.Mask;
  assert((mask == nullptr) == (!VUseMask));

  const auto                gridSpacing = userData.GridSpacing;
  const SampleGridSizeType  gridSizeForThread = workUnit.GridSize;
  const SampleGridIndexType gridIndexForThread = workUnit.GridIndex;

  /** Prepare for looping over the grid. */
  SampleGridIndexType index = gridIndexForThread;

  /** Ugly loop over the grid. */
  for (unsigned int t = 0; t < GetGridSizeValue<3>(gridSizeForThread); ++t)
  {
    for (unsigned int z = 0; z < GetGridSizeValue<2>(gridSizeForThread); ++z)
    {
      for (unsigned int y = 0; y < gridSizeForThread[1]; ++y)
      {
        for (unsigned int x = 0; x < gridSizeForThread[0]; ++x)
        {
          // Translate index to point.
          const auto point = inputImage.template TransformIndexToPhysicalPoint<SpacePrecisionType>(index);

          if constexpr (VUseMask)
          {
            if (mask->IsInsideInWorldSpace(point))
            {
              // Get sampled fixed image value.
              const auto pixel = inputImage.GetPixel(index);

              // Store sample in container.
              *samples = { point, static_cast<typename ImageSampleType::RealType>(pixel) };
              ++samples;
            }
          }
          else
          {
            // Get sampled fixed image value.
            const auto pixel = inputImage.GetPixel(index);

            // Store sample in container.
            *samples = { point, static_cast<typename ImageSampleType::RealType>(pixel) };
            ++samples;
          }

          // Jump to next position on grid.
          index[0] += gridSpacing[0];
        }
        JumpToNextGridPosition<1>(index, gridIndexForThread, gridSpacing);
      }
      JumpToNextGridPosition<2>(index, gridIndexForThread, gridSpacing);
    }
    JumpToNextGridPosition<3>(index, gridIndexForThread, gridSpacing);
  }

  if constexpr (VUseMask)
  {
    workUnit.NumberOfSamples = samples - workUnit.Samples;
  }

  return ITK_THREAD_RETURN_DEFAULT_VALUE;
}


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
    Math::Round<int64_t>(std::pow(fraction, 1.0 / indimd)));
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
