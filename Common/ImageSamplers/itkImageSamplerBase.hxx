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
#ifndef itkImageSamplerBase_hxx
#define itkImageSamplerBase_hxx

#include "itkImageSamplerBase.h"
#include <itkDeref.h>
#include <itkMultiThreaderBase.h>
#include <cassert>
#include <numeric> // For accumulate.


namespace itk
{

/**
 * ******************* SetMask *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::SetMask(const MaskType * _arg, unsigned int pos)
{
  if (m_MaskVector.size() < pos + 1)
  {
    m_MaskVector.resize(pos + 1);
    m_NumberOfMasks = pos + 1;
  }
  if (pos == 0)
  {
    m_Mask = _arg;
  }
  if (m_MaskVector[pos] != _arg)
  {
    m_MaskVector[pos] = _arg;

    /** The following line is not necessary, since the local
     * bounding box is already computed when SetImage() is called
     * in the elxRegistrationBase (when the mask spatial object
     * is constructed).
     */
    // m_Mask->ComputeLocalBoundingBox();
    this->Modified();
  }

} // SetMask()


/**
 * ******************* GetMask *******************
 */

template <typename TInputImage>
auto
ImageSamplerBase<TInputImage>::GetMask(unsigned int pos) const -> const MaskType *
{
  if (m_MaskVector.size() < pos + 1)
  {
    return nullptr;
  }
  return m_MaskVector[pos];

} // end GetMask()

/**
 * ******************* SetNumberOfMasks *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::SetNumberOfMasks(const unsigned int _arg)
{
  if (m_NumberOfMasks != _arg)
  {
    m_MaskVector.resize(_arg);
    m_NumberOfMasks = _arg;
    this->Modified();
  }

} // end SetNumberOfMasks()


/**
 * ******************* SetInputImageRegion *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::SetInputImageRegion(const InputImageRegionType _arg, unsigned int pos)
{
  if (m_InputImageRegionVector.size() < pos + 1)
  {
    m_InputImageRegionVector.resize(pos + 1);
    m_NumberOfInputImageRegions = pos + 1;
  }
  if (pos == 0)
  {
    m_InputImageRegion = _arg;
  }
  if (m_InputImageRegionVector[pos] != _arg)
  {
    m_InputImageRegionVector[pos] = _arg;
    this->Modified();
  }

} // SetInputImageRegion()


/**
 * ******************* GetInputImageRegion *******************
 */

template <typename TInputImage>
auto
ImageSamplerBase<TInputImage>::GetInputImageRegion(unsigned int pos) const -> const InputImageRegionType &
{
  if (m_InputImageRegionVector.size() < pos + 1)
  {
    return m_DummyInputImageRegion;
  }
  return m_InputImageRegionVector[pos];

} // end GetInputImageRegion()

/**
 * ******************* SetNumberOfInputImageRegions *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::SetNumberOfInputImageRegions(const unsigned int _arg)
{
  if (m_NumberOfInputImageRegions != _arg)
  {
    m_InputImageRegionVector.resize(_arg);
    m_NumberOfInputImageRegions = _arg;
    this->Modified();
  }

} // end SetNumberOfInputImageRegions()


/**
 * ******************* GenerateInputRequestedRegion *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::GenerateInputRequestedRegion()
{
  /** Check if input image was set. */
  if (this->GetNumberOfInputs() == 0)
  {
    itkExceptionMacro("ERROR: Input image not set");
  }

  /** Get a non-const reference to the input image. */
  auto & inputImage = const_cast<InputImageType &>(Deref(this->GetInput()));

  /** Get and set the region. */
  if (this->GetInputImageRegion().GetNumberOfPixels() != 0)
  {
    InputImageRegionType inputRequestedRegion = this->GetInputImageRegion();

    /** Crop the input requested region at the input's largest possible region. */
    if (inputRequestedRegion.Crop(inputImage.GetLargestPossibleRegion()))
    {
      inputImage.SetRequestedRegion(inputRequestedRegion);
    }
    else
    {
      /** Couldn't crop the region (requested region is outside the largest
       * possible region). Throw an exception.
       */

      /** Store what we tried to request (prior to trying to crop). */
      inputImage.SetRequestedRegion(inputRequestedRegion);

      /** Build an exception. */
      InvalidRequestedRegionError e(__FILE__, __LINE__);
      e.SetLocation(ITK_LOCATION);
      e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
      e.SetDataObject(&inputImage);
      throw e;
    }
  }
  else
  {
    inputImage.SetRequestedRegion(inputImage.GetLargestPossibleRegion());
    this->SetInputImageRegion(inputImage.GetLargestPossibleRegion());
  }

  /** Crop the region of the inputImage to the bounding box of the mask. */
  this->CropInputImageRegion();
  inputImage.SetRequestedRegion(m_CroppedInputImageRegion);

} // end GenerateInputRequestedRegion()


/**
 * ******************* SelectNewSamplesOnUpdate *******************
 */

template <typename TInputImage>
bool
ImageSamplerBase<TInputImage>::SelectNewSamplesOnUpdate()
{
  /** Set the Modified flag, such that on calling Update(),
   * the GenerateData method is executed again.
   * Return true to indicate that indeed new samples will be selected.
   * Inheriting subclasses may just return false and do nothing.
   */
  this->Modified();
  return true;

} // end SelectNewSamplesOnUpdate()


/**
 * ******************* IsInsideAllMasks *******************
 */

template <typename TInputImage>
bool
ImageSamplerBase<TInputImage>::IsInsideAllMasks(const InputImagePointType & point) const
{
  bool ret = true;
  for (unsigned int i = 0; i < m_NumberOfMasks; ++i)
  {
    ret &= this->GetMask(i)->IsInsideInWorldSpace(point);
  }

  return ret;

} // end IsInsideAllMasks()


/**
 * ******************* UpdateAllMasks *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::UpdateAllMasks()
{
  /** If the masks are generated by a filter, then make sure they are updated. */
  for (unsigned int i = 0; i < m_NumberOfMasks; ++i)
  {
    this->GetMask(i)->UpdateSource();
  }

} // end UpdateAllMasks()


/**
 * ******************* CheckInputImageRegions *******************
 */

template <typename TInputImage>
bool
ImageSamplerBase<TInputImage>::CheckInputImageRegions()
{
  bool ret = true;
  for (unsigned int i = 0; i < this->GetNumberOfInputImageRegions(); ++i)
  {
    ret &= this->GetInput(i)->GetLargestPossibleRegion().IsInside(this->GetInputImageRegion(i));
  }
  return ret;

} // end CheckInputImageRegions()


/**
 * ******************* CropInputImageRegion *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::CropInputImageRegion()
{
  /** Since we expect to be called from GenerateInputRequestedRegion(),
   * we can safely assume that m_InputImageRegion is either
   * the LargestPossibleRegion of InputImage or a valid subregion of it.
   *
   * If a mask was set, then compute the intersection of the
   * InputImageRegion and the BoundingBoxRegion.
   */
  m_CroppedInputImageRegion = m_InputImageRegion;
  if (!m_Mask.IsNull())
  {
    /** Get a handle to the input image. */
    InputImageConstPointer inputImage = this->GetInput();
    if (!inputImage)
    {
      return;
    }

    this->UpdateAllMasks();

    /** Get the indices of the bounding box extremes, based on the first mask.
     * Note that the bounding box is defined in terms of the mask
     * spacing and origin, and that we need a region in terms
     * of the inputImage indices.
     */

    using BoundingBoxType = typename MaskType::BoundingBoxType;
    using PointsContainerType = typename BoundingBoxType::PointsContainer;
    typename BoundingBoxType::ConstPointer bb = m_Mask->GetMyBoundingBoxInWorldSpace();
    auto                                   bbIndex = BoundingBoxType::New();
    const PointsContainerType *            cornersWorld = bb->GetPoints();
    auto                                   cornersIndex = PointsContainerType::New();
    cornersIndex->Reserve(cornersWorld->Size());
    auto itCW = cornersWorld->begin();
    auto itCI = cornersIndex->begin();
    while (itCW != cornersWorld->end())
    {
      *itCI = inputImage->template TransformPhysicalPointToContinuousIndex<InputImagePointValueType>(*itCW);
      ++itCI;
      ++itCW;
    }
    bbIndex->SetPoints(cornersIndex);
    bbIndex->ComputeBoundingBox();

    /** Create a bounding box region. */
    InputImageIndexType  minIndex, maxIndex;
    InputImageSizeType   size;
    InputImageRegionType boundingBoxRegion;
    for (unsigned int i = 0; i < InputImageDimension; ++i)
    {
      /** apply ceil/floor for max/min resp. to be sure that
       * the bounding box is not too small */
      maxIndex[i] = static_cast<IndexValueType>(std::ceil(bbIndex->GetMaximum()[i]));
      minIndex[i] = static_cast<IndexValueType>(std::floor(bbIndex->GetMinimum()[i]));
      size[i] = maxIndex[i] - minIndex[i] + 1;
    }
    boundingBoxRegion.SetIndex(minIndex);
    boundingBoxRegion.SetSize(size);

    /** Compute the intersection. */
    bool cropped = m_CroppedInputImageRegion.Crop(boundingBoxRegion);

    /** If the cropping return false, then the intersection is empty.
     * In this case m_CroppedInputImageRegion is unchanged,
     * but we would like to throw an exception.
     */
    if (!cropped)
    {
      itkExceptionMacro("ERROR: the bounding box of the mask lies entirely out of the InputImageRegion!");
    }
  }

} // end CropInputImageRegion()


/**
 * ******************* PrintSelf *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "NumberOfMasks" << m_NumberOfMasks << std::endl;
  os << indent << "Mask: " << m_Mask.GetPointer() << std::endl;
  os << indent << "MaskVector:" << std::endl;
  for (unsigned int i = 0; i < m_NumberOfMasks; ++i)
  {
    os << indent.GetNextIndent() << m_MaskVector[i].GetPointer() << std::endl;
  }

  os << indent << "NumberOfInputImageRegions" << m_NumberOfInputImageRegions << std::endl;
  os << indent << "InputImageRegion: " << m_InputImageRegion << std::endl;
  os << indent << "InputImageRegionVector:" << std::endl;
  for (unsigned int i = 0; i < m_NumberOfInputImageRegions; ++i)
  {
    os << indent.GetNextIndent() << m_InputImageRegionVector[i] << std::endl;
  }
  os << indent << "CroppedInputImageRegion" << m_CroppedInputImageRegion << std::endl;

} // end PrintSelf()


/**
 * ******************* Constructor *******************
 */

template <typename TInputImage>
ImageSamplerBase<TInputImage>::ImageSamplerBase()
{
  this->ProcessObject::SetNumberOfRequiredInputs(1);
  this->ProcessObject::SetNumberOfRequiredOutputs(1);
  this->ProcessObject::SetNthOutput(0, OutputVectorContainerType::New().GetPointer());

} // end Constructor


/**
 * ******************* MakeOutput *******************
 */

template <typename TInputImage>
DataObject::Pointer
ImageSamplerBase<TInputImage>::MakeOutput(ProcessObject::DataObjectPointerArraySizeType itkNotUsed(idx))
{
  OutputVectorContainerPointer outputVectorContainer = OutputVectorContainerType::New();
  return outputVectorContainer.GetPointer();
} // end MakeOutput()


/**
 * ******************* SetInput *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::SetInput(unsigned int idx, const InputImageType * input)
{
  // process object is not const-correct, the const_cast
  // is required here.
  this->ProcessObject::SetNthInput(idx, const_cast<InputImageType *>(input));
} // end SetInput()


/**
 * ******************* SetInput *******************
 */

template <typename TInputImage>
void
ImageSamplerBase<TInputImage>::SetInput(const InputImageType * input)
{
  this->ProcessObject::SetNthInput(0, const_cast<InputImageType *>(input));
} // end SetInput()


/**
 * ******************* GetInput *******************
 */

template <typename TInputImage>
auto
ImageSamplerBase<TInputImage>::GetInput() -> const InputImageType *
{
  return dynamic_cast<const InputImageType *>(this->ProcessObject::GetInput(0));
} // end GetInput()

/**
 * ******************* GetInput *******************
 */

template <typename TInputImage>
auto
ImageSamplerBase<TInputImage>::GetInput(unsigned int idx) -> const InputImageType *
{
  return dynamic_cast<const InputImageType *>(this->ProcessObject::GetInput(idx));
} // end GetInput()

/**
 * ******************* GetOutput *******************
 */

template <typename TInputImage>
auto
ImageSamplerBase<TInputImage>::GetOutput() -> OutputVectorContainerType *
{
  return dynamic_cast<OutputVectorContainerType *>(this->ProcessObject::GetOutput(0));
} // end GetOutput()


template <typename TInputImage>
auto
ImageSamplerBase<TInputImage>::SplitRegion(const InputImageRegionType & inputRegion,
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


} // end namespace itk

#endif // end #ifndef itkImageSamplerBase_hxx
