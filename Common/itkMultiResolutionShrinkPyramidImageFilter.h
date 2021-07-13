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
#ifndef itkMultiResolutionShrinkPyramidImageFilter_h
#define itkMultiResolutionShrinkPyramidImageFilter_h

#include "itkMultiResolutionPyramidImageFilter.h"

namespace itk
{

/** \class MultiResolutionShrinkPyramidImageFilter
 * \brief Framework for creating images in a multi-resolution
 * pyramid.
 *
 * MultiResolutionShrinkPyramidImageFilter simply shrinks the input images.
 * No smoothing or any other operation is performed. This is useful for
 * example for registering binary images.
 *
 * \sa ShrinkImageFilter
 *
 * \ingroup PyramidImageFilter Multithreaded Streamed
 */
template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT MultiResolutionShrinkPyramidImageFilter
  : public MultiResolutionPyramidImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef MultiResolutionShrinkPyramidImageFilter                      Self;
  typedef MultiResolutionPyramidImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self>                                           Pointer;
  typedef SmartPointer<const Self>                                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiResolutionShrinkPyramidImageFilter, MultiResolutionPyramidImageFilter);

  /** ImageDimension enumeration. */
  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Inherit types from Superclass. */
  typedef typename Superclass::ScheduleType           ScheduleType;
  typedef typename Superclass::InputImageType         InputImageType;
  typedef typename Superclass::OutputImageType        OutputImageType;
  typedef typename Superclass::InputImagePointer      InputImagePointer;
  typedef typename Superclass::OutputImagePointer     OutputImagePointer;
  typedef typename Superclass::InputImageConstPointer InputImageConstPointer;

  /** Overwrite the Superclass implementation: no padding required. */
  void
  GenerateInputRequestedRegion(void) override;

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(SameDimensionCheck, (Concept::SameDimension<ImageDimension, OutputImageDimension>));
  itkConceptMacro(OutputHasNumericTraitsCheck, (Concept::HasNumericTraits<typename TOutputImage::PixelType>));
  /** End concept checking */
#endif

protected:
  MultiResolutionShrinkPyramidImageFilter() = default;
  ~MultiResolutionShrinkPyramidImageFilter() override = default;

  /** Generate the output data. */
  void
  GenerateData(void) override;

private:
  MultiResolutionShrinkPyramidImageFilter(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiResolutionShrinkPyramidImageFilter.hxx"
#endif

#endif
