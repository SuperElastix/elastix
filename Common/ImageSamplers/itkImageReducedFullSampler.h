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

#ifndef __ImageReducedFullSampler_h
#define __ImageReducedFullSampler_h

#include "itkImageSamplerBase.h"

namespace itk
{
/** \class ImageReducedFullSampler
 *
 * \brief Samples all voxels in the InputImageRegion for groupwise registration.
 *
 * This ImageSampler samples all voxels in the InputImageRegion.
 * If a mask is given: only those voxels within the mask AND the
 * InputImageRegion.
 *
 * \ingroup ImageSamplers
 */

template <class TInputImage>
class ImageReducedFullSampler : public ImageSamplerBase<TInputImage>
{
public:
  /** Standard ITK-stuff. */
  typedef ImageReducedFullSampler       Self;
  typedef ImageSamplerBase<TInputImage> Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageReducedFullSampler, ImageSamplerBase);

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::DataObjectPointer            DataObjectPointer;
  typedef typename Superclass::OutputVectorContainerType    OutputVectorContainerType;
  typedef typename Superclass::OutputVectorContainerPointer OutputVectorContainerPointer;
  typedef typename Superclass::InputImageType               InputImageType;
  typedef typename Superclass::InputImagePointer            InputImagePointer;
  typedef typename Superclass::InputImageConstPointer       InputImageConstPointer;
  typedef typename Superclass::InputImageRegionType         InputImageRegionType;
  typedef typename Superclass::InputImagePixelType          InputImagePixelType;
  typedef typename Superclass::ImageSampleType              ImageSampleType;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;
  typedef typename Superclass::MaskType                     MaskType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass::InputImageDimension);

  itkStaticConstMacro(ReducedInputImageDimension, unsigned int, Superclass::InputImageDimension - 1);

  /** Other typdefs. */
  typedef typename InputImageType::IndexType InputImageIndexType;
  typedef typename InputImageType::SizeType  InputImageSizeType;
  typedef typename InputImageType::PointType InputImagePointType;

  /** Selecting new samples makes no sense if nothing changed.
   * The same samples would be selected anyway.
   */
  virtual bool
  SelectNewSamplesOnUpdate(void)
  {
    return false;
  }


  /** Returns whether the sampler supports SelectNewSamplesOnUpdate(). */
  virtual bool
  SelectingNewSamplesOnUpdateSupported(void) const
  {
    return false;
  }


protected:
  /** The constructor. */
  ImageReducedFullSampler() {}
  /** The destructor. */
  virtual ~ImageReducedFullSampler() {}

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const;

  /** Function that does the work. */
  virtual void
  GenerateData(void);

private:
  /** The private constructor. */
  ImageReducedFullSampler(const Self &); // purposely not implemented
  /** The private copy constructor. */
  void
  operator=(const Self &); // purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageReducedFullSampler.hxx"
#endif

#endif // end #ifndef __ImageReducedFullSampler_h
