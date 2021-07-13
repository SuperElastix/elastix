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
#ifndef itkImageFullSampler_h
#define itkImageFullSampler_h

#include "itkImageSamplerBase.h"

namespace itk
{
/** \class ImageFullSampler
 *
 * \brief Samples all voxels in the InputImageRegion.
 *
 * This ImageSampler samples all voxels in the InputImageRegion.
 * If a mask is given: only those voxels within the mask AND the
 * InputImageRegion.
 *
 * \ingroup ImageSamplers
 */

template <class TInputImage>
class ITK_TEMPLATE_EXPORT ImageFullSampler : public ImageSamplerBase<TInputImage>
{
public:
  /** Standard ITK-stuff. */
  typedef ImageFullSampler              Self;
  typedef ImageSamplerBase<TInputImage> Superclass;
  typedef SmartPointer<Self>            Pointer;
  typedef SmartPointer<const Self>      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageFullSampler, ImageSamplerBase);

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

  /** Other typdefs. */
  typedef typename InputImageType::IndexType InputImageIndexType;
  typedef typename InputImageType::PointType InputImagePointType;

  /** Selecting new samples makes no sense if nothing changed.
   * The same samples would be selected anyway.
   */
  bool
  SelectNewSamplesOnUpdate(void) override
  {
    return false;
  }


  /** Returns whether the sampler supports SelectNewSamplesOnUpdate(). */
  bool
  SelectingNewSamplesOnUpdateSupported(void) const override
  {
    return false;
  }


protected:
  /** The constructor. */
  ImageFullSampler() = default;
  /** The destructor. */
  ~ImageFullSampler() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Function that does the work. */
  void
  GenerateData(void) override;

  /** Multi-threaded function that does the work. */
  void
  ThreadedGenerateData(const InputImageRegionType & inputRegionForThread, ThreadIdType threadId) override;

private:
  /** The deleted copy constructor. */
  ImageFullSampler(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageFullSampler.hxx"
#endif

#endif // end #ifndef itkImageFullSampler_h
