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
#ifndef itkImageRandomSampler_h
#define itkImageRandomSampler_h

#include "itkImageRandomSamplerBase.h"

namespace itk
{
/** \class ImageRandomSampler
 *
 * \brief Samples randomly some voxels of an image.
 *
 * This image sampler randomly samples 'NumberOfSamples' voxels in
 * the InputImageRegion. Voxels may be selected multiple times.
 * If a mask is given, the sampler tries to find samples within the
 * mask. If the mask is very sparse, this may take some time. In this case,
 * consider using the ImageRandomSamplerSparseMask.
 *
 * \ingroup ImageSamplers
 */

template <class TInputImage>
class ITK_TEMPLATE_EXPORT ImageRandomSampler : public ImageRandomSamplerBase<TInputImage>
{
public:
  /** Standard ITK-stuff. */
  typedef ImageRandomSampler                  Self;
  typedef ImageRandomSamplerBase<TInputImage> Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageRandomSampler, ImageRandomSamplerBase);

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
  typedef typename Superclass::ImageSampleValueType         ImageSampleValueType;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;
  typedef typename Superclass::MaskType                     MaskType;
  typedef typename Superclass::InputImageSizeType           InputImageSizeType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass::InputImageDimension);

  /** Other typedefs. */
  typedef typename InputImageType::IndexType InputImageIndexType;
  typedef typename InputImageType::PointType InputImagePointType;

protected:
  /** The constructor. */
  ImageRandomSampler() = default;
  /** The destructor. */
  ~ImageRandomSampler() override = default;

  /** Functions that do the work. */
  void
  GenerateData(void) override;

  void
  ThreadedGenerateData(const InputImageRegionType & inputRegionForThread, ThreadIdType threadId) override;

private:
  /** The deleted copy constructor. */
  ImageRandomSampler(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageRandomSampler.hxx"
#endif

#endif // end #ifndef itkImageRandomSampler_h
