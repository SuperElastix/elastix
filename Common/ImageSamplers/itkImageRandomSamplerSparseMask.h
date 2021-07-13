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
#ifndef itkImageRandomSamplerSparseMask_h
#define itkImageRandomSamplerSparseMask_h

#include "itkImageRandomSamplerBase.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkImageFullSampler.h"

namespace itk
{
/** \class ImageRandomSamplerSparseMask
 *
 * \brief Samples randomly some voxels of an image.
 *
 * This version takes into account that the mask may be very small.
 * Also, it may be more efficient when very many different sample sets
 * of the same input image are required, because it does some precomputation.
 * \ingroup ImageSamplers
 */

template <class TInputImage>
class ITK_TEMPLATE_EXPORT ImageRandomSamplerSparseMask : public ImageRandomSamplerBase<TInputImage>
{
public:
  /** Standard ITK-stuff. */
  typedef ImageRandomSamplerSparseMask        Self;
  typedef ImageRandomSamplerBase<TInputImage> Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageRandomSamplerSparseMask, ImageRandomSamplerBase);

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

  /** The random number generator used to generate random indices. */
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;
  typedef typename RandomGeneratorType::Pointer                  RandomGeneratorPointer;

protected:
  typedef itk::ImageFullSampler<InputImageType>     InternalFullSamplerType;
  typedef typename InternalFullSamplerType::Pointer InternalFullSamplerPointer;

  /** The constructor. */
  ImageRandomSamplerSparseMask();
  /** The destructor. */
  ~ImageRandomSamplerSparseMask() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Function that does the work. */
  void
  GenerateData(void) override;

  /** Multi-threaded functionality that does the work. */
  void
  BeforeThreadedGenerateData(void) override;

  void
  ThreadedGenerateData(const InputImageRegionType & inputRegionForThread, ThreadIdType threadId) override;

  RandomGeneratorPointer     m_RandomGenerator;
  InternalFullSamplerPointer m_InternalFullSampler;

private:
  /** The deleted copy constructor. */
  ImageRandomSamplerSparseMask(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageRandomSamplerSparseMask.hxx"
#endif

#endif // end #ifndef itkImageRandomSamplerSparseMask_h
