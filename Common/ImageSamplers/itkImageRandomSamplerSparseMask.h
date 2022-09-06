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
  ITK_DISALLOW_COPY_AND_MOVE(ImageRandomSamplerSparseMask);

  /** Standard ITK-stuff. */
  using Self = ImageRandomSamplerSparseMask;
  using Superclass = ImageRandomSamplerBase<TInputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageRandomSamplerSparseMask, ImageRandomSamplerBase);

  /** Typedefs inherited from the superclass. */
  using typename Superclass::DataObjectPointer;
  using typename Superclass::OutputVectorContainerType;
  using typename Superclass::OutputVectorContainerPointer;
  using typename Superclass::InputImageType;
  using typename Superclass::InputImagePointer;
  using typename Superclass::InputImageConstPointer;
  using typename Superclass::InputImageRegionType;
  using typename Superclass::InputImagePixelType;
  using typename Superclass::ImageSampleType;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::MaskType;

  /** The input image dimension. */
  itkStaticConstMacro(InputImageDimension, unsigned int, Superclass::InputImageDimension);

  /** Other typdefs. */
  using InputImageIndexType = typename InputImageType::IndexType;
  using InputImagePointType = typename InputImageType::PointType;

  /** The random number generator used to generate random indices. */
  using RandomGeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
  using RandomGeneratorPointer = typename RandomGeneratorType::Pointer;

protected:
  using InternalFullSamplerType = itk::ImageFullSampler<InputImageType>;
  using InternalFullSamplerPointer = typename InternalFullSamplerType::Pointer;

  /** The constructor. */
  ImageRandomSamplerSparseMask() = default;
  /** The destructor. */
  ~ImageRandomSamplerSparseMask() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Function that does the work. */
  void
  GenerateData() override;

  /** Multi-threaded functionality that does the work. */
  void
  BeforeThreadedGenerateData() override;

  void
  ThreadedGenerateData(const InputImageRegionType & inputRegionForThread, ThreadIdType threadId) override;

  RandomGeneratorPointer     m_RandomGenerator{ RandomGeneratorType::GetInstance() };
  InternalFullSamplerPointer m_InternalFullSampler{ InternalFullSamplerType::New() };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageRandomSamplerSparseMask.hxx"
#endif

#endif // end #ifndef itkImageRandomSamplerSparseMask_h
