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
#ifndef itkGPUBSplineDecompositionImageFilter_h
#define itkGPUBSplineDecompositionImageFilter_h

#include "itkBSplineDecompositionImageFilter.h"
#include "itkGPUInPlaceImageFilter.h"

namespace itk
{
/** Create a helper GPU Kernel class for GPUBSplineDecompositionImageFilter */
itkGPUKernelClassMacro(GPUBSplineDecompositionImageFilterKernel);

/** \class GPUBSplineDecompositionImageFilter
 * \brief GPU version of BSplineDecompositionImageFilter.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT GPUBSplineDecompositionImageFilter
  : public GPUImageToImageFilter<TInputImage, TOutputImage, BSplineDecompositionImageFilter<TInputImage, TOutputImage>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUBSplineDecompositionImageFilter);

  /** Standard ITK-stuff. */
  using Self = GPUBSplineDecompositionImageFilter;
  using CPUSuperclass = BSplineDecompositionImageFilter<TInputImage, TOutputImage>;
  using GPUSuperclass = GPUImageToImageFilter<TInputImage, TOutputImage, CPUSuperclass>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUBSplineDecompositionImageFilter, GPUSuperclass);

  /** Superclass typedefs. */
  using OutputImageRegionType = typename GPUSuperclass::OutputImageRegionType;
  using OutputImagePixelType = typename GPUSuperclass::OutputImagePixelType;
  using SplinePolesVectorType = typename GPUSuperclass::SplinePolesVectorType;

  /** Some convenient typedefs. */
  using InputImageType = TInputImage;
  using InputImagePointer = typename InputImageType::Pointer;
  using InputImageConstPointer = typename InputImageType::ConstPointer;
  using InputImageRegionType = typename InputImageType::RegionType;
  using InputImagePixelType = typename InputImageType::PixelType;

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

protected:
  GPUBSplineDecompositionImageFilter();
  ~GPUBSplineDecompositionImageFilter() {}
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  virtual void
  GPUGenerateData();

private:
  std::size_t m_FilterGPUKernelHandle;
  std::size_t m_DeviceLocalMemorySize;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUBSplineDecompositionImageFilter.hxx"
#endif

#endif /* itkGPUBSplineDecompositionImageFilter_h */
