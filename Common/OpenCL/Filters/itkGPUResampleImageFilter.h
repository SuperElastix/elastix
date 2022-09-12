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
#ifndef itkGPUResampleImageFilter_h
#define itkGPUResampleImageFilter_h

#include "itkResampleImageFilter.h"

#include "itkGPUImageToImageFilter.h"
#include "itkGPUInterpolateImageFunction.h"
#include "itkGPUBSplineInterpolateImageFunction.h"
#include "itkGPUBSplineBaseTransform.h"
#include "itkGPUTransformBase.h"
#include "itkGPUCompositeTransformBase.h"

namespace itk
{
/** Create a helper GPU Kernel class for GPUResampleImageFilter */
itkGPUKernelClassMacro(GPUResampleImageFilterKernel);

/** \class GPUResampleImageFilter
 * \brief GPU version of ResampleImageFilter.
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup GPUCommon
 */
template <typename TInputImage,
          typename TOutputImage,
          typename TInterpolatorPrecisionType = float,
          typename TTransformPrecisionType = TInterpolatorPrecisionType>
class ITK_EXPORT GPUResampleImageFilter
  : public GPUImageToImageFilter<
      TInputImage,
      TOutputImage,
      ResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPUResampleImageFilter);

  /** Standard class typedefs. */
  using Self = GPUResampleImageFilter;
  using CPUSuperclass =
    ResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType, TTransformPrecisionType>;
  using GPUSuperclass = GPUImageToImageFilter<TInputImage, TOutputImage, CPUSuperclass>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUResampleImageFilter, GPUSuperclass);

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Some convenient typedefs. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using GPUInputImage = typename GPUTraits<TInputImage>::Type;
  using GPUOutputImage = typename GPUTraits<TOutputImage>::Type;
  using InterpolatorPrecisionType = TInterpolatorPrecisionType;

  /** Superclass typedefs. */
  using InterpolatorType = typename CPUSuperclass::InterpolatorType;
  using TransformType = typename CPUSuperclass::TransformType;
  using ExtrapolatorType = typename CPUSuperclass::ExtrapolatorType;
  using InputImageRegionType = typename CPUSuperclass::InputImageRegionType;
  using OutputImageRegionType = typename CPUSuperclass::OutputImageRegionType;
  using SizeType = typename CPUSuperclass::SizeType;
  using IndexType = typename CPUSuperclass::IndexType;

  using OutputImagePixelType = typename GPUSuperclass::OutputImagePixelType;

  /** Other typedefs. */
  using GPUKernelManagerPointer = typename OpenCLKernelManager::Pointer;
  using GPUDataManagerPointer = typename GPUDataManager::Pointer;
  using CompositeTransformBaseType = GPUCompositeTransformBase<InterpolatorPrecisionType, InputImageDimension>;

  /** Typedefs for the B-spline interpolator. */
  using GPUBSplineInterpolatorType = GPUBSplineInterpolateImageFunction<InputImageType, InterpolatorPrecisionType>;
  using GPUBSplineInterpolatorCoefficientImageType = typename GPUBSplineInterpolatorType::GPUCoefficientImageType;
  using GPUBSplineInterpolatorCoefficientImagePointer = typename GPUBSplineInterpolatorType::GPUCoefficientImagePointer;
  using GPUBSplineInterpolatorDataManagerPointer = typename GPUBSplineInterpolatorType::GPUDataManagerPointer;

  /** Typedefs for the B-spline transform. */
  using GPUBSplineBaseTransformType = GPUBSplineBaseTransform<InterpolatorPrecisionType, InputImageDimension>;

  /** Set the interpolator. */
  void
  SetInterpolator(InterpolatorType * _arg) override;

  /** Set the extrapolator. Not yet supported. */
  void
  SetExtrapolator(ExtrapolatorType * _arg) override;

  /** Set the transform. */
  void
  SetTransform(const TransformType * _arg) override;

  /** Set/Get the requested number of splits on OpenCL device.
   * Only works for 3D images. For 1D, 2D are always equal 1. */
  itkSetMacro(RequestedNumberOfSplits, unsigned int);
  itkGetConstMacro(RequestedNumberOfSplits, unsigned int);

protected:
  GPUResampleImageFilter();
  ~GPUResampleImageFilter() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  void
  GPUGenerateData() override;

  // Supported GPU transform types
  enum GPUTransformTypeEnum
  {
    IdentityTransform = 1,
    MatrixOffsetTransform,
    TranslationTransform,
    BSplineTransform,
    Else
  };

  /** Set arguments for the pre kernel manager. */
  void
  SetArgumentsForPreKernelManager(const typename GPUOutputImage::Pointer & output);

  /** Set arguments for the loop kernel manager. */
  void
  SetArgumentsForLoopKernelManager(const typename GPUInputImage::Pointer &  input,
                                   const typename GPUOutputImage::Pointer & output);

  /** Set arguments for the loop kernel manager. */
  void
  SetTransformParametersForLoopKernelManager(const std::size_t transformIndex);

  /** Set arguments for the post kernel manager. */
  void
  SetArgumentsForPostKernelManager(const typename GPUInputImage::Pointer &  input,
                                   const typename GPUOutputImage::Pointer & output);

  /** Set the B-spline transform coefficient images to the GPU. */
  void
  SetBSplineTransformCoefficientsToGPU(const std::size_t transformIndex);

  /** Get transform type. */
  const GPUTransformTypeEnum
  GetTransformType(const int & transformIndex) const;

  /** Check if a certain transform is present in the list of transforms. */
  bool
  HasTransform(const GPUTransformTypeEnum type) const;

  /** Get a handle to a certain transform type. */
  int
  GetTransformHandle(const GPUTransformTypeEnum type) const;

  /** Get a handle to the kernel given a handle to a transform. */
  bool
  GetKernelIdFromTransformId(const std::size_t & index, std::size_t & kernelId) const;

  /** Get the BSpline base transform. */
  GPUBSplineBaseTransformType *
  GetGPUBSplineBaseTransform(const std::size_t transformIndex);

private:
  GPUInterpolatorBase * m_InterpolatorBase;
  GPUTransformBase *    m_TransformBase;

  GPUDataManagerPointer m_InputGPUImageBase;
  GPUDataManagerPointer m_OutputGPUImageBase;
  GPUDataManagerPointer m_FilterParameters;
  GPUDataManagerPointer m_DeformationFieldBuffer;
  unsigned int          m_RequestedNumberOfSplits;

  using TransformHandle = std::pair<int, bool>;
  using TransformsHandle = std::map<GPUTransformTypeEnum, TransformHandle>;

#if 0
  class TransformKernelHelper
  {
    GPUTransformTypeEnum m_TransformType;
    std::string          m_TransformTypeAsString;
    std::: size_t        m_TransformKernelHandle;
    bool                 m_Compiled;
    TransformKernelHelper()
    {
      m_TransformType         = GPUResampleImageFilter::Else;
      m_TransformTypeAsString = "Else";
      m_TransformKernelHandle = 0; // ??
      m_Compiled              = false;
    }


  };

  std::vector< TransformKernelHelper > m_SupportedTransformKernels;
#endif

  std::vector<std::string> m_Sources;
  std::size_t              m_SourceIndex;

  std::size_t m_InterpolatorSourceLoadedIndex;
  std::size_t m_TransformSourceLoadedIndex;

  bool m_InterpolatorIsBSpline;
  bool m_TransformIsCombo;

  std::size_t      m_FilterPreGPUKernelHandle;
  TransformsHandle m_FilterLoopGPUKernelHandle;
  std::size_t      m_FilterPostGPUKernelHandle;

  // GPU kernel managers
  GPUKernelManagerPointer m_PreKernelManager;
  GPUKernelManagerPointer m_LoopKernelManager;
  GPUKernelManagerPointer m_PostKernelManager;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPUResampleImageFilter.hxx"
#endif

#endif /* itkGPUResampleImageFilter_h */
