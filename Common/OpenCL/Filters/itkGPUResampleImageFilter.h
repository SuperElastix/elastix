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
template <typename TInputImage, typename TOutputImage, typename TInterpolatorPrecisionType = float>
class ITK_EXPORT GPUResampleImageFilter
  : public GPUImageToImageFilter<TInputImage,
                                 TOutputImage,
                                 ResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType>>
{
public:
  /** Standard class typedefs. */
  typedef GPUResampleImageFilter                                                     Self;
  typedef ResampleImageFilter<TInputImage, TOutputImage, TInterpolatorPrecisionType> CPUSuperclass;
  typedef GPUImageToImageFilter<TInputImage, TOutputImage, CPUSuperclass>            GPUSuperclass;
  typedef SmartPointer<Self>                                                         Pointer;
  typedef SmartPointer<const Self>                                                   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUResampleImageFilter, GPUSuperclass);

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int, TOutputImage::ImageDimension);

  /** Some convenient typedefs. */
  typedef TInputImage                            InputImageType;
  typedef TOutputImage                           OutputImageType;
  typedef typename GPUTraits<TInputImage>::Type  GPUInputImage;
  typedef typename GPUTraits<TOutputImage>::Type GPUOutputImage;
  typedef TInterpolatorPrecisionType             InterpolatorPrecisionType;

  /** Superclass typedefs. */
  typedef typename CPUSuperclass::InterpolatorType      InterpolatorType;
  typedef typename CPUSuperclass::TransformType         TransformType;
  typedef typename CPUSuperclass::ExtrapolatorType      ExtrapolatorType;
  typedef typename CPUSuperclass::InputImageRegionType  InputImageRegionType;
  typedef typename CPUSuperclass::OutputImageRegionType OutputImageRegionType;
  typedef typename CPUSuperclass::SizeType              SizeType;
  typedef typename CPUSuperclass::IndexType             IndexType;

  typedef typename GPUSuperclass::OutputImagePixelType OutputImagePixelType;

  /** Other typedefs. */
  typedef typename OpenCLKernelManager::Pointer                                     GPUKernelManagerPointer;
  typedef typename GPUDataManager::Pointer                                          GPUDataManagerPointer;
  typedef GPUCompositeTransformBase<InterpolatorPrecisionType, InputImageDimension> CompositeTransformBaseType;

  /** Typedefs for the B-spline interpolator. */
  typedef GPUBSplineInterpolateImageFunction<InputImageType, InterpolatorPrecisionType> GPUBSplineInterpolatorType;
  typedef typename GPUBSplineInterpolatorType::GPUCoefficientImageType    GPUBSplineInterpolatorCoefficientImageType;
  typedef typename GPUBSplineInterpolatorType::GPUCoefficientImagePointer GPUBSplineInterpolatorCoefficientImagePointer;
  typedef typename GPUBSplineInterpolatorType::GPUDataManagerPointer      GPUBSplineInterpolatorDataManagerPointer;

  /** Typedefs for the B-spline transform. */
  typedef GPUBSplineBaseTransform<InterpolatorPrecisionType, InputImageDimension> GPUBSplineBaseTransformType;

  /** Set the interpolator. */
  virtual void
  SetInterpolator(InterpolatorType * _arg);

  /** Set the extrapolator. Not yet supported. */
  virtual void
  SetExtrapolator(ExtrapolatorType * _arg);

  /** Set the transform. */
  virtual void
  SetTransform(const TransformType * _arg);

  /** Set/Get the requested number of splits on OpenCL device.
   * Only works for 3D images. For 1D, 2D are always equal 1. */
  itkSetMacro(RequestedNumberOfSplits, unsigned int);
  itkGetConstMacro(RequestedNumberOfSplits, unsigned int);

protected:
  GPUResampleImageFilter();
  ~GPUResampleImageFilter() = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  virtual void
  GPUGenerateData(void);

  // Supported GPU transform types
  typedef enum
  {
    IdentityTransform = 1,
    MatrixOffsetTransform,
    TranslationTransform,
    BSplineTransform,
    Else
  } GPUTransformTypeEnum;

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
  GPUResampleImageFilter(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  GPUInterpolatorBase * m_InterpolatorBase;
  GPUTransformBase *    m_TransformBase;

  GPUDataManagerPointer m_InputGPUImageBase;
  GPUDataManagerPointer m_OutputGPUImageBase;
  GPUDataManagerPointer m_FilterParameters;
  GPUDataManagerPointer m_DeformationFieldBuffer;
  unsigned int          m_RequestedNumberOfSplits;

  typedef std::pair<int, bool>                            TransformHandle;
  typedef std::map<GPUTransformTypeEnum, TransformHandle> TransformsHandle;

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
