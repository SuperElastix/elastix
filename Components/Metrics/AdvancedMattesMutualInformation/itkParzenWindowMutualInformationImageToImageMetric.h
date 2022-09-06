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
#ifndef itkParzenWindowMutualInformationImageToImageMetric_h
#define itkParzenWindowMutualInformationImageToImageMetric_h

#include "itkParzenWindowHistogramImageToImageMetric.h"

#include "itkArray2D.h"

namespace itk
{

/**
 * \class ParzenWindowMutualInformationImageToImageMetric
 * \brief Computes the mutual information between two images to be
 * registered using the method of Mattes et al.
 *
 * ParzenWindowMutualInformationImageToImageMetric computes the mutual
 * information between a fixed and moving image to be registered.
 *
 * The calculations are based on the method of Mattes et al. [1,2],
 * where the probability density distribution are estimated using
 * Parzen histograms. Once the PDFs have been constructed, the
 * mutual information is obtained by double summing over the
 * discrete PDF values.
 *
 * Construction of the PDFs is implemented in the superclass
 * ParzenWindowHistogramImageToImageMetric.
 *
 * This implementation of the MattesMutualInformation is based on the
 * AdvancedImageToImageMetric, which means that:
 * \li It uses the ImageSampler-framework
 * \li It makes use of the compact support of B-splines, in case of B-spline transforms.
 * \li Image derivatives are computed using either the B-spline interpolator implementation
 * or by nearest neighbor interpolation of a precomputed central difference image.
 * \li A minimum number of samples that should map within the moving image (mask) can be specified.
 *
 * Notes:\n
 * 1. This class returns the negative mutual information value.\n
 * 2. This class in not thread safe due the private data structures
 *     used to the store the marginal and joint pdfs.
 *
 * References:\n
 * [1] "Nonrigid multimodality image registration"\n
 *      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank\n
 *      Medical Imaging 2001: Image Processing, 2001, pp. 1609-1620.\n
 * [2] "PET-CT Image Registration in the Chest Using Free-form Deformations"\n
 *      D. Mattes, D. R. Haynor, H. Vesselle, T. Lewellen and W. Eubank\n
 *      IEEE Transactions in Medical Imaging. To Appear.\n
 * [3] "Optimization of Mutual Information for MultiResolution Image
 *      Registration"\n
 *      P. Thevenaz and M. Unser\n
 *      IEEE Transactions in Image Processing, 9(12) December 2000.\n
 *
 * \ingroup Metrics
 * \sa ParzenWindowHistogramImageToImageMetric
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT ParzenWindowMutualInformationImageToImageMetric
  : public ParzenWindowHistogramImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ParzenWindowMutualInformationImageToImageMetric);

  /** Standard class typedefs. */
  using Self = ParzenWindowMutualInformationImageToImageMetric;
  using Superclass = ParzenWindowHistogramImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ParzenWindowMutualInformationImageToImageMetric, ParzenWindowHistogramImageToImageMetric);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::RealType;
  using typename Superclass::GradientPixelType;
  using typename Superclass::GradientImageType;
  using typename Superclass::GradientImagePointer;
  using typename Superclass::GradientImageFilterType;
  using typename Superclass::GradientImageFilterPointer;
  using typename Superclass::FixedImageMaskType;
  using typename Superclass::FixedImageMaskPointer;
  using typename Superclass::MovingImageMaskType;
  using typename Superclass::MovingImageMaskPointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;
  using typename Superclass::ParametersType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::MovingImageRegionType;
  using typename Superclass::ImageSamplerType;
  using typename Superclass::ImageSamplerPointer;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::FixedImageLimiterType;
  using typename Superclass::MovingImageLimiterType;
  using typename Superclass::FixedImageLimiterOutputType;
  using typename Superclass::MovingImageLimiterOutputType;
  using typename Superclass::MovingImageDerivativeScalesType;
  using typename Superclass::ThreaderType;
  using typename Superclass::ThreadInfoType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /**  Get the value. */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /** Set/get whether to apply the technique introduced by Nicholas Tustison; default: false */
  itkGetConstMacro(UseJacobianPreconditioning, bool);
  itkSetMacro(UseJacobianPreconditioning, bool);

protected:
  /** The constructor. */
  ParzenWindowMutualInformationImageToImageMetric();

  /** The destructor. */
  ~ParzenWindowMutualInformationImageToImageMetric() override = default;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  using typename Superclass::FixedImageIndexType;
  using typename Superclass::FixedImageIndexValueType;
  using typename Superclass::MovingImageIndexType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageContinuousIndexType;
  using typename Superclass::BSplineInterpolatorType;
  using typename Superclass::CentralDifferenceGradientFilterType;
  using typename Superclass::MovingImageDerivativeType;
  using typename Superclass::PDFValueType;
  using typename Superclass::PDFDerivativeValueType;
  using typename Superclass::MarginalPDFType;
  using typename Superclass::JointPDFType;
  using typename Superclass::JointPDFDerivativesType;
  using typename Superclass::IncrementalMarginalPDFType;
  using typename Superclass::JointPDFIndexType;
  using typename Superclass::JointPDFRegionType;
  using typename Superclass::JointPDFSizeType;
  using typename Superclass::JointPDFDerivativesIndexType;
  using typename Superclass::JointPDFDerivativesRegionType;
  using typename Superclass::JointPDFDerivativesSizeType;
  using typename Superclass::ParzenValueContainerType;
  using typename Superclass::KernelFunctionType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /**  Get the value and analytic derivative.
   * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == false.
   *
   * Implements a version that only loops once over the samples, but uses
   * a large block of memory to explicitly store the joint histogram derivative.
   * It's size is \#FixedHistogramBins * \#MovingHistogramBins * \#parameters * float.
   */
  void
  GetValueAndAnalyticDerivative(const ParametersType & parameters,
                                MeasureType &          value,
                                DerivativeType &       derivative) const override;

  /** Get the value and analytic derivative.
   * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == false
   * and UseExplicitPDFDerivatives == false.
   *
   * Implements a version that avoids the large memory allocation of the
   * explicit joint histogram derivative. This comes at the cost of looping
   * over the samples twice, instead of once. The first time does not require
   * GetJacobian() and moving image derivatives, however.
   */
  virtual void
  GetValueAndAnalyticDerivativeLowMemory(const ParametersType & parameters,
                                         MeasureType &          value,
                                         DerivativeType &       derivative) const;

  /**  Get the value and finite difference derivative.
   * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == true.
   *
   * This is really only here for experimental purposes.
   */
  void
  GetValueAndFiniteDifferenceDerivative(const ParametersType & parameters,
                                        MeasureType &          value,
                                        DerivativeType &       derivative) const override;

  /** Compute terms to implement preconditioning as proposed by Tustison et al. */
  virtual void
  ComputeJacobianPreconditioner(const TransformJacobianType &      jac,
                                const NonZeroJacobianIndicesType & nzji,
                                DerivativeType &                   preconditioner,
                                DerivativeType &                   divisor) const;

  /** Some initialization functions, called by Initialize. */
  void
  InitializeHistograms() override;

  /** Threading related parameters. */
  struct ParzenWindowMutualInformationMultiThreaderParameterType
  {
    Self * m_Metric;
  };
  ParzenWindowMutualInformationMultiThreaderParameterType m_ParzenWindowMutualInformationThreaderParameters;

  /** Multi-threaded versions of the ComputePDF function. */
  inline void
  ThreadedComputeDerivativeLowMemory(ThreadIdType threadId);

  /** Single-threadedly accumulate results. */
  inline void
  AfterThreadedComputeDerivativeLowMemory(DerivativeType & derivative) const;

  /** Helper function to launch the threads. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  ComputeDerivativeLowMemoryThreaderCallback(void * arg);

  /** Helper function to launch the threads. */
  void
  LaunchComputeDerivativeLowMemoryThreaderCallback() const;

private:
  /** Helper array for storing the values of the JointPDF ratios. */
  using PRatioType = double;
  using PRatioArrayType = Array2D<PRatioType>;
  mutable PRatioArrayType m_PRatioArray;

  /** Setting */
  bool m_UseJacobianPreconditioning;

  /** Helper function to compute the derivative for the low memory variant. */
  void
  ComputeDerivativeLowMemorySingleThreaded(DerivativeType & derivative) const;

  void
  ComputeDerivativeLowMemory(DerivativeType & derivative) const;

  /** Helper function to update the derivative for the low memory variant. */
  void
  UpdateDerivativeLowMemory(const RealType &                   fixedImageValue,
                            const RealType &                   movingImageValue,
                            const DerivativeType &             imageJacobian,
                            const NonZeroJacobianIndicesType & nzji,
                            DerivativeType &                   derivative) const;

  /** Helper function to compute m_PRatioArray in case of low memory consumption. */
  void
  ComputeValueAndPRatioArray(double & MI) const;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkParzenWindowMutualInformationImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkParzenWindowMutualInformationImageToImageMetric_h
