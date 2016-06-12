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
#ifndef __itkParzenWindowMutualInformationImageToImageMetric_H__
#define __itkParzenWindowMutualInformationImageToImageMetric_H__

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

template< class TFixedImage, class TMovingImage >
class ParzenWindowMutualInformationImageToImageMetric :
  public ParzenWindowHistogramImageToImageMetric< TFixedImage, TMovingImage >
{
public:

  /** Standard class typedefs. */
  typedef ParzenWindowMutualInformationImageToImageMetric Self;
  typedef ParzenWindowHistogramImageToImageMetric<
    TFixedImage, TMovingImage >                                       Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro(
    ParzenWindowMutualInformationImageToImageMetric,
    ParzenWindowHistogramImageToImageMetric );

  /** Typedefs from the superclass. */
  typedef typename
    Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::NumberOfParametersType     NumberOfParametersType;
  typedef typename Superclass::InterpolatorType           InterpolatorType;
  typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::DerivativeValueType        DerivativeValueType;
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType      MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType   ImageSampleContainerType;
  typedef typename
    Superclass::ImageSampleContainerPointer ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType  FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType MovingImageLimiterType;
  typedef typename
    Superclass::FixedImageLimiterOutputType FixedImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageLimiterOutputType MovingImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageDerivativeScalesType MovingImageDerivativeScalesType;
  typedef typename Superclass::ThreaderType   ThreaderType;
  typedef typename Superclass::ThreadInfoType ThreadInfoType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  /**  Get the value. */
  MeasureType GetValue( const ParametersType & parameters ) const;

  /** Set/get whether to apply the technique introduced by Nicholas Tustison; default: false */
  itkGetConstMacro( UseJacobianPreconditioning, bool );
  itkSetMacro( UseJacobianPreconditioning, bool );

protected:

  /** The constructor. */
  ParzenWindowMutualInformationImageToImageMetric();

  /** The destructor. */
  virtual ~ParzenWindowMutualInformationImageToImageMetric() {}

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  typedef typename Superclass::FixedImageIndexType                 FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType            FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType                MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType                 FixedImagePointType;
  typedef typename Superclass::MovingImagePointType                MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType      MovingImageContinuousIndexType;
  typedef typename Superclass::BSplineInterpolatorType             BSplineInterpolatorType;
  typedef typename Superclass::CentralDifferenceGradientFilterType CentralDifferenceGradientFilterType;
  typedef typename Superclass::MovingImageDerivativeType           MovingImageDerivativeType;
  typedef typename Superclass::PDFValueType                        PDFValueType;
  typedef typename Superclass::PDFDerivativeValueType              PDFDerivativeValueType;
  typedef typename Superclass::MarginalPDFType                     MarginalPDFType;
  typedef typename Superclass::JointPDFType                        JointPDFType;
  typedef typename Superclass::JointPDFDerivativesType             JointPDFDerivativesType;
  typedef typename Superclass::IncrementalMarginalPDFType          IncrementalMarginalPDFType;
  typedef typename Superclass::JointPDFIndexType                   JointPDFIndexType;
  typedef typename Superclass::JointPDFRegionType                  JointPDFRegionType;
  typedef typename Superclass::JointPDFSizeType                    JointPDFSizeType;
  typedef typename Superclass::JointPDFDerivativesIndexType        JointPDFDerivativesIndexType;
  typedef typename Superclass::JointPDFDerivativesRegionType       JointPDFDerivativesRegionType;
  typedef typename Superclass::JointPDFDerivativesSizeType         JointPDFDerivativesSizeType;
  typedef typename Superclass::ParzenValueContainerType            ParzenValueContainerType;
  typedef typename Superclass::KernelFunctionType                  KernelFunctionType;
  typedef typename Superclass::NonZeroJacobianIndicesType          NonZeroJacobianIndicesType;

  /**  Get the value and analytic derivative.
   * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == false.
   *
   * Implements a version that only loops once over the samples, but uses
   * a large block of memory to explicitly store the joint histogram derivative.
   * It's size is #FixedHistogramBins * #MovingHistogramBins * #parameters * float.
   */
  virtual void GetValueAndAnalyticDerivative(
    const ParametersType & parameters,
    MeasureType & value, DerivativeType & derivative ) const;

  /** Get the value and analytic derivative.
   * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == false
   * and UseExplicitPDFDerivatives == false.
   *
   * Implements a version that avoids the large memory allocation of the
   * explicit joint histogram derivative. This comes at the cost of looping
   * over the samples twice, instead of once. The first time does not require
   * GetJacobian() and moving image derivatives, however.
   */
  virtual void GetValueAndAnalyticDerivativeLowMemory(
    const ParametersType & parameters,
    MeasureType & value, DerivativeType & derivative ) const;

  /**  Get the value and finite difference derivative.
   * Called by GetValueAndDerivative if UseFiniteDifferenceDerivative == true.
   *
   * This is really only here for experimental purposes.
   */
  virtual void GetValueAndFiniteDifferenceDerivative(
    const ParametersType & parameters,
    MeasureType & value, DerivativeType & derivative ) const;

  /** Compute terms to implement preconditioning as proposed by Tustison et al. */
  virtual void ComputeJacobianPreconditioner(
    const TransformJacobianType & jac,
    const NonZeroJacobianIndicesType & nzji,
    DerivativeType & preconditioner,
    DerivativeType & divisor ) const;

  /** Some initialization functions, called by Initialize. */
  virtual void InitializeHistograms( void );

  /** Threading related parameters. */
  struct ParzenWindowMutualInformationMultiThreaderParameterType
  {
    Self * m_Metric;
  };
  ParzenWindowMutualInformationMultiThreaderParameterType m_ParzenWindowMutualInformationThreaderParameters;

  /** Multi-threaded versions of the ComputePDF function. */
  inline void ThreadedComputeDerivativeLowMemory( ThreadIdType threadId );

  /** Single-threadedly accumulate results. */
  inline void AfterThreadedComputeDerivativeLowMemory(
    DerivativeType & derivative ) const;

  /** Helper function to launch the threads. */
  static ITK_THREAD_RETURN_TYPE ComputeDerivativeLowMemoryThreaderCallback( void * arg );

  /** Helper function to launch the threads. */
  void LaunchComputeDerivativeLowMemoryThreaderCallback( void ) const;

private:

  /** The private constructor. */
  ParzenWindowMutualInformationImageToImageMetric( const Self & ); // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );                                  // purposely not implemented

  /** Helper array for storing the values of the JointPDF ratios. */
  typedef double                PRatioType;
  typedef Array2D< PRatioType > PRatioArrayType;
  mutable PRatioArrayType m_PRatioArray;

  /** Setting */
  bool m_UseJacobianPreconditioning;

  /** Helper function to compute the derivative for the low memory variant. */
  void ComputeDerivativeLowMemorySingleThreaded( DerivativeType & derivative ) const;

  void ComputeDerivativeLowMemory( DerivativeType & derivative ) const;

  /** Helper function to update the derivative for the low memory variant. */
  void UpdateDerivativeLowMemory(
    const RealType & fixedImageValue,
    const RealType & movingImageValue,
    const DerivativeType & imageJacobian,
    const NonZeroJacobianIndicesType & nzji,
    DerivativeType & derivative ) const;

  /** Helper function to compute m_PRatioArray in case of low memory consumption. */
  void ComputeValueAndPRatioArray( double & MI ) const;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkParzenWindowMutualInformationImageToImageMetric.hxx"
#endif

#endif // end #ifndef __itkParzenWindowMutualInformationImageToImageMetric_H__
