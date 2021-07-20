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

#ifndef itkParzenWindowNormalizedMutualInformationImageToImageMetric_h
#define itkParzenWindowNormalizedMutualInformationImageToImageMetric_h

#include "itkParzenWindowHistogramImageToImageMetric.h"

namespace itk
{

/**
 * \class ParzenWindowNormalizedMutualInformationImageToImageMetric
 * \brief Computes the normalized mutual information between two images to be
 * registered using a method based on Thevenaz&Unser [3].
 *
 * ParzenWindowNormalizedMutualInformationImageToImageMetric computes the
 * normalized mutual information between a fixed and moving image to be registered.
 * The calculations are based on the method of Mattes et al [1,2]
 * and Thevenaz&Unser [3], where the probability density distribution
 * are estimated using Parzen histograms. The expression for the
 * derivative is derived following [3].
 *
 * Construction of the PDFs is implemented in the superclass
 * ParzenWindowHistogramImageToImageMetric.
 *
 * This implementation of the NormalizedMutualInformation is based on the
 * AdvancedImageToImageMetric, which means that:
 * \li It uses the ImageSampler-framework
 * \li It makes use of the compact support of B-splines, in case of B-spline transforms.
 * \li Image derivatives are computed using either the B-spline interpolator's implementation
 * or by nearest neighbor interpolation of a precomputed central difference image.
 * \li A minimum number of samples that should map within the moving image (mask) can be specified.
 *
 * Notes:\n
 * 1. This class returns the negative normalized mutual information value.\n
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
class ITK_TEMPLATE_EXPORT ParzenWindowNormalizedMutualInformationImageToImageMetric
  : public ParzenWindowHistogramImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef ParzenWindowNormalizedMutualInformationImageToImageMetric          Self;
  typedef ParzenWindowHistogramImageToImageMetric<TFixedImage, TMovingImage> Superclass;
  typedef SmartPointer<Self>                                                 Pointer;
  typedef SmartPointer<const Self>                                           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ParzenWindowNormalizedMutualInformationImageToImageMetric, ParzenWindowHistogramImageToImageMetric);

  /** Typedefs from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType    CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType                 MovingImageType;
  typedef typename Superclass::MovingImagePixelType            MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer         MovingImageConstPointer;
  typedef typename Superclass::FixedImageType                  FixedImageType;
  typedef typename Superclass::FixedImageConstPointer          FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType            FixedImageRegionType;
  typedef typename Superclass::TransformType                   TransformType;
  typedef typename Superclass::TransformPointer                TransformPointer;
  typedef typename Superclass::InputPointType                  InputPointType;
  typedef typename Superclass::OutputPointType                 OutputPointType;
  typedef typename Superclass::TransformParametersType         TransformParametersType;
  typedef typename Superclass::TransformJacobianType           TransformJacobianType;
  typedef typename Superclass::InterpolatorType                InterpolatorType;
  typedef typename Superclass::InterpolatorPointer             InterpolatorPointer;
  typedef typename Superclass::RealType                        RealType;
  typedef typename Superclass::GradientPixelType               GradientPixelType;
  typedef typename Superclass::GradientImageType               GradientImageType;
  typedef typename Superclass::GradientImagePointer            GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType         GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer      GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType              FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer           FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType             MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer          MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                     MeasureType;
  typedef typename Superclass::DerivativeType                  DerivativeType;
  typedef typename Superclass::ParametersType                  ParametersType;
  typedef typename Superclass::FixedImagePixelType             FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType           MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType                ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer             ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType        ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer     ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType           FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType          MovingImageLimiterType;
  typedef typename Superclass::FixedImageLimiterOutputType     FixedImageLimiterOutputType;
  typedef typename Superclass::MovingImageLimiterOutputType    MovingImageLimiterOutputType;
  typedef typename Superclass::MovingImageDerivativeScalesType MovingImageDerivativeScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /**  Get the value: the negative normalized mutual information. */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /**  Get the value and derivatives for single valued optimizers. */
  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          Value,
                        DerivativeType &       Derivative) const override;

protected:
  /** The constructor. */
  ParzenWindowNormalizedMutualInformationImageToImageMetric() = default;

  /** The destructor. */
  ~ParzenWindowNormalizedMutualInformationImageToImageMetric() override = default;

  /** Print Self. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

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

  /** Replace the marginal probabilities by log(probabilities)
   * Changes the input pdf since they are not needed anymore! */
  virtual void
  ComputeLogMarginalPDF(MarginalPDFType & pdf) const;

  /** Compute the normalized mutual information and the jointEntropy
   * NMI = (Ef + Em) / Ej
   * Ef = fixed marginal entropy = - sum_k sum_i p(i,k) log pf(k)
   * Em = moving marginal entropy = - sum_k sum_i p(i,k) log pm(i)
   * Ej = joint entropy = - sum_k sum_i p(i,k) log p(i,k)
   */
  virtual MeasureType
  ComputeNormalizedMutualInformation(MeasureType & jointEntropy) const;

private:
  /** The deleted copy constructor. */
  ParzenWindowNormalizedMutualInformationImageToImageMetric(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkParzenWindowNormalizedMutualInformationImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkParzenWindowNormalizedMutualInformationImageToImageMetric_h
