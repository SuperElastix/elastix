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
  ITK_DISALLOW_COPY_AND_MOVE(ParzenWindowNormalizedMutualInformationImageToImageMetric);

  /** Standard class typedefs. */
  using Self = ParzenWindowNormalizedMutualInformationImageToImageMetric;
  using Superclass = ParzenWindowHistogramImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ParzenWindowNormalizedMutualInformationImageToImageMetric, ParzenWindowHistogramImageToImageMetric);

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
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkParzenWindowNormalizedMutualInformationImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkParzenWindowNormalizedMutualInformationImageToImageMetric_h
