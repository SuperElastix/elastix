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
#ifndef itkPCAMetric_h
#define itkPCAMetric_h

#include "itkAdvancedImageToImageMetric.h"

#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkImageRandomCoordinateSampler.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkExtractImageFilter.h"

using namespace std;

namespace itk
{
template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT PCAMetric : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(PCAMetric);

  /** Standard class typedefs. */
  using Self = PCAMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using typename Superclass::FixedImageRegionType;
  using FixedImageSizeType = typename FixedImageRegionType::SizeType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(PCAMetric, AdvancedImageToImageMetric);

  /** Set functions. */
  itkSetMacro(SampleLastDimensionRandomly, bool);
  itkSetMacro(NumSamplesLastDimension, unsigned int);
  itkSetMacro(NumAdditionalSamplesFixed, unsigned int);
  itkSetMacro(ReducedDimensionIndex, unsigned int);
  itkSetMacro(SubtractMean, bool);
  itkSetMacro(GridSize, FixedImageSizeType);
  itkSetMacro(TransformIsStackTransform, bool);
  itkSetMacro(NumEigenValues, unsigned int);
  itkSetMacro(UseDerivativeOfMean, bool);
  itkSetMacro(DeNoise, bool);
  itkSetMacro(VarNoise, double);

  /** Get functions. */
  itkGetConstMacro(SampleLastDimensionRandomly, bool);
  itkGetConstMacro(NumSamplesLastDimension, int);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageConstPointer;
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

  /** Get the value for single valued optimizers. */
  virtual MeasureType
  GetValue(const TransformParametersType & parameters) const;

  /** Get the derivatives of the match measure. */
  virtual void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & derivative) const;

  /** Get value and derivatives for multiple valued optimizers. */
  virtual void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation.   */

  virtual void
  Initialize();

protected:
  PCAMetric();
  virtual ~PCAMetric() {}
  void
  PrintSelf(std::ostream & os, Indent indent) const;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  using typename Superclass::FixedImageIndexType;
  using typename Superclass::FixedImageIndexValueType;
  using typename Superclass::MovingImageIndexType;
  using typename Superclass::FixedImagePointType;
  using FixedImageContinuousIndexType =
    typename itk::ContinuousIndex<CoordinateRepresentationType, FixedImageDimension>;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageContinuousIndexType;
  using typename Superclass::BSplineInterpolatorType;
  using typename Superclass::CentralDifferenceGradientFilterType;
  using typename Superclass::MovingImageDerivativeType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /** Computes the innerproduct of transform Jacobian with moving image gradient.
   * The results are stored in imageJacobian, which is supposed
   * to have the right size (same length as Jacobian's number of columns). */
  void
  EvaluateTransformJacobianInnerProduct(const TransformJacobianType &     jacobian,
                                        const MovingImageDerivativeType & movingImageDerivative,
                                        DerivativeType &                  imageJacobian) const;

  mutable vnl_vector<double> m_firstEigenVector;
  mutable vnl_vector<double> m_secondEigenVector;
  mutable vnl_vector<double> m_thirdEigenVector;
  mutable vnl_vector<double> m_fourthEigenVector;
  mutable vnl_vector<double> m_fifthEigenVector;
  mutable vnl_vector<double> m_sixthEigenVector;
  mutable vnl_vector<double> m_seventhEigenVector;
  mutable vnl_vector<double> m_eigenValues;
  mutable vnl_vector<double> m_normdCdmu;
  mutable int                m_NumberOfSamples;

private:
  /** Sample n random numbers from 0..m and add them to the vector. */
  void
  SampleRandom(const int n, const int m, std::vector<int> & numbers) const;

  /** Variables to control random sampling in last dimension. */
  bool         m_SampleLastDimensionRandomly;
  unsigned int m_NumSamplesLastDimension;
  unsigned int m_NumAdditionalSamplesFixed;
  unsigned int m_ReducedDimensionIndex;

  bool m_DeNoise;

  double m_VarNoise;

  /** Bool to determine if we want to subtract the mean derivate from the derivative elements. */
  bool m_SubtractMean;

  /** GridSize of B-spline transform. */
  FixedImageSizeType m_GridSize;

  /** Bool to indicate if the transform used is a stacktransform. Set by elx files. */
  bool m_TransformIsStackTransform;

  /** Integer to indicate how many eigenvalues you want to use in the metric */
  unsigned int m_NumEigenValues;

  bool m_UseDerivativeOfMean;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkPCAMetric.hxx"
#endif

#endif // end #ifndef itkPCAMetric_h
