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
  /** Standard class typedefs. */
  typedef PCAMetric                                             Self;
  typedef AdvancedImageToImageMetric<TFixedImage, TMovingImage> Superclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;

  typedef typename Superclass::FixedImageRegionType FixedImageRegionType;
  typedef typename FixedImageRegionType::SizeType   FixedImageSizeType;

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
  typedef typename Superclass::CoordinateRepresentationType    CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType                 MovingImageType;
  typedef typename Superclass::MovingImagePixelType            MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer         MovingImageConstPointer;
  typedef typename Superclass::FixedImageType                  FixedImageType;
  typedef typename Superclass::FixedImageConstPointer          FixedImageConstPointer;
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
  Initialize(void);

protected:
  PCAMetric();
  virtual ~PCAMetric() {}
  void
  PrintSelf(std::ostream & os, Indent indent) const;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  typedef typename Superclass::FixedImageIndexType      FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType     MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType      FixedImagePointType;
  typedef typename itk::ContinuousIndex<CoordinateRepresentationType, FixedImageDimension>
                                                                   FixedImageContinuousIndexType;
  typedef typename Superclass::MovingImagePointType                MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType      MovingImageContinuousIndexType;
  typedef typename Superclass::BSplineInterpolatorType             BSplineInterpolatorType;
  typedef typename Superclass::CentralDifferenceGradientFilterType CentralDifferenceGradientFilterType;
  typedef typename Superclass::MovingImageDerivativeType           MovingImageDerivativeType;
  typedef typename Superclass::NonZeroJacobianIndicesType          NonZeroJacobianIndicesType;

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
  PCAMetric(const Self &) = delete;
  void
  operator=(const Self &) = delete;

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
