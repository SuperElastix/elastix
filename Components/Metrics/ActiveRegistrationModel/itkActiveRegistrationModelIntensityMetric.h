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
#ifndef __itkActiveRegistrationModelImageIntensityMetric_h
#define __itkActiveRegistrationModelImageIntensityMetric_h

#include "itkAdvancedImageToImageMetric.h"

// Statismo includes
#include "itkDataManager.h"
#include "itkStatisticalModel.h"
#include "itkPCAModelBuilder.h"
#include "itkReducedVarianceModelBuilder.h"
#include "itkStandardImageRepresenter.h"

namespace itk
{

/** \class AdvancedMeanSquaresImageToImageMetric
 * \brief Compute Mean square difference between two images, based on AdvancedImageToImageMetric...
 *
 * This Class is templated over the type of the fixed and moving
 * images to be compared.
 *
 * This metric computes the sum of squared differenced between pixels in
 * the moving image and pixels in the fixed image. The spatial correspondance
 * between both images is established through a Transform. Pixel values are
 * taken from the Moving image. Their positions are mapped to the Fixed image
 * and result in general in non-grid position on it. Values at these non-grid
 * position of the Fixed image are interpolated using a user-selected Interpolator.
 *
 * This implementation of the MeanSquareDifference is based on the
 * AdvancedImageToImageMetric, which means that:
 * \li It uses the ImageSampler-framework
 * \li It makes use of the compact support of B-splines, in case of B-spline transforms.
 * \li Image derivatives are computed using either the B-spline interpolator's implementation
 * or by nearest neighbor interpolation of a precomputed central difference image.
 * \li A minimum number of samples that should map within the moving image (mask) can be specified.
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */

template< class TFixedImage, class TMovingImage >
class ActiveRegistrationModelIntensityMetric :
  public AdvancedImageToImageMetric< TFixedImage, TMovingImage >
{
public:

  /** Standard class typedefs. */
  typedef ActiveRegistrationModelIntensityMetric Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >                   Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageIntensityMetric, AdvancedImageToImageMetric );

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
  typedef typename Superclass::HessianValueType HessianValueType;
  typedef typename Superclass::HessianType      HessianType;
  typedef typename Superclass::ThreaderType     ThreaderType;
  typedef typename Superclass::ThreadInfoType   ThreadInfoType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  /** Get the value for single valued optimizers. */
  virtual MeasureType GetValue( const TransformParametersType & parameters ) const;

  // ActiveRegistrationModel typedefs
  typedef double                                                                  StatisticalModelScalarType;
  typedef vnl_vector< double >                                                    StatisticalModelVectorType;
  typedef vnl_matrix< double  >                                                   StatisticalModelMatrixType;

  typedef FixedImageType                                                          StatisticalModelImageType;
  typedef typename StatisticalModelImageType::Pointer                             StatisticalModelImagePointer;

  typedef StatisticalModel< StatisticalModelImageType >                           StatisticalModelType;
  typedef typename StatisticalModelType::Pointer                                  StatisticalModelPointer;

  typedef itk::StandardImageRepresenter<
          typename StatisticalModelImageType::PixelType,
          StatisticalModelImageType::ImageDimension >                             StatisticalModelRepresenterType;
  typedef typename StatisticalModelRepresenterType::Pointer                       StatisticalModelRepresenterPointer;

  typedef DataManager< StatisticalModelImageType >                                StatisticalModelDataManagerType;
  typedef typename StatisticalModelDataManagerType::Pointer                       StatisticalModelDataManagerPointer;

  typedef PCAModelBuilder< StatisticalModelImageType >                            StatisticalModelModelBuilderType;
  typedef typename StatisticalModelModelBuilderType::Pointer                      StatisticalModelBuilderPointer;

  typedef ReducedVarianceModelBuilder< StatisticalModelImageType >                ReducedVarianceModelBuilderType;
  typedef typename ReducedVarianceModelBuilderType::Pointer                       ReducedVarianceModelBuilderPointer;

  typedef unsigned int                                                            StatisticalModelIdType;

  typedef VectorContainer< StatisticalModelIdType, StatisticalModelMatrixType >   StatisticalModelMatrixContainerType;
  typedef typename StatisticalModelMatrixContainerType::Pointer                   StatisticalModelMatrixContainerPointer;
  typedef typename StatisticalModelMatrixContainerType::ConstPointer              StatisticalModelMatrixContainerConstPointer;
  typedef typename StatisticalModelMatrixContainerType::ConstIterator             StatisticalModelMatrixContainerConstIterator;

  typedef VectorContainer< StatisticalModelIdType, StatisticalModelVectorType >   StatisticalModelVectorContainerType;
  typedef typename StatisticalModelVectorContainerType::Pointer                   StatisticalModelVectorContainerPointer;
  typedef typename StatisticalModelVectorContainerType::ConstPointer              StatisticalModelVectorContainerConstPointer;
  typedef typename StatisticalModelVectorContainerType::ConstIterator             StatisticalModelVectorContainerConstIterator;

  typedef VectorContainer< StatisticalModelIdType, StatisticalModelScalarType >   StatisticalModelScalarContainerType;
  typedef typename StatisticalModelScalarContainerType::Pointer                   StatisticalModelScalarContainerPointer;
  typedef typename StatisticalModelScalarContainerType::ConstPointer              StatisticalModelScalarContainerConstPointer;
  typedef typename StatisticalModelScalarContainerType::ConstIterator             StatisticalModelScalarContainerConstIterator;

  typedef VectorContainer< StatisticalModelIdType, StatisticalModelRepresenterPointer > StatisticalModelRepresenterContainerType;
  typedef typename StatisticalModelRepresenterContainerType::Pointer              StatisticalModelRepresenterContainerPointer;
  typedef typename StatisticalModelRepresenterContainerType::ConstPointer         StatisticalModelRepresenterContainerConstPointer;
  typedef typename StatisticalModelRepresenterContainerType::ConstIterator        StatisticalModelRepresenterContainerConstIterator;

  itkSetConstObjectMacro( MeanVectorContainer, StatisticalModelVectorContainerType );
  itkGetConstObjectMacro( MeanVectorContainer, StatisticalModelVectorContainerType );

  itkSetConstObjectMacro( BasisMatrixContainer, StatisticalModelMatrixContainerType );
  itkGetConstObjectMacro( BasisMatrixContainer, StatisticalModelMatrixContainerType );

  itkSetConstObjectMacro( VarianceContainer, StatisticalModelVectorContainerType );
  itkGetConstObjectMacro( VarianceContainer, StatisticalModelVectorContainerType );

  itkSetConstObjectMacro( NoiseVarianceContainer, StatisticalModelScalarContainerType );
  itkGetConstObjectMacro( NoiseVarianceContainer, StatisticalModelScalarContainerType );

  itkSetConstObjectMacro( TotalVarianceContainer, StatisticalModelScalarContainerType );
  itkGetConstObjectMacro( TotalVarianceContainer, StatisticalModelScalarContainerType );

  itkSetConstObjectMacro( RepresenterContainer, StatisticalModelRepresenterContainerType );
  itkGetConstObjectMacro( RepresenterContainer, StatisticalModelRepresenterContainerType );

  /** Initialize the Metric by making sure that all the components are
  *  present and plugged together correctly.
  */
  virtual void Initialize( void );

  /** Get the derivatives of the match measure. */
  void GetDerivative( const TransformParametersType & parameters,
                      DerivativeType & Derivative ) const;

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
                              MeasureType & Value, DerivativeType & Derivative ) const;

  void GetValueAndFiniteDifferenceDerivative( const TransformParametersType & parameters,
                                              MeasureType& value,
                                              DerivativeType& derivative ) const;

  void GetModelValue( const StatisticalModelVectorType& meanVector,
                      const StatisticalModelMatrixType& basisMatrix,
                      const StatisticalModelScalarType& noiseVariance,
                      const StatisticalModelRepresenterPointer representer,
                      MeasureType & modelValue,
                      const TransformParametersType& parameters ) const;


  void GetModelFiniteDifferenceDerivative( const StatisticalModelVectorType& meanVector,
                                           const StatisticalModelMatrixType& basisMatrix,
                                           const StatisticalModelScalarType& noiseVariance,
                                           const StatisticalModelRepresenterPointer representer,
                                           DerivativeType& modelDerivative,
                                           const TransformParametersType & parameters ) const;

protected:

  ActiveRegistrationModelIntensityMetric();
  virtual ~ActiveRegistrationModelIntensityMetric(){}

  void PrintSelf( std::ostream & os, Indent indent ) const;

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
  typedef typename Superclass::NonZeroJacobianIndicesType          NonZeroJacobianIndicesType;

private:

  ActiveRegistrationModelIntensityMetric( const Self & ); // purposely not implemented
  void operator=( const Self & );       // purposely not implemented

  /**  Memory efficient computation of VV^T * ( T(mu) - mu ) */
  const StatisticalModelVectorType Reconstruct( const StatisticalModelVectorType& movingImageValues,
                                                const StatisticalModelMatrixType& subsampledBasisMatrix,
                                                const StatisticalModelScalarType& noiseVariance ) const;

  bool m_WriteReconstructedImageEachIteration;

  StatisticalModelVectorContainerConstPointer m_MeanVectorContainer;
  StatisticalModelMatrixContainerConstPointer m_BasisMatrixContainer;
  StatisticalModelVectorContainerConstPointer m_VarianceContainer;
  StatisticalModelScalarContainerConstPointer m_NoiseVarianceContainer;
  StatisticalModelScalarContainerConstPointer m_TotalVarianceContainer;
  StatisticalModelRepresenterContainerConstPointer m_RepresenterContainer;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkActiveRegistrationModelIntensityMetric.hxx"
#endif

#endif // end #ifndef __itkActiveRegistrationModelImageIntensityMetric_h
