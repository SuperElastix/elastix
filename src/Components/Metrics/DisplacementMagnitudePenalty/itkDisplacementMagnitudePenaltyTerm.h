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
#ifndef __itkDisplacementMagnitudePenaltyTerm_h
#define __itkDisplacementMagnitudePenaltyTerm_h

#include "itkTransformPenaltyTerm.h"

namespace itk
{

/**
 * \class DisplacementMagnitudePenaltyTerm
 * \brief A cost function that calculates \f$||T(x)-x||^2\f$.
 *
 * \ingroup Metrics
 */

template< class TFixedImage, class TScalarType >
class DisplacementMagnitudePenaltyTerm :
  public TransformPenaltyTerm< TFixedImage, TScalarType >
{
public:

  /** Standard ITK stuff. */
  typedef DisplacementMagnitudePenaltyTerm Self;
  typedef TransformPenaltyTerm<
    TFixedImage, TScalarType >                  Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( DisplacementMagnitudePenaltyTerm, TransformPenaltyTerm );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename Superclass::MovingImagePixelType         MovingImagePixelType;
  typedef typename Superclass::MovingImagePointer           MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer      MovingImageConstPointer;
  typedef typename Superclass::FixedImageType               FixedImageType;
  typedef typename Superclass::FixedImagePointer            FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType         FixedImageRegionType;
  typedef typename Superclass::TransformType                TransformType;
  typedef typename Superclass::TransformPointer             TransformPointer;
  typedef typename Superclass::InputPointType               InputPointType;
  typedef typename Superclass::OutputPointType              OutputPointType;
  typedef typename Superclass::TransformParametersType      TransformParametersType;
  typedef typename Superclass::TransformJacobianType        TransformJacobianType;
  typedef typename Superclass::InterpolatorType             InterpolatorType;
  typedef typename Superclass::InterpolatorPointer          InterpolatorPointer;
  typedef typename Superclass::RealType                     RealType;
  typedef typename Superclass::GradientPixelType            GradientPixelType;
  typedef typename Superclass::GradientImageType            GradientImageType;
  typedef typename Superclass::GradientImagePointer         GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType      GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer   GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType           FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer        FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType          MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer       MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                  MeasureType;
  typedef typename Superclass::DerivativeType               DerivativeType;
  typedef typename Superclass::DerivativeValueType          DerivativeValueType;
  typedef typename Superclass::ParametersType               ParametersType;
  typedef typename Superclass::FixedImagePixelType          FixedImagePixelType;
  typedef typename Superclass::ImageSampleContainerType     ImageSampleContainerType;
  typedef typename Superclass::ImageSampleContainerPointer  ImageSampleContainerPointer;
  typedef typename Superclass::ScalarType                   ScalarType;

  /** Typedefs from the AdvancedTransform. */
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

  /** Define the dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );

  /** Get the penalty term value.
   * \f[ Value = 1/N sum_x ||T(x) - x||^2 \f]
   */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /** Get the penalty term derivative.
   * Simply calls GetValueAndDerivative and returns the derivative. */
  virtual void GetDerivative( const ParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get the penalty term value and derivative.
   * \f[ Value = C(\mu) = 1/N sum_x ||T_{\mu}(x) - x||^2 \f]
   * \f[ Derivative = \frac{\partial C}{\partial\mu} = 2/N sum_x (T_{\mu}(x)-x)' \frac{\partial T}{\partial \mu} \f]
   */
  virtual void GetValueAndDerivative(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

protected:

  /** Typedefs for indices and points. */
  typedef typename Superclass::FixedImageIndexType            FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType       FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType           MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType            FixedImagePointType;
  typedef typename Superclass::MovingImagePointType           MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType MovingImageContinuousIndexType;
  typedef typename Superclass::NonZeroJacobianIndicesType     NonZeroJacobianIndicesType;

  /** The constructor. */
  DisplacementMagnitudePenaltyTerm();

  /** The destructor. */
  virtual ~DisplacementMagnitudePenaltyTerm() {}

  /** PrintSelf. *
  void PrintSelf( std::ostream& os, Indent indent ) const;*/

private:

  /** The private constructor. */
  DisplacementMagnitudePenaltyTerm( const Self & ); // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );                    // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkDisplacementMagnitudePenaltyTerm.hxx"
#endif

#endif // #ifndef __itkDisplacementMagnitudePenaltyTerm_h
