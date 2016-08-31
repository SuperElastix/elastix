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
#ifndef __itkTransformBendingEnergyPenaltyTerm_h
#define __itkTransformBendingEnergyPenaltyTerm_h

#include "itkTransformPenaltyTerm.h"
#include "itkImageGridSampler.h"

namespace itk
{

/**
 * \class TransformBendingEnergyPenaltyTerm
 * \brief A cost function that calculates the bending energy
 * of a transformation.
 *
 * The bending energy is defined as the sum of the spatial
 * second order derivatives of the transformation, as defined in
 * [1]. For rigid and affine transformation this energy is always
 * zero.
 *
 *
 * [1]: D. Rueckert, L. I. Sonoda, C. Hayes, D. L. G. Hill,
 *      M. O. Leach, and D. J. Hawkes, "Nonrigid registration
 *      using free-form deformations: Application to breast MR
 *      images", IEEE Trans. Med. Imaging 18, 712-721, 1999.\n
 * [2]: M. Staring and S. Klein,
 *      "Itk::Transforms supporting spatial derivatives"",
 *      Insight Journal, http://hdl.handle.net/10380/3215.
 *
 * \ingroup Metrics
 */

template< class TFixedImage, class TScalarType >
class TransformBendingEnergyPenaltyTerm :
  public TransformPenaltyTerm< TFixedImage, TScalarType >
{
public:

  /** Standard ITK stuff. */
  typedef TransformBendingEnergyPenaltyTerm Self;
  typedef TransformPenaltyTerm<
    TFixedImage, TScalarType >                  Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TransformBendingEnergyPenaltyTerm, TransformPenaltyTerm );

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
  typedef typename Superclass::NumberOfParametersType       NumberOfParametersType;
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
  typedef typename Superclass::ThreaderType                 ThreaderType;
  typedef typename Superclass::ThreadInfoType               ThreadInfoType;

  /** Typedef's for the B-spline transform. */
  typedef typename Superclass::CombinationTransformType       CombinationTransformType;
  typedef typename Superclass::BSplineOrder1TransformType     BSplineOrder1TransformType;
  typedef typename Superclass::BSplineOrder1TransformPointer  BSplineOrder1TransformPointer;
  typedef typename Superclass::BSplineOrder2TransformType     BSplineOrder2TransformType;
  typedef typename Superclass::BSplineOrder2TransformPointer  BSplineOrder2TransformPointer;
  typedef typename Superclass::BSplineOrder3TransformType     BSplineOrder3TransformType;
  typedef typename Superclass::BSplineOrder3TransformPointer  BSplineOrder3TransformPointer;

  /** Typedefs from the AdvancedTransform. */
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;
  typedef typename Superclass::HessianValueType   HessianValueType;
  typedef typename Superclass::HessianType        HessianType;

  /** Define the dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );

  /** Get the penalty term value. */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /** Get the penalty term derivative. */
  virtual void GetDerivative( const ParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get the penalty term value and derivative. */
  virtual void GetValueAndDerivativeSingleThreaded(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

  virtual void GetValueAndDerivative(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

  /** Get value and derivatives for each thread. */
  inline void ThreadedGetValueAndDerivative( ThreadIdType threadID );

  /** Gather the values and derivatives from all threads */
  inline void AfterThreadedGetValueAndDerivative(
    MeasureType & value, DerivativeType & derivative ) const;

  /** Experimental feature: compute SelfHessian */
  virtual void GetSelfHessian( const TransformParametersType & parameters, HessianType & H ) const;

  /** Default: 100000 */
  itkSetMacro( NumberOfSamplesForSelfHessian, unsigned int );
  itkGetConstMacro( NumberOfSamplesForSelfHessian, unsigned int );

protected:

  /** Typedefs for indices and points. */
  typedef typename Superclass::FixedImageIndexType            FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType       FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType           MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType            FixedImagePointType;
  typedef typename Superclass::MovingImagePointType           MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType MovingImageContinuousIndexType;
  typedef typename Superclass::NonZeroJacobianIndicesType     NonZeroJacobianIndicesType;

  /** Typedefs for SelfHessian */
  typedef ImageGridSampler< FixedImageType > SelfHessianSamplerType;

  /** The constructor. */
  TransformBendingEnergyPenaltyTerm();

  /** The destructor. */
  virtual ~TransformBendingEnergyPenaltyTerm() {}

private:

  /** The private constructor. */
  TransformBendingEnergyPenaltyTerm( const Self & ); // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );                    // purposely not implemented

  unsigned int m_NumberOfSamplesForSelfHessian;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTransformBendingEnergyPenaltyTerm.hxx"
#endif

#endif // #ifndef __itkTransformBendingEnergyPenaltyTerm_h
