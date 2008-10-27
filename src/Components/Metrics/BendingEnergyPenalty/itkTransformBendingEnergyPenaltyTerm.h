/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkTransformBendingEnergyPenaltyTerm_h
#define __itkTransformBendingEnergyPenaltyTerm_h

#include "itkTransformPenaltyTerm.h"

namespace itk
{

/** 
 * \class TransformBendingEnergyPenaltyTerm
 * \brief A cost function that calculates the bending energy
 * of a transformation.
 *
 * \par The bending energy is defined as the sum of the spatial
 * second order derivatives of the transformation, as defined in
 * [1]. For rigid and affine transformation this energy is always
 * zero. 
 *
 *
 * [1]: D. Rueckert, L. I. Sonoda, C. Hayes, D. L. G. Hill,
 *      M. O. Leach, and D. J. Hawkes, "Nonrigid registration
 *      using free-form deformations: Application to breast MR
 *      images", IEEE Trans. Med. Imaging 18, 712-721, 1999.
 *
 * \ingroup Metrics
 */

template< class TFixedImage, class TScalarType >
class TransformBendingEnergyPenaltyTerm
  : public TransformPenaltyTerm< TFixedImage, TScalarType >
{
public:

  /** Standard ITK stuff. */
  typedef TransformBendingEnergyPenaltyTerm			Self;
  typedef TransformPenaltyTerm<
    TFixedImage, TScalarType >                  Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TransformBendingEnergyPenaltyTerm, TransformPenaltyTerm );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass::MeasureType          MeasureType;
  typedef typename Superclass::DerivativeType       DerivativeType;
  typedef typename Superclass::ParametersType       ParametersType;
  typedef typename Superclass::TransformType        TransformType;
  typedef typename Superclass::FixedImageType       FixedImageType;
  typedef typename Superclass::ScalarType           ScalarType;

  /** Typedefs from the AdvancedTransform. */
  typedef typename Superclass::TransformType        TransformType;
  typedef typename Superclass
    ::NonZeroJacobianIndicesType                    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType  SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType                 JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType   SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType                  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType   InternalMatrixType;

  /** Define the dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int, FixedImageType::ImageDimension );

  /** Initialize the penalty term. *
  virtual void Initialize( void ) throw ( ExceptionObject );

  /** Get the penalty term value. */
  virtual MeasureType GetValue( const ParametersType & parameters ) const;

  /** Get the penalty term derivative. */
  virtual void GetDerivative( const ParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get the penalty term value and derivative. */
  virtual void GetValueAndDerivative(
    const ParametersType & parameters,
    MeasureType & value,
    DerivativeType & derivative ) const;

protected:

  /** The constructor. */
  TransformBendingEnergyPenaltyTerm();

  /** The destructor. */
  virtual ~TransformBendingEnergyPenaltyTerm() {};

  /** PrintSelf. *
  void PrintSelf( std::ostream& os, Indent indent ) const;*/

private:

  /** The private constructor. */
  TransformBendingEnergyPenaltyTerm( const Self& );	// purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self& );			  // purposely not implemented

}; // end class TransformBendingEnergyPenaltyTerm


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTransformBendingEnergyPenaltyTerm.txx"
#endif

#endif // #ifndef __itkTransformBendingEnergyPenaltyTerm_h

