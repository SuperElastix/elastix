/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedTransform.txx,v $
  Language:  C++
  Date:      $Date: 2007-11-20 20:08:16 $
  Version:   $Revision: 1.27 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkAdvancedTransform_hxx
#define _itkAdvancedTransform_hxx

#include "itkAdvancedTransform.h"

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template< class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
AdvancedTransform< TScalarType, NInputDimensions, NOutputDimensions >
::AdvancedTransform() : Superclass()
{
  this->m_HasNonZeroSpatialHessian           = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;

} // end Constructor


/**
 * ********************* Constructor ****************************
 */

template< class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
AdvancedTransform< TScalarType, NInputDimensions, NOutputDimensions >
::AdvancedTransform( NumberOfParametersType numberOfParameters ) :
  Superclass( numberOfParameters )
{
  this->m_HasNonZeroSpatialHessian           = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;
} // end Constructor


/**
 * ********************* EvaluateJacobianWithImageGradientProduct ****************************
 */

template< class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
void
AdvancedTransform< TScalarType, NInputDimensions, NOutputDimensions >
::EvaluateJacobianWithImageGradientProduct(
  const InputPointType & ipp,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType & imageJacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  /** Obtain the Jacobian. */
  JacobianType jacobian;//( SpaceDimension, );
  this->GetJacobian( ipp, jacobian, nonZeroJacobianIndices );

  /** Perform a full multiplication. */
  typedef typename JacobianType::const_iterator JacobianIteratorType;
  typedef typename DerivativeType::iterator     DerivativeIteratorType;
  JacobianIteratorType jac = jacobian.begin();
  imageJacobian.Fill( 0.0 );
  const unsigned int sizeImageJacobian = imageJacobian.GetSize();

  for( unsigned int dim = 0; dim < InputSpaceDimension; ++dim )
  {
    const double           imDeriv = movingImageGradient[ dim ];
    DerivativeIteratorType imjac   = imageJacobian.begin();

    for( unsigned int mu = 0; mu < sizeImageJacobian; ++mu )
    {
      ( *imjac ) += ( *jac ) * imDeriv;
      ++imjac;
      ++jac;
    }
  }

} // end EvaluateJacobianWithImageGradientProduct()


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template< class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
typename AdvancedTransform< TScalarType, NInputDimensions, NOutputDimensions >::NumberOfParametersType
AdvancedTransform< TScalarType, NInputDimensions, NOutputDimensions >
::GetNumberOfNonZeroJacobianIndices( void ) const
{
  return this->GetNumberOfParameters();

} // end GetNumberOfNonZeroJacobianIndices()


} // end namespace itk

#endif
