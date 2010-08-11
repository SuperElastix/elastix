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
#ifndef _itkAdvancedTransform_txx
#define _itkAdvancedTransform_txx

#include "itkAdvancedTransform.h"

namespace itk
{


/**
 * ********************* Constructor ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::AdvancedTransform()
{
  this->m_HasNonZeroSpatialHessian = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;

} // end Constructor


/**
 * ********************* Constructor ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::AdvancedTransform( unsigned int dimension, unsigned int numberOfParameters ) :
  Superclass( dimension, numberOfParameters )
{
  this->m_HasNonZeroSpatialHessian = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;
} // end Constructor


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
unsigned long
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetNumberOfNonZeroJacobianIndices( void ) const
{
  return this->GetNumberOfParameters();

} // end GetNumberOfNonZeroJacobianIndices()


/**
 * ********************* GetJacobian ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
const typename AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::JacobianType &
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetJacobian( const InputPointType & ) const
{
  itkExceptionMacro( << "Subclass should override this method" );
  // Next line is needed to avoid errors due to:
  // "function must return a value".
  return this->m_Jacobian;

} // end GetJacobian()


/**
 * ********************* GetJacobian ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
void
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetJacobian(
  const InputPointType &,
  JacobianType &,
  NonZeroJacobianIndicesType & ) const
{
  itkExceptionMacro( << "Subclass should override this method" );

} // end GetJacobian()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
void
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetSpatialJacobian(
  const InputPointType &,
  SpatialJacobianType & ) const
{
  itkExceptionMacro( << "Subclass should override this method" );

} // end GetSpatialJacobian()


/**
 * ********************* GetSpatialHessian ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
void
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetSpatialHessian(
  const InputPointType &,
  SpatialHessianType & ) const
{
  itkExceptionMacro( << "Subclass should override this method" );

} // end GetSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
void
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetJacobianOfSpatialJacobian(
  const InputPointType &,
  JacobianOfSpatialJacobianType &,
  NonZeroJacobianIndicesType & ) const
{
  itkExceptionMacro( << "Subclass should override this method" );

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
void
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetJacobianOfSpatialJacobian(
  const InputPointType &,
  SpatialJacobianType &,
  JacobianOfSpatialJacobianType &,
  NonZeroJacobianIndicesType & ) const
{
  itkExceptionMacro( << "Subclass should override this method" );

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
void
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetJacobianOfSpatialHessian(
  const InputPointType &,
  JacobianOfSpatialHessianType &,
  NonZeroJacobianIndicesType & ) const
{
  itkExceptionMacro( << "Subclass should override this method" );

} // end GetJacobianOfSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
void
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetJacobianOfSpatialHessian(
  const InputPointType &,
  SpatialHessianType &,
  JacobianOfSpatialHessianType &,
  NonZeroJacobianIndicesType & ) const
{
  itkExceptionMacro( << "Subclass should override this method" );

} // end GetJacobianOfSpatialHessian()


} // end namespace itk


#endif
