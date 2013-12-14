/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedTranslationTransform.txx,v $
  Language:  C++
  Date:      $Date: 2007-11-14 20:17:26 $
  Version:   $Revision: 1.31 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkAdvancedTranslationTransform_hxx
#define _itkAdvancedTranslationTransform_hxx

#include "itkAdvancedTranslationTransform.h"

namespace itk
{

// Constructor with default arguments
template< class TScalarType, unsigned int NDimensions >
AdvancedTranslationTransform< TScalarType, NDimensions >::AdvancedTranslationTransform() : Superclass( ParametersDimension )
{
  m_Offset.Fill( 0 );

  // The Jacobian of this transform is constant.
  // Therefore the m_Jacobian variable can be
  // initialized here and be shared among all the threads.
  this->m_LocalJacobian.SetSize( SpaceDimension, ParametersDimension );
  this->m_LocalJacobian.Fill( 0.0 );

  for( unsigned int i = 0; i < NDimensions; i++ )
  {
    this->m_LocalJacobian( i, i ) = 1.0;
  }

  /** SpatialJacobian is also constant */
  this->m_SpatialJacobian.SetIdentity();

  /** Nonzero Jacobian indices, for GetJacobian */
  this->m_NonZeroJacobianIndices.resize( ParametersDimension );
  for( unsigned int i = 0; i < ParametersDimension; ++i )
  {
    this->m_NonZeroJacobianIndices[ i ] = i;
  }

  /** Set to correct size. The elements are automatically initialized to 0 */
  this->m_JacobianOfSpatialJacobian.resize( ParametersDimension );
  this->m_JacobianOfSpatialHessian.resize( ParametersDimension );

  /** m_SpatialHessian is automatically initialized with zeros */
  this->m_HasNonZeroSpatialHessian           = false;
  this->m_HasNonZeroJacobianOfSpatialHessian = false;
}


// Destructor
template< class TScalarType, unsigned int NDimensions >
AdvancedTranslationTransform< TScalarType, NDimensions >::
~AdvancedTranslationTransform()
{
  return;
}


// Set the parameters
template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >
::SetParameters( const ParametersType & parameters )
{
  bool modified = false;
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    if( m_Offset[ i ] != parameters[ i ] )
    {
      m_Offset[ i ] = parameters[ i ];
      modified      = true;
    }
  }
  if( modified )
  {
    this->Modified();
  }
}


// Get the parameters
template< class TScalarType, unsigned int NDimensions >
const typename AdvancedTranslationTransform< TScalarType, NDimensions >::ParametersType
& AdvancedTranslationTransform< TScalarType, NDimensions >
::GetParameters( void ) const
{
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    this->m_Parameters[ i ] = this->m_Offset[ i ];
  }
  return this->m_Parameters;
}

// Print self
template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Offset: " << m_Offset << std::endl;
}


// Compose with another affine transformation
template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >::Compose( const Self * other, bool )
{
  this->Translate( other->m_Offset );
  return;
}


// Compose with a translation
template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >::Translate( const OutputVectorType & offset, bool )
{
  ParametersType newOffset( SpaceDimension );
  for( unsigned int i = 0; i < SpaceDimension; i++ )
  {
    newOffset[ i ] = m_Offset[ i ] + offset[ i ];
  }
  this->SetParameters( newOffset );
  return;
}


// Transform a point
template< class TScalarType, unsigned int NDimensions >
typename AdvancedTranslationTransform< TScalarType, NDimensions >::OutputPointType
AdvancedTranslationTransform< TScalarType, NDimensions >::TransformPoint( const InputPointType & point ) const
{
  return point + m_Offset;
}


// Transform a vector
template< class TScalarType, unsigned int NDimensions >
typename AdvancedTranslationTransform< TScalarType, NDimensions >::OutputVectorType
AdvancedTranslationTransform< TScalarType, NDimensions >::TransformVector( const InputVectorType & vect ) const
{
  return vect;
}


// Transform a vnl_vector_fixed
template< class TScalarType, unsigned int NDimensions >
typename AdvancedTranslationTransform< TScalarType, NDimensions >::OutputVnlVectorType
AdvancedTranslationTransform< TScalarType, NDimensions >::TransformVector( const InputVnlVectorType & vect ) const
{
  return vect;
}


// Transform a CovariantVector
template< class TScalarType, unsigned int NDimensions >
typename AdvancedTranslationTransform< TScalarType, NDimensions >::OutputCovariantVectorType
AdvancedTranslationTransform< TScalarType, NDimensions >::TransformCovariantVector( const InputCovariantVectorType & vect ) const
{
  return vect;
}


// return an inverse transformation
template< class TScalarType, unsigned int NDimensions >
bool
AdvancedTranslationTransform< TScalarType, NDimensions >::GetInverse( Self * inverse ) const
{
  if( !inverse )
  {
    return false;
  }

  inverse->m_Offset = -m_Offset;
  return true;
}


/**
 * ********************* GetJacobian ****************************
 */

template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >
::GetJacobian(
  const InputPointType & itkNotUsed( p ),
  JacobianType & j,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  j                      = this->m_LocalJacobian;
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
} // end GetJacobian()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >
::GetSpatialJacobian(
  const InputPointType & itkNotUsed( p ),
  SpatialJacobianType & sj ) const
{
  /** Return pre-stored spatial Jacobian */
  sj = this->m_SpatialJacobian;
} // end GetSpatialJacobian()


/**
 * ********************* GetSpatialHessian ****************************
 */

template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >
::GetSpatialHessian(
  const InputPointType & itkNotUsed( p ),
  SpatialHessianType & sh ) const
{
  /** The SpatialHessian contains only zeros. */
  sh = this->m_SpatialHessian;
} // end GetSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >
::GetJacobianOfSpatialJacobian(
  const InputPointType & itkNotUsed( p ),
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  jsj                    = this->m_JacobianOfSpatialJacobian;
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >
::GetJacobianOfSpatialJacobian(
  const InputPointType & itkNotUsed( p ),
  SpatialJacobianType & sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  sj                     = this->m_SpatialJacobian;
  jsj                    = this->m_JacobianOfSpatialJacobian;
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >
::GetJacobianOfSpatialHessian(
  const InputPointType & itkNotUsed( p ),
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  /** The JacobianOfSpatialHessian contains only zeros.*/
  jsh                    = this->m_JacobianOfSpatialHessian;
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;
} // end GetJacobianOfSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >
::GetJacobianOfSpatialHessian(
  const InputPointType & itkNotUsed( p ),
  SpatialHessianType & sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const
{
  /** The Hessian and the JacobianOfSpatialHessian contain only zeros. */
  sh                     = this->m_SpatialHessian;
  jsh                    = this->m_JacobianOfSpatialHessian;
  nonZeroJacobianIndices = this->m_NonZeroJacobianIndices;

} // end GetJacobianOfSpatialHessian()


// Set the parameters for an Identity transform of this class
template< class TScalarType, unsigned int NDimensions >
void
AdvancedTranslationTransform< TScalarType, NDimensions >
::SetIdentity( void )
{
  m_Offset.Fill( 0.0 );
}


} // namespace

#endif
