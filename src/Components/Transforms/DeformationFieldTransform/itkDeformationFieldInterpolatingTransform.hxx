/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef _itkDeformationFieldInterpolatingTransform_hxx
#define _itkDeformationFieldInterpolatingTransform_hxx

#include "itkDeformationFieldInterpolatingTransform.h"

namespace itk
{

// Constructor with default arguments
template< class TScalarType, unsigned int NDimensions, class TComponentType >
DeformationFieldInterpolatingTransform< TScalarType, NDimensions,  TComponentType >::DeformationFieldInterpolatingTransform() :
  Superclass( OutputSpaceDimension )
{
  this->m_DeformationField     = 0;
  this->m_ZeroDeformationField = DeformationFieldType::New();
  typename DeformationFieldType::SizeType dummySize;
  dummySize.Fill( 0 );
  this->m_ZeroDeformationField->SetRegions( dummySize );
  this->SetIdentity();

} // end Constructor


// Destructor
template< class TScalarType, unsigned int NDimensions, class TComponentType >
DeformationFieldInterpolatingTransform< TScalarType, NDimensions,  TComponentType >::
~DeformationFieldInterpolatingTransform()
{} // end Destructor

// Transform a point
template< class TScalarType, unsigned int NDimensions, class TComponentType >
typename DeformationFieldInterpolatingTransform< TScalarType, NDimensions,  TComponentType >::
OutputPointType
DeformationFieldInterpolatingTransform< TScalarType, NDimensions,  TComponentType >
::TransformPoint( const InputPointType & point ) const
{
  InputContinuousIndexType cindex;
  this->m_DeformationFieldInterpolator->ConvertPointToContinuousIndex(
    point, cindex );

  if( this->m_DeformationFieldInterpolator->IsInsideBuffer( cindex ) )
  {
    InterpolatorOutputType vec
      = this->m_DeformationFieldInterpolator->EvaluateAtContinuousIndex( cindex );
    OutputPointType outpoint;
    for( unsigned int i = 0; i < InputSpaceDimension; ++i )
    {
      outpoint[ i ] = point[ i ] + static_cast< ScalarType >( vec[ i ] );
    }
    return outpoint;

  }
  else
  {
    return point;
  }
}


// Set the deformation field
template< class TScalarType, unsigned int NDimensions, class TComponentType >
void
DeformationFieldInterpolatingTransform< TScalarType, NDimensions,  TComponentType >
::SetDeformationField( DeformationFieldType * _arg )
{
  itkDebugMacro( "setting DeformationField to " << _arg );
  if( this->m_DeformationField != _arg )
  {
    this->m_DeformationField = _arg;
    this->Modified();
  }
  if( this->m_DeformationFieldInterpolator.IsNotNull() )
  {
    this->m_DeformationFieldInterpolator->SetInputImage(
      this->m_DeformationField );
  }
}


// Set the deformation field interpolator
template< class TScalarType, unsigned int NDimensions, class TComponentType >
void
DeformationFieldInterpolatingTransform< TScalarType, NDimensions,  TComponentType >
::SetDeformationFieldInterpolator( DeformationFieldInterpolatorType * _arg )
{
  itkDebugMacro( "setting DeformationFieldInterpolator to " << _arg );
  if( this->m_DeformationFieldInterpolator != _arg )
  {
    this->m_DeformationFieldInterpolator = _arg;
    this->Modified();
  }
  if( this->m_DeformationFieldInterpolator.IsNotNull() )
  {
    this->m_DeformationFieldInterpolator->SetInputImage(
      this->m_DeformationField );
  }
}


// Print self
template< class TScalarType, unsigned int NDimensions, class TComponentType >
void
DeformationFieldInterpolatingTransform< TScalarType, NDimensions,  TComponentType >::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "DeformationField: " << this->m_DeformationField << std::endl;
  os << indent << "ZeroDeformationField: " << this->m_ZeroDeformationField << std::endl;
  os << indent << "DeformationFieldInterpolator: " << this->m_DeformationFieldInterpolator << std::endl;
}


// Set the parameters for an Identity transform of this class
template< class TScalarType, unsigned int NDimensions, class TComponentType >
void
DeformationFieldInterpolatingTransform< TScalarType, NDimensions,  TComponentType >::SetIdentity( void )
{
  if( this->m_DeformationFieldInterpolator.IsNull() )
  {
    this->m_DeformationFieldInterpolator
      = DefaultDeformationFieldInterpolatorType::New();
  }
  this->SetDeformationField( this->m_ZeroDeformationField );

} // end SetIdentity()


} // end namespace itk

#endif
