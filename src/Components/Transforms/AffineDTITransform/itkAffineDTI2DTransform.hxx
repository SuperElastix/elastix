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
  Module:    $RCSfile: itkAffineDTI3DTransform.txx,v $
  Language:  C++
  Date:      $Date: 2008-10-13 15:36:31 $
  Version:   $Revision: 1.24 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAffineDTI2DTransform_hxx
#define __itkAffineDTI2DTransform_hxx

#include "itkAffineDTI2DTransform.h"

namespace itk
{

// Constructor with default arguments
template< class TScalarType >
AffineDTI2DTransform< TScalarType >
::AffineDTI2DTransform() :
  Superclass( ParametersDimension )
{
  this->m_Angle.Fill( itk::NumericTraits< ScalarType >::Zero );
  this->m_Shear.Fill( itk::NumericTraits< ScalarType >::Zero );
  this->m_Scale.Fill( itk::NumericTraits< ScalarType >::One );

  this->PrecomputeJacobianOfSpatialJacobian();
}


// Constructor with default arguments
template< class TScalarType >
AffineDTI2DTransform< TScalarType >
::AffineDTI2DTransform( const MatrixType & matrix,
  const OutputPointType & offset )
{
  this->SetMatrix( matrix );

  OffsetType off;
  off[ 0 ] = offset[ 0 ];
  off[ 1 ] = offset[ 1 ];
  off[ 2 ] = offset[ 2 ];
  this->SetOffset( off );

  // this->ComputeMatrix?
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Constructor with arguments
template< class TScalarType >
AffineDTI2DTransform< TScalarType >
::AffineDTI2DTransform( unsigned int spaceDimension,
  unsigned int parametersDimension ) :
  Superclass( spaceDimension, parametersDimension )
{
  this->m_Angle.Fill( itk::NumericTraits< ScalarType >::Zero );
  this->m_Shear.Fill( itk::NumericTraits< ScalarType >::Zero );
  this->m_Scale.Fill( itk::NumericTraits< ScalarType >::One );
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Set Angles etc
template< class TScalarType >
void
AffineDTI2DTransform< TScalarType >
::SetVarAngleScaleShear(
  ScalarArrayType angle,
  ScalarArrayType shear,
  ScalarArrayType scale )
{
  this->m_Angle = angle;
  this->m_Shear = shear;
  this->m_Scale = scale;
}


// Set Parameters
template< class TScalarType >
void
AffineDTI2DTransform< TScalarType >
::SetParameters( const ParametersType & parameters )
{
  itkDebugMacro( << "Setting parameters " << parameters );

  this->m_Angle[ 0 ] = parameters[ 0 ];
  this->m_Shear[ 0 ] = parameters[ 1 ];
  this->m_Shear[ 1 ] = parameters[ 2 ];
  this->m_Scale[ 0 ] = parameters[ 3 ];
  this->m_Scale[ 1 ] = parameters[ 4 ];
  this->ComputeMatrix();

  // Transfer the translation part
  OutputVectorType newTranslation;
  newTranslation[ 0 ] = parameters[ 5 ];
  newTranslation[ 1 ] = parameters[ 6 ];
  this->SetVarTranslation( newTranslation );
  this->ComputeOffset();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();

  itkDebugMacro( << "After setting parameters " );
}


// Get Parameters
template< class TScalarType >
const typename AffineDTI2DTransform< TScalarType >::ParametersType
& AffineDTI2DTransform< TScalarType >
::GetParameters( void ) const
{
  this->m_Parameters[ 0 ]  = this->m_Angle[ 0 ];
  this->m_Parameters[ 1 ]  = this->m_Shear[ 0 ];
  this->m_Parameters[ 2 ]  = this->m_Shear[ 1 ];
  this->m_Parameters[ 3 ]  = this->m_Scale[ 0 ];
  this->m_Parameters[ 4 ]  = this->m_Scale[ 1 ];
  this->m_Parameters[ 5 ]  = this->GetTranslation()[ 0 ];
  this->m_Parameters[ 6 ] = this->GetTranslation()[ 1 ];

  return this->m_Parameters;
}

// SetIdentity()
template< class TScalarType >
void
AffineDTI2DTransform< TScalarType >
::SetIdentity( void )
{
  Superclass::SetIdentity();
  this->m_Angle.Fill( itk::NumericTraits< ScalarType >::Zero );
  this->m_Shear.Fill( itk::NumericTraits< ScalarType >::Zero );
  this->m_Scale.Fill( itk::NumericTraits< ScalarType >::One );
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Compute angles from the rotation matrix
template< class TScalarType >
void
AffineDTI2DTransform< TScalarType >
::ComputeMatrixParameters( void )
{
  // let's hope we don't need it :)
  itkExceptionMacro( << "This function has not been implemented yet!" );
  this->ComputeMatrix();
}


// Compute the matrix
template< class TScalarType >
void
AffineDTI2DTransform< TScalarType >
::ComputeMatrix( void )
{
  // need to check if angles are in the right order
  const double cx  = vcl_cos( this->m_Angle[ 0 ] );
  const double sx  = vcl_sin( this->m_Angle[ 0 ] );
  const double gx  = this->m_Shear[ 0 ];
  const double gy  = this->m_Shear[ 1 ];
  const double ssx = this->m_Scale[ 0 ];
  const double ssy = this->m_Scale[ 1 ];

  /** NB: opposite definition as in EulerTransform */

  MatrixType Rotation;
  Rotation[ 0 ][ 0 ] =  cx; Rotation[ 0 ][ 1 ] = sx;
  Rotation[ 1 ][ 0 ] = -sx; Rotation[ 1 ][ 1 ] = cx;

  MatrixType ShearX;
  ShearX[ 0 ][ 0 ] = 1; ShearX[ 0 ][ 1 ] = gx; 
  ShearX[ 1 ][ 0 ] = 0; ShearX[ 1 ][ 1 ] = 1; 

  MatrixType ShearY;
  ShearY[ 0 ][ 0 ] = 1;  ShearY[ 0 ][ 1 ] = 0; 
  ShearY[ 1 ][ 0 ] = gy; ShearY[ 1 ][ 1 ] = 1; 

  MatrixType Scale;
  Scale[ 0 ][ 0 ] = ssx; Scale[ 0 ][ 1 ] = 0; 
  Scale[ 1 ][ 0 ] = 0;   Scale[ 1 ][ 1 ] = ssy; 

  this->SetVarMatrix(
    Rotation * ShearX * ShearY  * Scale );

  this->PrecomputeJacobianOfSpatialJacobian();
}


// Set parameters
template< class TScalarType >
void
AffineDTI2DTransform< TScalarType >
::GetJacobian( const InputPointType & p,
  JacobianType & j,
  NonZeroJacobianIndicesType & nzji ) const
{
  j.SetSize( OutputSpaceDimension, ParametersDimension );
  j.Fill( 0.0 );
  const JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;

  /** Compute dR/dmu * (p-c) */
  const InputVectorType pp = p - this->GetCenter();
  for( unsigned int dim = 0; dim < 5; dim++ )
  {
    const InputVectorType column = jsj[ dim ] * pp;
    for( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      j( i, dim ) = column[ i ];
    }
  }

  // compute derivatives for the translation part
  const unsigned int blockOffset = 5;
  for( unsigned int dim = 0; dim < SpaceDimension; dim++ )
  {
    j[ dim ][ blockOffset + dim ] = 1.0;
  }

  nzji = this->m_NonZeroJacobianIndices;
}


// Precompute Jacobian of Spatial Jacobian
template< class TScalarType >
void
AffineDTI2DTransform< TScalarType >
::PrecomputeJacobianOfSpatialJacobian( void )
{
  if( ParametersDimension < 7 )
  {
    /** Some subclass has a different number of parameters */
    return;
  }

  /** The Jacobian of spatial Jacobian is constant over inputspace, so is precomputed */
  JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;
  jsj.resize( ParametersDimension );

  // need to check if angles are in the right order
  const double cx  = vcl_cos( this->m_Angle[ 0 ] );
  const double sx  = vcl_sin( this->m_Angle[ 0 ] );
  const double gx  = this->m_Shear[ 0 ];
  const double gy  = this->m_Shear[ 1 ];
  const double ssx = this->m_Scale[ 0 ];
  const double ssy = this->m_Scale[ 1 ];

  /** derivatives: */
  const double cxd = -vcl_sin( this->m_Angle[ 0 ] );
  const double sxd = vcl_cos( this->m_Angle[ 0 ] );

  /** NB: opposite definition as in EulerTransform */
  MatrixType Rotation;
  Rotation[ 0 ][ 0 ] =  cx; Rotation[ 0 ][ 1 ] = sx; 
  Rotation[ 1 ][ 0 ] = -sx; Rotation[ 1 ][ 1 ] = cx; 

  MatrixType ShearX;
  ShearX[ 0 ][ 0 ] = 1; ShearX[ 0 ][ 1 ] = gx; 
  ShearX[ 1 ][ 0 ] = 0; ShearX[ 1 ][ 1 ] = 1; 

  MatrixType ShearY;
  ShearY[ 0 ][ 0 ] = 1;  ShearY[ 0 ][ 1 ] = 0; 
  ShearY[ 1 ][ 0 ] = gy; ShearY[ 1 ][ 1 ] = 1; 

  MatrixType ScaleX;
  ScaleX[ 0 ][ 0 ] = ssx; ScaleX[ 0 ][ 1 ] = 0; 
  ScaleX[ 1 ][ 0 ] = 0;   ScaleX[ 1 ][ 1 ] = 1; 
  
  MatrixType ScaleY;
  ScaleY[ 0 ][ 0 ] = 1; ScaleY[ 0 ][ 1 ] = 0; 
  ScaleY[ 1 ][ 0 ] = 0; ScaleY[ 1 ][ 1 ] = ssy;

  /** Derivative matrices: */
  MatrixType RotationXd;
  RotationXd[ 0 ][ 0 ] =  cxd; RotationXd[ 0 ][ 1 ] = sxd; 
  RotationXd[ 1 ][ 0 ] = -sxd; RotationXd[ 1 ][ 1 ] = cxd; 

  MatrixType ShearXd;
  ShearXd[ 0 ][ 0 ] = 0; ShearXd[ 0 ][ 1 ] = 1; 
  ShearXd[ 1 ][ 0 ] = 0; ShearXd[ 1 ][ 1 ] = 0; 
  
  MatrixType ShearYd;
  ShearYd[ 0 ][ 0 ] = 0; ShearYd[ 0 ][ 1 ] = 0; 
  ShearYd[ 1 ][ 0 ] = 1; ShearYd[ 1 ][ 1 ] = 0; 

  MatrixType ScaleXd;
  ScaleXd[ 0 ][ 0 ] = 1; ScaleXd[ 0 ][ 1 ] = 0; 
  ScaleXd[ 1 ][ 0 ] = 0; ScaleXd[ 1 ][ 1 ] = 0; 
  
  MatrixType ScaleYd;
  ScaleYd[ 0 ][ 0 ] = 0; ScaleYd[ 0 ][ 1 ] = 0; 
  ScaleYd[ 1 ][ 0 ] = 0; ScaleYd[ 1 ][ 1 ] = 1;


  jsj[ 0 ] = RotationXd * ShearX * ShearY * ScaleX * ScaleY;
  jsj[ 1 ] = Rotation * ShearXd * ShearY * ScaleX * ScaleY;
  jsj[ 2 ] = Rotation * ShearX * ShearYd * ScaleX * ScaleY;
  jsj[ 3 ] = Rotation * ShearX * ShearY * ScaleXd * ScaleY;
  jsj[ 4 ] = Rotation * ShearX * ShearY * ScaleX * ScaleYd;

  /** Translation parameters: */
  for( unsigned int par = 5; par < 7; ++par )
  {
    jsj[ par ].Fill( 0.0 );
  }
}


// Print self
template< class TScalarType >
void
AffineDTI2DTransform< TScalarType >::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Angles: " << this->m_Angle  << std::endl;
  os << indent << "Shear: " << this->m_Shear  << std::endl;
  os << indent << "Scale: " << this->m_Scale  << std::endl;
}


} // namespace

#endif
