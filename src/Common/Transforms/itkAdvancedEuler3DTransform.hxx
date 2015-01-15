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
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedEuler3DTransform.txx,v $
  Language:  C++
  Date:      $Date: 2008-10-13 15:36:31 $
  Version:   $Revision: 1.24 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedEuler3DTransform_hxx
#define __itkAdvancedEuler3DTransform_hxx

#include "itkAdvancedEuler3DTransform.h"

namespace itk
{

// Constructor with default arguments
template< class TScalarType >
AdvancedEuler3DTransform< TScalarType >
::AdvancedEuler3DTransform() :
  Superclass( ParametersDimension )
{
  m_ComputeZYX = false;
  m_AngleX     = m_AngleY = m_AngleZ = NumericTraits< ScalarType >::Zero;
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Constructor with default arguments
template< class TScalarType >
AdvancedEuler3DTransform< TScalarType >
::AdvancedEuler3DTransform( const MatrixType & matrix,
  const OutputPointType & offset )
{
  m_ComputeZYX = false;
  this->SetMatrix( matrix );

  OffsetType off;
  off[ 0 ] = offset[ 0 ];
  off[ 1 ] = offset[ 1 ];
  off[ 2 ] = offset[ 2 ];
  this->SetOffset( off );

  this->PrecomputeJacobianOfSpatialJacobian();
}


// Constructor with arguments
template< class TScalarType >
AdvancedEuler3DTransform< TScalarType >
::AdvancedEuler3DTransform( unsigned int parametersDimension ) :
  Superclass( parametersDimension )
{
  m_ComputeZYX = false;
  m_AngleX     = m_AngleY = m_AngleZ = NumericTraits< ScalarType >::Zero;
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Set Angles
template< class TScalarType >
void
AdvancedEuler3DTransform< TScalarType >
::SetVarRotation( ScalarType angleX, ScalarType angleY, ScalarType angleZ )
{
  this->m_AngleX = angleX;
  this->m_AngleY = angleY;
  this->m_AngleZ = angleZ;
}


// Set Parameters
template< class TScalarType >
void
AdvancedEuler3DTransform< TScalarType >
::SetParameters( const ParametersType & parameters )
{
  itkDebugMacro( << "Setting parameters " << parameters );

  // Set angles with parameters
  m_AngleX = parameters[ 0 ];
  m_AngleY = parameters[ 1 ];
  m_AngleZ = parameters[ 2 ];
  this->ComputeMatrix();

  // Transfer the translation part
  OutputVectorType newTranslation;
  newTranslation[ 0 ] = parameters[ 3 ];
  newTranslation[ 1 ] = parameters[ 4 ];
  newTranslation[ 2 ] = parameters[ 5 ];
  this->SetVarTranslation( newTranslation );
  this->ComputeOffset();

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();

  itkDebugMacro( << "After setting parameters " );
}


// Get Parameters
template< class TScalarType >
const typename AdvancedEuler3DTransform< TScalarType >::ParametersType
& AdvancedEuler3DTransform< TScalarType >
::GetParameters( void ) const
{
  this->m_Parameters[ 0 ] = m_AngleX;
  this->m_Parameters[ 1 ] = m_AngleY;
  this->m_Parameters[ 2 ] = m_AngleZ;
  this->m_Parameters[ 3 ] = this->GetTranslation()[ 0 ];
  this->m_Parameters[ 4 ] = this->GetTranslation()[ 1 ];
  this->m_Parameters[ 5 ] = this->GetTranslation()[ 2 ];

  return this->m_Parameters;
}

// Set Rotational Part
template< class TScalarType >
void
AdvancedEuler3DTransform< TScalarType >
::SetRotation( ScalarType angleX, ScalarType angleY, ScalarType angleZ )
{
  m_AngleX = angleX;
  m_AngleY = angleY;
  m_AngleZ = angleZ;
  this->ComputeMatrix();
  this->ComputeOffset();
}


// Compose
template< class TScalarType >
void
AdvancedEuler3DTransform< TScalarType >
::SetIdentity( void )
{
  Superclass::SetIdentity();
  m_AngleX = 0;
  m_AngleY = 0;
  m_AngleZ = 0;
  this->PrecomputeJacobianOfSpatialJacobian();
}


// Compute angles from the rotation matrix
template< class TScalarType >
void
AdvancedEuler3DTransform< TScalarType >
::ComputeMatrixParameters( void )
{
  if( m_ComputeZYX )
  {
    m_AngleY = -asin( this->GetMatrix()[ 2 ][ 0 ] );
    double C = vcl_cos( m_AngleY );
    if( vcl_fabs( C ) > 0.00005 )
    {
      double x = this->GetMatrix()[ 2 ][ 2 ] / C;
      double y = this->GetMatrix()[ 2 ][ 1 ] / C;
      m_AngleX = vcl_atan2( y, x );
      x        = this->GetMatrix()[ 0 ][ 0 ] / C;
      y        = this->GetMatrix()[ 1 ][ 0 ] / C;
      m_AngleZ = vcl_atan2( y, x );
    }
    else
    {
      m_AngleX = 0;
      double x = this->GetMatrix()[ 1 ][ 1 ];
      double y = -this->GetMatrix()[ 0 ][ 1 ];
      m_AngleZ = vcl_atan2( y, x );
    }
  }
  else
  {
    m_AngleX = vcl_asin( this->GetMatrix()[ 2 ][ 1 ] );
    double A = vcl_cos( m_AngleX );
    if( vcl_fabs( A ) > 0.00005 )
    {
      double x = this->GetMatrix()[ 2 ][ 2 ] / A;
      double y = -this->GetMatrix()[ 2 ][ 0 ] / A;
      m_AngleY = vcl_atan2( y, x );

      x        = this->GetMatrix()[ 1 ][ 1 ] / A;
      y        = -this->GetMatrix()[ 0 ][ 1 ] / A;
      m_AngleZ = vcl_atan2( y, x );
    }
    else
    {
      m_AngleZ = 0;
      double x = this->GetMatrix()[ 0 ][ 0 ];
      double y = this->GetMatrix()[ 1 ][ 0 ];
      m_AngleY = vcl_atan2( y, x );
    }
  }
  this->ComputeMatrix();
}


// Compute the matrix
template< class TScalarType >
void
AdvancedEuler3DTransform< TScalarType >
::ComputeMatrix( void )
{
  // need to check if angles are in the right order
  const double cx = vcl_cos( m_AngleX );
  const double sx = vcl_sin( m_AngleX );
  const double cy = vcl_cos( m_AngleY );
  const double sy = vcl_sin( m_AngleY );
  const double cz = vcl_cos( m_AngleZ );
  const double sz = vcl_sin( m_AngleZ );

  Matrix< TScalarType, 3, 3 > RotationX;
  RotationX[ 0 ][ 0 ] = 1; RotationX[ 0 ][ 1 ] = 0; RotationX[ 0 ][ 2 ] = 0;
  RotationX[ 1 ][ 0 ] = 0; RotationX[ 1 ][ 1 ] = cx; RotationX[ 1 ][ 2 ] = -sx;
  RotationX[ 2 ][ 0 ] = 0; RotationX[ 2 ][ 1 ] = sx; RotationX[ 2 ][ 2 ] = cx;

  Matrix< TScalarType, 3, 3 > RotationY;
  RotationY[ 0 ][ 0 ] = cy; RotationY[ 0 ][ 1 ] = 0; RotationY[ 0 ][ 2 ] = sy;
  RotationY[ 1 ][ 0 ] = 0; RotationY[ 1 ][ 1 ] = 1; RotationY[ 1 ][ 2 ] = 0;
  RotationY[ 2 ][ 0 ] = -sy; RotationY[ 2 ][ 1 ] = 0; RotationY[ 2 ][ 2 ] = cy;

  Matrix< TScalarType, 3, 3 > RotationZ;
  RotationZ[ 0 ][ 0 ] = cz; RotationZ[ 0 ][ 1 ] = -sz; RotationZ[ 0 ][ 2 ] = 0;
  RotationZ[ 1 ][ 0 ] = sz; RotationZ[ 1 ][ 1 ] = cz; RotationZ[ 1 ][ 2 ] = 0;
  RotationZ[ 2 ][ 0 ] = 0; RotationZ[ 2 ][ 1 ] = 0; RotationZ[ 2 ][ 2 ] = 1;

  /** Aply the rotation first around Y then X then Z */
  if( m_ComputeZYX )
  {
    this->SetVarMatrix( RotationZ * RotationY * RotationX );
  }
  else
  {
    // Like VTK transformation order
    this->SetVarMatrix( RotationZ * RotationX * RotationY );
  }

  this->PrecomputeJacobianOfSpatialJacobian();
}


// Get Jacobian
template< class TScalarType >
void
AdvancedEuler3DTransform< TScalarType >::GetJacobian( const InputPointType & p,
  JacobianType & j,
  NonZeroJacobianIndicesType & nzji ) const
{
  // Initialize the Jacobian. Resizing is only performed when needed.
  // Filling with zeros is needed because the lower loops only visit
  // the nonzero positions.
  j.SetSize( OutputSpaceDimension, ParametersDimension );
  j.Fill( 0.0 );

  /** Compute dR/dmu * (p-c) */
  const InputVectorType                 pp  = p - this->GetCenter();
  const JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;
  for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
  {
    const InputVectorType column = jsj[ dim ] * pp;
    for( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      j( i, dim ) = column[ i ];
    }
  }

  // compute derivatives for the translation part
  const unsigned int blockOffset = 3;
  for( unsigned int dim = 0; dim < SpaceDimension; ++dim )
  {
    j[ dim ][ blockOffset + dim ] = 1.0;
  }

  // Copy the constant nonZeroJacobianIndices
  nzji = this->m_NonZeroJacobianIndices;
}


// Precompute Jacobian of Spatial Jacobian
template< class TScalarType >
void
AdvancedEuler3DTransform< TScalarType >
::PrecomputeJacobianOfSpatialJacobian( void )
{
  if( ParametersDimension < 6 )
  {
    /** Some subclass has a different number of parameters */
    return;
  }

  /** The Jacobian of spatial Jacobian is constant over inputspace, so is precomputed */
  JacobianOfSpatialJacobianType & jsj = this->m_JacobianOfSpatialJacobian;
  jsj.resize( ParametersDimension );
  const double cx = vcl_cos( m_AngleX );
  const double sx = vcl_sin( m_AngleX );
  const double cy = vcl_cos( m_AngleY );
  const double sy = vcl_sin( m_AngleY );
  const double cz = vcl_cos( m_AngleZ );
  const double sz = vcl_sin( m_AngleZ );

  /** derivatives: */
  const double cxd = -vcl_sin( m_AngleX );
  const double sxd = vcl_cos( m_AngleX );
  const double cyd = -vcl_sin( m_AngleY );
  const double syd = vcl_cos( m_AngleY );
  const double czd = -vcl_sin( m_AngleZ );
  const double szd = vcl_cos( m_AngleZ );

  /** rotation matrices */
  Matrix< TScalarType, 3, 3 > RotationX;
  RotationX[ 0 ][ 0 ] = 1; RotationX[ 0 ][ 1 ] = 0; RotationX[ 0 ][ 2 ] = 0;
  RotationX[ 1 ][ 0 ] = 0; RotationX[ 1 ][ 1 ] = cx; RotationX[ 1 ][ 2 ] = -sx;
  RotationX[ 2 ][ 0 ] = 0; RotationX[ 2 ][ 1 ] = sx; RotationX[ 2 ][ 2 ] = cx;

  Matrix< TScalarType, 3, 3 > RotationY;
  RotationY[ 0 ][ 0 ] = cy; RotationY[ 0 ][ 1 ] = 0; RotationY[ 0 ][ 2 ] = sy;
  RotationY[ 1 ][ 0 ] = 0; RotationY[ 1 ][ 1 ] = 1; RotationY[ 1 ][ 2 ] = 0;
  RotationY[ 2 ][ 0 ] = -sy; RotationY[ 2 ][ 1 ] = 0; RotationY[ 2 ][ 2 ] = cy;

  Matrix< TScalarType, 3, 3 > RotationZ;
  RotationZ[ 0 ][ 0 ] = cz; RotationZ[ 0 ][ 1 ] = -sz; RotationZ[ 0 ][ 2 ] = 0;
  RotationZ[ 1 ][ 0 ] = sz; RotationZ[ 1 ][ 1 ] = cz; RotationZ[ 1 ][ 2 ] = 0;
  RotationZ[ 2 ][ 0 ] = 0; RotationZ[ 2 ][ 1 ] = 0; RotationZ[ 2 ][ 2 ] = 1;

  /** derivative matrices */
  Matrix< TScalarType, 3, 3 > RotationXd;
  RotationXd[ 0 ][ 0 ] = 0; RotationXd[ 0 ][ 1 ] = 0; RotationXd[ 0 ][ 2 ] = 0;
  RotationXd[ 1 ][ 0 ] = 0; RotationXd[ 1 ][ 1 ] = cxd; RotationXd[ 1 ][ 2 ] = -sxd;
  RotationXd[ 2 ][ 0 ] = 0; RotationXd[ 2 ][ 1 ] = sxd; RotationXd[ 2 ][ 2 ] = cxd;

  Matrix< TScalarType, 3, 3 > RotationYd;
  RotationYd[ 0 ][ 0 ] = cyd; RotationYd[ 0 ][ 1 ] = 0; RotationYd[ 0 ][ 2 ] = syd;
  RotationYd[ 1 ][ 0 ] = 0; RotationYd[ 1 ][ 1 ] = 0; RotationYd[ 1 ][ 2 ] = 0;
  RotationYd[ 2 ][ 0 ] = -syd; RotationYd[ 2 ][ 1 ] = 0; RotationYd[ 2 ][ 2 ] = cyd;

  Matrix< TScalarType, 3, 3 > RotationZd;
  RotationZd[ 0 ][ 0 ] = czd; RotationZd[ 0 ][ 1 ] = -szd; RotationZd[ 0 ][ 2 ] = 0;
  RotationZd[ 1 ][ 0 ] = szd; RotationZd[ 1 ][ 1 ] = czd; RotationZd[ 1 ][ 2 ] = 0;
  RotationZd[ 2 ][ 0 ] = 0; RotationZd[ 2 ][ 1 ] = 0; RotationZd[ 2 ][ 2 ] = 0;

  /** Aply the rotation first around Y then X then Z */
  if( m_ComputeZYX )
  {
    jsj[ 0 ] = RotationZ * RotationY * RotationXd;
    jsj[ 1 ] = RotationZ * RotationYd * RotationX;
    jsj[ 2 ] = RotationZd * RotationY * RotationX;

  }
  else
  {
    // Like VTK transformation order
    jsj[ 0 ] = RotationZ * RotationXd * RotationY;
    jsj[ 1 ] = RotationZ * RotationX * RotationYd;
    jsj[ 2 ] = RotationZd * RotationX * RotationY;
  }

  /** Translation parameters: */
  for( unsigned int par = 3; par < ParametersDimension; ++par )
  {
    jsj[ par ].Fill( 0.0 );
  }
}


// Print self
template< class TScalarType >
void
AdvancedEuler3DTransform< TScalarType >::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Euler's angles: AngleX=" << m_AngleX
     << " AngleY=" << m_AngleY
     << " AngleZ=" << m_AngleZ
     << std::endl;
  os << indent << "m_ComputeZYX = " << m_ComputeZYX << std::endl;
}


} // namespace

#endif
