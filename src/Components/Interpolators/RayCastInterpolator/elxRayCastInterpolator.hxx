/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxRayCastInterpolator_hxx
#define __elxRayCastInterpolator_hxx

#include "elxRayCastInterpolator.h"

namespace elastix
{

/*
 * ***************** BeforeAll *****************
 */

template< class TElastix >
int
RayCastInterpolator< TElastix >
::BeforeAll( void )
{
  // Check if 2D-3D
  if( this->m_Elastix->GetFixedImage()->GetImageDimension() != 3 )
  {
    itkExceptionMacro( << "The RayCastInterpolator expects the fixed image to be 3D." );
    return 1;
  }
  if( this->m_Elastix->GetMovingImage()->GetImageDimension() != 3 )
  {
    itkExceptionMacro( << "The RayCastInterpolator expects the moving image to be 3D." );
    return 1;
  }

  return 0;
} // end BeforeAll()


/*
 * ***************** BeforeRegistration *****************
 */

template< class TElastix >
void
RayCastInterpolator< TElastix >
::BeforeRegistration( void )
{
  this->m_CombinationTransform = CombinationTransformType::New();
  this->m_CombinationTransform->SetUseComposition( true );

  typedef typename elastix::OptimizerBase< TElastix >::ITKBaseType::ParametersType ParametersType;
  unsigned int            numberofparameters = this->m_Elastix->GetElxTransformBase()->GetAsITKBaseType()->GetNumberOfParameters();
  TransformParametersType preParameters( numberofparameters );
  preParameters.Fill( 0.0 );

  for( unsigned int i = 0; i < numberofparameters; i++ )
  {
    bool ret = this->GetConfiguration()->ReadParameter( preParameters[ i ],
      "PreParameters", this->GetComponentLabel(), i, 0 );
    if( !ret )
    {
      std::cerr << " Error, not enough PreParameters are given" << std::endl;
    }
  }

  this->m_PreTransform = EulerTransformType::New();
  this->m_PreTransform->SetParameters( preParameters );
  this->m_CombinationTransform->SetInitialTransform( this->m_PreTransform );
  this->m_CombinationTransform->SetCurrentTransform( this->m_Elastix->GetElxTransformBase()->GetAsITKBaseType() );

  this->SetTransform( this->m_CombinationTransform );

  PointType focalPoint;
  focalPoint.Fill( 0. );

  for( unsigned int i = 0; i < this->m_Elastix->GetFixedImage()->GetImageDimension(); i++ )
  {
    bool ret = this->GetConfiguration()->ReadParameter( focalPoint[ i ],
      "FocalPoint", this->GetComponentLabel(), i, 0 );
    if( !ret )
    {
      std::cerr << "Error, FocalPoint not assigned" << std::endl;
    }
  }

  this->SetFocalPoint( focalPoint );

} // end BeforeRegistration()


/*
 * ***************** BeforeEachResolution *****************
 */

template< class TElastix >
void
RayCastInterpolator< TElastix >
::BeforeEachResolution( void )
{
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  double threshold = 0.;
  this->GetConfiguration()->ReadParameter( threshold, "Threshold", this->GetComponentLabel(), level, 0 );
  this->SetThreshold( threshold );

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxRayCastInterpolator_hxx
