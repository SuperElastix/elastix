/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxRayCastResampleInterpolator_hxx
#define __elxRayCastResampleInterpolator_hxx

#include "elxRayCastResampleInterpolator.h"
#include "itkImageFileWriter.h"

namespace elastix
{

/*
 * ***************** BeforeAll *****************
 */

template< class TElastix >
void
RayCastResampleInterpolator< TElastix >
::InitializeRayCastInterpolator( void )
{

  this->m_CombinationTransform = CombinationTransformType::New();
  this->m_CombinationTransform->SetUseComposition( true );

  this->m_PreTransform = EulerTransformType::New();
  unsigned int            numberofparameters = this->m_PreTransform->GetNumberOfParameters();
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

  typename EulerTransformType::InputPointType centerofrotation;
  centerofrotation.Fill( 0.0 );

  for( unsigned int i = 0; i < this->m_Elastix->GetMovingImage()->GetImageDimension(); i++ )
  {
    this->GetConfiguration()->ReadParameter( centerofrotation[ i ],
      "CenterOfRotationPoint", this->GetComponentLabel(), i, 0 );
  }

  this->m_PreTransform->SetParameters( preParameters );
  this->m_PreTransform->SetCenter( centerofrotation );
  this->m_CombinationTransform->SetInitialTransform( this->m_PreTransform );
  this->m_CombinationTransform->SetCurrentTransform(
    this->m_Elastix->GetElxTransformBase()->GetAsITKBaseType() );
  this->SetTransform( this->m_CombinationTransform );
  this->SetInputImage( this->m_Elastix->GetMovingImage() );

  PointType focalPoint;
  focalPoint.Fill( 0.0 );

  for( unsigned int i = 0; i < this->m_Elastix->GetFixedImage()->GetImageDimension(); i++ )
  {
    bool ret = this->GetConfiguration()->ReadParameter( focalPoint[ i ],
      "FocalPoint", this->GetComponentLabel(), i, 0 );
    if( !ret )
    {
      std::cerr << " Error, FocalPoint not assigned" << std::endl;
    }
  }

  this->SetFocalPoint( focalPoint );
  this->m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->SetTransform(
    this->m_CombinationTransform );
  double threshold = 0.;
  this->GetConfiguration()->ReadParameter( threshold, "Threshold", 0 );
  this->SetThreshold( threshold );

} // end InitializeRayCastInterpolator()


/*
 * ***************** BeforeAll *****************
 */

template< class TElastix >
int
RayCastResampleInterpolator< TElastix >
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

}   // end BeforeAll()


/*
 * ***************** BeforeRegistration *****************
 */

template< class TElastix >
void
RayCastResampleInterpolator< TElastix >
::BeforeRegistration( void )
{

  this->InitializeRayCastInterpolator();

}   // end BeforeRegistration()


/*
 * ***************** ReadFromFile *****************
 */

template< class TElastix >
void
RayCastResampleInterpolator< TElastix >
::ReadFromFile( void )
{

  /** Call ReadFromFile of the ResamplerBase. */
  this->Superclass2::ReadFromFile();
  this->InitializeRayCastInterpolator();

}   // end ReadFromFile()


/**
 * ******************* WriteToFile ******************************
 */

template< class TElastix >
void
RayCastResampleInterpolator< TElastix >
::WriteToFile( void ) const
{

  /** Call WriteToFile of the ResamplerBase. */
  this->Superclass2::WriteToFile();

  PointType focalpoint = this->GetFocalPoint();

  xout[ "transpar" ] << "(" << "FocalPoint ";
  for( unsigned int i = 0; i < this->m_Elastix->GetMovingImage()->GetImageDimension(); i++ )
  {
    xout[ "transpar" ] << focalpoint[ i ] << " ";
  }
  xout[ "transpar" ] << ")" << std::endl;

  TransformParametersType preParameters = this->m_PreTransform->GetParameters();

  xout[ "transpar" ] << "(" << "PreParameters ";

  unsigned int numberofparameters = preParameters.GetSize();
  for( unsigned int i = 0; i < numberofparameters; i++ )
  {
    xout[ "transpar" ] << preParameters[ i ] << " ";
  }
  xout[ "transpar" ] << ")" << std::endl;

  double threshold = this->GetThreshold();
  xout[ "transpar" ] << "(Threshold "
                     << threshold << ")" << std::endl;

}   // end WriteToFile()


} // end namespace elastix

#endif // end #ifndef __elxRayCastResampleInterpolator_hxx
