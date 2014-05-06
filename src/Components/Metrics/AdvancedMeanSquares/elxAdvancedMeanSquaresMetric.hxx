/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxAdvancedMeanSquaresMetric_HXX__
#define __elxAdvancedMeanSquaresMetric_HXX__

#include "elxAdvancedMeanSquaresMetric.h"
#include "itkTimeProbe.h"


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
AdvancedMeanSquaresMetric< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of AdvancedMeanSquares metric took: "
         << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
AdvancedMeanSquaresMetric< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Get and set the normalization. */
  bool useNormalization = false;
  this->GetConfiguration()->ReadParameter( useNormalization,
    "UseNormalization", this->GetComponentLabel(), level, 0 );
  this->SetUseNormalization( useNormalization );

  /** Experimental options for SelfHessian */

  /** Set the number of samples used to compute the SelfHessian */
  unsigned int numberOfSamplesForSelfHessian = 100000;
  this->GetConfiguration()->ReadParameter( numberOfSamplesForSelfHessian,
    "NumberOfSamplesForSelfHessian", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfSamplesForSelfHessian( numberOfSamplesForSelfHessian );

  /** Set the smoothing sigma used to compute the SelfHessian */
  double selfHessianSmoothingSigma = 1.0;
  this->GetConfiguration()->ReadParameter( selfHessianSmoothingSigma,
    "SelfHessianSmoothingSigma", this->GetComponentLabel(), level, 0 );
  this->SetSelfHessianSmoothingSigma( selfHessianSmoothingSigma );

  /** Set the smoothing sigma used to compute the SelfHessian */
  double selfHessianNoiseRange = 1.0;
  this->GetConfiguration()->ReadParameter( selfHessianNoiseRange,
    "SelfHessianNoiseRange", this->GetComponentLabel(), level, 0 );
  this->SetSelfHessianNoiseRange( selfHessianNoiseRange );

  /** Set moving image derivative scales. */
  this->SetUseMovingImageDerivativeScales( false );
  MovingImageDerivativeScalesType movingImageDerivativeScales;
  movingImageDerivativeScales.Fill( 1.0 );
  bool usescales = true;
  for( unsigned int i = 0; i < MovingImageDimension; ++i )
  {
    usescales &= this->GetConfiguration()->ReadParameter(
      movingImageDerivativeScales[ i ], "MovingImageDerivativeScales",
      this->GetComponentLabel(), i, -1, false );
  }
  if( usescales )
  {
    this->SetUseMovingImageDerivativeScales( true );
    this->SetMovingImageDerivativeScales( movingImageDerivativeScales );
    elxout << "Multiplying moving image derivatives by: "
           << movingImageDerivativeScales << std::endl;
  }

  /** Select the use of an OpenMP implementation for GetValueAndDerivative. */
  std::string useOpenMP = this->m_Configuration->GetCommandLineArgument( "-useOpenMP_SSD" );
  if( useOpenMP == "true" )
  {
    this->SetUseOpenMP( true );
  }

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxAdvancedMeanSquaresMetric_HXX__
