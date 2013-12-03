/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxAdvancedMattesMutualInformationMetric_HXX__
#define __elxAdvancedMattesMutualInformationMetric_HXX__

#include "elxAdvancedMattesMutualInformationMetric.h"

#include "itkHardLimiterFunction.h"
#include "itkExponentialLimiterFunction.h"
#include <string>
#include "vnl/vnl_math.h"

namespace elastix
{

/**
 * ****************** Constructor ***********************
 */

template< class TElastix >
AdvancedMattesMutualInformationMetric< TElastix >
::AdvancedMattesMutualInformationMetric()
{
  this->m_CurrentIteration = 0.0;
  this->m_Param_c          = 1.0;
  this->m_Param_gamma      = 0.101;
  this->SetUseDerivative( true );

}   // end Constructor()


/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
AdvancedMattesMutualInformationMetric< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  TimerPointer timer = TimerType::New();
  timer->StartTimer();
  this->Superclass1::Initialize();
  timer->StopTimer();
  elxout << "Initialization of AdvancedMattesMutualInformation metric took: "
         << static_cast< long >( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

}   // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
AdvancedMattesMutualInformationMetric< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Get and set the number of histogram bins. */
  unsigned int numberOfHistogramBins = 32;
  this->GetConfiguration()->ReadParameter( numberOfHistogramBins,
    "NumberOfHistogramBins", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfFixedHistogramBins( numberOfHistogramBins );
  this->SetNumberOfMovingHistogramBins( numberOfHistogramBins );

  unsigned int numberOfFixedHistogramBins  = numberOfHistogramBins;
  unsigned int numberOfMovingHistogramBins = numberOfHistogramBins;
  this->GetConfiguration()->ReadParameter( numberOfFixedHistogramBins,
    "NumberOfFixedHistogramBins", this->GetComponentLabel(), level, 0 );
  this->GetConfiguration()->ReadParameter( numberOfMovingHistogramBins,
    "NumberOfMovingHistogramBins", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfFixedHistogramBins( numberOfFixedHistogramBins );
  this->SetNumberOfMovingHistogramBins( numberOfMovingHistogramBins );

  /** Set limiters. */
  typedef itk::HardLimiterFunction< RealType, FixedImageDimension >         FixedLimiterType;
  typedef itk::ExponentialLimiterFunction< RealType, MovingImageDimension > MovingLimiterType;
  this->SetFixedImageLimiter( FixedLimiterType::New() );
  this->SetMovingImageLimiter( MovingLimiterType::New() );

  /** Get and set the limit range ratios. */
  double fixedLimitRangeRatio  = 0.01;
  double movingLimitRangeRatio = 0.01;
  this->GetConfiguration()->ReadParameter( fixedLimitRangeRatio,
    "FixedLimitRangeRatio", this->GetComponentLabel(), level, 0 );
  this->GetConfiguration()->ReadParameter( movingLimitRangeRatio,
    "MovingLimitRangeRatio", this->GetComponentLabel(), level, 0 );
  this->SetFixedLimitRangeRatio( fixedLimitRangeRatio );
  this->SetMovingLimitRangeRatio( movingLimitRangeRatio );

  /** Set B-spline Parzen kernel orders. */
  unsigned int fixedKernelBSplineOrder  = 0;
  unsigned int movingKernelBSplineOrder = 3;
  this->GetConfiguration()->ReadParameter( fixedKernelBSplineOrder,
    "FixedKernelBSplineOrder", this->GetComponentLabel(), level, 0 );
  this->GetConfiguration()->ReadParameter( movingKernelBSplineOrder,
    "MovingKernelBSplineOrder", this->GetComponentLabel(), level, 0 );
  this->SetFixedKernelBSplineOrder( fixedKernelBSplineOrder );
  this->SetMovingKernelBSplineOrder( movingKernelBSplineOrder );

  /** Set whether a low memory consumption should be used. */
  bool useFastAndLowMemoryVersion = true;
  this->GetConfiguration()->ReadParameter( useFastAndLowMemoryVersion,
    "UseFastAndLowMemoryVersion", this->GetComponentLabel(), level, 0 );
  this->SetUseExplicitPDFDerivatives( !useFastAndLowMemoryVersion );

  /** Set whether to use Nick Tustison's preconditioning technique. */
  bool useJacobianPreconditioning = false;
  this->GetConfiguration()->ReadParameter( useJacobianPreconditioning,
    "UseJacobianPreconditioning", this->GetComponentLabel(), level, 0 );
  this->SetUseJacobianPreconditioning( useJacobianPreconditioning );

  /** Set whether a finite difference derivative should be used. */
  bool useFiniteDifferenceDerivative = false;
  this->GetConfiguration()->ReadParameter( useFiniteDifferenceDerivative,
    "FiniteDifferenceDerivative", this->GetComponentLabel(), level, 0 );
  this->SetUseFiniteDifferenceDerivative( useFiniteDifferenceDerivative );

  /** Prepare for computing the perturbation gain c_k. */
  this->SetCurrentIteration( 0 );
  if( useFiniteDifferenceDerivative )
  {
    double c     = 1.0;
    double gamma = 0.101;
    this->GetConfiguration()->ReadParameter( c, "SP_c",
      this->GetComponentLabel(), level, 0 );
    this->GetConfiguration()->ReadParameter( gamma, "SP_gamma",
      this->GetComponentLabel(), level, 0 );
    this->SetParam_c( c );
    this->SetParam_gamma( gamma );
    this->SetFiniteDifferencePerturbation( this->Compute_c( 0 ) );
  }

  /** Set whether to use moving image derivative scales. */
  this->SetUseMovingImageDerivativeScales( false );
  MovingImageDerivativeScalesType movingImageDerivativeScales;
  movingImageDerivativeScales.Fill( 1.0 );
  bool usescales = false;
  for( unsigned int i = 0; i < MovingImageDimension; ++i )
  {
    usescales |= this->GetConfiguration()->ReadParameter(
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

}   // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration ***********************
 */

template< class TElastix >
void
AdvancedMattesMutualInformationMetric< TElastix >
::AfterEachIteration( void )
{
  if( this->GetUseFiniteDifferenceDerivative() )
  {
    this->m_CurrentIteration++;
    this->SetFiniteDifferencePerturbation(
      this->Compute_c( this->m_CurrentIteration ) );
  }
}    // end AfterEachIteration()


/**
 * ************************** Compute_c *************************
 *
 * This function computes the parameter a at iteration k, as
 * in the finite difference optimizer.
 */

template< class TElastix >
double
AdvancedMattesMutualInformationMetric< TElastix >
::Compute_c( unsigned long k ) const
{
  return static_cast< double >(
    this->m_Param_c / vcl_pow( k + 1, this->m_Param_gamma ) );

}   // end Compute_c()


} // end namespace elastix

#endif // end #ifndef __elxAdvancedMattesMutualInformationMetric_HXX__
