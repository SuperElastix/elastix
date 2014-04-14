/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxAdaptiveStochasticGradientDescent_hxx
#define __elxAdaptiveStochasticGradientDescent_hxx

#include "elxAdaptiveStochasticGradientDescent.h"

#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <utility>
#include "itkAdvancedImageToImageMetric.h"
#include "elxTimer.h"

namespace elastix
{

/**
 * ********************** Constructor ***********************
 */

template< class TElastix >
AdaptiveStochasticGradientDescent< TElastix >
::AdaptiveStochasticGradientDescent()
{
  this->m_MaximumNumberOfSamplingAttempts  = 0;
  this->m_CurrentNumberOfSamplingAttempts  = 0;
  this->m_PreviousErrorAtIteration         = 0;
  this->m_AutomaticParameterEstimationDone = false;

  this->m_AutomaticParameterEstimation = false;
  this->m_MaximumStepLength            = 1.0;

  this->m_NumberOfGradientMeasurements    = 0;
  this->m_NumberOfJacobianMeasurements    = 0;
  this->m_NumberOfSamplesForExactGradient = 100000;
  this->m_SigmoidScaleFactor              = 0.1;

  this->m_RandomGenerator   = RandomGeneratorType::GetInstance();
  this->m_AdvancedTransform = 0;

  this->m_UseNoiseCompensation        = true;
  this->m_OriginalButSigmoidToDefault = false;

} // Constructor


/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::BeforeRegistration( void )
{
  /** Add the target cell "stepsize" to xout["iteration"]. */
  xout[ "iteration" ].AddTargetCell( "2:Metric" );
  xout[ "iteration" ].AddTargetCell( "3a:Time" );
  xout[ "iteration" ].AddTargetCell( "3b:StepSize" );
  xout[ "iteration" ].AddTargetCell( "4:||Gradient||" );

  /** Format the metric and stepsize as floats. */
  xl::xout[ "iteration" ][ "2:Metric" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "3a:Time" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "3b:StepSize" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "4:||Gradient||" ] << std::showpoint << std::fixed;

  this->m_SettingsVector.clear();

  /** Temporary?: Use the multi-threaded version or not. Default true. */
  std::string tmp = this->m_Configuration->GetCommandLineArgument( "-mto" ); // mto: multi-threaded optimizers
  if( tmp == "true" || tmp == "" )
  {
    this->SetUseMultiThread( true );
    std::string  tmp2        = this->m_Configuration->GetCommandLineArgument( "-threads" );
    unsigned int nrOfThreads = atoi( tmp2.c_str() );
    if( tmp2 != "" )
    {
      this->SetNumberOfThreads( nrOfThreads );
    }
  }
  else { this->SetUseMultiThread( false ); }

  // Use eigen or openmp for multi-threading? Default false = use itk threads.
  tmp = this->m_Configuration->GetCommandLineArgument( "-useEigen" );
  if( tmp == "true" ) { this->SetUseEigen( true ); }
  tmp = this->m_Configuration->GetCommandLineArgument( "-useOpenMP" );
  if( tmp == "true" ) { this->SetUseOpenMP( true ); }

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level = static_cast< unsigned int >(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  const unsigned int P = this->GetElastix()->GetElxTransformBase()
    ->GetAsITKBaseType()->GetNumberOfParameters();

  /** Set the maximumNumberOfIterations. */
  SizeValueType maximumNumberOfIterations = 500;
  this->GetConfiguration()->ReadParameter( maximumNumberOfIterations,
    "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfIterations( maximumNumberOfIterations );

  /** Set the gain parameter A. */
  double A = 20.0;
  this->GetConfiguration()->ReadParameter( A,
    "SP_A", this->GetComponentLabel(), level, 0 );
  this->SetParam_A( A );

  /** Set the MaximumNumberOfSamplingAttempts. */
  SizeValueType maximumNumberOfSamplingAttempts = 0;
  this->GetConfiguration()->ReadParameter( maximumNumberOfSamplingAttempts,
    "MaximumNumberOfSamplingAttempts", this->GetComponentLabel(), level, 0 );
  this->SetMaximumNumberOfSamplingAttempts( maximumNumberOfSamplingAttempts );
  if( maximumNumberOfSamplingAttempts > 5 )
  {
    elxout[ "warning" ]
      << "\nWARNING: You have set MaximumNumberOfSamplingAttempts to "
      << maximumNumberOfSamplingAttempts << ".\n"
      << "  This functionality is known to cause problems (stack overflow) for large values.\n"
      << "  If elastix stops or segfaults for no obvious reason, reduce this value.\n"
      << "  You may select the RandomSparseMask image sampler to fix mask-related problems.\n"
      << std::endl;
  }

  /** Set/Get the initial time. Default: 0.0. Should be >=0. */
  double initialTime = 0.0;
  this->GetConfiguration()->ReadParameter( initialTime,
    "SigmoidInitialTime", this->GetComponentLabel(), level, 0 );
  this->SetInitialTime( initialTime );

  /** Set the maximum band size of the covariance matrix. */
  this->m_MaxBandCovSize = 192;
  this->GetConfiguration()->ReadParameter( this->m_MaxBandCovSize,
    "MaxBandCovSize", this->GetComponentLabel(), level, 0 );

  /** Set the number of random samples used to estimate the structure of the covariance matrix. */
  this->m_NumberOfBandStructureSamples = 10;
  this->GetConfiguration()->ReadParameter( this->m_NumberOfBandStructureSamples,
    "NumberOfBandStructureSamples", this->GetComponentLabel(), level, 0 );

  /** Set/Get whether the adaptive step size mechanism is desired. Default: true
   * NB: the setting is turned of in case of UseRandomSampleRegion=true.
   * Deprecated alias UseCruzAcceleration is also still supported.
   */
  bool useAdaptiveStepSizes = true;
  this->GetConfiguration()->ReadParameter( useAdaptiveStepSizes,
    "UseCruzAcceleration", this->GetComponentLabel(), level, 0, false );
  this->GetConfiguration()->ReadParameter( useAdaptiveStepSizes,
    "UseAdaptiveStepSizes", this->GetComponentLabel(), level, 0 );
  this->SetUseAdaptiveStepSizes( useAdaptiveStepSizes );

  /** Set whether automatic gain estimation is required; default: true. */
  this->m_AutomaticParameterEstimation = true;
  this->GetConfiguration()->ReadParameter( this->m_AutomaticParameterEstimation,
    "AutomaticParameterEstimation", this->GetComponentLabel(), level, 0 );

  if( this->m_AutomaticParameterEstimation )
  {
    /** Set the maximum step length: the maximum displacement of a voxel in mm.
     * Compute default value: mean spacing of fixed and moving image.
     */
    const unsigned int fixdim = this->GetElastix()->FixedDimension;
    const unsigned int movdim = this->GetElastix()->MovingDimension;
    double             sum    = 0.0;
    for( unsigned int d = 0; d < fixdim; ++d )
    {
      sum += this->GetElastix()->GetFixedImage()->GetSpacing()[ d ];
    }
    for( unsigned int d = 0; d < movdim; ++d )
    {
      sum += this->GetElastix()->GetMovingImage()->GetSpacing()[ d ];
    }
    this->m_MaximumStepLength = sum / static_cast< double >( fixdim + movdim );
    /** Read user setting */
    this->GetConfiguration()->ReadParameter( this->m_MaximumStepLength,
      "MaximumStepLength", this->GetComponentLabel(), level, 0 );

    /** Number of gradients N to estimate the average square magnitudes
     * of the exact gradient and the approximation error.
     * A value of 0 (default) means automatic estimation.
     */
    this->m_NumberOfGradientMeasurements = 0;
    this->GetConfiguration()->ReadParameter(
      this->m_NumberOfGradientMeasurements,
      "NumberOfGradientMeasurements",
      this->GetComponentLabel(), level, 0 );

    /** Set the number of Jacobian measurements M.
     * By default, if nothing specified by the user, M is determined as:
     * M = max( 1000, nrofparams );
     * This is a rather crude rule of thumb, which seems to work in practice.
     */
    this->m_NumberOfJacobianMeasurements = vnl_math_max(
      static_cast< unsigned int >( 1000 ), static_cast< unsigned int >( P ) );
    this->GetConfiguration()->ReadParameter(
      this->m_NumberOfJacobianMeasurements,
      "NumberOfJacobianMeasurements",
      this->GetComponentLabel(), level, 0 );

    /** Set the number of image samples used to compute the 'exact' gradient.
     * By default, if nothing supplied by the user, 100000. This works in general.
     * If the image is smaller, the number of samples is automatically reduced later.
     */
    this->m_NumberOfSamplesForExactGradient = 100000;
    this->GetConfiguration()->ReadParameter(
      this->m_NumberOfSamplesForExactGradient,
      "NumberOfSamplesForExactGradient",
      this->GetComponentLabel(), level, 0 );

    /** Set/Get the scaling factor zeta of the sigmoid width. Large values
      * cause a more wide sigmoid. Default: 0.1. Should be >0.
      */
    double sigmoidScaleFactor = 0.1;
    this->GetConfiguration()->ReadParameter( sigmoidScaleFactor,
      "SigmoidScaleFactor", this->GetComponentLabel(), level, 0 );
    this->m_SigmoidScaleFactor = sigmoidScaleFactor;

  } // end if automatic parameter estimation
  else
  {
    /** If no automatic parameter estimation is used, a and alpha also need
     * to be specified.
     */
    double a     = 400.0; // arbitrary guess
    double alpha = 0.602;
    this->GetConfiguration()->ReadParameter( a, "SP_a", this->GetComponentLabel(), level, 0 );
    this->GetConfiguration()->ReadParameter( alpha, "SP_alpha", this->GetComponentLabel(), level, 0 );
    this->SetParam_a( a );
    this->SetParam_alpha( alpha );

    /** Set/Get the maximum of the sigmoid. Should be >0. Default: 1.0. */
    double sigmoidMax = 1.0;
    this->GetConfiguration()->ReadParameter( sigmoidMax,
      "SigmoidMax", this->GetComponentLabel(), level, 0 );
    this->SetSigmoidMax( sigmoidMax );

    /** Set/Get the minimum of the sigmoid. Should be <0. Default: -0.8. */
    double sigmoidMin = -0.8;
    this->GetConfiguration()->ReadParameter( sigmoidMin,
      "SigmoidMin", this->GetComponentLabel(), level, 0 );
    this->SetSigmoidMin( sigmoidMin );

    /** Set/Get the scaling of the sigmoid width. Large values
     * cause a more wide sigmoid. Default: 1e-8. Should be >0.
     */
    double sigmoidScale = 1e-8;
    this->GetConfiguration()->ReadParameter( sigmoidScale,
      "SigmoidScale", this->GetComponentLabel(), level, 0 );
    this->SetSigmoidScale( sigmoidScale );

  } // end else: no automatic parameter estimation

} // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration *************************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::AfterEachIteration( void )
{
  /** Print some information. */
  xl::xout[ "iteration" ][ "2:Metric" ] << this->GetValue();
  xl::xout[ "iteration" ][ "3a:Time" ] << this->GetCurrentTime();
  xl::xout[ "iteration" ][ "3b:StepSize" ] << this->GetLearningRate();
  bool asFastAsPossible = false;
  if( asFastAsPossible )
  {
    xl::xout[ "iteration" ][ "4:||Gradient||" ] << "---";
  }
  else
  {
    xl::xout[ "iteration" ][ "4:||Gradient||" ] << this->GetGradient().magnitude();
  }

  /** Select new spatial samples for the computation of the metric. */
  if( this->GetNewSamplesEveryIteration() )
  {
    this->SelectNewSamples();
  }

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution *************************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::AfterEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level = static_cast< unsigned int >(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  /**
   * typedef enum {
   *   MaximumNumberOfIterations,
   *   MetricError,
   *   MinimumStepSize } StopConditionType;
   */
  std::string stopcondition;

  switch( this->GetStopCondition() )
  {

    case MaximumNumberOfIterations:
      stopcondition = "Maximum number of iterations has been reached";
      break;

    case MetricError:
      stopcondition = "Error in metric";
      break;

    case MinimumStepSize:
      stopcondition = "The minimum step length has been reached";
      break;

    default:
      stopcondition = "Unknown";
      break;
  }

  /** Print the stopping condition. */
  elxout << "Stopping condition: " << stopcondition << "." << std::endl;

  /** Store the used parameters, for later printing to screen. */
  SettingsType settings;
  settings.a     = this->GetParam_a();
  settings.A     = this->GetParam_A();
  settings.alpha = this->GetParam_alpha();
  settings.fmax  = this->GetSigmoidMax();
  settings.fmin  = this->GetSigmoidMin();
  settings.omega = this->GetSigmoidScale();
  this->m_SettingsVector.push_back( settings );

  /** Print settings that were used in this resolution. */
  SettingsVectorType tempSettingsVector;
  tempSettingsVector.push_back( settings );
  elxout
    << "Settings of " << this->elxGetClassName()
    << " in resolution " << level << ":" << std::endl;
  this->PrintSettingsVector( tempSettingsVector );

} // end AfterEachResolution()


/**
 * ******************* AfterRegistration ************************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::AfterRegistration( void )
{
  /** Print the best metric value. */

  double bestValue = this->GetValue();
  elxout << std::endl
         << "Final metric value  = "
         << bestValue
         << std::endl;

  elxout
    << "Settings of " << this->elxGetClassName()
    << " for all resolutions:" << std::endl;
  this->PrintSettingsVector( this->m_SettingsVector );

} // end AfterRegistration()


/**
 * ****************** StartOptimization *************************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::StartOptimization( void )
{
  /** Check if the entered scales are correct and != [ 1 1 1 ...]. */
  this->SetUseScales( false );
  const ScalesType & scales = this->GetScales();
  if( scales.GetSize() == this->GetInitialPosition().GetSize() )
  {
    ScalesType unit_scales( scales.GetSize() );
    unit_scales.Fill( 1.0 );
    if( scales != unit_scales )
    {
      /** only then: */
      this->SetUseScales( true );
    }
  }

  this->m_AutomaticParameterEstimationDone = false;

  this->Superclass1::StartOptimization();

} //end StartOptimization()


/**
 * ********************** ResumeOptimization **********************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::ResumeOptimization( void )
{
  /** The following code relies on the fact that all
  * components have been set up and that the initial
  * position has been set, so must be called in this
  * function. */

  if( this->GetAutomaticParameterEstimation()
    && !this->m_AutomaticParameterEstimationDone )
  {
    this->AutomaticParameterEstimation();
    // hack
    this->m_AutomaticParameterEstimationDone = true;
  }

  this->Superclass1::ResumeOptimization();

} // end ResumeOptimization()


/**
 * ****************** MetricErrorResponse *************************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::MetricErrorResponse( itk::ExceptionObject & err )
{
  if( this->GetCurrentIteration() != this->m_PreviousErrorAtIteration )
  {
    this->m_PreviousErrorAtIteration        = this->GetCurrentIteration();
    this->m_CurrentNumberOfSamplingAttempts = 1;
  }
  else
  {
    this->m_CurrentNumberOfSamplingAttempts++;
  }

  if( this->m_CurrentNumberOfSamplingAttempts <= this->m_MaximumNumberOfSamplingAttempts )
  {
    this->SelectNewSamples();
    this->ResumeOptimization();
  }
  else
  {
    /** Stop optimisation and pass on exception. */
    this->Superclass1::MetricErrorResponse( err );
  }

} // end MetricErrorResponse()


/**
 * ******************* AutomaticParameterEstimation **********************
 * Select different method to estimate some reasonable values for the parameters
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::AutomaticParameterEstimation( void )
{
  /** Setup timers. */
  tmr::Timer::Pointer timer1 = tmr::Timer::New();

  /** Total time. */
  timer1->StartTimer();
  elxout << "Starting automatic parameter estimation for "
         << this->elxGetClassName()
         << " ..." << std::endl;

  /** Decide which method is to be used. */
  std::string asgdParameterEstimationMethod = "Original";
  this->GetConfiguration()->ReadParameter( asgdParameterEstimationMethod,
    "ASGDParameterEstimationMethod", this->GetComponentLabel(), 0, 0 );

  /** Perform automatic optimizer parameter estimation by the desired method. */
  if( asgdParameterEstimationMethod == "Original" )
  {
    /** Original ASGD estimation method. */
    this->m_OriginalButSigmoidToDefault = false;
    this->AutomaticParameterEstimationOriginal();
  }
  else if( asgdParameterEstimationMethod == "OriginalButSigmoidToDefault" )
  {
    /** Original ASGD estimation method, but keeping the sigmoid parameters fixed. */
    this->m_OriginalButSigmoidToDefault = true;
    this->AutomaticParameterEstimationOriginal();
  }
  else if( asgdParameterEstimationMethod == "DisplacementDistribution" )
  {
    /** Accelerated parameter estimation method. */
    this->AutomaticParameterEstimationUsingDisplacementDistribution();
  }

  /** Print the elapsed time. */
  timer1->StopTimer();
  elxout << "Automatic parameter estimation took "
         << timer1->PrintElapsedTimeDHMS()
         << std::endl;

} // end AutomaticParameterEstimation()


/**
 * ******************* AutomaticParameterEstimationOriginal **********************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::AutomaticParameterEstimationOriginal( void )
{
  tmr::Timer::Pointer timer2 = tmr::Timer::New();
  tmr::Timer::Pointer timer3 = tmr::Timer::New();

  /** Get the user input. */
  const double delta = this->GetMaximumStepLength();

  /** Compute the Jacobian terms. */
  double TrC    = 0.0;
  double TrCC   = 0.0;
  double maxJJ  = 0.0;
  double maxJCJ = 0.0;

  /** Get current position to start the parameter estimation. */
  this->GetRegistration()->GetAsITKBaseType()->GetTransform()->SetParameters(
    this->GetCurrentPosition() );

  /** Cast to advanced metric type. */
  typedef typename ElastixType::MetricBaseType::AdvancedMetricType MetricType;
  MetricType * testPtr = dynamic_cast< MetricType * >(
    this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType() );
  if( !testPtr )
  {
    itkExceptionMacro( << "ERROR: AdaptiveStochasticGradientDescent expects "
                       << "the metric to be of type AdvancedImageToImageMetric!" );
  }

  /** Construct computeJacobianTerms to initialize the parameter estimation. */
  typename ComputeJacobianTermsType::Pointer computeJacobianTerms = ComputeJacobianTermsType::New();
  computeJacobianTerms->SetFixedImage( testPtr->GetFixedImage() );
  computeJacobianTerms->SetFixedImageRegion( testPtr->GetFixedImageRegion() );
  computeJacobianTerms->SetFixedImageMask( testPtr->GetFixedImageMask() );
  computeJacobianTerms->SetTransform(
    this->GetRegistration()->GetAsITKBaseType()->GetTransform() );
  computeJacobianTerms->SetMaxBandCovSize( this->m_MaxBandCovSize );
  computeJacobianTerms->SetNumberOfBandStructureSamples(
    this->m_NumberOfBandStructureSamples );
  computeJacobianTerms->SetNumberOfJacobianMeasurements(
    this->m_NumberOfJacobianMeasurements );

  /** Check if use scales. */
  bool useScales = this->GetUseScales();
  if( useScales )
  {
    computeJacobianTerms->SetScales( this->m_ScaledCostFunction->GetScales() );
    computeJacobianTerms->SetUseScales( true );
  }
  else
  {
    computeJacobianTerms->SetUseScales( false );
  }

  /** Compute the Jacobian terms. */
  elxout << "  Computing JacobianTerms ..." << std::endl;
  timer2->StartTimer();
  computeJacobianTerms->ComputeParameters( TrC, TrCC, maxJJ, maxJCJ );
  timer2->StopTimer();
  elxout << "  Computing the Jacobian terms took "
         << timer2->PrintElapsedTimeDHMS()
         << std::endl;

  /** Determine number of gradient measurements such that
   * E + 2\sqrt(Var) < K E
   * with
   * E = E(1/N \sum_n g_n^T g_n) = sigma_1^2 TrC
   * Var = Var(1/N \sum_n g_n^T g_n) = 2 sigma_1^4 TrCC / N
   * K = 1.5
   * We enforce a minimum of 2.
   */
  timer3->StartTimer();
  if( this->m_NumberOfGradientMeasurements == 0 )
  {
    const double K = 1.5;
    if( TrCC > 1e-14 && TrC > 1e-14 )
    {
      this->m_NumberOfGradientMeasurements = static_cast< unsigned int >(
        vcl_ceil( 8.0 * TrCC / TrC / TrC / ( K - 1 ) / ( K - 1 ) ) );
    }
    else
    {
      this->m_NumberOfGradientMeasurements = 2;
    }
    this->m_NumberOfGradientMeasurements = vnl_math_max(
      static_cast< SizeValueType >( 2 ),
      this->m_NumberOfGradientMeasurements );
    elxout << "  NumberOfGradientMeasurements to estimate sigma_i: "
           << this->m_NumberOfGradientMeasurements << std::endl;
  }

  /** Measure square magnitude of exact gradient and approximation error. */
  const double sigma4factor = 1.0;
  double       sigma4       = 0.0;
  double       gg           = 0.0;
  double       ee           = 0.0;
  if( maxJJ > 1e-14 )
  {
    sigma4 = sigma4factor * delta / vcl_sqrt( maxJJ );
  }
  this->SampleGradients(
    this->GetScaledCurrentPosition(), sigma4, gg, ee );
  timer3->StopTimer();
  elxout << "  Sampling the gradients took "
         << timer3->PrintElapsedTimeDHMS()
         << std::endl;

  /** Determine parameter settings. */
  double sigma1 = 0.0;
  double sigma3 = 0.0;
  /** Estimate of sigma such that empirical norm^2 equals theoretical:
   * gg = 1/N sum_n g_n' g_n
   * sigma = gg / TrC
   */
  if( gg > 1e-14 && TrC > 1e-14 )
  {
    sigma1 = vcl_sqrt( gg / TrC );
  }
  if( ee > 1e-14 && TrC > 1e-14 )
  {
    sigma3 = vcl_sqrt( ee / TrC );
  }

  const double alpha = 1.0;
  const double A     = this->GetParam_A();
  double       a_max = 0.0;
  if( sigma1 > 1e-14 && maxJCJ > 1e-14 )
  {
    a_max = A * delta / sigma1 / vcl_sqrt( maxJCJ );
  }
  const double noisefactor = sigma1 * sigma1
    / ( sigma1 * sigma1 + sigma3 * sigma3 + 1e-14 );
  const double a = a_max * noisefactor;

  const double omega = vnl_math_max( 1e-14,
    this->m_SigmoidScaleFactor * sigma3 * sigma3 * vcl_sqrt( TrCC ) );
  const double fmax = 1.0;
  const double fmin = -0.99 + 0.98 * noisefactor;

  /** Set parameters in superclass. */
  this->SetParam_a( a );
  this->SetParam_alpha( alpha );

  /** Set parameters for original method. */
  if( !this->m_OriginalButSigmoidToDefault )
  {
    this->SetSigmoidMax( fmax );
    this->SetSigmoidMin( fmin );
    this->SetSigmoidScale( omega );
  }
} // end AutomaticParameterEstimationOriginal()


/**
 * *************** AutomaticParameterEstimationUsingDisplacementDistribution *****
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::AutomaticParameterEstimationUsingDisplacementDistribution( void )
{
  tmr::Timer::Pointer timer4 = tmr::Timer::New();
  tmr::Timer::Pointer timer5 = tmr::Timer::New();

  /** Get current position to start the parameter estimation. */
  this->GetRegistration()->GetAsITKBaseType()->GetTransform()->SetParameters(
    this->GetCurrentPosition() );

  /** Get the user input. */
  const double delta = this->GetMaximumStepLength();
  double       maxJJ = 0;

  /** Cast to advanced metric type. */
  typedef typename ElastixType::MetricBaseType::AdvancedMetricType MetricType;
  MetricType * testPtr = dynamic_cast< MetricType * >(
    this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType() );
  if( !testPtr )
  {
    itkExceptionMacro( << "ERROR: AdaptiveStochasticGradientDescent expects "
                       << "the metric to be of type AdvancedImageToImageMetric!" );
  }

  /** Construct computeJacobianTerms to initialize the parameter estimation. */
  typename ComputeDisplacementDistributionType::Pointer
  computeDisplacementDistribution = ComputeDisplacementDistributionType::New();
  computeDisplacementDistribution->SetFixedImage( testPtr->GetFixedImage() );
  computeDisplacementDistribution->SetFixedImageRegion( testPtr->GetFixedImageRegion() );
  computeDisplacementDistribution->SetFixedImageMask( testPtr->GetFixedImageMask() );
  computeDisplacementDistribution->SetTransform(
    this->GetRegistration()->GetAsITKBaseType()->GetTransform() );
  computeDisplacementDistribution->SetCostFunction( this->m_CostFunction );
  computeDisplacementDistribution->SetNumberOfJacobianMeasurements(
    this->m_NumberOfJacobianMeasurements );

  /** Check if use scales. */
  if( this->GetUseScales() )
  {
    computeDisplacementDistribution->SetUseScales( true );
    computeDisplacementDistribution->SetScales( this->m_ScaledCostFunction->GetScales() );
    //this setting is not successful. Because copying the scales from ASGD to computeDisplacementDistribution is failed.
  }
  else
  {
    computeDisplacementDistribution->SetUseScales( false );
  }

  double      jacg                                = 0.0;
  std::string maximumDisplacementEstimationMethod = "2sigma";
  this->GetConfiguration()->ReadParameter( maximumDisplacementEstimationMethod,
    "MaximumDisplacementEstimationMethod", this->GetComponentLabel(), 0, 0 );

  /** Compute the Jacobian terms. */
  elxout << "  Computing displacement distribution ..." << std::endl;
  timer4->StartTimer();
  computeDisplacementDistribution->ComputeDistributionTerms(
    this->GetScaledCurrentPosition(), jacg, maxJJ,
    maximumDisplacementEstimationMethod );
  timer4->StopTimer();
  elxout << "  Computing the displacement distribution took "
         << timer4->PrintElapsedTimeDHMS()
         << std::endl;

  /** Initial of the variables. */
  double       a     = 0.0;
  const double A     = this->GetParam_A();
  const double alpha = 1.0;

  this->m_UseNoiseCompensation = true;
  this->GetConfiguration()->ReadParameter( this->m_UseNoiseCompensation,
    "NoiseCompensation", this->GetComponentLabel(), 0, 0 );

  /** Use noise Compensation factor or not. */
  if( this->m_UseNoiseCompensation == true )
  {
    double sigma4       = 0.0;
    double gg           = 0.0;
    double ee           = 0.0;
    double sigma4factor = 1.0;

    /** Sample the grid and random sampler container to estimate the noise factor.*/
    if( this->m_NumberOfGradientMeasurements == 0 )
    {
      this->m_NumberOfGradientMeasurements = vnl_math_max(
        static_cast< SizeValueType >( 2 ),
        this->m_NumberOfGradientMeasurements );
      elxout << "  NumberOfGradientMeasurements to estimate sigma_i: "
             << this->m_NumberOfGradientMeasurements << std::endl;
    }
    timer5->StartTimer();
    if( maxJJ > 1e-14 )
    {
      sigma4 = sigma4factor * delta / vcl_sqrt( maxJJ );
    }
    this->SampleGradients( this->GetScaledCurrentPosition(), sigma4, gg, ee );

    double noisefactor = gg / ( gg + ee );
    a =  delta * vcl_pow( A + 1.0, alpha ) / jacg * noisefactor;
    timer5->StopTimer();
    elxout << "  Compute the noise compensation took "
           << timer5->PrintElapsedTimeDHMS()
           << std::endl;
  }
  else
  {
    a = delta * vcl_pow( A + 1.0, alpha ) / jacg;
  }

  /** Set parameters in superclass. */
  this->SetParam_a( a );
  this->SetParam_alpha( alpha );

} // end AutomaticParameterEstimationUsingDisplacementDistribution()


/**
 * ******************** SampleGradients **********************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::SampleGradients( const ParametersType & mu0,
  double perturbationSigma, double & gg, double & ee )
{
  /** Some shortcuts. */
  const unsigned int M = this->GetElastix()->GetNumberOfMetrics();

  /** Variables for sampler support. Each metric may have a sampler. */
  std::vector< bool >                                useRandomSampleRegionVec( M, false );
  std::vector< ImageRandomSamplerBasePointer >       randomSamplerVec( M, 0 );
  std::vector< ImageRandomCoordinateSamplerPointer > randomCoordinateSamplerVec( M, 0 );
  std::vector< ImageGridSamplerPointer >             gridSamplerVec( M, 0 );

  /** If new samples every iteration, get each sampler, and check if it is
   * a kind of random sampler. If yes, prepare an additional grid sampler
   * for the exact gradients, and set the stochasticgradients flag to true.
   */
  bool stochasticgradients = false;
  if( this->GetNewSamplesEveryIteration() )
  {
    for( unsigned int m = 0; m < M; ++m )
    {
      /** Get the sampler. */
      ImageSamplerBasePointer sampler
        = this->GetElastix()->GetElxMetricBase( m )->GetAdvancedMetricImageSampler();
      randomSamplerVec[ m ]
        = dynamic_cast< ImageRandomSamplerBaseType * >( sampler.GetPointer() );
      randomCoordinateSamplerVec[ m ]
        = dynamic_cast< ImageRandomCoordinateSamplerType * >( sampler.GetPointer() );

      if( randomSamplerVec[ m ].IsNotNull() )
      {
        /** At least one of the metric has a random sampler. */
        stochasticgradients |= true;

        /** If the sampler is a randomCoordinateSampler set the UseRandomSampleRegion
         * property to false temporarily. It disturbs the parameter estimation.
         * At the end of this function the original setting is set back.
         * Also, the AdaptiveStepSize mechanism is turned off when any of the samplers
         * has UseRandomSampleRegion==true.
         * \todo Extend ASGD to really take into account random region sampling.
         * \todo This does not work for the MultiInputRandomCoordinateImageSampler,
         * because it does not inherit from the RandomCoordinateImageSampler 
         */
        if( randomCoordinateSamplerVec[ m ].IsNotNull() )
        {
          useRandomSampleRegionVec[ m ]
            = randomCoordinateSamplerVec[ m ]->GetUseRandomSampleRegion();
          if( useRandomSampleRegionVec[ m ] )
          {
            if( this->GetUseAdaptiveStepSizes() )
            {
              xl::xout[ "warning" ]
                << "WARNING: UseAdaptiveStepSizes is turned off, "
                << "because UseRandomSampleRegion is set to \"true\"."
                << std::endl;
              this->SetUseAdaptiveStepSizes( false );
            }
          }
          /** Do not turn it off yet, as it would go wrong if you multiple metrics are using
           * all the same sampler. */
          //randomCoordinateSamplerVec[ m ]->SetUseRandomSampleRegion( false );

        } // end if random coordinate sampler

        /** Set up the grid sampler for the "exact" gradients.
         * Copy settings from the random sampler and update.
         */
        gridSamplerVec[ m ] = ImageGridSamplerType::New();
        gridSamplerVec[ m ]->SetInput( randomSamplerVec[ m ]->GetInput() );
        gridSamplerVec[ m ]->SetInputImageRegion( randomSamplerVec[ m ]->GetInputImageRegion() );
        gridSamplerVec[ m ]->SetMask( randomSamplerVec[ m ]->GetMask() );
        gridSamplerVec[ m ]->SetNumberOfSamples( this->m_NumberOfSamplesForExactGradient );
        gridSamplerVec[ m ]->Update();

      } // end if random sampler

    } // end for loop over metrics

    /** Start a second loop over all metrics to turn off the random region sampling. */     
    for( unsigned int m = 0; m < M; ++m )
    {
      if( randomCoordinateSamplerVec[ m ].IsNotNull() )
      {
        randomCoordinateSamplerVec[ m ]->SetUseRandomSampleRegion( false );
      }   
    } // end loop over metrics

  }   // end if NewSamplesEveryIteration.

#ifndef _ELASTIX_BUILD_LIBRARY
  /** Prepare for progress printing. */
  ProgressCommandPointer progressObserver = ProgressCommandType::New();
  progressObserver->SetUpdateFrequency(
    this->m_NumberOfGradientMeasurements, this->m_NumberOfGradientMeasurements );
  progressObserver->SetStartString( "  Progress: " );
#endif
  elxout << "  Sampling gradients ..." << std::endl;

  /** Initialize some variables for storing gradients and their magnitudes. */
  const unsigned int P = this->GetElastix()->GetElxTransformBase()
    ->GetAsITKBaseType()->GetNumberOfParameters();
  DerivativeType approxgradient( P );
  DerivativeType exactgradient( P );
  DerivativeType diffgradient;
  double         exactgg = 0.0;
  double         diffgg  = 0.0;

  /** Compute gg for some random parameters. */
  for( unsigned int i = 0; i < this->m_NumberOfGradientMeasurements; ++i )
  {
#ifndef _ELASTIX_BUILD_LIBRARY
    /** Show progress 0-100% */
    progressObserver->UpdateAndPrintProgress( i );
#endif
    /** Generate a perturbation, according to:
     *    \mu_i ~ N( \mu_0, perturbationsigma^2 I ).
     */
    ParametersType perturbedMu0 = mu0;
    this->AddRandomPerturbation( perturbedMu0, perturbationSigma );

    /** Compute contribution to exactgg and diffgg. */
    if( stochasticgradients )
    {
      /** Set grid sampler(s) and get exact derivative. */
      for( unsigned int m = 0; m < M; ++m )
      {
        if( gridSamplerVec[ m ].IsNotNull() )
        {
          this->GetElastix()->GetElxMetricBase( m )
          ->SetAdvancedMetricImageSampler( gridSamplerVec[ m ] );
        }
      }
      this->GetScaledDerivativeWithExceptionHandling( perturbedMu0, exactgradient );

      /** Set random sampler(s), select new spatial samples and get approximate derivative. */
      for( unsigned int m = 0; m < M; ++m )
      {
        if( randomSamplerVec[ m ].IsNotNull() )
        {
          this->GetElastix()->GetElxMetricBase( m )->
          SetAdvancedMetricImageSampler( randomSamplerVec[ m ] );
        }
      }
      this->SelectNewSamples();
      this->GetScaledDerivativeWithExceptionHandling( perturbedMu0, approxgradient );

      /** Compute error vector. */
      diffgradient = exactgradient - approxgradient;

      /** Compute g^T g and e^T e */
      exactgg += exactgradient.squared_magnitude();
      diffgg  += diffgradient.squared_magnitude();
    }
    else // no stochastic gradients
    {
      /** Get exact gradient. */
      this->GetScaledDerivativeWithExceptionHandling( perturbedMu0, exactgradient );

      /** Compute g^T g. NB: diffgg=0. */
      exactgg += exactgradient.squared_magnitude();
    } // end else: no stochastic gradients

  } // end for loop over gradient measurements
#ifdef _ELASTIX_BUILD_LIBARY
  progressObserver->PrintProgress( 1.0 );
#endif
  /** Compute means. */
  exactgg /= this->m_NumberOfGradientMeasurements;
  diffgg  /= this->m_NumberOfGradientMeasurements;

  /** For output: gg and ee.
   * gg and ee will be divided by Pd, but actually need to be divided by
   * the rank, in case of maximum likelihood. In case of no maximum likelihood,
   * the rank equals Pd.
   */
  gg = exactgg;
  ee = diffgg;

  /** Set back useRandomSampleRegion flag to what it was. */
  for( unsigned int m = 0; m < M; ++m )
  {
    if( randomCoordinateSamplerVec[ m ].IsNotNull() )
    {
      randomCoordinateSamplerVec[ m ]
      ->SetUseRandomSampleRegion( useRandomSampleRegionVec[ m ] );
    }
  }

} // end SampleGradients()


/**
 * **************** PrintSettingsVector **********************
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::PrintSettingsVector( const SettingsVectorType & settings ) const
{
  const unsigned long nrofres = settings.size();

  /** Print to log file */
  elxout << "( SP_a ";
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].a << " ";
  }
  elxout << ")\n";

  elxout << "( SP_A ";
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].A << " ";
  }
  elxout << ")\n";

  elxout << "( SP_alpha ";
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].alpha << " ";
  }
  elxout << ")\n";

  elxout << "( SigmoidMax ";
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].fmax << " ";
  }
  elxout << ")\n";

  elxout << "( SigmoidMin ";
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].fmin << " ";
  }
  elxout << ")\n";

  elxout << "( SigmoidScale ";
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].omega << " ";
  }
  elxout << ")\n";

  elxout << std::endl;

} // end PrintSettingsVector()


/**
 * ****************** CheckForAdvancedTransform **********************
 * Check if the transform is of type AdvancedTransform.
 * If so, we can speed up derivative calculations by only inspecting
 * the parameters in the support region of a point.
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::CheckForAdvancedTransform( void )
{
  typename TransformType::Pointer transform = this->GetRegistration()
    ->GetAsITKBaseType()->GetTransform();

  AdvancedTransformType * testPtr = dynamic_cast< AdvancedTransformType * >(
    transform.GetPointer() );
  if( !testPtr )
  {
    this->m_AdvancedTransform = 0;
    itkDebugMacro( "Transform is not Advanced" );
    itkExceptionMacro( << "The automatic parameter estimation of the ASGD "
                       << "optimizer works only with advanced transforms" );
  }
  else
  {
    this->m_AdvancedTransform = testPtr;
    itkDebugMacro( "Transform is Advanced" );
  }

} // end CheckForAdvancedTransform()


/**
 * *************** GetScaledDerivativeWithExceptionHandling ***************
 * Helper function, used by SampleGradients.
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::GetScaledDerivativeWithExceptionHandling(
  const ParametersType & parameters, DerivativeType & derivative )
{
  double dummyvalue = 0;
  try
  {
    this->GetScaledValueAndDerivative( parameters, dummyvalue, derivative );
  }
  catch( itk::ExceptionObject & err )
  {
    this->m_StopCondition = MetricError;
    this->StopOptimization();
    throw err;
  }

} // end GetScaledDerivativeWithExceptionHandling()


/**
 * *************** AddRandomPerturbation ***************
 * Helper function, used by SampleGradients.
 */

template< class TElastix >
void
AdaptiveStochasticGradientDescent< TElastix >
::AddRandomPerturbation( ParametersType & parameters, double sigma )
{
  /** Add delta ~ sigma * N(0,I) to the input parameters. */
  for( unsigned int p = 0; p < parameters.GetSize(); ++p )
  {
    parameters[ p ] += sigma * this->m_RandomGenerator->GetNormalVariate( 0.0, 1.0 );
  }

} // end AddRandomPerturbation()


} // end namespace elastix

#endif // end #ifndef __elxAdaptiveStochasticGradientDescent_hxx
