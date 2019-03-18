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
#ifndef __elxPreconditionedStochasticGradientDescent_hxx
#define __elxPreconditionedStochasticGradientDescent_hxx

#include "elxPreconditionedStochasticGradientDescent.h"

#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <utility>
#include "itkAdvancedImageToImageMetric.h"
#include "itkTimeProbe.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace elastix
{

/**
 * ********************** Constructor ***********************
 */

template< class TElastix >
PreconditionedStochasticGradientDescent< TElastix >
::PreconditionedStochasticGradientDescent()
{
  this->m_MaximumNumberOfSamplingAttempts  = 0;
  this->m_CurrentNumberOfSamplingAttempts  = 0;
  this->m_PreviousErrorAtIteration         = 0;
  this->m_AutomaticParameterEstimationDone = false;

  this->m_AutomaticParameterEstimation = false;
  this->m_MaximumStepLength            = 1.0;
  this->m_MaximumStepLengthRatio       = 1.0;
  this->m_RegularizationKappa          = 0.8;
  this->m_ConditionNumber              = 2.0;
  this->m_NoiseFactor                  = 1.0;

  this->m_NumberOfGradientMeasurements    = 0;
  this->m_NumberOfJacobianMeasurements    = 0;
  this->m_NumberOfSamplesForPrecondition  = 0;
  this->m_NumberOfSamplesForNoiseCompensationFactor = 0;
  this->m_NumberOfSpatialSamples          = 5000;
  this->m_SigmoidScaleFactor              = 0.1;
  this->m_GlobalStepSize                  = 0;

  this->m_RandomGenerator   = RandomGeneratorType::GetInstance();
  this->m_AdvancedTransform = 0;

  this->m_UseNoiseCompensation = true;

} // Constructor


/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
PreconditionedStochasticGradientDescent< TElastix >
::BeforeRegistration( void )
{
  /** Add the target cell "stepsize" to xout["iteration"]. */
  xout[ "iteration" ].AddTargetCell( "2:Metric" );
  xout[ "iteration" ].AddTargetCell( "3a:Time" );
  xout[ "iteration" ].AddTargetCell( "3b:StepSize" );
  xout[ "iteration" ].AddTargetCell( "4a:||Gradient||" );
  xout[ "iteration" ].AddTargetCell( "4b:||SearchDirection||" );

  /** Format the metric and stepsize as floats. */
  xl::xout[ "iteration" ][ "2:Metric" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "3a:Time" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "3b:StepSize" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "4:||Gradient||" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "4b:||SearchDirection||" ] << std::showpoint << std::fixed;

  this->m_SettingsVector.clear();

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
PreconditionedStochasticGradientDescent< TElastix >
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

  double a = 1.0;
  this->GetConfiguration()->ReadParameter( a, "SP_a", this->GetComponentLabel(), level, 0 );
  this->SetParam_a(a);

  /** Set the MaximumNumberOfSamplingAttempts. check if needed? */
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

  /** Set/Get the initial time. Default: 0.0. Should be >= 0. */
  double initialTime = 0.0;
  this->GetConfiguration()->ReadParameter( initialTime,
    "SigmoidInitialTime", this->GetComponentLabel(), level, 0 );
  this->SetInitialTime( initialTime );

  /** Set/Get whether the adaptive step size mechanism is desired. Default: true
   * NB: the setting is turned off in case of UseRandomSampleRegion == true.
   */
  /** Set whether automatic gain estimation is required; default: true. */
  this->m_AutomaticParameterEstimation = true;
  this->GetConfiguration()->ReadParameter( this->m_AutomaticParameterEstimation,
    "AutomaticParameterEstimation", this->GetComponentLabel(), level, 0 );

  std::string stepSizeStrategy = "Adaptive";
  this->GetConfiguration()->ReadParameter(stepSizeStrategy,
    "StepSizeStrategy", this->GetComponentLabel(), 0, 0 );
  this->m_StepSizeStrategy = stepSizeStrategy;

  if( this->m_AutomaticParameterEstimation )
  {
    /** Read user setting. */
    this->GetConfiguration()->ReadParameter( this->m_MaximumStepLengthRatio,
      "MaximumStepLengthRatio", this->GetComponentLabel(), level, 0 );

    /** Set the maximum step length: the maximum displacement of a voxel in mm.
     * Compute default value: mean in-plane spacing of fixed and moving image.
     */
    const unsigned int fixdim = vnl_math_min( (unsigned int) this->GetElastix()->FixedDimension, (unsigned int) 2);
    const unsigned int movdim = vnl_math_min( (unsigned int) this->GetElastix()->MovingDimension, (unsigned int) 2);
    double             sum    = 0.0;
    for( unsigned int d = 0; d < fixdim; ++d )
    {
      sum += this->GetElastix()->GetFixedImage()->GetSpacing()[ d ];
    }
    for( unsigned int d = 0; d < movdim; ++d )
    {
      sum += this->GetElastix()->GetMovingImage()->GetSpacing()[ d ];
    }
    this->m_MaximumStepLength = this->m_MaximumStepLengthRatio * sum / static_cast< double >( fixdim + movdim );

    /** Read user setting. */
    this->GetConfiguration()->ReadParameter( this->m_MaximumStepLength,
      "MaximumStepLength", this->GetComponentLabel(), level, 0 );

    /** Number of gradients N to estimate the average magnitudes
     * of the exact preconditioned gradient and the approximation error.
     */
    this->m_NumberOfGradientMeasurements = 0;
    this->GetConfiguration()->ReadParameter(
      this->m_NumberOfGradientMeasurements,
      "NumberOfGradientMeasurements",
      this->GetComponentLabel(), level, 0 );
    this->m_NumberOfGradientMeasurements = vnl_math_max(
      static_cast< SizeValueType >( 2 ),
      this->m_NumberOfGradientMeasurements );

    /** Set the number of Jacobian measurements M.
     * By default, if nothing specified by the user, M is determined as:
     * M = max( 1000, nrofparams );
     * This is a rather crude rule of thumb, which seems to work in practice.
     */
    this->m_NumberOfJacobianMeasurements = vnl_math_max(
      static_cast< unsigned int >( 5000 ), static_cast< unsigned int >( 2 * P ) );
    this->GetConfiguration()->ReadParameter(
      this->m_NumberOfJacobianMeasurements,
      "NumberOfJacobianMeasurements",
      this->GetComponentLabel(), level, 0 );

    /** Set the NumberOfSpatialSamples. */
    unsigned long numberOfSpatialSamples = 5000;
    this->GetConfiguration()->ReadParameter(numberOfSpatialSamples,
      "NumberOfSpatialSamples", this->GetComponentLabel(), level, 0);
    this->m_NumberOfSpatialSamples = numberOfSpatialSamples;

    /** Set the number of samples for precondition matrix computation.
     * By default, if nothing specified by the user, M is determined as:
     * P = max( 1000, nrofparams );
     * This is a rather crude rule of thumb, which seems to work in practice.
     */
    this->m_NumberOfSamplesForPrecondition = vnl_math_max(
      static_cast< unsigned int >( 1000 ), static_cast< unsigned int >( P ) );
    this->GetConfiguration()->ReadParameter(
      this->m_NumberOfSamplesForPrecondition,
      "NumberOfSamplesForPrecondition",
      this->GetComponentLabel(), level, 0 );

    /** Set the number of image samples used to compute the 'exact' gradient.
     * By default, if nothing supplied by the user, 100000. This works in general.
     * If the image is smaller, the number of samples is automatically reduced later.
     */
    this->m_NumberOfSamplesForNoiseCompensationFactor = 100000;
    this->GetConfiguration()->ReadParameter(
      this->m_NumberOfSamplesForNoiseCompensationFactor,
      "NumberOfSamplesForNoiseCompensationFactor",
      this->GetComponentLabel(), level, 0 );

    /** Set/Get the scaling factor zeta of the sigmoid width. Large values
      * cause a more wide sigmoid. Default: 0.1. Should be > 0.
      */
    double sigmoidScaleFactor = 0.1;
    this->GetConfiguration()->ReadParameter( sigmoidScaleFactor,
      "SigmoidScaleFactor", this->GetComponentLabel(), level, 0 );
    this->m_SigmoidScaleFactor = sigmoidScaleFactor;

    /** Set the regularization factor kappa. */
    this->m_RegularizationKappa = 0.8;
    this->GetConfiguration()->ReadParameter(this->m_RegularizationKappa,
      "RegularizationKappa", this->GetComponentLabel(), level, 0);

    /** Set the regularization factor kappa. */
    this->m_ConditionNumber = 2.0;
    this->GetConfiguration()->ReadParameter( this->m_ConditionNumber,
      "ConditionNumber", this->GetComponentLabel(), level, 0 );

  } // end if automatic parameter estimation
  else
  {
    /** If no automatic parameter estimation is used, a and alpha also need
     * to be specified.
     */
    double a     = 1.0; // arbitrary guess
    double alpha = 0.602;
    this->GetConfiguration()->ReadParameter( a, "SP_a", this->GetComponentLabel(), level, 0 );
    this->GetConfiguration()->ReadParameter( alpha, "SP_alpha", this->GetComponentLabel(), level, 0 );
    this->SetParam_a( a );
    this->SetParam_alpha( alpha );

    /** Set/Get the maximum of the sigmoid. Should be > 0. Default: 1.0. */
    double sigmoidMax = 1.0;
    this->GetConfiguration()->ReadParameter( sigmoidMax,
      "SigmoidMax", this->GetComponentLabel(), level, 0 );
    this->SetSigmoidMax( sigmoidMax );

    /** Set/Get the minimum of the sigmoid. Should be < 0. Default: -0.8. */
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
PreconditionedStochasticGradientDescent< TElastix >
::AfterEachIteration( void )
{
  /** Print some information. */
  xl::xout[ "iteration" ][ "2:Metric" ] << this->GetValue();
  xl::xout[ "iteration" ][ "3a:Time" ] << this->GetCurrentTime();
  xl::xout[ "iteration" ][ "3b:StepSize" ] << this->GetLearningRate() * this->m_NoiseFactor;

  bool asFastAsPossible = false;
  if( asFastAsPossible )
  {
    xl::xout[ "iteration" ][ "4a:||Gradient||" ] << "---";
    xl::xout[ "iteration" ][ "4b:||SearchDirection||" ] << "---";
  }
  else
  {
    xl::xout[ "iteration" ][ "4a:||Gradient||" ] << this->GetGradient().magnitude();
    xl::xout[ "iteration" ][ "4b:||SearchDirection||" ] << this->GetSearchDirection().magnitude();
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
PreconditionedStochasticGradientDescent< TElastix >
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
PreconditionedStochasticGradientDescent< TElastix >
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
PreconditionedStochasticGradientDescent< TElastix >
::StartOptimization( void )
{
  /** As this optimizer estimates the scales itself, no other scales are used. */
  this->SetUseScales( false );

  this->m_AutomaticParameterEstimationDone = false;
  this->Superclass1::StartOptimization();

} // end StartOptimization()


/**
 * ********************** AdvanceOneStep **********************
 */

template <class TElastix>
void
PreconditionedStochasticGradientDescent< TElastix >
::AdvanceOneStep( void )
{
  /** Get space dimension. */
  const unsigned int spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Compute and set the learning rate. */
  const double lamda = this->GetParam_a() / ( 1.0 + this->Superclass1::GetCurrentTime() / this->GetParam_A() );
  this->SetLearningRate( lamda );

  DerivativeType & searchDirection = this->m_SearchDirection;

  /** Get a reference to the previously allocated newPosition. */
  ParametersType & newPosition = this->m_ScaledCurrentPosition;

  /** Get a reference to the current position. */
  const ParametersType & currentPosition = this->GetScaledCurrentPosition();

  /** Update the new position. */
  const double lamda2 = lamda * this->m_NoiseFactor;
  for( unsigned int j = 0; j < spaceDimension; ++j )
  {
    searchDirection[ j ] = this->m_PreconditionVector[ j ] * this->m_Gradient[ j ];
    newPosition[ j ] = currentPosition[ j ] - lamda2 * searchDirection[ j ];
  }

  this->Superclass1::UpdateCurrentTime();
  this->InvokeEvent( itk::IterationEvent() );

} // end AdvanceOneStep()


/**
 * ********************** ResumeOptimization **********************
 */

template< class TElastix >
void
PreconditionedStochasticGradientDescent< TElastix >
::ResumeOptimization( void )
{
  /** The following code relies on the fact that all components have been set up and
   * that the initial position has been set, so must be called in this function.
   */
  if( this->GetAutomaticParameterEstimation()
    && !this->m_AutomaticParameterEstimationDone )
  {
    this->AutomaticPreconditionerEstimation();
    this->m_AutomaticParameterEstimationDone = true; // hack
  }

  this->Superclass1::ResumeOptimization();

} // end ResumeOptimization()


/**
 * ****************** MetricErrorResponse *************************
 */

template< class TElastix >
void
PreconditionedStochasticGradientDescent< TElastix >
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
    /** Stop optimization and pass on exception. */
    this->Superclass1::MetricErrorResponse( err );
  }

} // end MetricErrorResponse()


/**
 * ******************* AutomaticPreconditionerEstimation **********************
 */

template< class TElastix >
void
PreconditionedStochasticGradientDescent< TElastix >
::AutomaticPreconditionerEstimation( void )
{
  /** Total time. */
  itk::TimeProbe timer;
  timer.Start();
  elxout << "Starting preconditioner estimation for "
         << this->elxGetClassName() << " ..." << std::endl;

  /** Get current position to start the parameter estimation. */
  this->GetRegistration()->GetAsITKBaseType()->GetTransform()->SetParameters(
    this->GetCurrentPosition() );

  /** Get the number of parameters. */
  unsigned int P = static_cast< unsigned int >(
    this->GetRegistration()->GetAsITKBaseType()->GetTransform()->GetNumberOfParameters() );

  this->m_SearchDirection = ParametersType( P );
  this->m_SearchDirection.Fill( 0.0 );// if the print out is not needed, this could be removed. YQ
  /** Get the current resolution level. */
  unsigned int level = static_cast< unsigned int >(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  /** Cast to advanced metric type. */
  typedef typename ElastixType::MetricBaseType::AdvancedMetricType MetricType;
  MetricType * testPtr = dynamic_cast< MetricType * >(
    this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType() );
  if( !testPtr )
  {
    itkExceptionMacro( << "ERROR: PreconditionedStochasticGradientDescent expects "
                       << "the metric to be of type AdvancedImageToImageMetric!" );
  }

  /** Getting pointers to the samplers. */
  const unsigned int M = this->GetElastix()->GetNumberOfMetrics();
  std::vector< ImageSamplerBasePointer >  originalSampler( M );
  for( unsigned int m = 0; m < M; ++m )
  {
    ImageSamplerBasePointer sampler =
      this->GetElastix()->GetElxMetricBase(m)->GetAdvancedMetricImageSampler();
    originalSampler[ m ] = dynamic_cast< ImageSamplerBaseType * >( sampler.GetPointer() );
  }

  /** Create a random sampler with more samples that can be used for the pre-conditioner computation. */
  //std::vector< ImageRandomCoordinateSamplerPointer > preconditionSamplers( M, 0 ); // very slow, leave this for reminder. YQ
  std::vector< ImageRandomSamplerPointer > preconditionSamplers( M );
  for( unsigned int m = 0; m < M; ++m )
  {
    ImageSamplerBasePointer sampler =
      this->GetElastix()->GetElxMetricBase( m )->GetAdvancedMetricImageSampler();
    //preconditionSamplers[ m ] = ImageRandomCoordinateSamplerType::New();
    preconditionSamplers[ m ] = ImageRandomSamplerType::New();
    preconditionSamplers[ m ]->SetInput( sampler->GetInput() );
    preconditionSamplers[ m ]->SetInputImageRegion( sampler->GetInputImageRegion() );
    preconditionSamplers[ m ]->SetMask( sampler->GetMask() );
    preconditionSamplers[ m ]->SetNumberOfSamples( this->m_NumberOfSamplesForPrecondition );
    preconditionSamplers[ m ]->Update();
    this->GetElastix()->GetElxMetricBase( m )
      ->SetAdvancedMetricImageSampler( preconditionSamplers[ m ] );
  }

  /** Construct preconditionerEstimator to initialize the preconditioner estimation. */
  PreconditionerEstimationPointer preconditionerEstimator = PreconditionerEstimationType::New();
  preconditionerEstimator->SetFixedImage( testPtr->GetFixedImage() );
  preconditionerEstimator->SetFixedImageRegion( testPtr->GetFixedImageRegion() );
  preconditionerEstimator->SetFixedImageMask( testPtr->GetFixedImageMask() );
  preconditionerEstimator->SetTransform(
    this->GetRegistration()->GetAsITKBaseType()->GetTransform() );
  preconditionerEstimator->SetCostFunction( this->m_CostFunction );
  preconditionerEstimator->SetNumberOfJacobianMeasurements(
    this->m_NumberOfJacobianMeasurements );
  preconditionerEstimator->SetRegularizationKappa( this->m_RegularizationKappa );
  preconditionerEstimator->SetMaximumStepLength( this->m_MaximumStepLength );
  preconditionerEstimator->SetConditionNumber( this->m_ConditionNumber );
  preconditionerEstimator->SetUseScales( false ); // Make sure scales are not used

  /** Construct the preconditioner and initialize. */
  this->m_PreconditionVector = ParametersType( P );
  this->m_PreconditionVector.Fill( 0.0 );

  /** Compute the preconditioner. */
  itk::TimeProbe timer_P; timer_P.Start();
  elxout << "  Computing preconditioner ..." << std::endl;
  double maxJJ = 0; // needed for the noise compensation term

  bool useJacobiType = false;
  this->GetConfiguration()->ReadParameter( useJacobiType,
    "JacobiTypePreconditioner", this->GetComponentLabel(), level, 0 );

  if( useJacobiType )
  {
    preconditionerEstimator->ComputeJacobiTypePreconditioner( this->GetScaledCurrentPosition(),
      maxJJ, this->m_PreconditionVector );
  }
  else
  {
    preconditionerEstimator->Compute( this->GetScaledCurrentPosition(),
      maxJJ, this->m_PreconditionVector );
  }

  timer_P.Stop();
  elxout << "  Computing the preconditioner took "
    << this->ConvertSecondsToDHMS( timer_P.GetMean(), 6 )
    << std::endl;

#if 0
  elxout << std::scientific;
  elxout << "The preconditioner: [ ";
  for( unsigned int i = 0; i < P; ++i ) elxout << m_PreconditionVector[ i ] << " ";
  elxout << "]" <<  std::endl;
  elxout << std::fixed;
#endif

  /** Set the sampler back to the original. */
  for( unsigned int m = 0; m < M; ++m )
  {
    this->GetElastix()->GetElxMetricBase( m )->SetAdvancedMetricImageSampler( originalSampler[ m ] );
  }

  /** This part is for PSGD-Jacobian type preconditioner, automatic etimation of the step size. */
  double      jacg = 0.0;
  if( useJacobiType )
  {
    itk::TimeProbe timer4;
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

    std::string maximumDisplacementEstimationMethod = "2sigma";
    this->GetConfiguration()->ReadParameter(maximumDisplacementEstimationMethod,
      "MaximumDisplacementEstimationMethod", this->GetComponentLabel(), 0, 0 );

    /** Compute the Jacobian terms. */
    elxout << "  Computing displacement distribution ..." << std::endl;
    timer4.Start();
    computeDisplacementDistribution->Compute(
      this->GetScaledCurrentPosition(), jacg, maxJJ,
      maximumDisplacementEstimationMethod );
    timer4.Stop();
    elxout << "  Computing the displacement distribution took "
      << this->ConvertSecondsToDHMS( timer4.GetMean(), 6 ) << std::endl;
  }

  /** Sample the fixed image to estimate the noise factor. */
  itk::TimeProbe timer_noise; timer_noise.Start();
  double sigma4factor = 1.0;
  double sigma4       = 0.0;
  elxout << "  The estimated MaxJJ is: " << maxJJ << std::endl;
  if( maxJJ > 1e-14 )
  {
    sigma4 = sigma4factor * this->m_MaximumStepLength / std::sqrt( maxJJ );
  }
  double gg           = 0.0;
  double ee           = 0.0;
  this->SampleGradients( this->GetScaledCurrentPosition(), sigma4, gg, ee );
  this->m_NoiseFactor = gg / ( gg + ee + 1e-14 );
  timer_noise.Stop();
  elxout << "  The MaxJJ used for noisefactor is: " << maxJJ << std::endl;
  elxout << "  The NoiseFactor is: " << m_NoiseFactor << std::endl;
  elxout << "  Compute the noise compensation took "
         << this->ConvertSecondsToDHMS( timer_noise.GetMean(), 6 )
         << std::endl;

  // MS: the following can probably be removed or moved.
  // YQ: these variables are used to update the time for adaptive step size.
  // See in itkPreconditionedASGDOptimizer.cxx
  /** Initial of the variables. */
  const double alpha = 1.0;
  const double fmax = 1.0;
  const double fmin = -0.8;
  /** Set parameters in superclass. */
  this->SetParam_alpha( alpha );
  this->SetSigmoidMax( fmax );
  this->SetSigmoidMin( fmin );

  /** Initial of the variables. */
  double       a = 1.0;
  const double A = this->GetParam_A();
  const double delta = this->GetMaximumStepLength();

  if( useJacobiType )
  {
    a = delta * std::pow( A + 1.0, alpha ) / ( jacg + 1e-14 );
  }
  this->SetParam_a( a );

  /** Print the elapsed time. */
  timer.Stop();
  elxout << "Automatic preconditioner estimation took "
    << this->ConvertSecondsToDHMS( timer.GetMean(), 2 )
    << std::endl;

} // end AutomaticPreconditionerEstimation()


/**
 * ******************** SampleGradients **********************
 */

template< class TElastix >
void
PreconditionedStochasticGradientDescent< TElastix >
::SampleGradients( const ParametersType & mu0,
  double perturbationSigma, double & gg, double & ee )
{
  /** Some shortcuts. */
  const unsigned int M = this->GetElastix()->GetNumberOfMetrics();

  /** Variables for sampler support. Each metric may have a sampler. */
  std::vector< bool >                                useRandomSampleRegionVec( M, false );
  std::vector< ImageRandomSamplerBasePointer >       randomSamplerVec( M );
  std::vector< ImageRandomCoordinateSamplerPointer > randomCoordinateSamplerVec( M );
  std::vector< ImageGridSamplerPointer >             gridSamplerVec( M );

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
            if ( this->m_StepSizeStrategy == "Adaptive" )
            {
              xl::xout[ "warning" ]
                << "WARNING: StepSizeStrategy is set to Constant, "
                << "because UseRandomSampleRegion is set to \"true\"."
                << std::endl;
              this->m_StepSizeStrategy = "Constant";
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
        gridSamplerVec[ m ]->SetNumberOfSamples( this->m_NumberOfSamplesForNoiseCompensationFactor);
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
  DerivativeType searchDirection( P );
  DerivativeType diffgradient;
  double         exactgg = 0.0;
  double         diffgg  = 0.0;
  double         approxgg  = 0.0;

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

      for( unsigned int i = 0; i < P; ++i )
      {
        searchDirection[ i ] = exactgradient[ i ] * this->m_PreconditionVector[ i ];
      }
      exactgg += inner_product(searchDirection, exactgradient);

      /** Set random sampler(s), select new spatial samples and get approximate derivative. */
      for( unsigned int m = 0; m < M; ++m )
      {
        if( randomSamplerVec[ m ].IsNotNull() )
        {
          this->GetElastix()->GetElxMetricBase( m )
            ->SetAdvancedMetricImageSampler( randomSamplerVec[ m ] );
        }
      }
      this->SelectNewSamples();
      this->GetScaledDerivativeWithExceptionHandling( perturbedMu0, approxgradient );

      /** Compute error vector. */
      diffgradient = exactgradient - approxgradient;
      for( unsigned int i = 0; i < P; ++i )
      {
        searchDirection[ i ] = diffgradient[ i ] * this->m_PreconditionVector[ i ];
      }
      approxgg = inner_product( searchDirection, diffgradient );
      diffgg += approxgg;
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
  gg = vnl_math_abs( exactgg );
  ee = vnl_math_abs( diffgg );

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
PreconditionedStochasticGradientDescent< TElastix >
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
 */

template< class TElastix >
void
PreconditionedStochasticGradientDescent< TElastix >
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
 */

template< class TElastix >
void
PreconditionedStochasticGradientDescent< TElastix >
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
 */

template< class TElastix >
void
PreconditionedStochasticGradientDescent< TElastix >
::AddRandomPerturbation( ParametersType & parameters, double sigma )
{
  /** Add delta ~ sigma * N(0,I) to the input parameters. */
  for( unsigned int p = 0; p < parameters.GetSize(); ++p )
  {
    parameters[ p ] += sigma * this->m_RandomGenerator->GetNormalVariate( 0.0, 1.0 );
  }

} // end AddRandomPerturbation()


} // end namespace elastix

#endif // end #ifndef __elxPreconditionedStochasticGradientDescent_hxx
