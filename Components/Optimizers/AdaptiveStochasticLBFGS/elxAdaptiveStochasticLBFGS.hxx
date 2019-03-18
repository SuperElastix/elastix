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
#ifndef __elxAdaptiveStochasticLBFGS_hxx
#define __elxAdaptiveStochasticLBFGS_hxx

#include "elxAdaptiveStochasticLBFGS.h"

#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <utility>
#include "itkAdvancedImageToImageMetric.h"
#include "itkTimeProbe.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

#ifdef ELASTIX_USE_EIGEN
#include <Eigen/Dense>
#include <Eigen/Core>
#endif


namespace elastix
{

/**
 * ********************** Constructor ***********************
 */

template <class TElastix>
AdaptiveStochasticLBFGS<TElastix>
::AdaptiveStochasticLBFGS()
{
  this->m_MaximumNumberOfSamplingAttempts = 0;
  this->m_CurrentNumberOfSamplingAttempts = 0;
  this->m_PreviousErrorAtIteration = 0;
  this->m_AutomaticParameterEstimationDone = false;

  this->m_AutomaticParameterEstimation = false;
  this->m_AutomaticLBFGSStepsizeEstimation = false;
  this->m_MaximumStepLength = 1.0;

  this->m_NumberOfGradientMeasurements = 0;
  this->m_NumberOfJacobianMeasurements = 0;
  this->m_NumberOfSamplesForExactGradient = 100000;
  this->m_NumberOfSpatialSamples = 5000;
  this->m_NumberOfInnerLoopSamples = 10;
  this->m_SigmoidScaleFactor = 0.1;
  this->m_NoiseFactor =0.8;

  this->m_LBFGSMemory = 10;
  this->m_OutsideIterations = 10;

  this->m_CurrentT  = 0;
  this->m_PreviousT = 0;
  this->m_Bound     = 0;
  this->m_GradientMagnitudeTolerance = 0.000001;
  this->m_WindowScale = 5;

  this->m_RandomGenerator = RandomGeneratorType::GetInstance();
  this->m_AdvancedTransform = 0;

  this->m_UseNoiseCompensation            = true;
  this->m_OriginalButSigmoidToDefault     = false;
  this->m_UseAdaptiveLBFGSStepSizes       = false;
  this->m_UseSearchDirForAdaptiveStepSize = false;

  //this->m_LearningRate = 1.0;
} // end Constructor


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::BeforeRegistration( void )
{
  /** Add the target cell "stepsize" to xout["iteration"]. */
  xout["iteration"].AddTargetCell("2:Metric");
  xout["iteration"].AddTargetCell("3a:Time");
  xout["iteration"].AddTargetCell("3b:StepSize");
  xout[ "iteration" ].AddTargetCell( "4a:||Gradient||" );
  xout[ "iteration" ].AddTargetCell( "4b:||SearchDir||" );

  /** Format the metric and stepsize as floats. */
  xl::xout["iteration"]["2:Metric"]         << std::showpoint << std::fixed;
  xl::xout["iteration"]["3a:Time"]          << std::showpoint << std::fixed;
  xl::xout["iteration"]["3b:StepSize"]      << std::showpoint << std::fixed;
  xl::xout["iteration"]["4a:||Gradient||"]  << std::showpoint << std::fixed;
  xl::xout["iteration"]["4b:||SearchDir||"] << std::showpoint << std::fixed;

  this->m_SettingsVector.clear();

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void AdaptiveStochasticLBFGS<TElastix>
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level = static_cast<unsigned int>(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  const unsigned int P = this->GetElastix()->GetElxTransformBase()
    ->GetAsITKBaseType()->GetNumberOfParameters();

  /** Set the LBFGSMemory. */
  SizeValueType memory = 5;
  this->GetConfiguration()->ReadParameter( memory,
    "LBFGSMemory", this->GetComponentLabel(), level, 0 );
  this->m_LBFGSMemory = memory;

  /** Set the updateFrequenceL. */
  SizeValueType updateFrequenceL = 5;
  this->GetConfiguration()->ReadParameter( updateFrequenceL,
    "UpdateFrequenceL", this->GetComponentLabel(), level, 0 );
  this->m_UpdateFrequenceL = updateFrequenceL;

  /** Set the maximumNumberOfIterations. */
  SizeValueType maximumNumberOfIterations = 500;
  this->GetConfiguration()->ReadParameter( maximumNumberOfIterations,
    "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
  this->m_OutsideIterations = maximumNumberOfIterations;
  this->SetNumberOfIterations( this->m_OutsideIterations );

  /** Set the numberOfInnerLoopSamples. */
  SizeValueType numberOfInnerLoopSamples = 10;
  this->GetConfiguration()->ReadParameter( numberOfInnerLoopSamples,
    "NumberOfInnerLoopSamples", this->GetComponentLabel(), level, 0 );
  this->m_NumberOfInnerLoopSamples = numberOfInnerLoopSamples;

  /** Set the NumberOfSpatialSamples. */
  unsigned long numberOfSpatialSamples = 5000;
  this->GetConfiguration()->ReadParameter( numberOfSpatialSamples,
    "NumberOfSpatialSamples", this->GetComponentLabel(), level, 0 );
  this->m_NumberOfSpatialSamples = numberOfSpatialSamples;

  /** Set the gain parameter A. */
  double A = 20.0;
  this->GetConfiguration()->ReadParameter( A,
    "SP_A", this->GetComponentLabel(), level, 0 );
  this->SetParam_A( A );

  /** Set the gain parameter beta. */
  double beta = 20.0;
  this->GetConfiguration()->ReadParameter( beta,
    "SP_beta", this->GetComponentLabel(), level, 0 );
  this->SetParam_beta( beta );

  /** Set the MaximumNumberOfSamplingAttempts. */
  SizeValueType maximumNumberOfSamplingAttempts = 0;
  this->GetConfiguration()->ReadParameter( maximumNumberOfSamplingAttempts,
    "MaximumNumberOfSamplingAttempts", this->GetComponentLabel(), level, 0 );
  this->SetMaximumNumberOfSamplingAttempts( maximumNumberOfSamplingAttempts );
  if ( maximumNumberOfSamplingAttempts > 5 )
  {
    elxout["warning"]
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

  /** Set whether the adaptive LBFGS step size mechanism is desired. Default: true */
  bool useAdaptiveLBFGSStepSizes = true;
  this->GetConfiguration()->ReadParameter( useAdaptiveLBFGSStepSizes,
    "UseAdaptiveLBFGSStepSizes", this->GetComponentLabel(), level, 0 );
  this->m_UseAdaptiveLBFGSStepSizes = useAdaptiveLBFGSStepSizes;

  /** Set whether automatic gain estimation is required; default: true. */
  this->m_AutomaticParameterEstimation = true;
  this->GetConfiguration()->ReadParameter( this->m_AutomaticParameterEstimation,
    "AutomaticParameterEstimation", this->GetComponentLabel(), level, 0 );

  /** Set whether automatic gain estimation is required; default: true. */
  this->m_AutomaticLBFGSStepsizeEstimation = true;
  this->GetConfiguration()->ReadParameter( this->m_AutomaticLBFGSStepsizeEstimation,
    "AutomaticLBFGSstepsizeEstimation", this->GetComponentLabel(), level, 0 );

  /** Set the GradientMagnitudeTolerance *
  double gradientMagnitudeTolerance = 0.000001;
  this->m_Configuration->ReadParameter( gradientMagnitudeTolerance,
    "GradientMagnitudeTolerance", this->GetComponentLabel(), level, 0 );
  this->SetGradientMagnitudeTolerance( gradientMagnitudeTolerance );

  /** Set the scale of the windowa for H0. */
  double windowScale = 5;
  this->m_Configuration->ReadParameter( windowScale,
    "WindowScale", this->GetComponentLabel(), level, 0 );
  this->m_WindowScale = windowScale;

  std::string stepSizeStrategy = "Adaptive";
  this->GetConfiguration()->ReadParameter( stepSizeStrategy,
    "StepSizeStrategy", this->GetComponentLabel(), 0, 0 );
  this->m_StepSizeStrategy = stepSizeStrategy;

  if ( this->m_AutomaticParameterEstimation )
  {
    /** Set the maximum step length: the maximum displacement of a voxel in mm.
     * Compute default value: mean in-plane spacing of fixed and moving image.
     */
    const unsigned int fixdim = vnl_math_min( (unsigned int) this->GetElastix()->FixedDimension, (unsigned int) 2 );
    const unsigned int movdim = vnl_math_min( (unsigned int) this->GetElastix()->MovingDimension, (unsigned int) 2 );
    double             sum    = 0.0;
    for (unsigned int d = 0; d < fixdim; ++d )
    {
      sum += this->GetElastix()->GetFixedImage()->GetSpacing()[d];
    }
    for (unsigned int d = 0; d < movdim; ++d )
    {
      sum += this->GetElastix()->GetMovingImage()->GetSpacing()[d];
    }
    this->m_MaximumStepLength = sum / static_cast<double>( fixdim + movdim );

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
      static_cast<unsigned int>(1000), static_cast<unsigned int>(P) );
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

    /** Set/Get the scaling of the sigmoid width. Large values
     * cause a more wide sigmoid. Default: 1e-8. Should be >0.
     */
    double sigmoidScale = 1e-8;
    this->GetConfiguration()->ReadParameter( sigmoidScale,
      "SigmoidScale", this->GetComponentLabel(), level, 0 );
    this->SetSigmoidScale( sigmoidScale );

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

  } // end if automatic parameter estimation
  else
  {
    /** If no automatic parameter estimation is used, a and alpha also need
     * to be specified.
     */
    double a = 400.0; // arbitrary guess
    double alpha = 0.602;
    this->GetConfiguration()->ReadParameter(a, "SP_a", this->GetComponentLabel(), level, 0 );
    this->GetConfiguration()->ReadParameter(alpha, "SP_alpha", this->GetComponentLabel(), level, 0 );
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

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::AfterEachIteration( void )
{
  /** Print some information. */
  xl::xout["iteration"]["2:Metric"]          << this->GetValue();
  xl::xout["iteration"]["3a:Time"]           << this->GetCurrentTime();
  xl::xout["iteration"]["3b:StepSize"]       << this->GetLearningRate();
  xl::xout["iteration"]["4a:||Gradient||"]   << this->GetGradient().magnitude();
  xl::xout["iteration"]["4b:||SearchDir||"]  << this->m_SearchDir.magnitude();

  /** Select new spatial samples for the computation of the metric. */
  if( this->GetNewSamplesEveryIteration() )
  {
    this->SelectNewSamples();
  }

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution *************************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::AfterEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level = static_cast<unsigned int>(
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
  case MetricError :
    stopcondition = "Error in metric";
    break;

  case MaximumNumberOfIterations :
    stopcondition = "Maximum number of iterations has been reached";
    break;

  case InvalidDiagonalMatrix :
    stopcondition = "The InvalidDiagonalMatrix";
    break;

  case GradientMagnitudeTolerance :
    stopcondition = "The gradient magnitude has (nearly) vanished";
    break;

  case MinimumStepSize:
    stopcondition = "The last step size was (nearly) zero";
    break;

  default:
    stopcondition = "Unknown";
    break;
  }

  /** Print the stopping condition. */
  elxout << "Stopping condition: " << stopcondition << "." << std::endl;
  this->m_CurrentTime = 0.0;

  /** Store the used parameters, for later printing to screen. */
  SettingsType settings;
  settings.a = this->GetParam_a();
  settings.A = this->GetParam_A();
  settings.alpha = this->GetParam_alpha();
  settings.fmax = this->GetSigmoidMax();
  settings.fmin = this->GetSigmoidMin();
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

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
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

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::StartOptimization( void )
{
  /** Reset some variables */
  this->m_CurrentT  = 0;
  this->m_PreviousT = 0;
  this->m_Bound     = 0;

  /** Check if the entered scales are correct and != [ 1 1 1 ...]. */
  this->SetUseScales( false );
  const ScalesType & scales = this->GetScales();
  if ( scales.GetSize() == this->GetInitialPosition().GetSize() )
  {
    ScalesType unit_scales( scales.GetSize() );
    unit_scales.Fill(1.0);
    if ( scales != unit_scales )
    {
      /** only then: */
      this->SetUseScales( true );
    }
  }

  this->m_AutomaticParameterEstimationDone = false;

  itkDebugMacro("StartOptimization");

  this->m_CurrentIteration   = 0;

  /** Get the number of parameters; checks also if a cost function has been set at all.
   * if not: an exception is thrown.
   */
  this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Resize Rho, S and Y. */
  this->m_Rho.SetSize( this->m_LBFGSMemory );
  this->m_HessianFillValue.SetSize( this->m_LBFGSMemory );
  this->m_HessianFillValue.fill( 0.0 );
  this->m_S.resize( this->m_LBFGSMemory );
  this->m_Y.resize( this->m_LBFGSMemory );

  /** Initialize the scaledCostFunction with the currently set scales */
  this->InitializeScales();

  /** Set the current position as the scaled initial position */
  this->SetCurrentPosition( this->GetInitialPosition() );

  /** Reset the current time to initial time. */
  this->ResetCurrentTimeToInitialTime();

  this->ResumeOptimization();

} // end StartOptimization()


/**
 * ********************** LBFGSUpdate **********************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::LBFGSUpdate( void )
{
  itkDebugMacro( "LBFGSUpdate" );

  /** Get space dimension. */
  const unsigned int spaceDimension
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Get a reference to the previously allocated newPosition. */
  ParametersType & newPosition = this->m_ScaledCurrentPosition;

  /** Get a reference to the current position. */
  const ParametersType & currentPosition = this->GetScaledCurrentPosition();

  /** Update the new position. */
  const double learningRate = this->GetLearningRate();
  for( unsigned int j = 0; j < spaceDimension; ++j )
  {
    newPosition[ j ] = currentPosition[ j ] + learningRate * this->m_SearchDir[ j ];
  }

  this->InvokeEvent( itk::IterationEvent() );
} // end LBFGSUpdate()


/**
 * ********************** AdvanceOneStep **********************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::AdvanceOneStep( void )
{
  itkDebugMacro( "AdvanceOneStep" );

  /** Get space dimension. */
  const unsigned int spaceDimension
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Get a reference to the previously allocated newPosition. */
  ParametersType & newPosition = this->m_ScaledCurrentPosition;

  /** Get a reference to the current position. */
  const ParametersType & currentPosition = this->GetScaledCurrentPosition();

  /** Update the new position. */
  const double learningRate = this->GetLearningRate();
  for( unsigned int j = 0; j < spaceDimension; ++j )
  {
    newPosition[ j ] = currentPosition[ j ] - learningRate * this->m_Gradient[ j ];
  }

  this->InvokeEvent( itk::IterationEvent() );

} // end AdvanceOneStep()


/**
 * ********************** StopOptimization **********************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::StopOptimization( void )
{
  itkDebugMacro( "StopOptimization" );

  this->m_Stop = true;
  this->InvokeEvent( itk::EndEvent() );
} // end StopOptimization()


/**
 * ********************** ResumeOptimization **********************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::ResumeOptimization( void )
{
  /** The following code relies on the fact that all
   * components have been set up and that the initial
   * position has been set, so must be called in this
   * function.
   */
  if( this->GetAutomaticParameterEstimation()
    && !this->m_AutomaticParameterEstimationDone )
  {
    this->AutomaticParameterEstimation();
    // hack
    this->m_AutomaticParameterEstimationDone = true;
  }

  itkDebugMacro( "ResumeOptimization" );

  this->m_Stop = false;

  InvokeEvent( itk::StartEvent() );

  /** Set the NumberOfSpatialSamples. */
  SizeValueType numberOfSpatialSamples = this->m_NumberOfSpatialSamples;

  SizeValueType spaceDimension
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  this->m_Gradient  = DerivativeType( spaceDimension );   // check this
  this->m_Gradient.fill(0);
  this->m_SearchDir = DerivativeType( spaceDimension );   // check this
  this->m_SearchDir.fill(0);
  DerivativeType meanCurrentCurvatureGradient( spaceDimension );
  DerivativeType previousCurvatureGradient( spaceDimension );

  ParametersType previousCurvaturePosition = this->GetScaledCurrentPosition();
  ParametersType meanCurrentCurvaturePosition;

  ParametersType s;
  DerivativeType y;

  /** Getting pointers to the samplers. */
  const unsigned int M = this->GetElastix()->GetNumberOfMetrics();
  std::vector< ImageSamplerBasePointer >  originalSampler( M );
  for( unsigned int m = 0; m < M; ++m )
  {
    ImageSamplerBasePointer sampler =
      this->GetElastix()->GetElxMetricBase(m)->GetAdvancedMetricImageSampler();
    originalSampler[ m ] = dynamic_cast< ImageSamplerBaseType * >( sampler.GetPointer() );
  }

  /** Get the sampler that is used for the curvature pair update. */
  std::string curvatureSamplerType = "Random";
  this->GetConfiguration()->ReadParameter( curvatureSamplerType,
    "CurvatureSampler", this->GetComponentLabel(), 0, 0 );

  /** Create some samplers that can be used for the curvature computation. */
  std::vector< ImageSamplerBasePointer > curvatureSamplers( M );
  std::vector< ImageRandomSamplerPointer > randomSamplerVec( M );
  std::vector< ImageRandomCoordinateSamplerPointer > randomCoordinateSamplerVec( M );
  std::vector< ImageGridSamplerPointer > gridSamplerVec( M );
  for( unsigned int m = 0; m < M; ++m )
  {
    ImageSamplerBasePointer sampler =
      this->GetElastix()->GetElxMetricBase(m)->GetAdvancedMetricImageSampler();
    curvatureSamplers[ m ] = dynamic_cast< ImageSamplerBaseType * >( sampler.GetPointer() );
    curvatureSamplers[ m ]-> SetNumberOfSamples( this->m_NumberOfInnerLoopSamples );
  }

  /** Loop over the iterations. */
  while( !this->m_Stop )
  {
    /** Every iteration we update the mean position over a block of L iterations
     * with the previous position.
     */
    const ParametersType & previousPosition = this->GetScaledCurrentPosition();
    if( this->m_CurrentIteration % this->m_UpdateFrequenceL == 0 )
    {
      meanCurrentCurvaturePosition = previousPosition;
    }
    else
    {
      meanCurrentCurvaturePosition += previousPosition;
    }

    /** Compute the stochastic gradient. */
    this->GetScaledValueAndDerivative(
      this->GetScaledCurrentPosition(), this->m_Value, this->m_Gradient );

    /** Main optimization updates. */
    if( this->m_CurrentIteration < 2 * this->m_UpdateFrequenceL )
    {
      /** The first 2L iterations we use the ASGD update scheme. */

      /** We compute the learning rate lambda. */
      this->SetLearningRate( this->Superclass1::Compute_a( this->Superclass1::GetCurrentTime() ) );

      /** Perform gradient descent. */
      this->AdvanceOneStep();

      /** Update the time. */
      this->Superclass1::UpdateCurrentTime();
    }
    else if( m_CurrentIteration < m_NumberOfIterations )
    {
      /** After the first 2L iterations we proceed with stochastic LBFGS updates. */

      /** Compute the new search direction, using the current stochastic gradient.
       * We use the previously computed curvature variables to get a Hessian scaled
       * search direction.
       */
      this->ComputeSearchDirection( this->m_Gradient, this->m_SearchDir );

      /** Step size determination part 1: automatically estimate the initial step size. */
      if( this->GetAutomaticLBFGSStepsizeEstimation()
        && (this->m_CurrentIteration + 0) % this->m_UpdateFrequenceL == 0 )
        //&& (this->m_CurrentIteration  == this->m_UpdateFrequenceL * 2) )
      {
        // This takes a significant amount of time!!
        itk::TimeProbe timer_ape; timer_ape.Start();
        this->AutomaticLBFGSStepsizeEstimation();
        timer_ape.Stop();
        elxout << "ape: " << timer_ape.GetMean() * 1000.0 << " ms" << std::endl;

        // Currently not used, so outcommented
        //this->m_SearchLengthScale = this->m_SearchDir.magnitude();

        /** Get the current resolution level. */
        unsigned int level = static_cast<unsigned int>(
          this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

        /** Set whether the Search Direction used for adaptive step size mechanism is desired. Default: false */
        bool useSearchDirForAdaptiveStepSize = false;
        this->GetConfiguration()->ReadParameter( useSearchDirForAdaptiveStepSize,
          "UseSearchDirForAdaptiveStepSize", this->GetComponentLabel(), level, 0, false );
        this->SetUseSearchDirForAdaptiveStepSize( useSearchDirForAdaptiveStepSize );

        /** Reset the current time to initial time. */
        this->ResetCurrentTimeToInitialTime();

      } // end if AutomaticLBFGSstepsizeEstimation

      /** Step size determination part 2: use an adaptive scheme for time or not. */
//      if( this->m_UseAdaptiveLBFGSStepSizes )
//      {
        this->SetLearningRate( this->Superclass1::Compute_beta( this->Superclass1::GetCurrentTime() ) );
        this->Superclass1::UpdateCurrentTime();
//      }
//       else
//       {
//         this->SetLearningRate( this->Superclass1::Compute_beta( this->m_CurrentIteration ) );
//       }

      /** Update the position using the quasi-Newton method. */
      this->LBFGSUpdate();

    } // end if update rule

    /**
     * CURVATURE FOR STOCHASTIC LBFGS
     *
     * Every L iterations we compute the curvature pairs and store them into M-size memory.
     */

    if( (this->m_CurrentIteration + 1) % this->m_UpdateFrequenceL == 0 )
    {
      /** Compute the mean curvature position. */
      meanCurrentCurvaturePosition /= this->m_UpdateFrequenceL;

      /** Select a new sampler for the curvature update. */
      for( unsigned int m = 0; m < M; ++m )
      {
        curvatureSamplers[ m ]->Update();
        this->GetElastix()->GetElxMetricBase( m )
          ->SetAdvancedMetricImageSampler( curvatureSamplers[ m ] );
      }

      /** Get current and previous curvature gradients. */
      itk::TimeProbe timer_gvad; timer_gvad.Start();
      this->GetScaledDerivativeWithExceptionHandling( meanCurrentCurvaturePosition, meanCurrentCurvatureGradient );
      this->GetScaledDerivativeWithExceptionHandling( previousCurvaturePosition, previousCurvatureGradient );
      timer_gvad.Stop();
      elxout << "gvad: " << timer_gvad.GetMean() * 1000.0 << " ms" << std::endl;

      /** Set the sampler back to the original. */
      for( unsigned int m = 0; m < M; ++m )
      {
        this->GetElastix()->GetElxMetricBase( m )->SetAdvancedMetricImageSampler( originalSampler[ m ] );
      }

      /** Compute s and y and store them. */
      s = meanCurrentCurvaturePosition - previousCurvaturePosition;
      y = meanCurrentCurvatureGradient - previousCurvatureGradient;
      this->StoreCurrentPoint( s, y );

      /** Update previous. */
      previousCurvaturePosition = meanCurrentCurvaturePosition;

      /** Update the time. */
      this->m_PreviousT = this->m_CurrentT;
      this->m_CurrentT++;
      if( this->m_CurrentT >= this->m_LBFGSMemory ){ this->m_CurrentT = 0; }

      /** Update the bound. */
      if( this->m_Bound < this->m_LBFGSMemory ){ this->m_Bound++; }
    } // end if curvature pair update.

    /** StopOptimization may have been called. */
    if( this->m_Stop )
    {
      break;
    }

    this->m_CurrentIteration++;

    if( m_CurrentIteration >= m_NumberOfIterations )
    {
      this->m_StopCondition = MaximumNumberOfIterations;
      this->StopOptimization();
      break;
    }
  } // end while iterations

} // end ResumeOptimization()


/**
 * ****************** MetricErrorResponse *************************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::MetricErrorResponse( itk::ExceptionObject & err )
{
  if ( this->GetCurrentIteration() != this->m_PreviousErrorAtIteration )
  {
    this->m_PreviousErrorAtIteration = this->GetCurrentIteration();
    this->m_CurrentNumberOfSamplingAttempts = 1;
  }
  else
  {
    this->m_CurrentNumberOfSamplingAttempts++;
  }

  if ( this->m_CurrentNumberOfSamplingAttempts <= this->m_MaximumNumberOfSamplingAttempts )
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
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::AutomaticParameterEstimation( void )
{
  /** Setup timers. */
  itk::TimeProbe timer1;

  /** Total time. */
  timer1.Start();
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
  timer1.Stop();
  elxout << "Automatic parameter estimation took "
    << this->ConvertSecondsToDHMS( timer1.GetMean(), 6 )
    << std::endl;

} // end AutomaticParameterEstimation()


/**
 * ******************* AutomaticParameterEstimationOriginal **********************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::AutomaticParameterEstimationOriginal( void )
{
  itk::TimeProbe timer2, timer3;

  /** Get the user input. */
  const double delta = this->GetMaximumStepLength();

  /** Compute the Jacobian terms. */
  double TrC = 0.0;
  double TrCC = 0.0;
  double maxJJ = 0.0;
  double maxJCJ = 0.0;

  /** Get current position to start the parameter estimation. */
  this->GetRegistration()->GetAsITKBaseType()->GetTransform()->SetParameters(
    this->GetCurrentPosition() );

  /** Cast to advanced metric type. */
  typedef typename ElastixType::MetricBaseType::AdvancedMetricType MetricType;
  MetricType * testPtr = dynamic_cast<MetricType *>(
    this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType() );
  if( !testPtr )
  {
    itkExceptionMacro( << "ERROR: AdaptiveStochasticLBFGS expects "
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
  timer2.Start();
  computeJacobianTerms->Compute( TrC, TrCC, maxJJ, maxJCJ );
  timer2.Stop();
  elxout << "  Computing the Jacobian terms took "
    << this->ConvertSecondsToDHMS( timer2.GetMean(), 6 )
    << std::endl;

  /** Determine number of gradient measurements such that
   * E + 2\sqrt(Var) < K E
   * with
   * E = E(1/N \sum_n g_n^T g_n) = sigma_1^2 TrC
   * Var = Var(1/N \sum_n g_n^T g_n) = 2 sigma_1^4 TrCC / N
   * K = 1.5
   * We enforce a minimum of 2.
   */
  timer3.Start();
  if( this->m_NumberOfGradientMeasurements == 0 )
  {
    const double K = 1.5;
    if( TrCC > 1e-14 && TrC > 1e-14 )
    {
      this->m_NumberOfGradientMeasurements = static_cast<unsigned int>(
        std::ceil( 8.0 * TrCC / TrC / TrC / (K-1) / (K-1) ) );
    }
    else
    {
      this->m_NumberOfGradientMeasurements = 2;
    }
    this->m_NumberOfGradientMeasurements = vnl_math_max(
      static_cast<SizeValueType>( 2 ),
      this->m_NumberOfGradientMeasurements );
    elxout << "  NumberOfGradientMeasurements to estimate sigma_i: "
      << this->m_NumberOfGradientMeasurements << std::endl;
  }

  /** Measure square magnitude of exact gradient and approximation error. */
  const double sigma4factor = 1.0;
  double sigma4 = 0.0;
  double gg = 0.0;
  double ee = 0.0;
  if( maxJJ > 1e-14 )
  {
    sigma4 = sigma4factor * delta / std::sqrt( maxJJ );
  }
  this->SampleGradients(
    this->GetScaledCurrentPosition(), sigma4, gg, ee );
  timer3.Stop();
  elxout << "  Sampling the gradients took "
    << this->ConvertSecondsToDHMS( timer3.GetMean(), 6 )
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
    sigma1 = std::sqrt( gg / TrC );
  }
  if( ee > 1e-14 && TrC > 1e-14 )
  {
    sigma3 = std::sqrt( ee / TrC );
  }

  const double alpha = 1.0;
  const double A = this->GetParam_A();
  double a_max = 0.0;
  if( sigma1 > 1e-14 && maxJCJ > 1e-14 )
  {
    a_max = A * delta / sigma1 / std::sqrt( maxJCJ );
  }
  const double noisefactor = sigma1 * sigma1
    / ( sigma1 * sigma1 + sigma3 * sigma3 + 1e-14 );
  const double a = a_max * noisefactor;

  const double omega = vnl_math_max( 1e-14,
    this->m_SigmoidScaleFactor * sigma3 * sigma3 * std::sqrt( TrCC ) );
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

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::AutomaticParameterEstimationUsingDisplacementDistribution( void )
{
  itk::TimeProbe timer4 ,timer5;

  /** Get current position to start the parameter estimation. */
  this->GetRegistration()->GetAsITKBaseType()->GetTransform()->SetParameters(
    this->GetCurrentPosition() );

  /** Get the user input. */
  const double delta = this->GetMaximumStepLength();
  double maxJJ = 0;

  /** Cast to advanced metric type. */
  typedef typename ElastixType::MetricBaseType::AdvancedMetricType MetricType;
  MetricType * testPtr = dynamic_cast<MetricType *>(
    this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType() );
  if( !testPtr )
  {
    itkExceptionMacro( << "ERROR: AdaptiveStochasticLBFGS expects "
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

  double jacg = 0.0;
  std::string maximumDisplacementEstimationMethod = "2sigma";
  this->GetConfiguration()->ReadParameter( maximumDisplacementEstimationMethod,
    "MaximumDisplacementEstimationMethod", this->GetComponentLabel(), 0, 0 );

  /** Compute the Jacobian terms. */
  elxout << "  Computing displacement distribution ..." << std::endl;
  timer4.Start();
  computeDisplacementDistribution->Compute(
    this->GetScaledCurrentPosition(), jacg, maxJJ,
    maximumDisplacementEstimationMethod );
  timer4.Stop();
  elxout << "  Computing the displacement distribution took "
    << this->ConvertSecondsToDHMS( timer4.GetMean(), 6 )
    << std::endl;

  /** Initial of the variables. */
  double a = 0.0;
  const double A = this->GetParam_A();
  const double alpha = 1.0;

  this->m_UseNoiseCompensation = true;
  this->GetConfiguration()->ReadParameter( this->m_UseNoiseCompensation,
    "NoiseCompensation", this->GetComponentLabel(), 0, 0 );

  /** Use noise Compensation factor or not. */
  if( this->m_UseNoiseCompensation == true )
  {
    double sigma4 = 0.0;
    double gg = 0.0;
    double ee = 0.0;
    double sigma4factor = 1.0;

    /** Sample the grid and random sampler container to estimate the noise factor.*/
    if( this->m_NumberOfGradientMeasurements == 0 )
    {
      this->m_NumberOfGradientMeasurements = vnl_math_max(
        static_cast<SizeValueType>( 2 ),
        this->m_NumberOfGradientMeasurements );
      elxout << "  NumberOfGradientMeasurements to estimate sigma_i: "
        << this->m_NumberOfGradientMeasurements << std::endl;
    }
    timer5.Start();
    if( maxJJ > 1e-14 )
    {
      sigma4 = sigma4factor * delta / std::sqrt( maxJJ );
    }
    this ->SampleGradients( this->GetScaledCurrentPosition(), sigma4, gg, ee );

    double noisefactor = gg / ( gg + ee );
    this->m_NoiseFactor = noisefactor;
    a =  delta * std::pow( A + 1.0, alpha ) / jacg * noisefactor;
    timer5.Stop();
    elxout << "  Compute the noise compensation took "
      << this->ConvertSecondsToDHMS( timer5.GetMean(), 6 )
      << std::endl;
  }
  else
  {
    a = delta * std::pow( A + 1.0, alpha ) / jacg;
  }

  /** Set parameters in superclass. */
  this->SetParam_a( a );
  this->SetParam_alpha( alpha );

} // end AutomaticParameterEstimationUsingDisplacementDistribution()


/**
 * *************** AutomaticLBFGSStepsizeEstimation *****
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::AutomaticLBFGSStepsizeEstimation( void )
{
  /** Get current position to start the parameter estimation. */
  this->GetRegistration()->GetAsITKBaseType()->GetTransform()->SetParameters(
    this->GetCurrentPosition() );

  /** Get the user input. */
  const double delta = this->GetMaximumStepLength();
  double maxJJ = 0;

  /** Cast to advanced metric type. */
  typedef typename ElastixType::MetricBaseType::AdvancedMetricType MetricType;
  MetricType * testPtr = dynamic_cast<MetricType *>(
    this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType() );
  if( !testPtr )
  {
    itkExceptionMacro( << "ERROR: AdaptiveStochasticLBFGS expects "
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

  double jacg = 0.0;
  std::string maximumDisplacementEstimationMethod = "2sigma";
  this->GetConfiguration()->ReadParameter( maximumDisplacementEstimationMethod,
    "MaximumDisplacementEstimationMethod", this->GetComponentLabel(), 0, 0 );

  /** Compute the Jacobian terms. */
  computeDisplacementDistribution->ComputeUsingSearchDirection(
    this->m_SearchDir, jacg, maxJJ,
    maximumDisplacementEstimationMethod );

  /** Initial of the variables. */
  double a = 0.0;
  const double A = this->GetParam_A();
  const double alpha = 1.0;

  this->m_UseNoiseCompensation = true;
  this->GetConfiguration()->ReadParameter( this->m_UseNoiseCompensation,
    "NoiseCompensation", this->GetComponentLabel(), 0, 0 );

  /** Use noise Compensation factor or not. */
  if( this->m_UseNoiseCompensation == true )
  {
    /** This use the time t_k = 0. */
    a =  delta * std::pow( A + 1.0, alpha ) / jacg * this->m_NoiseFactor;
    /** Here we change the initial time to t_k = current time, so we keep a zero initial time. */
    //double t = this->Superclass1::GetCurrentTime();
    //a =  delta * std::pow( A + 1.0 + t, alpha ) / jacg * noisefactor;
  }
  else
  {
    a = delta * std::pow( A + 1.0, alpha ) / jacg;
  }

  /** Set parameters in superclass. */
  this->SetParam_beta( a );
  this->SetParam_alpha( alpha );

} // end AutomaticLBFGSStepsizeEstimation()


/**
 * ******************** SampleGradients **********************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::SampleGradients( const ParametersType & mu0,
  double perturbationSigma, double & gg, double & ee )
{
  /** Some shortcuts. */
  const unsigned int M = this->GetElastix()->GetNumberOfMetrics();

  /** Variables for sampler support. Each metric may have a sampler. */
  std::vector< bool >                                 useRandomSampleRegionVec( M, false );
  std::vector< ImageRandomSamplerBasePointer >        randomSamplerVec( M );
  std::vector< ImageRandomCoordinateSamplerPointer >  randomCoordinateSamplerVec( M );
  std::vector< ImageGridSamplerPointer >              gridSamplerVec( M );

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
      ImageSamplerBasePointer sampler =
        this->GetElastix()->GetElxMetricBase( m )->GetAdvancedMetricImageSampler();
      randomSamplerVec[m] =
        dynamic_cast< ImageRandomSamplerBaseType * >( sampler.GetPointer() );
      randomCoordinateSamplerVec[m] =
        dynamic_cast< ImageRandomCoordinateSamplerType * >( sampler.GetPointer() );

      if( randomSamplerVec[m].IsNotNull() )
      {
        /** At least one of the metric has a random sampler. */
        stochasticgradients |= true;

        /** If the sampler is a randomCoordinateSampler set the UseRandomSampleRegion
         * property to false temporarily. It disturbs the parameter estimation.
         * At the end of this function the original setting is set back.
         * Also, the AdaptiveStepSize mechanism is turned off when any of the samplers
         * has UseRandomSampleRegion==true.
         * \todo Extend ASGD to really take into account random region sampling.
         */
        if( randomCoordinateSamplerVec[ m ].IsNotNull() )
        {
          useRandomSampleRegionVec[ m ]
            = randomCoordinateSamplerVec[ m ]->GetUseRandomSampleRegion();
          if( useRandomSampleRegionVec[ m ] )
          {
            if( this->GetUseAdaptiveStepSizes() )
            {
              xl::xout["warning"]
                << "WARNING: UseAdaptiveStepSizes is turned off, "
                << "because UseRandomSampleRegion is set to \"true\"."
                << std::endl;
              this->SetUseAdaptiveStepSizes( false );
            }
          }
          randomCoordinateSamplerVec[ m ]->SetUseRandomSampleRegion( false );

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
  } // end if NewSamplesEveryIteration.

#ifndef _ELASTIX_BUILD_LIBRARY
  /** Prepare for progress printing. */
  ProgressCommandPointer progressObserver = ProgressCommandType::New();
  progressObserver->SetUpdateFrequency(
    this->m_NumberOfGradientMeasurements, this->m_NumberOfGradientMeasurements );
  progressObserver->SetStartString( "  Progress: " );
#endif
  //elxout << "  Sampling gradients ..." << std::endl;

  /** Initialize some variables for storing gradients and their magnitudes. */
  const unsigned int P = this->GetElastix()->GetElxTransformBase()
    ->GetAsITKBaseType()->GetNumberOfParameters();
  DerivativeType approxgradient( P );
  DerivativeType exactgradient( P );
  DerivativeType diffgradient;
  double exactgg = 0.0;
  double diffgg = 0.0;

  /** Compute gg for some random parameters. */
  for( unsigned int i = 0 ; i < this->m_NumberOfGradientMeasurements; ++i )
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
      diffgg += diffgradient.squared_magnitude();
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
  diffgg /= this->m_NumberOfGradientMeasurements;

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

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::PrintSettingsVector( const SettingsVectorType & settings ) const
{
  const unsigned long nrofres = settings.size();

  /** Print to log file */
  elxout << "( SP_a " ;
  for ( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].a << " ";
  }
  elxout << ")\n" ;

  elxout << "( SP_A " ;
  for ( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].A << " ";
  }
  elxout << ")\n" ;

  elxout << "( SP_alpha " ;
  for ( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[i].alpha << " ";
  }
  elxout << ")\n" ;

  elxout << "( SigmoidMax " ;
  for ( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[i].fmax << " ";
  }
  elxout << ")\n" ;

  elxout << "( SigmoidMin " ;
  for ( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[i].fmin << " ";
  }
  elxout << ")\n" ;

  elxout << "( SigmoidScale " ;
  for ( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[i].omega << " ";
  }
  elxout << ")\n" ;

  elxout << std::endl;

} // end PrintSettingsVector()


/**
 * ****************** CheckForAdvancedTransform **********************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::CheckForAdvancedTransform( void )
{
  typename TransformType::Pointer transform = this->GetRegistration()
    ->GetAsITKBaseType()->GetTransform();

  AdvancedTransformType * testPtr = dynamic_cast<AdvancedTransformType *>(
    transform.GetPointer() );
  if ( !testPtr )
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

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::GetScaledDerivativeWithExceptionHandling(
  const ParametersType & parameters, DerivativeType & derivative )
{
  double dummyvalue = 0;
  try
  {
    this->GetScaledValueAndDerivative( parameters, dummyvalue, derivative );
  }
  catch ( itk::ExceptionObject & err )
  {
    this->m_StopCondition = MetricError;
    this->StopOptimization();
    throw err;
  }

} // end GetScaledDerivativeWithExceptionHandling()


/**
 * *************** AddRandomPerturbation ***************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::AddRandomPerturbation( ParametersType & parameters, double sigma )
{
  /** Add delta ~ sigma * N(0,I) to the input parameters. */
  for ( unsigned int p = 0; p < parameters.GetSize(); ++p )
  {
    parameters[ p ] += sigma * this->m_RandomGenerator->GetNormalVariate( 0.0, 1.0 );
  }

} // end AddRandomPerturbation()


/**
 * ********************* StoreCurrentPoint ************************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::StoreCurrentPoint(  const ParametersType & step,
  const DerivativeType & grad_dif )
{
  itkDebugMacro( "StoreCurrentPoint" );

//   const double rho = 1.0 / inner_product( step, grad_dif ) ; // 1/ys
//   const double  ys = 1.0 / rho;
  const double  ys = inner_product( step, grad_dif ) ; // 1/ys;
  const double rho = 1.0 / ys;
  const double  yy = grad_dif.squared_magnitude();

  double fill_value = ys / yy;
  if( fill_value < 0.0 )
  {
//    fill_value = this->m_GradientMagnitudeTolerance;
    this->m_StopCondition = InvalidDiagonalMatrix;
    this->StopOptimization();
  }

  this->m_S[ this->m_CurrentT ]   = step;     // expensive copy
  this->m_Y[ this->m_CurrentT ]   = grad_dif; // expensive copy
  this->m_Rho[ this->m_CurrentT ] = rho;
  this->m_HessianFillValue[ this->m_CurrentT ] = fill_value;


  elxout << "parameter difference s: " << step.magnitude() << std::endl;
  elxout << "gradient difference y: " << grad_dif.magnitude() << std::endl;
  elxout << "rho: " << this->m_Rho[ this->m_CurrentT ] << std::endl;
  elxout << "New H0: " << fill_value << std::endl;

} // end StoreCurrentPoint()


/**
 * ********************* ComputeDiagonalMatrix ********************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::ComputeDiagonalMatrix( DiagonalMatrixType & diag_H0 )
{
  diag_H0.SetSize(
    this->GetScaledCostFunction()->GetNumberOfParameters() );

  double curent_fill_value = 1.0;
  double oldest_fill_value = 1.0;
  double fill_value = 1.0;
  if( this->m_Bound > 0 )
  {
//     curent_fill_value = this->m_HessianFillValue[this->m_PreviousT];
//     oldest_fill_value = this->m_HessianFillValue[this->m_CurrentT];
//     fill_value = vnl_math_max(curent_fill_value,oldest_fill_value);

    fill_value = this->m_HessianFillValue[this->m_PreviousT];
  }

  elxout << "H0: " << fill_value << std::endl;
  diag_H0.Fill( fill_value );

} // end ComputeDiagonalMatrix()


/**
 * *********************** ComputeSearchDirection ************************
 */

template <class TElastix>
void
AdaptiveStochasticLBFGS<TElastix>
::ComputeSearchDirection(
  const DerivativeType & gradient,
  DerivativeType & searchDir )
{
  itkDebugMacro( "ComputeSearchDirection" );

  /** Assumes m_Rho, m_S, and m_Y are up-to-date at m_PreviousPoint */
  typedef itk::Array< double > AlphaType;
  AlphaType alpha( this->m_LBFGSMemory );

  const unsigned int numberOfParameters = gradient.GetSize();

#if 0
  /** this step can be emitted when s y did not change. */
  /** refine it later. */
  // omitting probably won't save a lot of time, only a fill over the diagonal
  DiagonalMatrixType H0;
  this->ComputeDiagonalMatrix( H0 );
#else
  // We can simply only return the fill_value and completely skip the diagonal matrix construction
  double fill_value = 1.0;
  if( this->m_Bound > 0 )
  {
    fill_value = this->m_HessianFillValue[ this->m_PreviousT ];
  }
#endif

  // The following line constructs a new negated vector which is then
  // copied to searchDir. The construction and copying can be avoided
  // as searchDir is already allocated.
  //searchDir = -gradient;
  for( unsigned int j = 0; j < numberOfParameters; ++j )
  {
    searchDir[ j ] = -gradient[ j ];
  }

  int cp = static_cast< int >( this->m_CurrentT );

  for( unsigned int i = 0; i < this->m_Bound; ++i )
  {
    --cp;
    if( cp == -1 )
    {
      cp = this->m_LBFGSMemory - 1;
    }
    const double sq = inner_product( this->m_S[ cp ], searchDir );
    alpha[ cp ] = this->m_Rho[ cp ] * sq;
    const double &         alpha_cp = alpha[ cp ];
    const DerivativeType & y        = this->m_Y[ cp ];
    for( unsigned int j = 0; j < numberOfParameters; ++j )
    {
      searchDir[ j ] -= alpha_cp * y[ j ];
    }
  }

  for( unsigned int j = 0; j < numberOfParameters; ++j )
  {
#if 0
    searchDir[ j ] *= H0[ j ];
#else
    searchDir[ j ] *= fill_value;
#endif
  }

  for( unsigned int i = 0; i < this->m_Bound; ++i )
  {
    const double           yr             = inner_product( this->m_Y[ cp ], searchDir );
    const double           beta           = this->m_Rho[ cp ] * yr;
    const double           alpha_min_beta = alpha[ cp ] - beta;
    const ParametersType & s              = this->m_S[ cp ];
    for( unsigned int j = 0; j < numberOfParameters; ++j )
    {
      searchDir[ j ] += alpha_min_beta * s[ j ];
    }
    ++cp;
    if( static_cast< unsigned int >( cp ) == this->m_LBFGSMemory )
    {
      cp = 0;
    }
  }

  /** Normalize if no information about previous steps is available yet */
  if( this->m_Bound == 0 )
  {
    searchDir /= gradient.magnitude();
  }

} // end ComputeSearchDirection()


} // end namespace elastix

#endif // end #ifndef __elxAdaptiveStochasticLBFGS_hxx
