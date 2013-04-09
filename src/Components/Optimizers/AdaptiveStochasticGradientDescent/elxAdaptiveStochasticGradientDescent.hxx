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
#include "vnl/vnl_math.h"
#include "vnl/vnl_fastops.h"
#include "vnl/vnl_diag_matrix.h"
#include "vnl/vnl_sparse_matrix.h"
#include "vnl/vnl_matlab_filewrite.h"
#include "itkAdvancedImageToImageMetric.h"
#include "elxTimer.h"

namespace elastix
{

/**
 * ********************** Constructor ***********************
 */

template <class TElastix>
AdaptiveStochasticGradientDescent<TElastix>
::AdaptiveStochasticGradientDescent()
{
  this->m_MaximumNumberOfSamplingAttempts = 0;
  this->m_CurrentNumberOfSamplingAttempts = 0;
  this->m_PreviousErrorAtIteration = 0;
  this->m_AutomaticParameterEstimationDone = false;

  this->m_AutomaticParameterEstimation = false;
  this->m_MaximumStepLength = 1.0;

  this->m_NumberOfGradientMeasurements = 0;
  this->m_NumberOfJacobianMeasurements = 0;
  this->m_NumberOfSamplesForExactGradient = 100000;
  this->m_SigmoidScaleFactor = 0.1;

  this->m_RandomGenerator = RandomGeneratorType::GetInstance();
  this->m_AdvancedTransform = 0;

} // Constructor


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
::BeforeRegistration( void )
{
  /** Add the target cell "stepsize" to xout["iteration"]. */
  xout["iteration"].AddTargetCell("2:Metric");
  xout["iteration"].AddTargetCell("3a:Time");
  xout["iteration"].AddTargetCell("3b:StepSize");
  xout["iteration"].AddTargetCell("4:||Gradient||");

  /** Format the metric and stepsize as floats. */
  xl::xout["iteration"]["2:Metric"]   << std::showpoint << std::fixed;
  xl::xout["iteration"]["3a:Time"] << std::showpoint << std::fixed;
  xl::xout["iteration"]["3b:StepSize"] << std::showpoint << std::fixed;
  xl::xout["iteration"]["4:||Gradient||"] << std::showpoint << std::fixed;

  this->m_SettingsVector.clear();

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void AdaptiveStochasticGradientDescent<TElastix>
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level = static_cast<unsigned int>(
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

  /** Set whether automatic gain estimation is required; default: true. */
  this->m_AutomaticParameterEstimation = true;
  this->GetConfiguration()->ReadParameter( this->m_AutomaticParameterEstimation,
    "AutomaticParameterEstimation", this->GetComponentLabel(), level, 0 );

  if ( this->m_AutomaticParameterEstimation )
  {
    /** Set the maximum step length: the maximum displacement of a voxel in mm.
     * Compute default value: mean spacing of fixed and moving image.
     */
    const unsigned int fixdim = this->GetElastix()->FixedDimension;
    const unsigned int movdim = this->GetElastix()->MovingDimension;
    double sum = 0.0;
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
AdaptiveStochasticGradientDescent<TElastix>
::AfterEachIteration( void )
{
  /** Print some information. */
  xl::xout["iteration"]["2:Metric"] << this->GetValue();
  xl::xout["iteration"]["3a:Time"] << this->GetCurrentTime();
  xl::xout["iteration"]["3b:StepSize"] << this->GetLearningRate();
  xl::xout["iteration"]["4:||Gradient||"] << this->GetGradient().magnitude();

  /** Select new spatial samples for the computation of the metric. */
  if ( this->GetNewSamplesEveryIteration() )
  {
    this->SelectNewSamples();
  }

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution *************************
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
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

  case MaximumNumberOfIterations :
    stopcondition = "Maximum number of iterations has been reached";
    break;

  case MetricError :
    stopcondition = "Error in metric";
    break;

  case MinimumStepSize :
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
AdaptiveStochasticGradientDescent<TElastix>
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
AdaptiveStochasticGradientDescent<TElastix>
::StartOptimization( void )
{
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

  this->Superclass1::StartOptimization();

} //end StartOptimization()


/**
 * ********************** ResumeOptimization **********************
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
::ResumeOptimization( void )
{
  /** The following code relies on the fact that all
  * components have been set up and that the initial
  * position has been set, so must be called in this
  * function. */

  if ( this->GetAutomaticParameterEstimation()
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

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
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
 * Estimates some reasonable values for the parameters
 * SP_a, SP_alpha (=1), SigmoidMin, SigmoidMax (=1), and SigmoidScale.
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
::AutomaticParameterEstimation( void )
{
  /** Setup timers. */
  tmr::Timer::Pointer timer1 = tmr::Timer::New();
  tmr::Timer::Pointer timer2 = tmr::Timer::New();
  tmr::Timer::Pointer timer3 = tmr::Timer::New();

  /** Total time. */
  timer1->StartTimer();
  elxout << "Starting automatic parameter estimation for "
    << this->elxGetClassName()
    << " ..." << std::endl;

  /** Get the user input. */
  const double delta = this->GetMaximumStepLength();

  /** Compute the Jacobian terms. */
  double TrC = 0.0;
  double TrCC = 0.0;
  double maxJJ = 0.0;
  double maxJCJ = 0.0;
  timer2->StartTimer();
  this->ComputeJacobianTerms( TrC, TrCC, maxJJ, maxJCJ );
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
  if ( this->m_NumberOfGradientMeasurements == 0 )
  {
    const double K = 1.5;
    if ( TrCC > 1e-14 && TrC > 1e-14 )
    {
      this->m_NumberOfGradientMeasurements = static_cast<unsigned int>(
        vcl_ceil( 8.0 * TrCC / TrC / TrC / (K-1) / (K-1) ) );
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
  if ( maxJJ > 1e-14 )
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
  if ( gg > 1e-14 && TrC > 1e-14 )
  {
    sigma1 = vcl_sqrt( gg / TrC );
  }
  if ( ee > 1e-14 && TrC > 1e-14 )
  {
    sigma3 = vcl_sqrt( ee / TrC );
  }

  const double alpha = 1.0;
  const double A = this->GetParam_A();
  double a_max = 0.0;
  if ( sigma1 > 1e-14 && maxJCJ > 1e-14 )
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
  this->SetSigmoidMax( fmax );
  this->SetSigmoidMin( fmin );
  this->SetSigmoidScale( omega );

  /** Print the elapsed time. */
  timer1->StopTimer();
  elxout << "Automatic parameter estimation took "
    << timer1->PrintElapsedTimeDHMS()
    << std::endl;

} // end AutomaticParameterEstimation()


/**
 * ******************** SampleGradients **********************
 */

/** Measure some derivatives, exact and approximated. Returns
 * the squared magnitude of the gradient and approximation error.
 * Needed for the automatic parameter estimation.
 */
template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
::SampleGradients( const ParametersType & mu0,
  double perturbationSigma, double & gg, double & ee )
{
  /** Some shortcuts. */
  const unsigned int M = this->GetElastix()->GetNumberOfMetrics();

  /** Variables for sampler support. Each metric may have a sampler. */
  std::vector< bool >                                 useRandomSampleRegionVec( M, false );
  std::vector< ImageRandomSamplerBasePointer >        randomSamplerVec( M, 0 );
  std::vector< ImageRandomCoordinateSamplerPointer >  randomCoordinateSamplerVec( M, 0 );
  std::vector< ImageGridSamplerPointer >              gridSamplerVec( M, 0 );

  /** If new samples every iteration, get each sampler, and check if it is
   * a kind of random sampler. If yes, prepare an additional grid sampler
   * for the exact gradients, and set the stochasticgradients flag to true.
   */
  bool stochasticgradients = false;
  if ( this->GetNewSamplesEveryIteration() )
  {
    for ( unsigned int m = 0; m < M; ++m )
    {
      /** Get the sampler. */
      ImageSamplerBasePointer sampler =
        this->GetElastix()->GetElxMetricBase( m )->GetAdvancedMetricImageSampler();
      randomSamplerVec[m] =
        dynamic_cast< ImageRandomSamplerBaseType * >( sampler.GetPointer() );
      randomCoordinateSamplerVec[m] =
        dynamic_cast< ImageRandomCoordinateSamplerType * >( sampler.GetPointer() );

      if ( randomSamplerVec[m].IsNotNull() )
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
        if ( randomCoordinateSamplerVec[ m ].IsNotNull() )
        {
          useRandomSampleRegionVec[ m ]
          = randomCoordinateSamplerVec[ m ]->GetUseRandomSampleRegion();
          if ( useRandomSampleRegionVec[ m ] )
          {
            if ( this->GetUseAdaptiveStepSizes() )
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
  elxout << "  Sampling gradients ..." << std::endl;

  /** Initialize some variables for storing gradients and their magnitudes. */
  DerivativeType approxgradient;
  DerivativeType exactgradient;
  DerivativeType diffgradient;
  double exactgg = 0.0;
  double diffgg = 0.0;

  /** Compute gg for some random parameters. */
  for ( unsigned int i = 0 ; i < this->m_NumberOfGradientMeasurements; ++i )
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
    if ( stochasticgradients )
    {
      /** Set grid sampler(s) and get exact derivative. */
      for ( unsigned int m = 0; m < M; ++m )
      {
        if ( gridSamplerVec[ m ].IsNotNull() )
        {
          this->GetElastix()->GetElxMetricBase( m )
            ->SetAdvancedMetricImageSampler( gridSamplerVec[ m ] );
        }
      }
      this->GetScaledDerivativeWithExceptionHandling( perturbedMu0, exactgradient );

      /** Set random sampler(s), select new spatial samples and get approximate derivative. */
      for ( unsigned int m = 0; m < M; ++m )
      {
        if ( randomSamplerVec[ m ].IsNotNull() )
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
  for ( unsigned int m = 0; m < M; ++m )
  {
    if ( randomCoordinateSamplerVec[ m ].IsNotNull() )
    {
      randomCoordinateSamplerVec[ m ]
        ->SetUseRandomSampleRegion( useRandomSampleRegionVec[ m ] );
    }
  }

} // end SampleGradients()


/**
 * ******************** ComputeJacobianTerms **********************
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
::ComputeJacobianTerms( double & TrC, double & TrCC,
  double & maxJJ, double & maxJCJ )
{
  /** This function computes four terms needed for the automatic parameter
   * estimation. The equation number refers to the IJCV paper.
   * Term 1: TrC, which is the trace of the covariance matrix, needed in (34):
   *    C = 1/n \sum_{i=1}^n J_i^T J_i    (25)
   *    with n the number of samples, J_i the Jacobian of the i-th sample.
   * Term 2: TrCC, which is the Frobenius norm of C, needed in (60):
   *    ||C||_F^2 = trace( C^T C )
   * To compute equations (47) and (54) we need the four sub-terms:
   *    A: trace( J_j C J_j^T )  in (47)
   *    B: || J_j C J_j^T ||_F   in (47)
   *    C: || J_j ||_F^2         in (54)
   *    D: || J_j J_j^T ||_F     in (54)
   * Term 3: maxJJ, see (47)
   * Term 4: maxJCJ, see (54)
   */

  typedef double                                      CovarianceValueType;
  typedef itk::Array2D<CovarianceValueType>           CovarianceMatrixType;
  typedef vnl_sparse_matrix<CovarianceValueType>      SparseCovarianceMatrixType;
  typedef typename SparseCovarianceMatrixType::row    SparseRowType;
  typedef typename SparseCovarianceMatrixType::pair_t SparseCovarianceElementType;
  typedef itk::Array<SizeValueType>                   NonZeroJacobianIndicesExpandedType;
  typedef vnl_diag_matrix<CovarianceValueType>        DiagCovarianceMatrixType;
  typedef vnl_vector<CovarianceValueType>             JacobianColumnType;

  /** Initialize. */
  TrC = TrCC = maxJJ = maxJCJ = 0.0;

  this->CheckForAdvancedTransform();

  /** Get samples. */
  ImageSampleContainerPointer sampleContainer = 0;
  this->SampleFixedImageForJacobianTerms( sampleContainer );
  const SizeValueType nrofsamples = sampleContainer->Size();
  const double n = static_cast<double>( nrofsamples );

  /** Get the number of parameters. */
  const unsigned int P = static_cast<unsigned int>(
    this->GetScaledCurrentPosition().GetSize() );

  /** Get transform and set current position. */
  typename TransformType::Pointer transform = this->GetRegistration()
    ->GetAsITKBaseType()->GetTransform();
  transform->SetParameters( this->GetCurrentPosition() );
  const unsigned int outdim = transform->GetOutputSpaceDimension();

  /** Get scales vector */
  const ScalesType & scales = this->m_ScaledCostFunction->GetScales();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();
  unsigned int samplenr = 0;

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType sizejacind
    = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  JacobianType jacj( outdim, sizejacind );
  jacj.Fill( 0.0 );
  NonZeroJacobianIndicesType jacind( sizejacind );
  jacind[ 0 ] = 0;
  if ( sizejacind > 1 ) jacind[ 1 ] = 0;
  NonZeroJacobianIndicesType prevjacind = jacind;

  /** Initialize covariance matrix. Sparse, diagonal, and band form. */
  SparseCovarianceMatrixType cov( P, P );
  DiagCovarianceMatrixType diagcov( P, 0.0 );
  CovarianceMatrixType bandcov;

  /** For temporary storage of J'J. */
  CovarianceMatrixType jactjac( sizejacind, sizejacind );
  jactjac.Fill( 0.0 );
#ifndef _ELASTIX_BUILD_LIBRARY
  /** Prepare for progress printing. */
  ProgressCommandPointer progressObserver = ProgressCommandType::New();
  progressObserver->SetUpdateFrequency( nrofsamples * 2, 100 );
  progressObserver->SetStartString( "  Progress: " );
  elxout << "  Computing JacobianTerms ..." << std::endl;
#endif
  /** Variables for the band cov matrix. */
  const unsigned int maxbandcovsize = m_MaxBandCovSize;
  const unsigned int nrOfBandStructureSamples = m_NumberOfBandStructureSamples;

  /** DifHist is a histogram of absolute parameterNrDifferences that
   * occur in the nonzerojacobianindex vectors.
   * DifHist2 is another way of storing the histogram, as a vector
   * of pairs. pair.first = Frequency, pair.second = parameterNrDifference.
   * This is useful for sorting.
   */
  typedef std::vector<unsigned int>             DifHistType;
  typedef std::pair<unsigned int, unsigned int> FreqPairType;
  typedef std::vector<FreqPairType>             DifHist2Type;
  DifHistType difHist( P, 0 );
  DifHist2Type difHist2;

  /** Try to guess the band structure of the covariance matrix.
   * A 'band' is a series of elements cov(p,q) with constant q-p.
   * In the loop below, on a few positions in the image the Jacobian
   * is computed. The nonzerojacobianindices are inspected to figure out
   * which values of q-p occur often. This is done by making a histogram.
   * The histogram is then sorted and the most occurring bands
   * are determined. The covariance elements in these bands will not
   * be stored in the sparse matrix structure 'cov', but in the band
   * matrix 'bandcov', which is much faster.
   * Only after the bandcov and cov have been filled (by looping over
   * all Jacobian measurements in the sample container, the bandcov
   * matrix is injected in the cov matrix, for easy further calculations,
   * and the bandcov matrix is deleted.
   */
  unsigned int onezero = 0;
  for ( unsigned int s = 0; s < nrOfBandStructureSamples; ++s )
  {
    /** Semi-randomly get some samples from the sample container. */
    const unsigned int samplenr = ( s + 1 ) * nrofsamples
      / ( nrOfBandStructureSamples + 2 + onezero );
    onezero = 1 - onezero; // introduces semi-randomness

    /** Read fixed coordinates and get Jacobian J_j. */
    const FixedImagePointType & point
      = sampleContainer->GetElement(samplenr).m_ImageCoordinates;
    this->m_AdvancedTransform->GetJacobian( point, jacj, jacind );

    /** Skip invalid Jacobians in the beginning, if any. */
    if ( sizejacind > 1 )
    {
      if ( jacind[ 0 ] == jacind[ 1 ] )
      {
        continue;
      }
    }

    /** Fill the histogram of parameter nr differences. */
    for ( unsigned int i = 0; i < sizejacind; ++i )
    {
      const int jacindi = static_cast<int>( jacind[ i ] );
      for ( unsigned int j = i; j < sizejacind; ++j )
      {
        const int jacindj = static_cast<int>( jacind[ j ] );
        difHist[ static_cast<unsigned int>( vcl_abs( jacindj - jacindi ) ) ]++;
      }
    }
  }

  /** Copy the nonzero elements of the difHist to a vector pairs. */
  for ( unsigned int p = 0; p < P; ++p )
  {
    const unsigned int freq = difHist[ p ];
    if ( freq != 0 )
    {
      difHist2.push_back( FreqPairType( freq, p ) );
    }
  }
  difHist.resize( 0 );

  /** Compute the number of bands. */
  elxout << "  Estimated band size covariance matrix: " << difHist2.size() << std::endl;
  const unsigned int bandcovsize = vnl_math_min( maxbandcovsize,
    static_cast<unsigned int>(difHist2.size()) );
  elxout << "  Used band size covariance matrix: " << bandcovsize << std::endl;
  /** Maps parameterNrDifference (q-p) to colnr in bandcov. */
  std::vector<unsigned int> bandcovMap( P, bandcovsize );
  /** Maps colnr in bandcov to parameterNrDifference (q-p). */
  std::vector<unsigned int> bandcovMap2( bandcovsize, P );

  /** Sort the difHist2 based on the frequencies. */
  std::sort( difHist2.begin(), difHist2.end() );

  /** Determine the bands that are expected to be most dominant. */
  DifHist2Type::iterator difHist2It = difHist2.end();
  for ( unsigned int b = 0; b < bandcovsize; ++b )
  {
    --difHist2It;
    bandcovMap[ difHist2It->second ] = b;
    bandcovMap2[ b ] = difHist2It->second;
  }

  /** Initialize band matrix. */
  bandcov = CovarianceMatrixType( P, bandcovsize );
  bandcov.Fill( 0.0 );

  /**
   *    TERM 1
   *
   * Loop over image and compute Jacobian.
   * Compute C = 1/n \sum_i J_i^T J_i
   * Possibly apply scaling afterwards.
   */
  jacind[ 0 ] = 0;
  if ( sizejacind > 1 ) jacind[ 1 ] = 0;
  for ( iter = begin; iter != end; ++iter )
  {
#ifndef _ELASTIX_BUILD_LIBRARY
    /** Print progress 0-50%. */
    progressObserver->UpdateAndPrintProgress( samplenr );
#endif
    ++samplenr;

    /** Read fixed coordinates and get Jacobian J_j. */
    const FixedImagePointType & point = (*iter).Value().m_ImageCoordinates;
    this->m_AdvancedTransform->GetJacobian( point, jacj, jacind  );

    /** Skip invalid Jacobians in the beginning, if any. */
    if ( sizejacind > 1 )
    {
      if ( jacind[ 0 ] == jacind[ 1 ] )
      {
        continue;
      }
    }

    if ( jacind == prevjacind )
    {
      /** Update sum of J_j^T J_j. */
      vnl_fastops::inc_X_by_AtA( jactjac, jacj );
    }
    else
    {
      /** The following should only be done after the first sample. */
      if ( iter != begin )
      {
        /** Update covariance matrix. */
        for ( unsigned int pi = 0; pi < sizejacind; ++pi )
        {
          const unsigned int p = prevjacind[ pi ];

          for ( unsigned int qi = 0; qi < sizejacind; ++qi )
          {
            const unsigned int q = prevjacind[ qi ];
            /** Exploit symmetry: only fill upper triangular part. */
            if ( q >= p )
            {
              const double tempval = jactjac( pi, qi ) / n;
              if ( vcl_abs( tempval ) > 1e-14 )
              {
                const unsigned int bandindex = bandcovMap[ q - p ];
                if ( bandindex < bandcovsize )
                {
                  bandcov( p, bandindex ) +=tempval;
                }
                else
                {
                  cov( p, q ) += tempval;
                }
              }
            }
          } // qi
        } // pi
      } // end if

      /** Initialize jactjac by J_j^T J_j. */
      vnl_fastops::AtA( jactjac, jacj );

      /** Remember nonzerojacobian indices. */
      prevjacind = jacind;
    } // end else

  } // end iter loop: end computation of covariance matrix

  /** Update covariance matrix once again to include last jactjac updates
   * \todo: a bit ugly that this loop is copied from above.
   */
  for ( unsigned int pi = 0; pi < sizejacind; ++pi )
  {
    const unsigned int p = prevjacind[ pi ];
    for ( unsigned int qi = 0; qi < sizejacind; ++qi )
    {
      const unsigned int q = prevjacind[ qi ];
      if ( q >= p )
      {
        const double tempval = jactjac( pi, qi ) / n;
        if ( vcl_abs( tempval ) > 1e-14 )
        {
           const unsigned int bandindex  = bandcovMap[ q - p ];
           if ( bandindex < bandcovsize )
           {
             bandcov( p, bandindex ) +=tempval;
           }
           else
           {
             cov( p, q ) += tempval;
           }
        }
      }
    } // qi
  } // pi

  /** Copy the bandmatrix into the sparse matrix and empty the bandcov matrix.
   * \todo: perhaps work further with this bandmatrix instead.
   */
  for( unsigned int p = 0; p < P; ++p )
  {
    for ( unsigned int b = 0; b < bandcovsize; ++b )
    {
      const double tempval = bandcov( p, b );
      if ( vcl_abs( tempval ) > 1e-14 )
      {
        const unsigned int q = p + bandcovMap2[ b ];
        cov( p, q ) = tempval;
      }
    }
  }
  bandcov.set_size( 0, 0 );

  /** Apply scales. */
  if ( this->GetUseScales() )
  {
    for ( unsigned int p = 0; p < P; ++p )
    {
      cov.scale_row( p, 1.0 / scales[ p ] );
    }
    /**  \todo: this might be faster with get_row instead of the iterator */
    cov.reset();
    bool notfinished = cov.next();
    while ( notfinished )
    {
      const int col = cov.getcolumn();
      cov( cov.getrow(), col ) /= scales[ col ];
      notfinished = cov.next();
    }
  }

  /** Compute TrC = trace(C), and diagcov. */
  for ( unsigned int p = 0; p < P; ++p )
  {
    if ( !cov.empty_row( p ) )
    {
      //avoid creation of element if the row is empty
      CovarianceValueType & covpp = cov( p, p );
      TrC += covpp;
      diagcov[ p ] = covpp;
    }
  }

  /**
   *    TERM 2
   *
   * Compute TrCC = ||C||_F^2.
   */
  cov.reset();
  bool notfinished2 = cov.next();
  while ( notfinished2 )
  {
    TrCC += vnl_math_sqr( cov.value() );
    notfinished2 = cov.next();
  }

  /** Symmetry: multiply by 2 and subtract sumsqr(diagcov). */
  TrCC *= 2.0;
  TrCC -= diagcov.diagonal().squared_magnitude();

  /**
   *    TERM 3 and 4
   *
   * Compute maxJJ and maxJCJ
   * \li maxJJ = max_j [ ||J_j||_F^2 + 2\sqrt{2} || J_j J_j^T ||_F ]
   * \li maxJCJ = max_j [ Tr( J_j C J_j^T ) + 2\sqrt{2} || J_j C J_j^T ||_F ]
   */
  maxJJ = 0.0;
  maxJCJ = 0.0;
  const double sqrt2 = vcl_sqrt( static_cast<double>( 2.0 ) );
  JacobianType jacjjacj( outdim, outdim );
  JacobianType jacjcov( outdim, sizejacind );
  DiagCovarianceMatrixType diagcovsparse( sizejacind );
  JacobianType jacjdiagcov( outdim, sizejacind );
  JacobianType jacjdiagcovjacj( outdim, outdim );
  JacobianType jacjcovjacj( outdim, outdim );
  NonZeroJacobianIndicesExpandedType jacindExpanded( P );

  samplenr = 0;
  for ( iter = begin; iter != end; ++iter )
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = (*iter).Value().m_ImageCoordinates;
    this->m_AdvancedTransform->GetJacobian( point, jacj, jacind  );

    /** Apply scales, if necessary. */
    if ( this->GetUseScales() )
    {
      for ( unsigned int pi = 0; pi < sizejacind; ++pi )
      {
        const unsigned int p = jacind[ pi ];
        jacj.scale_column( pi, 1.0 / scales[ p ] );
      }
    }

    /** Compute 1st part of JJ: ||J_j||_F^2. */
    double JJ_j = vnl_math_sqr( jacj.frobenius_norm() );

    /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F. */
    vnl_fastops::ABt( jacjjacj, jacj, jacj );
    JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

    /** Max_j [JJ_j]. */
    maxJJ = vnl_math_max( maxJJ, JJ_j );

    /** Compute JCJ_j. */
    double JCJ_j = 0.0;

    /** J_j C = jacjC. */
    jacjcov.Fill( 0.0 );

    /** Store the nonzero Jacobian indices in a different format
     * and create the sparse diagcov.
     */
    jacindExpanded.Fill( sizejacind );
    for ( unsigned int pi = 0; pi < sizejacind; ++pi )
    {
      const unsigned int p = jacind[ pi ];
      jacindExpanded[ p ] = pi;
      diagcovsparse[ pi ] = diagcov[ p ];
    }

    /** We below calculate jacjC = J_j cov^T, but later we will correct
     * for this using:
     * J C J' = J (cov + cov' - diag(cov')) J'.
     * (NB: cov now still contains only the upper triangular part of C)
     */
    for ( unsigned int pi = 0; pi < sizejacind; ++pi )
    {
      const unsigned int p = jacind[ pi ];
      if ( !cov.empty_row( p ) )
      {
        SparseRowType & covrowp = cov.get_row( p );
        typename SparseRowType::iterator covrowpit;

        /** Loop over row p of the sparse cov matrix. */
        for ( covrowpit = covrowp.begin(); covrowpit != covrowp.end(); ++covrowpit )
        {
          const unsigned int q = (*covrowpit).first;
          const unsigned int qi = jacindExpanded[ q ];

          if ( qi < sizejacind )
          {
            /** If found, update the jacjC matrix. */
            const CovarianceValueType covElement = (*covrowpit).second;
            for ( unsigned int dx = 0; dx < outdim; ++dx )
            {
              jacjcov[ dx ][ pi ] += jacj[ dx ][ qi ] * covElement;
            } //dx
          } // if qi < sizejacind
        } // for covrow

      } // if not empty row
    } // pi

    /** J_j C J_j^T  = jacjCjacj.
     * But note that we actually compute J_j cov' J_j^T
     */
    vnl_fastops::ABt( jacjcovjacj, jacjcov, jacj );

    /** jacjCjacj = jacjCjacj+ jacjCjacj' - jacjdiagcovjacj */
    jacjdiagcov = jacj * diagcovsparse;
    vnl_fastops::ABt( jacjdiagcovjacj, jacjdiagcov, jacj );
    jacjcovjacj += jacjcovjacj.transpose();
    jacjcovjacj -= jacjdiagcovjacj;

    /** Compute 1st part of JCJ: Tr( J_j C J_j^T ). */
    for ( unsigned int d = 0; d < outdim; ++d )
    {
      JCJ_j += jacjcovjacj[ d ][ d ];
    }

    /** Compute 2nd part of JCJ_j: 2 \sqrt{2} || J_j C J_j^T ||_F. */
    JCJ_j += 2.0 * sqrt2 * jacjcovjacj.frobenius_norm();

    /** Max_j [JCJ_j]. */
    maxJCJ = vnl_math_max( maxJCJ, JCJ_j );
#ifndef _ELASTIX_BUILD_LIBRARY
    /** Show progress 50-100%. */
    progressObserver->UpdateAndPrintProgress( samplenr + nrofsamples );
#endif
    ++samplenr;

  } // end loop over sample container
#ifndef _ELASTIX_BUILD_LIBRARY
  /** Finalize progress information. */
  progressObserver->PrintProgress( 1.0 );
#endif
} // end ComputeJacobianTerms()


/**
 * **************** SampleFixedImageForJacobianTerms *******************
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
::SampleFixedImageForJacobianTerms(
  ImageSampleContainerPointer & sampleContainer )
{
  typedef typename ElastixType::MetricBaseType::AdvancedMetricType MetricType;
  MetricType * testPtr = dynamic_cast<MetricType *>(
    this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType() );
  if ( !testPtr )
  {
    itkExceptionMacro( << "ERROR: AdaptiveStochasticGradientDescent expects "
      << "the metric to be of type AdvancedImageToImageMetric!" );
  }

  /** Set up grid sampler. */
  ImageGridSamplerPointer sampler = ImageGridSamplerType::New();
  sampler->SetInput( testPtr->GetFixedImage() );
  sampler->SetInputImageRegion( testPtr->GetFixedImageRegion() );
  sampler->SetMask( testPtr->GetFixedImageMask() );

  /** Determine grid spacing of sampler such that the desired
   * NumberOfJacobianMeasurements is achieved approximately.
   * Note that the actually obtained number of samples may be lower, due to masks.
   * This is taken into account at the end of this function.
   */
  SizeValueType nrofsamples = this->m_NumberOfJacobianMeasurements;
  sampler->SetNumberOfSamples( nrofsamples );

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  sampleContainer = sampler->GetOutput();
  nrofsamples = sampleContainer->Size();

  if ( nrofsamples == 0 )
  {
    itkExceptionMacro(
      << "No valid voxels (0/" << this->m_NumberOfJacobianMeasurements
      << ") found to estimate the AdaptiveStochasticGradientDescent parameters." );
  }

} // end SampleFixedImageForJacobianTerms()


/**
 * **************** PrintSettingsVector **********************
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
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
 * Check if the transform is of type AdvancedTransform.
 * If so, we can speed up derivative calculations by only inspecting
 * the parameters in the support region of a point.
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
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
 * Helper function, used by SampleGradients.
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
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
 * Helper function, used by SampleGradients.
 */

template <class TElastix>
void
AdaptiveStochasticGradientDescent<TElastix>
::AddRandomPerturbation( ParametersType & parameters, double sigma )
{
  /** Add delta ~ sigma * N(0,I) to the input parameters. */
  for ( unsigned int p = 0; p < parameters.GetSize(); ++p )
  {
    parameters[ p ] += sigma * this->m_RandomGenerator->GetNormalVariate( 0.0, 1.0 );
  }

} // end AddRandomPerturbation()


} // end namespace elastix

#endif // end #ifndef __elxAdaptiveStochasticGradientDescent_hxx

