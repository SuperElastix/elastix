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
#ifndef __elxPreconditionedGradientDescent_hxx
#define __elxPreconditionedGradientDescent_hxx

#include "elxPreconditionedGradientDescent.h"

#include "itkAdvancedImageToImageMetric.h"
#include "itkTimeProbe.h"
#include <iomanip>
#include <string>

namespace elastix
{
/**
 * ***************** Constructor ***********************
 */

template <class TElastix>
PreconditionedGradientDescent<TElastix>
::PreconditionedGradientDescent()
{
  this->m_MaximumNumberOfSamplingAttempts = 0;
  this->m_CurrentNumberOfSamplingAttempts = 0;
  this->m_PreviousErrorAtIteration = 0;
  this->m_PreconditionMatrixSet = false;
  this->m_AutomaticParameterEstimationDone = false;
  this->m_AutomaticParameterEstimation = false;
  this->m_NumberOfGradientMeasurements = 0;
  this->m_NumberOfSamplesForExactGradient = 100000;
  this->m_SigmoidScaleFactor = 0.1;
  this->m_RandomGenerator = RandomGeneratorType::New();

} // end Constructor()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
PreconditionedGradientDescent<TElastix>
::BeforeRegistration( void )
{
  /** Add the target cell "stepsize" to xout["iteration"].*/
  xout["iteration"].AddTargetCell("2:Metric");
  xout["iteration"].AddTargetCell("3a:Time");
  xout["iteration"].AddTargetCell("3b:StepSize");
  xout["iteration"].AddTargetCell("4a:||Gradient||");
  xout["iteration"].AddTargetCell("4b:||SearchDir||");

  /** Format the metric and stepsize as floats */
  xl::xout["iteration"]["2:Metric"]   << std::showpoint << std::fixed;
  xl::xout["iteration"]["3a:Time"] << std::showpoint << std::fixed;
  xl::xout["iteration"]["3b:StepSize"] << std::showpoint << std::fixed;
  xl::xout["iteration"]["4a:||Gradient||"] << std::showpoint << std::fixed;
  xl::xout["iteration"]["4b:||SearchDir||"] << std::showpoint << std::fixed;

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
PreconditionedGradientDescent<TElastix>
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level = static_cast<unsigned int>(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  /** Set the maximumNumberOfIterations. */
  unsigned int maximumNumberOfIterations = 500;
  this->GetConfiguration()->ReadParameter( maximumNumberOfIterations,
    "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfIterations( maximumNumberOfIterations );

  /** Set the gain parameter A. */
  double A = 20.0;
  this->GetConfiguration()->ReadParameter( A,
    "SP_A", this->GetComponentLabel(), level, 0 );
  this->SetParam_A( A );

  /** Set the MaximumNumberOfSamplingAttempts. */
  unsigned int maximumNumberOfSamplingAttempts = 0;
  this->GetConfiguration()->ReadParameter( maximumNumberOfSamplingAttempts,
    "MaximumNumberOfSamplingAttempts", this->GetComponentLabel(), level, 0 );
  this->SetMaximumNumberOfSamplingAttempts( maximumNumberOfSamplingAttempts );

  /** Set/Get the initial time. Default: 0.0. Should be >=0. */
  double initialTime = 0.0;
  this->GetConfiguration()->ReadParameter( initialTime,
    "SigmoidInitialTime", this->GetComponentLabel(), level, 0 );
  this->SetInitialTime( initialTime );

  /** Set/Get whether the adaptive step size mechanism is desired. Default: true
   * NB: the setting is turned of in case of UseRandomSampleRegion=true.
   */
  bool useAdaptiveStepSizes = true;
  this->GetConfiguration()->ReadParameter( useAdaptiveStepSizes,
    "UseAdaptiveStepSizes", this->GetComponentLabel(), level, 0 );
  this->SetUseAdaptiveStepSizes( useAdaptiveStepSizes );

  /** Set the diagonal weight for the precondition matrix */
  double diagonalWeight = 1e-6;
  this->GetConfiguration()->ReadParameter( diagonalWeight,
    "DiagonalWeight", this->GetComponentLabel(), level, 0 );
  this->SetDiagonalWeight( diagonalWeight );

  /** Set the minimum element magnitude for the gradient */
  double minimumGradientElementMagnitude = 1e-10;
  this->GetConfiguration()->ReadParameter( minimumGradientElementMagnitude,
    "MinimumGradientElementMagnitude", this->GetComponentLabel(), level, 0 );
  this->SetMinimumGradientElementMagnitude( minimumGradientElementMagnitude );

  /** Set whether automatic gain estimation is required; default: true. */
  this->m_AutomaticParameterEstimation = true;
  this->GetConfiguration()->ReadParameter( this->m_AutomaticParameterEstimation,
    "AutomaticParameterEstimation", this->GetComponentLabel(), level, 0 );

  if( this->m_AutomaticParameterEstimation )
  {
    /** Number of gradients N to estimate the average square magnitudes
     * of the exact gradient and the approximation error.
     * A value of 0 (default) means automatic estimation.
     */
    this->m_NumberOfGradientMeasurements = 0;
    this->GetConfiguration()->ReadParameter(
      this->m_NumberOfGradientMeasurements,
      "NumberOfGradientMeasurements",
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
     * cause a more wide sigmoid. Default: 0.1. Should be > 0.
     */
    double sigmoidScaleFactor = 0.1;
    this->GetConfiguration()->ReadParameter( sigmoidScaleFactor,
      "SigmoidScaleFactor", this->GetComponentLabel(), level, 0 );
    this->m_SigmoidScaleFactor = sigmoidScaleFactor;

  } // end if automatic parameter estimation
  else
  {
    /** If no automatic parameter estimation is used, a and alpha also need
     * to be specified. Try to guess reasonable values as defaults.
     */
    const double noisefactor = 0.2; //eta

    double alpha = 1.0;
    this->GetConfiguration()->ReadParameter(alpha, "SP_alpha",
      this->GetComponentLabel(), level, 0 );
    this->SetParam_alpha( alpha );

    double a = 2.0 * noisefactor * std::pow( this->GetParam_A() + 1.0, alpha );
    this->GetConfiguration()->ReadParameter(a, "SP_a",
      this->GetComponentLabel(), level, 0 );
    this->SetParam_a( a );

    /** Set/Get the maximum of the sigmoid. Should be >0. Default: 1.0. */
    double sigmoidMax = 1.0;
    this->GetConfiguration()->ReadParameter( sigmoidMax,
      "SigmoidMax", this->GetComponentLabel(), level, 0 );
    this->SetSigmoidMax( sigmoidMax );

    /** Set/Get the minimum of the sigmoid. Should be <0. Default ~ -0.8. */
    double sigmoidMin = -0.99 + 0.98 * noisefactor;
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
PreconditionedGradientDescent<TElastix>
::AfterEachIteration( void )
{
  /** Print some information. */
  xl::xout["iteration"]["2:Metric"]   << this->GetValue();
  xl::xout["iteration"]["3a:Time"] << this->GetCurrentTime();
  xl::xout["iteration"]["3b:StepSize"] << this->GetLearningRate();
  xl::xout["iteration"]["4a:||Gradient||"] << this->GetGradient().magnitude();
  xl::xout["iteration"]["4b:||SearchDir||"] << this->GetSearchDirection().magnitude();

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
PreconditionedGradientDescent<TElastix>
::AfterEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level = static_cast<unsigned int>(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  /** enum StopConditionType {  MaximumNumberOfIterations, MetricError } */
  std::string stopcondition;
  switch( this->GetStopCondition() )
  {
  case MaximumNumberOfIterations :
    stopcondition = "Maximum number of iterations has been reached";
    break;

  case MetricError :
    stopcondition = "Error in metric";
    break;

  default:
    stopcondition = "Unknown";
    break;
  }

  /** Print the stopping condition */
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
PreconditionedGradientDescent<TElastix>
::AfterRegistration( void )
{
  /** Print the best metric value */
  double bestValue = this->GetValue();
  elxout
    << std::endl
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
PreconditionedGradientDescent<TElastix>
::StartOptimization( void )
{
  /** Check if the entered scales are correct and != [ 1 1 1 ...] */
  this->SetUseScales( false );
  const ScalesType & scales = this->GetScales();
  if( scales.GetSize() == this->GetInitialPosition().GetSize() )
  {
    ScalesType unit_scales( scales.GetSize() );
    unit_scales.Fill(1.0);
    if( scales != unit_scales )
    {
      /** only then: */
      this->SetUseScales( true );
    }
  }

  /** \todo: this class probably does not work properly in
   * combination with scales.
   */

  /** Reset these values. */
  this->m_CurrentNumberOfSamplingAttempts = 0;
  this->m_PreviousErrorAtIteration = 0;
  this->m_PreconditionMatrixSet = false;
  this->m_AutomaticParameterEstimationDone = false;

  /** Superclass implementation. */
  this->Superclass1::StartOptimization();

} // end StartOptimization()


/**
 * ********************** ResumeOptimization **********************
 */

template <class TElastix>
void
PreconditionedGradientDescent<TElastix>
::ResumeOptimization( void )
{
  /** The following code relies on the fact that all
   * components have been set up and that the initial
   * position has been set, so must be called in this
   * function.
   */

  if( !this->m_PreconditionMatrixSet )
  {
    this->SetSelfHessian();
    this->m_PreconditionMatrixSet = true;
  }

  if( this->GetAutomaticParameterEstimation()
    && !this->m_AutomaticParameterEstimationDone )
  {
    this->AutomaticParameterEstimation();
    this->m_AutomaticParameterEstimationDone = true;
  }

  this->Superclass1::ResumeOptimization();

} // end ResumeOptimization()


/**
 * ********************** SetSelfHessian **********************
 */

template <class TElastix>
void
PreconditionedGradientDescent<TElastix>
::SetSelfHessian( void )
{
  /** If it works, think about a more generic solution */
  typedef itk::AdvancedImageToImageMetric<
    FixedImageType, MovingImageType >                 MetricWithSelfHessianType;

  /** Total time. */
  itk::TimeProbe timer;
  timer.Start();

  PreconditionType H;

  /* Get metric as metric with self Hessian. */
  const MetricWithSelfHessianType * metricWithSelfHessian = dynamic_cast<
    const MetricWithSelfHessianType *>( this->GetCostFunction() );

  if( metricWithSelfHessian == 0 )
  {
    itkExceptionMacro( <<
      "The PreconditionedGradientDescent optimizer can only be used with metrics that derive from the AdvancedImageToImage metric!" );
  }

  elxout << "Computing SelfHessian." << std::endl;
  try
  {
    metricWithSelfHessian->GetSelfHessian( this->GetCurrentPosition(), H );
  }
  catch( itk::ExceptionObject & err )
  {
    this->m_StopCondition = MetricError;
    this->StopOptimization();
    throw err;
  }

  timer.Stop();
  elxout << "Computing SelfHessian took: "
    << this->ConvertSecondsToDHMS( timer.GetMean(), 6 )
    << std::endl;

  timer.Start();
  elxout << "Computing Cholesky decomposition of SelfHessian." << std::endl;
  this->SetPreconditionMatrix( H );
  elxout << "Sparsity: " << this->GetSparsity() << std::endl;
  elxout << "Largest eigenvalue: " << this->GetLargestEigenValue() << std::endl;
  elxout << "Condition number: " << this->GetConditionNumber() << std::endl;
  timer.Stop();

  elxout << "Computing Cholesky decomposition took: "
    << this->ConvertSecondsToDHMS( timer.GetMean(), 6 )
    << std::endl;

} // end SetSelfHessian()


/**
 * ****************** MetricErrorResponse *************************
 */

template <class TElastix>
void
PreconditionedGradientDescent<TElastix>
::MetricErrorResponse( itk::ExceptionObject & err )
{
  if( this->GetCurrentIteration() != this->m_PreviousErrorAtIteration )
  {
    this->m_PreviousErrorAtIteration = this->GetCurrentIteration();
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
 * ******************* AutomaticParameterEstimation **********************
 */

template <class TElastix>
void
PreconditionedGradientDescent<TElastix>
::AutomaticParameterEstimation( void )
{
  /** Setup timer. */
  itk::TimeProbe timer;
  timer.Start();
  elxout << "Starting automatic parameter estimation ..." << std::endl;

  const unsigned int P =
    this->GetScaledCostFunction()->GetNumberOfParameters();
  const double Pd = static_cast<double>( P );

  /** Determine number of gradient measurements such that
   * E + 2\sqrt(Var) < K E
   * with
   * E = E(1/N \sum_n g_n^T g_n) = sigma_1^2 TrC
   * Var = Var(1/N \sum_n g_n^T g_n) = 2 sigma_1^4 TrCC / N
   * K = 1.5
   * We enforce a minimum of 2.
   * In the case of APSGD, TrC=TrCC=P (apart from scalar factor),
   * which simplifies things, compared to ASGD.
   * For the meaning of TrC and TrCC see ASGD.
   */
  if( this->m_NumberOfGradientMeasurements == 0 )
  {
    const double K = 1.5;
    this->m_NumberOfGradientMeasurements = static_cast<unsigned int>(
      std::ceil( 8.0 / P / (K-1) / (K-1) ) );
    this->m_NumberOfGradientMeasurements = vnl_math_max(
      static_cast<unsigned int>( 2 ),
      this->m_NumberOfGradientMeasurements );
    elxout << "  NumberOfGradientMeasurements to estimate sigma_i: "
      << this->m_NumberOfGradientMeasurements << std::endl;
  }

  /** Measure sigma1 and sigma2
   * (note: these are actually (sigma_1)^2 and (sigma_2)^2.
   */
  double sigma1 = 0.0;
  double sigma2 = 0.0;
  this->SampleGradients( this->GetScaledCurrentPosition(), sigma1, sigma2 );

  /** Determine parameter settings. */
  const double alpha = 1.0;
  const double A = this->GetParam_A();
  double a_max = 2.0 * A;
  const double noisefactor = sigma1 / ( 2.0 * sigma1  + sigma2 + 1e-14 );
  const double a = a_max * noisefactor;

  const double omega = vnl_math_max( 1e-14,
    this->m_SigmoidScaleFactor * ( sigma1 + sigma2 ) * std::sqrt( Pd ) );
  const double fmax = 1.0;
  const double fmin = -0.99 + 0.98 * noisefactor;

  /** Set parameters in superclass. */
  this->SetParam_a( a );
  this->SetParam_alpha( alpha );
  this->SetSigmoidMax( fmax );
  this->SetSigmoidMin( fmin );
  this->SetSigmoidScale( omega );

  /** Print the elapsed time. */
  timer.Stop();
  elxout << "Automatic parameter estimation took "
    << this->ConvertSecondsToDHMS( timer.GetMean(), 6 )
    << std::endl;

} // end AutomaticParameterEstimation()


/**
 * ******************** SampleGradients **********************
 */

template <class TElastix>
void
PreconditionedGradientDescent<TElastix>
::SampleGradients( const ParametersType & mu0,
  double & sigma1, double & sigma2 )
{
  /** Some shortcuts. \todo: P is used in the comments to denote the preconditioner...*/
  const unsigned int M = this->GetElastix()->GetNumberOfMetrics();
  const unsigned int P =
    this->GetScaledCostFunction()->GetNumberOfParameters();
  const double Pd = static_cast<double>(P);

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

        /** Set up the grid samper for the "exact" gradients.
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

  /** Prepare for progress printing. */
  elxout << "  Sampling exact gradient..." << std::endl;

  /** Initialize some variables for storing gradients and their magnitudes. */
  DerivativeType gradient( P );
  DerivativeType searchDirection( P );

  /** g_0' P g_0  */
  double exactgg = 0.0;

  /** Compute exactgg and sigma1. */
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
  }
  this->GetScaledDerivativeWithExceptionHandling( mu0, gradient );
  this->CholmodSolve( gradient, searchDirection );
  exactgg += inner_product( gradient, searchDirection ); // gPg
  sigma1 = exactgg / Pd;
  elxout << "sigma1 " << sigma1 << " exactgg: " << exactgg << std::endl;

  /** If all samplers are deterministic, simply set sigma2 to sigma1. */
  sigma2 = sigma1;

  /** Otherwise, sample some metric derivatives at random positions. */
  if( stochasticgradients )
  {
    /** Prepare for progress printing. */
    ProgressCommandPointer progressObserver = ProgressCommandType::New();
    progressObserver->SetUpdateFrequency(
      this->m_NumberOfGradientMeasurements, this->m_NumberOfGradientMeasurements );
    progressObserver->SetStartString( "  Progress: " );
    elxout << "  Sampling approximate gradients..." << std::endl;

    /** Set random sampler(s). */
    for( unsigned int m = 0; m < M; ++m )
    {
      if( randomSamplerVec[ m ].IsNotNull() )
      {
        this->GetElastix()->GetElxMetricBase( m )->
          SetAdvancedMetricImageSampler( randomSamplerVec[ m ] );
      }
    }

    double perturbationSigma = 0.0;
    if( sigma1 > 0 )
    {
      perturbationSigma = std::sqrt( sigma1 );
    }

    /** sum_n g_n' P g_n */
    double approxgg = 0.0;

    /** Compute gg for some random parameters. */
    ParametersType perturbedMu0 = mu0;
    for( unsigned int i = 0 ; i < this->m_NumberOfGradientMeasurements; ++i )
    {
      /** Show progress 0-100% */
      progressObserver->UpdateAndPrintProgress( i );

      /** Generate a perturbation, according to:
       *    \mu_i - \mu_0 ~ L^{-T} N( 0, sigma1 I ) = perturbationSigma L^{-T} N( 0, I ),
       * where L is the cholesky decomposition of H.
       */
      this->AddRandomPerturbation( mu0, perturbedMu0, perturbationSigma );

      /** Measure approximated gradient */
      this->SelectNewSamples();
      this->GetScaledDerivativeWithExceptionHandling( perturbedMu0, gradient );

      /** Compute g'Pg */
      this->CholmodSolve( gradient, searchDirection );
      approxgg += inner_product( gradient, searchDirection ); // gPg

      elxout << "approxgg: " << approxgg << std::endl;

    } // end for loop over gradient measurements

    /** For output: sigma2.
     * sigma1 and 2 are divided by Pd, but actually need to be divided by
     * the rank, in case of maximum likelihood. In case of no maximum likelihood,
     * the rank equals Pd.
     */
    sigma2 = approxgg / ( this->m_NumberOfGradientMeasurements * Pd ) ;

    progressObserver->PrintProgress( 1.0 );

  } // end if stochastic gradient sampling

  elxout << "sigma2 " << sigma2 <<  std::endl;

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
PreconditionedGradientDescent<TElastix>
::PrintSettingsVector( const SettingsVectorType & settings ) const
{
  const unsigned long nrofres = settings.size();

  /** Print to log file */
  elxout << "( SP_a " ;
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].a << " ";
  }
  elxout << ")\n" ;

  elxout << "( SP_A " ;
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[ i ].A << " ";
  }
  elxout << ")\n" ;

  elxout << "( SP_alpha " ;
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[i].alpha << " ";
  }
  elxout << ")\n" ;

  elxout << "( SigmoidMax " ;
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[i].fmax << " ";
  }
  elxout << ")\n" ;

  elxout << "( SigmoidMin " ;
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[i].fmin << " ";
  }
  elxout << ")\n" ;

  elxout << "( SigmoidScale " ;
  for( unsigned int i = 0; i < nrofres; ++i )
  {
    elxout << settings[i].omega << " ";
  }
  elxout << ")\n" ;

  elxout << std::endl;

} // end PrintSettingsVector()


/**
 * *************** GetScaledDerivativeWithExceptionHandling ***************
 */

template <class TElastix>
void
PreconditionedGradientDescent<TElastix>
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

template <class TElastix>
void
PreconditionedGradientDescent<TElastix>
::AddRandomPerturbation( const ParametersType & initialParameters,
  ParametersType & perturbedParameters, double sigma  )
{
  const unsigned int P = initialParameters.GetSize();

  /** Create nu ~ sigma * N(0,I). */
  ParametersType tempParameters( P );
  ParametersType tempParameters2( P );
  perturbedParameters.SetSize( P );
  for( unsigned int p = 0; p < P; ++p )
  {
    tempParameters[ p ] = sigma * this->m_RandomGenerator->GetNormalVariate( 0.0, 1.0 );
  }

  /** Compute (\mu - \mu0) = Permutation' L^{-T} (\nu - \nu0) */
  this->CholmodSolve( tempParameters, tempParameters2, CHOLMOD_Lt );
  this->CholmodSolve( tempParameters2, perturbedParameters, CHOLMOD_Pt );

  /** Add initial parameters */
  perturbedParameters += initialParameters;

} // end AddRandomPerturbation()


} // end namespace elastix

#endif // end #ifndef __elxPreconditionedGradientDescent_hxx
