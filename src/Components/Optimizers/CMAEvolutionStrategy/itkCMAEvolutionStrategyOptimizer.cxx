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

#ifndef __itkCMAEvolutionStrategyOptimizer_cxx
#define __itkCMAEvolutionStrategyOptimizer_cxx

#include "itkCMAEvolutionStrategyOptimizer.h"
#include "itkSymmetricEigenAnalysis.h"
#include "vnl/vnl_math.h"
#include <algorithm>
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"

namespace itk
{

/**
 * ******************** Constructor *************************
 */

CMAEvolutionStrategyOptimizer::CMAEvolutionStrategyOptimizer()
{
  itkDebugMacro( "Constructor" );

  this->m_RandomGenerator = RandomGeneratorType::GetInstance();

  this->m_CurrentValue     = NumericTraits< MeasureType >::Zero;
  this->m_CurrentIteration = 0;
  this->m_StopCondition    = Unknown;
  this->m_Stop             = false;

  this->m_UseCovarianceMatrixAdaptation = true;
  this->m_PopulationSize                = 0;
  this->m_NumberOfParents               = 0;
  this->m_UpdateBDPeriod                = 1;

  this->m_EffectiveMu                        = 0.0;
  this->m_ConjugateEvolutionPathConstant     = 0.0;
  this->m_SigmaDampingConstant               = 0.0;
  this->m_CovarianceMatrixAdaptationConstant = 0.0;
  this->m_EvolutionPathConstant              = 0.0;
  this->m_CovarianceMatrixAdaptationWeight   = 0.0;
  this->m_ExpectationNormNormalDistribution  = 0.0;
  this->m_HistoryLength                      = 0;
  this->m_CurrentMaximumD                    = 1.0;
  this->m_CurrentMinimumD                    = 1.0;

  this->m_CurrentSigma = 0.0;
  this->m_Heaviside    = false;

  this->m_MaximumNumberOfIterations  = 100;
  this->m_UseDecayingSigma           = false;
  this->m_InitialSigma               = 1.0;
  this->m_SigmaDecayA                = 50;
  this->m_SigmaDecayAlpha            = 0.602;
  this->m_RecombinationWeightsPreset = "superlinear";
  this->m_MaximumDeviation           = NumericTraits< double >::max();
  this->m_MinimumDeviation           = 0.0;
  this->m_PositionToleranceMin       = 1e-12;
  this->m_PositionToleranceMax       = 1e8;
  this->m_ValueTolerance             = 1e-12;

}   // end constructor


/**
 * ******************* PrintSelf *********************
 */

void
CMAEvolutionStrategyOptimizer::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  //os << indent << "m_RandomGenerator: " << this->m_RandomGenerator << std::endl;
  os << indent << "m_CurrentValue: " << this->m_CurrentValue << std::endl;
  os << indent << "m_CurrentIteration: " << this->m_CurrentIteration << std::endl;
  os << indent << "m_StopCondition: " << this->m_StopCondition << std::endl;
  os << indent << "m_Stop: " << this->m_Stop << std::endl;

  os << indent << "m_UseCovarianceMatrixAdaptation: " << this->m_UseCovarianceMatrixAdaptation << std::endl;
  os << indent << "m_PopulationSize: " << this->m_PopulationSize << std::endl;
  os << indent << "m_NumberOfParents: " << this->m_NumberOfParents << std::endl;
  os << indent << "m_UpdateBDPeriod: " << this->m_UpdateBDPeriod << std::endl;

  os << indent << "m_EffectiveMu: " << this->m_EffectiveMu << std::endl;
  os << indent << "m_ConjugateEvolutionPathConstant: " << this->m_ConjugateEvolutionPathConstant << std::endl;
  os << indent << "m_SigmaDampingConstant: " << this->m_SigmaDampingConstant << std::endl;
  os << indent << "m_CovarianceMatrixAdaptationConstant: " << this->m_CovarianceMatrixAdaptationConstant << std::endl;
  os << indent << "m_EvolutionPathConstant: " << this->m_EvolutionPathConstant << std::endl;
  os << indent << "m_CovarianceMatrixAdaptationWeight: " << this->m_CovarianceMatrixAdaptationWeight << std::endl;
  os << indent << "m_ExpectationNormNormalDistribution: " << this->m_ExpectationNormNormalDistribution << std::endl;
  os << indent << "m_HistoryLength: " << this->m_HistoryLength << std::endl;

  os << indent << "m_CurrentSigma: " << this->m_CurrentSigma << std::endl;
  os << indent << "m_Heaviside: " << this->m_Heaviside << std::endl;
  os << indent << "m_CurrentMaximumD: " << this->m_CurrentMaximumD << std::endl;
  os << indent << "m_CurrentMinimumD: " << this->m_CurrentMinimumD << std::endl;

  os << indent << "m_MaximumNumberOfIterations: " << this->m_MaximumNumberOfIterations << std::endl;
  os << indent << "m_UseDecayingSigma: " << this->m_UseDecayingSigma << std::endl;
  os << indent << "m_InitialSigma: " << this->m_InitialSigma << std::endl;
  os << indent << "m_SigmaDecayA: " << this->m_SigmaDecayA << std::endl;
  os << indent << "m_SigmaDecayAlpha: " << this->m_SigmaDecayAlpha << std::endl;
  os << indent << "m_RecombinationWeightsPreset: " << this->m_RecombinationWeightsPreset << std::endl;
  os << indent << "m_MaximumDeviation: " << this->m_MaximumDeviation << std::endl;
  os << indent << "m_MinimumDeviation: " << this->m_MinimumDeviation << std::endl;
  os << indent << "m_PositionToleranceMin: " << this->m_PositionToleranceMin << std::endl;
  os << indent << "m_PositionToleranceMax: " << this->m_PositionToleranceMax << std::endl;
  os << indent << "m_ValueTolerance: " << this->m_ValueTolerance << std::endl;

  os << indent << "m_RecombinationWeights: " << this->m_RecombinationWeights << std::endl;
  os << indent << "m_C: " << this->m_C << std::endl;
  os << indent << "m_B: " << this->m_B << std::endl;
  os << indent << "m_D: " << this->m_D.diagonal() << std::endl;

  // template:
  //os << indent << ": " << this-> << std::endl;

}   // end PrintSelf;


/**
 * ******************* StartOptimization *********************
 */

void
CMAEvolutionStrategyOptimizer::StartOptimization()
{
  itkDebugMacro( "StartOptimization" );

  /** Reset some variables */
  this->m_CurrentValue     = NumericTraits< MeasureType >::Zero;
  this->m_CurrentIteration = 0;
  this->m_Stop             = false;
  this->m_StopCondition    = Unknown;

  /** Get the number of parameters; checks also if a cost function has been set at all.
  * if not: an exception is thrown */
  this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Initialize the scaledCostFunction with the currently set scales */
  this->InitializeScales();

  /** Set the current position as the scaled initial position */
  this->SetCurrentPosition( this->GetInitialPosition() );

  /** Compute default values for a lot of constants */
  this->InitializeConstants();

  /** Resize/Initialize variables used that are updated during optimisation */
  this->InitializeProgressVariables();

  /** Resize/Initialize B, C, and D */
  this->InitializeBCD();

  if( !this->m_Stop )
  {
    this->ResumeOptimization();
  }

}   // end StartOptimization


/**
 * ******************* ResumeOptimization *********************
 */

void
CMAEvolutionStrategyOptimizer::ResumeOptimization()
{
  itkDebugMacro( "ResumeOptimization" );

  this->m_Stop          = false;
  this->m_StopCondition = Unknown;

  this->InvokeEvent( StartEvent() );

  try
  {
    this->m_CurrentValue = this->GetScaledValue( this->GetScaledCurrentPosition() );
  }
  catch( ExceptionObject & err )
  {
    this->m_StopCondition = MetricError;
    this->StopOptimization();
    throw err;
  }

  /** Test if not by chance we are already converged */
  bool convergence = this->TestConvergence( true );
  if( convergence )
  {
    this->StopOptimization();
  }

  /** Start iterating */
  while( !this->m_Stop )
  {
    this->GenerateOffspring();
    this->SortCostFunctionValues();

    /** Something may have gone wrong during evaluation of the cost function values */
    if( this->m_Stop )
    {
      break;
    }

    this->AdvanceOneStep();

    /** Something may have gone wrong during evalution of the current value */
    if( this->m_Stop )
    {
      break;
    }

    /** Give the user opportunity to observe progress (current value/position/sigma etc.) */
    this->InvokeEvent( IterationEvent() );

    if( this->m_Stop )
    {
      break;
    }

    /** Prepare for next iteration */
    this->UpdateConjugateEvolutionPath();
    this->UpdateHeaviside();
    this->UpdateEvolutionPath();
    this->UpdateC();
    this->UpdateSigma();
    this->UpdateBD();
    this->FixNumericalErrors();

    /** Test if convergence has occured in some sense */
    convergence = this->TestConvergence( false );
    if( convergence )
    {
      this->StopOptimization();
      break;
    }

    /** Next iteration */
    ++( this->m_CurrentIteration );

  }   // end while !m_Stop

}   // end ResumeOptimization


/**
 * *********************** StopOptimization *****************************
 */

void
CMAEvolutionStrategyOptimizer::StopOptimization()
{
  itkDebugMacro( "StopOptimization" );
  this->m_Stop = true;
  this->InvokeEvent( EndEvent() );
}   // end StopOptimization()


/**
 * ****************** InitializeConstants *********************
 */

void
CMAEvolutionStrategyOptimizer::InitializeConstants( void )
{
  itkDebugMacro( "InitializeConstants" );

  /** Get the number of parameters from the cost function */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** m_PopulationSize (if not provided by the user) */
  if( this->m_PopulationSize == 0 )
  {
    this->m_PopulationSize = 4 + static_cast< unsigned int >(
      vcl_floor( 3.0 * vcl_log( static_cast< double >( numberOfParameters ) ) ) );
  }

  /** m_NumberOfParents (if not provided by the user) */
  if( this->m_NumberOfParents == 0 )
  {
    this->m_NumberOfParents = this->m_PopulationSize / 2;
  }

  /** Some casts/aliases: */
  const unsigned int N       = numberOfParameters;
  const double       Nd      = static_cast< double >( N );
  const unsigned int lambda  = this->m_PopulationSize;
  const double       lambdad = static_cast< double >( lambda );
  const unsigned int mu      = this->m_NumberOfParents;
  const double       mud     = static_cast< double >( mu );

  /** m_RecombinationWeights */
  this->m_RecombinationWeights.SetSize( mu );
  this->m_RecombinationWeights.Fill( 1.0 );   // "equal" preset
  if( this->m_RecombinationWeightsPreset == "linear" )
  {
    for( unsigned int i = 0; i < mu; ++i )
    {
      this->m_RecombinationWeights[ i ]
        = mud + 1.0 - static_cast< double >( i + 1 );
    }
  }
  else if( this->m_RecombinationWeightsPreset == "superlinear" )
  {
    const double logmud = vcl_log( mud + 1.0 );
    for( unsigned int i = 0; i < mu; ++i )
    {
      this->m_RecombinationWeights[ i ]
        = logmud - vcl_log( static_cast< double >( i + 1 ) );
    }
  }
  this->m_RecombinationWeights /= this->m_RecombinationWeights.sum();

  /** m_EffectiveMu */
  this->m_EffectiveMu = 1.0 / this->m_RecombinationWeights.squared_magnitude();
  if( this->m_EffectiveMu >= lambdad )
  {
    itkExceptionMacro( << "The RecombinationWeights have unreasonable values!" );
  }
  /** alias: */
  const double mueff = this->m_EffectiveMu;

  /** m_ConjugateEvolutionPathConstant (c_\sigma) */
  this->m_ConjugateEvolutionPathConstant = ( mueff + 2.0 ) / ( Nd + mueff + 3.0 );

  /** m_SigmaDampingConstant */
  this->m_SigmaDampingConstant = this->m_ConjugateEvolutionPathConstant
    + ( 1.0 + 2.0 * vnl_math_max( 0.0, vcl_sqrt( ( mueff - 1.0 ) / ( Nd + 1.0 ) ) - 1.0 ) )
    * vnl_math_max( 0.3, 1.0 - Nd / static_cast< double >( this->m_MaximumNumberOfIterations ) );

  /** m_CovarianceMatrixAdaptationWeight (\mu_cov)*/
  this->m_CovarianceMatrixAdaptationWeight = mueff;
  /** alias: */
  const double mucov = this->m_CovarianceMatrixAdaptationWeight;

  /** m_CovarianceMatrixAdaptationConstant (c_cov) */
  this->m_CovarianceMatrixAdaptationConstant
    = ( 1.0 / mucov ) * 2.0 / vnl_math_sqr( Nd + vcl_sqrt( 2.0 ) )
    + ( 1.0 - 1.0 / mucov )
    * vnl_math_min( 1.0, ( 2.0 * mueff - 1.0 ) / ( vnl_math_sqr( Nd + 2.0 ) + mueff ) );
  /** alias: */
  const double c_cov = this->m_CovarianceMatrixAdaptationConstant;

  /** Update only every 'period' iterations */
  if( this->m_UpdateBDPeriod == 0 )
  {
    this->m_UpdateBDPeriod = static_cast< unsigned int >( vcl_floor( 1.0 / c_cov / Nd / 10.0 ) );
  }
  this->m_UpdateBDPeriod = vnl_math_max( static_cast< unsigned int >( 1 ), this->m_UpdateBDPeriod );
  if( this->m_UpdateBDPeriod >= this->m_MaximumNumberOfIterations )
  {
    this->SetUseCovarianceMatrixAdaptation( false );
  }

  /** m_EvolutionPathConstant (c_c)*/
  this->m_EvolutionPathConstant = 4.0 / ( Nd + 4.0 );

  /** m_ExpectationNormNormalDistribution */
  this->m_ExpectationNormNormalDistribution = vcl_sqrt( Nd )
    * ( 1.0 - 1.0 / ( 4.0 * Nd ) + 1.0 / ( 21.0 * vnl_math_sqr( Nd ) ) );

  /** m_HistoryLength */
  this->m_HistoryLength = static_cast< unsigned long >( vnl_math_min(
    this->GetMaximumNumberOfIterations(),
    10 + static_cast< unsigned long >( vcl_ceil( 3.0 * 10.0 * Nd / lambdad ) ) ) );

}   // end InitializeConstants


/**
 * ****************** InitializeProgressVariables *********************
 */

void
CMAEvolutionStrategyOptimizer::InitializeProgressVariables( void )
{
  itkDebugMacro( "InitializeProgressVariables" );

  /** Get the number of parameters from the cost function */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Some casts/aliases: */
  const unsigned int N      = numberOfParameters;
  const unsigned int lambda = this->m_PopulationSize;

  /** CurrentSigma */
  this->m_CurrentSigma = this->GetInitialSigma();

  /** Heaviside */
  this->m_Heaviside = 0.0;

  /** m_SearchDirs */
  ParametersType zeroParam( N );
  zeroParam.Fill( 0.0 );
  this->m_SearchDirs.clear();
  this->m_SearchDirs.resize( lambda, zeroParam );

  /** m_NormalizedSearchDirs */
  this->m_NormalizedSearchDirs.clear();
  this->m_NormalizedSearchDirs.resize( lambda, zeroParam );

  /** m_CostFunctionValues */
  this->m_CostFunctionValues.clear();

  /** m_CurrentScaledStep */
  this->m_CurrentScaledStep.SetSize( N );
  this->m_CurrentScaledStep.Fill( 0.0 );

  /** m_CurrentNormalizedStep */
  this->m_CurrentNormalizedStep.SetSize( N );
  this->m_CurrentNormalizedStep.Fill( 0.0 );

  /** m_EvolutionPath */
  this->m_EvolutionPath.SetSize( N );
  this->m_EvolutionPath.Fill( 0.0 );

  /** m_ConjugateEvolutionPath */
  this->m_ConjugateEvolutionPath.SetSize( N );
  this->m_ConjugateEvolutionPath.Fill( 0.0 );

  /** m_MeasureHistory */
  this->m_MeasureHistory.clear();

  /** Maximum and minimum square root eigenvalues */
  this->m_CurrentMaximumD = 1.0;
  this->m_CurrentMinimumD = 1.0;

}   // end InitializeProgressVariables


/**
 * ****************** InitializeBCD *********************
 */

void
CMAEvolutionStrategyOptimizer::InitializeBCD( void )
{
  itkDebugMacro( "InitializeBCD" );

  if( this->GetUseCovarianceMatrixAdaptation() )
  {
    /** Get the number of parameters from the cost function */
    const unsigned int numberOfParameters
      = this->GetScaledCostFunction()->GetNumberOfParameters();

    /** Some casts/aliases: */
    const unsigned int N = numberOfParameters;

    /** Resize */
    this->m_B.SetSize( N, N );
    this->m_C.SetSize( N, N );
    this->m_D.set_size( N );

    /** Initialize */
    this->m_B.Fill( 0.0 );
    this->m_C.Fill( 0.0 );
    this->m_B.fill_diagonal( 1.0 );
    this->m_C.fill_diagonal( 1.0 );
    this->m_D.fill( 1.0 );
  }
  else
  {
    /** Clear */
    this->m_B.SetSize( 0, 0 );
    this->m_C.SetSize( 0, 0 );
    this->m_D.clear();
  }

}   // end InitializeBCD


/**
 * ****************** GenerateOffspring *********************
 */

void
CMAEvolutionStrategyOptimizer::GenerateOffspring( void )
{
  itkDebugMacro( "GenerateOffspring" );

  /** Get the number of parameters from the cost function */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Some casts/aliases: */
  const unsigned int N      = numberOfParameters;
  const unsigned int lambda = this->m_PopulationSize;

  /** Clear the old values */
  this->m_CostFunctionValues.clear();

  /** Fill the m_NormalizedSearchDirs and SearchDirs */
  unsigned int lam       = 0;
  unsigned int nrOfFails = 0;
  while( lam < lambda )
  {
    /** draw from distribution N(0,I) */
    for( unsigned int par = 0; par < N; ++par )
    {
      this->m_NormalizedSearchDirs[ lam ][ par ]
        = this->m_RandomGenerator->GetNormalVariate();
    }
    /** Make like it was drawn from N(0,C) */
    if( this->GetUseCovarianceMatrixAdaptation() )
    {
      this->m_SearchDirs[ lam ] = this->m_B * ( this->m_D * this->m_NormalizedSearchDirs[ lam ] );
    }
    else
    {
      this->m_SearchDirs[ lam ] = this->m_NormalizedSearchDirs[ lam ];
    }
    /** Make like it was drawn from N( 0, sigma^2 C ) */
    this->m_SearchDirs[ lam ] *= this->m_CurrentSigma;

    /** Compute the cost function */
    MeasureType costFunctionValue = 0.0;
    /** x_lam = m + d_lam */
    ParametersType x_lam = this->GetScaledCurrentPosition();
    x_lam += this->m_SearchDirs[ lam ];
    try
    {
      costFunctionValue = this->GetScaledValue( x_lam );
    }
    catch( ExceptionObject & err )
    {
      ++nrOfFails;
      /** try another parameter vector if we haven't tried that for 10 times already */
      if( nrOfFails <= 10 )
      {
        continue;
      }
      else
      {
        this->m_StopCondition = MetricError;
        this->StopOptimization();
        throw err;
      }
    }
    /** Successfull cost function evaluation */
    this->m_CostFunctionValues.push_back(
      MeasureIndexPairType( costFunctionValue, lam ) );

    /** Reset the number of failed cost function evaluations */
    nrOfFails = 0;

    /** next offspring member */
    ++lam;
  }

}   // end GenerateOffspring


/**
 * ****************** SortCostFunctionValues *********************
 */

void
CMAEvolutionStrategyOptimizer::SortCostFunctionValues( void )
{
  itkDebugMacro( "SortCostFunctionValues" );

  /** Sort the cost function values in order of increasing cost function value */
  std::sort( this->m_CostFunctionValues.begin(), this->m_CostFunctionValues.end() );

  /** Store the best value in the history, and remove the oldest entry of the
   * the history if the history exceeds the HistoryLength */
  this->m_MeasureHistory.push_front( this->m_CostFunctionValues[ 0 ].first );
  if( this->m_MeasureHistory.size() > this->m_HistoryLength )
  {
    this->m_MeasureHistory.pop_back();
  }

}   // end SortCostFunctionValues


/**
 * ****************** AdvanceOneStep *********************
 */

void
CMAEvolutionStrategyOptimizer::AdvanceOneStep( void )
{
  itkDebugMacro( "AdvanceOneStep" );

  /** Some casts/aliases: */
  const unsigned int mu = this->m_NumberOfParents;

  /** Compute the CurrentScaledStep, using the RecombinationWeights and
   * the sorted CostFunctionValues-vector.
   * On the fly, also compute the CurrentNormalizedStep */
  this->m_CurrentScaledStep.Fill( 0.0 );
  this->m_CurrentNormalizedStep.Fill( 0.0 );
  for( unsigned int m = 0; m < mu; ++m )
  {
    const unsigned int lam    = this->m_CostFunctionValues[ m ].second;
    const double       weight = this->m_RecombinationWeights[ m ];
    this->m_CurrentScaledStep     += ( weight * this->m_SearchDirs[ lam ] );
    this->m_CurrentNormalizedStep += ( weight * this->m_NormalizedSearchDirs[ lam ] );
  }

  /** Set the new current position */
  ParametersType newPos = this->GetScaledCurrentPosition();
  newPos += this->GetCurrentScaledStep();
  this->SetScaledCurrentPosition( newPos );

  /** Compute the cost function at the new position */
  try
  {
    this->m_CurrentValue = this->GetScaledValue( this->GetScaledCurrentPosition() );
  }
  catch( ExceptionObject & err )
  {
    this->m_StopCondition = MetricError;
    this->StopOptimization();
    throw err;
  }
}   // end AdvanceOneStep


/**
 * ****************** UpdateConjugateEvolutionPath *********************
 */

void
CMAEvolutionStrategyOptimizer::UpdateConjugateEvolutionPath( void )
{
  itkDebugMacro( "UpdateConjugateEvolutionPath" );

  /** Some casts/aliases: */
  const double c_sigma = this->m_ConjugateEvolutionPathConstant;

  /** Update p_sigma */
  const double factor = vcl_sqrt( c_sigma * ( 2.0 - c_sigma ) * this->m_EffectiveMu );
  this->m_ConjugateEvolutionPath *= ( 1.0 - c_sigma );
  if( this->GetUseCovarianceMatrixAdaptation() )
  {
    this->m_ConjugateEvolutionPath
      += ( factor * ( this->m_B * this->m_CurrentNormalizedStep ) );
  }
  else
  {
    this->m_ConjugateEvolutionPath += ( factor * this->m_CurrentNormalizedStep );
  }

}   // end UpdateConjugateEvolutionPath


/**
 * ****************** UpdateHeaviside *********************
 */

void
CMAEvolutionStrategyOptimizer::UpdateHeaviside( void )
{
  itkDebugMacro( "UpdateHeaviside" );

  /** Get the number of parameters from the cost function */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Some casts/aliases: */
  const unsigned int N       = numberOfParameters;
  const double       Nd      = static_cast< double >( N );
  const double       c_sigma = this->m_ConjugateEvolutionPathConstant;
  const int          nextit  = static_cast< int >( this->GetCurrentIteration() + 1 );
  const double       chiN    = this->m_ExpectationNormNormalDistribution;

  /** Compute the Heaviside function: */
  this->m_Heaviside = false;
  const double normps        = this->m_ConjugateEvolutionPath.magnitude();
  const double denom         = vcl_sqrt( 1.0 - vcl_pow( 1.0 - c_sigma, 2 * nextit ) );
  const double righthandside = 1.5 + 1.0 / ( Nd - 0.5 );
  if( ( normps / denom / chiN ) < righthandside )
  {
    this->m_Heaviside = true;
  }

}   // end UpdateHeaviside


/**
 * ****************** UpdateEvolutionPath *********************
 */

void
CMAEvolutionStrategyOptimizer::UpdateEvolutionPath( void )
{
  itkDebugMacro( "UpdateEvolutionPath" );

  /** Some casts/aliases: */
  const double c_c = this->m_EvolutionPathConstant;

  /** Compute the evolution path p_c */
  this->m_EvolutionPath *= ( 1.0 - c_c );
  if( this->m_Heaviside )
  {
    const double factor
                           = vcl_sqrt( c_c * ( 2.0 - c_c ) * this->m_EffectiveMu ) / this->m_CurrentSigma;
    this->m_EvolutionPath += ( factor * this->m_CurrentScaledStep );
  }

}   // end UpdateEvolutionPath


/**
 * ****************** UpdateC *********************
 */

void
CMAEvolutionStrategyOptimizer::UpdateC( void )
{
  itkDebugMacro( "UpdateC" );

  if( !( this->GetUseCovarianceMatrixAdaptation() ) )
  {
    /** We don't need C */
    return;
  }

  /** Get the number of parameters from the cost function */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Some casts/aliases: */
  const unsigned int N      = numberOfParameters;
  const unsigned int mu     = this->m_NumberOfParents;
  const double       c_c    = this->m_EvolutionPathConstant;
  const double       c_cov  = this->m_CovarianceMatrixAdaptationConstant;
  const double       mu_cov = this->m_CovarianceMatrixAdaptationWeight;
  const double       sigma  = this->m_CurrentSigma;

  /** Multiply old m_C with some factor */
  double oldCfactor = 1.0 - c_cov;
  if( !this->m_Heaviside )
  {
    oldCfactor += ( c_cov * c_c * ( 2.0 - c_c ) / mu_cov );
  }
  this->m_C *= oldCfactor;

  /** Do rank-one update */
  const double rankonefactor = c_cov / mu_cov;
  for( unsigned int i = 0; i < N; ++i )
  {
    const double evolutionPath_i = this->m_EvolutionPath[ i ];
    for( unsigned int j = 0; j < N; ++j )
    {
      const double update = rankonefactor * evolutionPath_i * this->m_EvolutionPath[ j ];
      this->m_C[ i ][ j ] += update;
    }
  }

  /** Do rank-mu update */
  const double rankmufactor = c_cov * ( 1.0 - 1.0 / mu_cov );
  for( unsigned int m = 0; m < mu; ++m )
  {
    const unsigned int lam               = this->m_CostFunctionValues[ m ].second;
    const double       sqrtweight        = vcl_sqrt( this->m_RecombinationWeights[ m ] );
    ParametersType     weightedSearchDir = this->m_SearchDirs[ lam ];
    weightedSearchDir *= ( sqrtweight / sigma );
    for( unsigned int i = 0; i < N; ++i )
    {
      const double weightedSearchDir_i = weightedSearchDir[ i ];
      for( unsigned int j = 0; j < N; ++j )
      {
        const double update = rankmufactor * weightedSearchDir_i * weightedSearchDir[ j ];
        this->m_C[ i ][ j ] += update;
      }
    }
  }   // end for m

}   // end UpdateC


/**
 * ****************** UpdateSigma *********************
 */

void
CMAEvolutionStrategyOptimizer::UpdateSigma( void )
{
  itkDebugMacro( "UpdateSigma" );

  if( this->GetUseDecayingSigma() )
  {
    const double it  = static_cast< double >( this->GetCurrentIteration() );
    const double num = vcl_pow( this->m_SigmaDecayA + it, this->m_SigmaDecayAlpha );
    const double den = vcl_pow( this->m_SigmaDecayA + it + 1.0, this->m_SigmaDecayAlpha );
    this->m_CurrentSigma *= num / den;
  }
  else
  {
    const double normps  = this->m_ConjugateEvolutionPath.magnitude();
    const double chiN    = this->m_ExpectationNormNormalDistribution;
    const double c_sigma = this->m_ConjugateEvolutionPathConstant;
    const double d_sigma = this->m_SigmaDampingConstant;
    this->m_CurrentSigma *= vcl_exp( ( normps / chiN - 1.0 ) * c_sigma / d_sigma );
  }

}   // end UpdateSigma


/**
 * ****************** UpdateBD *********************
 */

void
CMAEvolutionStrategyOptimizer::UpdateBD( void )
{
  itkDebugMacro( "UpdateBD" );

  /** Get the number of parameters from the cost function */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Some casts/aliases: */
  const unsigned int N      = numberOfParameters;
  const int          nextit = static_cast< int >( this->GetCurrentIteration() + 1 );

  /** Update only every 'm_UpdateBDPeriod' iterations */
  unsigned int periodover = nextit % this->m_UpdateBDPeriod;

  if( !( this->GetUseCovarianceMatrixAdaptation() ) || ( periodover != 0 ) )
  {
    /** We don't need to update B and D */
    return;
  }

  typedef itk::SymmetricEigenAnalysis<
    CovarianceMatrixType,
    EigenValueMatrixType,
    CovarianceMatrixType >                      EigenAnalysisType;

  /** In the itkEigenAnalysis only the upper triangle of the matrix will be accessed, so
   * we do not need to make sure the matrix is symmetric, like in the
   * matlab code. Just run the eigenAnalysis! */
  EigenAnalysisType eigenAnalysis( N );
  unsigned int      returncode = 0;
  returncode = eigenAnalysis.ComputeEigenValuesAndVectors( this->m_C, this->m_D, this->m_B );
  if( returncode != 0 )
  {
    itkExceptionMacro( << "EigenAnalysis failed while computing eigenvalue nr: " << returncode );
  }

  /** itk eigen analysis returns eigen vectors in rows... */
  this->m_B.inplace_transpose();

  /**  limit condition of C to 1e10 + 1, and avoid negative eigenvalues */
  const double largeNumber = 1e10;
  double       dmax        = this->m_D.diagonal().max_value();
  double       dmin        = this->m_D.diagonal().min_value();
  if( dmin < 0.0 )
  {
    const double diagadd = dmax / largeNumber;
    for( unsigned int i = 0; i < N; ++i )
    {
      if( this->m_D[ i ] < 0.0 )
      {
        this->m_D[ i ] = 0.0;
      }
      this->m_C[ i ][ i ] += diagadd;
      this->m_D[ i ]      += diagadd;
    }
  }

  dmax = this->m_D.diagonal().max_value();
  dmin = this->m_D.diagonal().min_value();
  if( dmax > dmin * largeNumber )
  {
    const double diagadd = dmax / largeNumber  - dmin;
    for( unsigned int i = 0; i < N; ++i )
    {
      this->m_C[ i ][ i ] += diagadd;
      this->m_D[ i ]      += diagadd;
    }
  }

  /** the D matrix is supposed to contain the square root of the eigen values */
  for( unsigned int i = 0; i < N; ++i )
  {
    this->m_D[ i ] = vcl_sqrt( this->m_D[ i ] );
  }

  /** Keep for the user */
  this->m_CurrentMaximumD = this->m_D.diagonal().max_value();
  this->m_CurrentMinimumD = this->m_D.diagonal().min_value();

}   // end UpdateBD


/**
 * **************** FixNumericalErrors ********************
 */

void
CMAEvolutionStrategyOptimizer::FixNumericalErrors( void )
{
  itkDebugMacro( "FixNumericalErrors" );

  /** Get the number of parameters from the cost function */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Some casts/aliases: */
  const unsigned int N               = numberOfParameters;
  const double       c_sigma         = this->m_ConjugateEvolutionPathConstant;
  const double       c_cov           = this->m_CovarianceMatrixAdaptationConstant;
  const double       d_sigma         = this->m_SigmaDampingConstant;
  const double       strange_factor  = vcl_exp( 0.05 + c_sigma / d_sigma );
  const double       strange_factor2 = vcl_exp( 0.2 + c_sigma / d_sigma );
  const unsigned int nextit          = this->m_CurrentIteration + 1;

  /** Check if m_MaximumDeviation and m_MinimumDeviation are satisfied. This
   * check is different depending on the m_UseCovarianceMatrixAdaptation flag */
  if( this->GetUseCovarianceMatrixAdaptation() )
  {
    /** Check for too large deviation */
    for( unsigned int i = 0; i < N; ++i )
    {
      const double sqrtCii   = vcl_sqrt( this->m_C[ i ][ i ] );
      const double actualDev = this->m_CurrentSigma * sqrtCii;
      if( actualDev > this->m_MaximumDeviation )
      {
        this->m_CurrentSigma = this->m_MaximumDeviation / sqrtCii;
      }
    }

    /** Check for too small deviation */
    bool minDevViolated = false;
    for( unsigned int i = 0; i < N; ++i )
    {
      const double sqrtCii   = vcl_sqrt( this->m_C[ i ][ i ] );
      const double actualDev = this->m_CurrentSigma * sqrtCii;
      if( actualDev < this->m_MinimumDeviation )
      {
        this->m_CurrentSigma = this->m_MinimumDeviation / sqrtCii;
        minDevViolated       = true;
      }
    }
    if( minDevViolated )
    {
      /** \todo: does this make sense if m_UseDecayingSigma == true??
       * Anyway, we have to do something, in order to satisfy the minimum deviation */
      this->m_CurrentSigma *= strange_factor;
    }
  }
  else
  {
    /** If no covariance matrix adaptation is used, the check becomes simpler */

    /** Check for too large deviation */
    double actualDev = this->m_CurrentSigma;
    if( actualDev > this->m_MaximumDeviation )
    {
      this->m_CurrentSigma = this->m_MaximumDeviation;
    }
    /** Check for too small deviation */
    bool minDevViolated = false;
    actualDev = this->m_CurrentSigma;
    if( actualDev < this->m_MinimumDeviation )
    {
      this->m_CurrentSigma = this->m_MinimumDeviation;
      minDevViolated       = true;
    }
    if( minDevViolated  )
    {
      /** \todo: does this make sense if m_UseDecayingSigma == true??
       * Anyway, we have to do something, in order to satisfy the minimum deviation */
      this->m_CurrentSigma *= strange_factor;
    }

  }   // end else: no covariance matrix adaptation

  /** Adjust too low coordinate axis deviations that would cause numerical
   * problems (because of finite precision of the datatypes). This check
   * is different depending on the m_UseCovarianceMatrixAdaptation flag */
  const ParametersType & param                        = this->GetScaledCurrentPosition();
  bool                   numericalProblemsEncountered = false;
  if( this->GetUseCovarianceMatrixAdaptation() )
  {
    /** Check for numerically too small deviation */
    for( unsigned int i = 0; i < N; ++i )
    {
      const double actualDev = 0.2 * this->m_CurrentSigma * vcl_sqrt( this->m_C[ i ][ i ] );
      if( param[ i ] == ( param[ i ] + actualDev ) )
      {
        /** The parameters wouldn't change after perturbation, because
         * of too low precision. Increase the problematic diagonal element of C */
        this->m_C[ i ][ i ]         *= ( 1.0 + c_cov );
        numericalProblemsEncountered = true;
      }
    }   // end for i
  }
  else
  {
    const double actualDev = 0.2 * this->m_CurrentSigma;
    for( unsigned int i = 0; i < N; ++i )
    {
      if( param[ i ] == ( param[ i ] + actualDev ) )
      {
        /** The parameters wouldn't change after perturbation, because
        * of too low precision. Increase the sigma (equivalent to
        * increasing a diagonal element of C^0.5).  */
        this->m_CurrentSigma        *= vcl_sqrt( 1.0 + c_cov );
        numericalProblemsEncountered = true;
      }
    }
  }   // end else: no covariance matrix adaptation
  if( numericalProblemsEncountered )
  {
    /** \todo: does this make sense if m_UseDecayingSigma == true??
     * Anyway, we have to do something, in order to solve the numerical problems */
    this->m_CurrentSigma *= strange_factor;
  }

  /** Check if "main axis standard deviation sigma*D(i,i) has effect" (?),
   * with i = 1+floor(mod(countiter,N))
   * matlabcode: if all( xmean == xmean + 0.1*sigma*B*D(:,i) )
   * B*D(:,i) = i'th column of B times eigenvalue = i'th eigenvector * eigenvalue[i]
   * In the code below: colnr=i-1 (zero-based indexing). */
  bool               numericalProblemsEncountered2 = false;
  const unsigned int colnr                         = static_cast< unsigned int >( nextit % N );
  if( this->GetUseCovarianceMatrixAdaptation() )
  {
    const double sigDcol = 0.1 * this->m_CurrentSigma * this->m_D[ colnr ];
    //const ParametersType actualDevVector = sigDcol * this->m_B.get_column(colnr);
    const ParametersType::VnlVectorType actualDevVector = sigDcol * this->m_B.get_column( colnr );
    if( param == ( param + actualDevVector ) )
    {
      numericalProblemsEncountered2 = true;
    }
  }
  else
  {
    /** B and D are not used, so can be considered identity matrices.
     * This simplifies the check */
    const double sigDcol = 0.1 * this->m_CurrentSigma;
    if( param[ colnr ] == ( param[ colnr ] + sigDcol ) )
    {
      numericalProblemsEncountered2 = true;
    }
  }   // end else: no covariance matrix adaptation
  if( numericalProblemsEncountered2 )
  {
    /** \todo: does this make sense if m_UseDecayingSigma == true??
     * Anyway, we have to do something, in order to solve the numerical problems */
    this->m_CurrentSigma *= strange_factor2;
  }

  /** Adjust step size in case of equal function values (flat fitness) */

  /** The indices of the two population members whose cost function will
   * be compared */
  const unsigned int populationMemberA = 0;
  const unsigned int populationMemberB = static_cast< unsigned int >(
    vcl_ceil( 0.1 + static_cast< double >( this->m_PopulationSize ) / 4.0 ) );
  /** If they are the same: increase sigma with a magic factor */
  if( this->m_CostFunctionValues[ populationMemberA ].first ==
    this->m_CostFunctionValues[ populationMemberB ].first )
  {
    this->m_CurrentSigma *= strange_factor2;
  }

  /** Check if the best function value changes over iterations */
  if( this->m_MeasureHistory.size() > 1 )
  {
    const MeasureType maxhist = *max_element(
      this->m_MeasureHistory.begin(), this->m_MeasureHistory.end() );
    const MeasureType minhist = *min_element(
      this->m_MeasureHistory.begin(), this->m_MeasureHistory.end() );
    if( maxhist == minhist )
    {
      this->m_CurrentSigma *= strange_factor2;
    }
  }

}   // end FixNumericalErrors


/**
 * ********************* TestConvergence ************************
 */

bool
CMAEvolutionStrategyOptimizer::TestConvergence( bool firstCheck )
{
  itkDebugMacro( "TestConvergence" );

  /** Get the number of parameters from the cost function */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Some casts/aliases: */
  const unsigned int N = numberOfParameters;

  /** Check if the maximum number of iterations will not be exceeded in the following iteration */
  if( ( this->GetCurrentIteration() + 1 ) >= this->GetMaximumNumberOfIterations() )
  {
    this->m_StopCondition = MaximumNumberOfIterations;
    return true;
  }

  /** Check if the step was not too large:
   * if ( sigma * sqrt(C[i,i]) > PositionToleranceMax*sigma0   for any i ) */
  const double tolxmax      = this->m_PositionToleranceMax * this->m_InitialSigma;
  bool         stepTooLarge = false;
  if( this->GetUseCovarianceMatrixAdaptation() )
  {
    for( unsigned int i = 0; i < N; ++i )
    {
      const double sqrtCii  = vcl_sqrt( this->m_C[ i ][ i ] );
      const double stepsize =  this->m_CurrentSigma * sqrtCii;
      if( stepsize > tolxmax )
      {
        stepTooLarge = true;
        break;
      }
    }   // end for i
  }
  else
  {
    const double sqrtCii  = 1.0;
    const double stepsize =  this->m_CurrentSigma * sqrtCii;
    if( stepsize > tolxmax )
    {
      stepTooLarge = true;
    }
  }   // end else: if no covariance matrix adaptation
  if( stepTooLarge )
  {
    this->m_StopCondition = PositionToleranceMax;
    return true;
  }

  /** Check for zero steplength (should never happen):
   * if ( sigma * D[i] <= 0  for all i  ) */
  bool zeroStep = false;
  if( this->GetUseCovarianceMatrixAdaptation() )
  {
    if( ( this->m_CurrentSigma * this->m_D.diagonal().max_value() ) <= 0.0 )
    {
      zeroStep = true;
    }
  }
  else
  {
    if( this->m_CurrentSigma <= 0.0 )
    {
      zeroStep = true;
    }
  }
  if( zeroStep )
  {
    this->m_StopCondition = ZeroStepLength;
    return true;
  }

  /** The very first convergence check can not test for everything yet */
  if( firstCheck )
  {
    return false;
  }

  /** Check if the step was not too small:
   * if ( sigma * max( abs(p_c[i]), sqrt(C[i,i]) ) < PositionToleranceMin*sigma0  for all i ) */
  const double tolxmin      = this->m_PositionToleranceMin * this->m_InitialSigma;
  bool         stepTooSmall = true;
  for( unsigned int i = 0; i < N; ++i )
  {
    const double pci     = vcl_abs( this->m_EvolutionPath[ i ] );
    double       sqrtCii = 1.0;
    if( this->m_UseCovarianceMatrixAdaptation )
    {
      sqrtCii = vcl_sqrt( this->m_C[ i ][ i ] );
    }
    const double stepsize =  this->m_CurrentSigma * vnl_math_max( pci, sqrtCii );
    if( stepsize > tolxmin )
    {
      stepTooSmall = false;
      break;
    }
  }
  if( stepTooSmall )
  {
    this->m_StopCondition = PositionToleranceMin;
    return true;
  }

  /** Check if the best function value changes over iterations */
  if( this->m_MeasureHistory.size() > 10 )
  {
    const MeasureType maxhist = *max_element(
      this->m_MeasureHistory.begin(), this->m_MeasureHistory.end() );
    const MeasureType minhist = *min_element(
      this->m_MeasureHistory.begin(), this->m_MeasureHistory.end() );
    if( ( maxhist - minhist ) < this->m_ValueTolerance )
    {
      this->m_StopCondition = ValueTolerance;
      return true;
    }
  }

  return false;

}   // end TestConvergence


} // end namespace itk

#endif // #ifndef __itkCMAEvolutionStrategyOptimizer_cxx
