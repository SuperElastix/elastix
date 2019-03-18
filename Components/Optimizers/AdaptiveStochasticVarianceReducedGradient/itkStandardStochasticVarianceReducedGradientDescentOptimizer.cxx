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
#ifndef __itkStandardStochasticVarianceReducedGradientOptimizer_cxx
#define __itkStandardStochasticVarianceReducedGradientOptimizer_cxx

#include "itkStandardStochasticVarianceReducedGradientDescentOptimizer.h"

#include "vnl/vnl_math.h"

namespace itk
{

/**
 * ************************* Constructor ************************
 */

StandardStochasticVarianceReducedGradientOptimizer
::StandardStochasticVarianceReducedGradientOptimizer()
{
  this->m_Param_a = 1.0;
  this->m_Param_A = 1.0;
  this->m_Param_alpha = 0.602;

  this->m_CurrentTime = 0.0;
  this->m_InitialTime = 0.0;

} // end Constructor


/**
 * ********************** StartOptimization *********************
 */

void StandardStochasticVarianceReducedGradientOptimizer::StartOptimization( void )
{
  this->m_CurrentTime = this->m_InitialTime;
  this->Superclass::StartOptimization();
} // end StartOptimization


/**
 * ******************** AdvanceOneStep **************************
 */

void
StandardStochasticVarianceReducedGradientOptimizer
::AdvanceOneStep( void )
{

  this->SetLearningRate( this->Compute_a( this->m_CurrentTime ) );

  this->Superclass::AdvanceOneStep();

  this->UpdateCurrentTime();

} // end AdvanceOneStep


/**
 * ************************** Compute_a *************************
 */

double StandardStochasticVarianceReducedGradientOptimizer
::Compute_a( double k ) const
{
  return static_cast<double>(
    this->m_Param_a / std::pow( this->m_Param_A + k + 1.0, this->m_Param_alpha ) );

} // end Compute_a


/**
 * ************************** Compute_beta *************************
 */

double StandardStochasticVarianceReducedGradientOptimizer
::Compute_beta( double k ) const
{
  return static_cast<double>(
    this->m_Param_beta / std::pow( this->m_Param_A + k + 1.0, this->m_Param_alpha ) );

} // end Compute_beta


/**
 * ************************** UpdateCurrentTime ********************
 */

void StandardStochasticVarianceReducedGradientOptimizer
::UpdateCurrentTime( void )
{
  /** Simply Robbins-Monro: time=iterationnr. */
  this->m_CurrentTime += 1.0;

} // end UpdateCurrentTime


} // end namespace itk

#endif // end #ifndef __itkStandardStochasticVarianceReducedGradientOptimizer_cxx
