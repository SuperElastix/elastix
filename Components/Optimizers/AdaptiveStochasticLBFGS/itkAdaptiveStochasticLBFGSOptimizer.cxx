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
#ifndef __itkAdaptiveStochasticLBFGSOptimizer_cxx
#define __itkAdaptiveStochasticLBFGSOptimizer_cxx

#include "itkAdaptiveStochasticLBFGSOptimizer.h"

#include "vnl/vnl_math.h"
#include "itkSigmoidImageFilter.h"

namespace itk
{

/**
 * ************************* Constructor ************************
 */

AdaptiveStochasticLBFGSOptimizer
::AdaptiveStochasticLBFGSOptimizer()
{
  this->m_UseAdaptiveStepSizes = true;
  this->m_SigmoidMax = 1.0;
  this->m_SigmoidMin = -0.8;
  this->m_SigmoidScale = 1e-8;
  this->m_SearchLengthScale = 10;

} // end Constructor


/**
 * ************************** UpdateCurrentTime ********************
 */

void
AdaptiveStochasticLBFGSOptimizer
::UpdateCurrentTime( void )
{
  typedef itk::Functor::Sigmoid<double, double> SigmoidType;

  /** Create a sigmoid function. */
  SigmoidType sigmoid;
  sigmoid.SetOutputMaximum( this->GetSigmoidMax() );
  sigmoid.SetOutputMinimum( this->GetSigmoidMin() );
  sigmoid.SetAlpha( this->GetSigmoidScale() );
  const double beta = this->GetSigmoidScale() *
    std::log( - this->GetSigmoidMax() / this->GetSigmoidMin() );
  sigmoid.SetBeta( beta );

  /** to be cleaned about the logical issues. */
  /** Adaptive step size. */
  if( this->m_UseAdaptiveStepSizes && (!this->m_UseSearchDirForAdaptiveStepSize) )
  {
    if( this->GetCurrentIteration() > 0 )
    {
      /** Formula (2) in Cruz: <g_k, g_{k-1}>. */
      const double inprod = inner_product( this->m_PreviousGradient, this->GetGradient() );
      this->m_CurrentTime += sigmoid( -inprod );
      this->m_CurrentTime = vnl_math_max( 0.0, this->m_CurrentTime );
    }

    /** Save for next iteration */
    this->m_PreviousGradient = this->GetGradient();
  }
  else if( this->m_UseAdaptiveStepSizes && this->m_UseSearchDirForAdaptiveStepSize )
  {
    if( (this->GetCurrentIteration()+1) % this->m_UpdateFrequenceL != 0 )
    {
      /** Formula (2) in Cruz: <d_k, d_{k-1}>. */
      const DerivativeType & searchDir = this->GetSearchDir();
      //const double inprod = inner_product( this->m_PreviousSearchDir,  searchDir );
      /** test <g_k, d_k>, only using the information of current gradient and search direction. */
      const double inprod = inner_product( this->GetGradient(),  searchDir );
      this->m_CurrentTime += sigmoid( -inprod );
      this->m_CurrentTime = vnl_math_max( 0.0, this->m_CurrentTime );
    }

    /** Save for next iteration */
    this->m_PreviousSearchDir = this->GetSearchDir();
    this->m_PreviousGradient = this->GetGradient();
  }
  /** Decaying or constant step size. */
  else if ( this->m_StepSizeStrategy == "Decaying")
  {
    /** Almost Robbins-Monro: time = time + E_0.
    * If you want the parameter estimation but no adaptive stuff,
    * this may be use useful:  */
    //this->m_CurrentTime += ( this->GetSigmoidMax() + this->GetSigmoidMin() ) / 2.0;
    this->m_CurrentTime += 1.0;
  }
  else if ( this->m_StepSizeStrategy == "Constant")
  {
    this->m_CurrentTime = 0.0;
  }
} // end UpdateCurrentTime()


} // end namespace itk

#endif // end #ifndef __itkAdaptiveStochasticLBFGSOptimizer_cxx
