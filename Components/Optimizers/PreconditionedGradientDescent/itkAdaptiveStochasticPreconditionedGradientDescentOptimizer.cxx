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
#ifndef __itkAdaptiveStochasticPreconditionedGradientDescentOptimizer_cxx
#define __itkAdaptiveStochasticPreconditionedGradientDescentOptimizer_cxx

#include "itkAdaptiveStochasticPreconditionedGradientDescentOptimizer.h"

#include "vnl/vnl_math.h"

namespace itk
{
/**
 * ************************* Constructor ************************
 */

AdaptiveStochasticPreconditionedGradientDescentOptimizer
::AdaptiveStochasticPreconditionedGradientDescentOptimizer()
{
  this->m_UseAdaptiveStepSizes = true;
  this->m_SigmoidMax = 1.0;
  this->m_SigmoidMin = -0.8;
  this->m_SigmoidScale = 1e-8;
} // end Constructor


/**
 * ************************** UpdateCurrentTime ********************
 */

void
AdaptiveStochasticPreconditionedGradientDescentOptimizer
::UpdateCurrentTime( void )
{
  typedef itk::Functor::Sigmoid<double, double> SigmoidType;

  if( this->m_UseAdaptiveStepSizes )
  {
    if( this->GetCurrentIteration() > 0 )
    {
      /** Make sigmoid function
       * Compute beta such that sigmoid(0)=0
       * We assume Max>0, min<0 */
      SigmoidType sigmoid;
      sigmoid.SetOutputMaximum( this->GetSigmoidMax() );
      sigmoid.SetOutputMinimum( this->GetSigmoidMin() );
      sigmoid.SetAlpha( this->GetSigmoidScale() );
      const double beta = this->GetSigmoidScale() *
        std::log( - this->GetSigmoidMax() / this->GetSigmoidMin() );
      sigmoid.SetBeta( beta );

      /** Formula (2) in Cruz */
      const double inprod = inner_product(
        this->m_PreviousSearchDirection, this->GetGradient() );
      this->m_CurrentTime += sigmoid(-inprod);
      this->m_CurrentTime = vnl_math_max( 0.0, this->m_CurrentTime );
    }

    /** Save for next iteration */
    this->m_PreviousSearchDirection = this->GetSearchDirection();
  }
  else
  {
    /** Almost Robbins-Monro: time = time + E_0.
     * If you want the parameter estimation but no adaptive stuff,
     * this may be use useful:  */
    this->m_CurrentTime += ( this->GetSigmoidMax() + this->GetSigmoidMin() ) / 2.0;
  }

} // end UpdateCurrentTime()


} // end namespace itk

#endif // end #ifndef __itkAdaptiveStochasticPreconditionedGradientDescentOptimizer_cxx
