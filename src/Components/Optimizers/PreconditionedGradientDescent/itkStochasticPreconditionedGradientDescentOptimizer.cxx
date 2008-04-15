/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkStochasticPreconditionedGradientDescentOptimizer_cxx
#define __itkStochasticPreconditionedGradientDescentOptimizer_cxx

#include "itkStochasticPreconditionedGradientDescentOptimizer.h"
#include "vnl/vnl_math.h"

namespace itk
{

  /**
  * ************************* Constructor ************************
  */

  StochasticPreconditionedGradientDescentOptimizer
    ::StochasticPreconditionedGradientDescentOptimizer()
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

  void StochasticPreconditionedGradientDescentOptimizer::StartOptimization(void)
  {
    this->m_CurrentTime = this->m_InitialTime;
    this->Superclass::StartOptimization();
  } // end StartOptimization


  /**
  * ******************** AdvanceOneStep **************************
  */

  void StochasticPreconditionedGradientDescentOptimizer::AdvanceOneStep(void)
  {

    this->SetLearningRate( this->Compute_a( this->m_CurrentTime ) );

    this->Superclass::AdvanceOneStep();

    this->UpdateCurrentTime();

  } // end AdvanceOneStep


  /**
  * ************************** Compute_a *************************
  *
  * This function computes the parameter a at iteration/time k, as
  * described by Spall.
  */

  double StochasticPreconditionedGradientDescentOptimizer
    ::Compute_a(double k) const
  { 
    return static_cast<double>(
      this->m_Param_a / vcl_pow( this->m_Param_A + k + 1.0, this->m_Param_alpha ) );

  } // end Compute_a


  /**
  * ************************** UpdateCurrentTime ********************
  *
  * This function computes the input for the Compute_a function.
  */

  void StochasticPreconditionedGradientDescentOptimizer
    ::UpdateCurrentTime( void )
  {    
    /** Simply Robbins-Monro: time=iterationnr. */
    this->m_CurrentTime += 1.0;

  } // end UpdateCurrentTime


} // end namespace itk

#endif // end #ifndef __itkStochasticPreconditionedGradientDescentOptimizer_cxx

