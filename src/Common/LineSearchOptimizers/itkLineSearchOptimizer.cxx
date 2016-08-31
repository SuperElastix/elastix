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

#ifndef __itkLineSearchOptimizer_cxx
#define __itkLineSearchOptimizer_cxx

#include "itkLineSearchOptimizer.h"
#include "itkNumericTraits.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

LineSearchOptimizer
::LineSearchOptimizer()
{
  this->m_CurrentStepLength         = NumericTraits< double >::Zero;
  this->m_MinimumStepLength         = NumericTraits< double >::Zero;
  this->m_MaximumStepLength         = NumericTraits< double >::max();
  this->m_InitialStepLengthEstimate = NumericTraits< double >::One;

} // end Constructor


/**
 * ***************** SetCurrentStepLength *************
 *
 * Set the current step length AND the current position, where
 * the current position is computed as:
 * m_CurrentPosition =
 * m_InitialPosition + StepLength * m_LineSearchDirection
 */

void
LineSearchOptimizer
::SetCurrentStepLength( double step )
{
  itkDebugMacro( "Setting current step length to " << step );

  this->m_CurrentStepLength = step;

  ParametersType         newPosition        =  this->GetInitialPosition();
  const unsigned int     numberOfParameters = newPosition.GetSize();
  const ParametersType & LSD                = this->GetLineSearchDirection();

  for( unsigned int i = 0; i < numberOfParameters; ++i )
  {
    newPosition[ i ] += ( step * LSD[ i ] );
  }

  this->SetCurrentPosition( newPosition );

} // end SetCurrentStepLength()


/**
 * ******************** DirectionalDerivative **************************
 *
 * Computes the inner product of the argument and the line search direction
 */

double
LineSearchOptimizer
::DirectionalDerivative( const DerivativeType & derivative ) const
{
  /** Easy, thanks to the functions defined in vnl_vector.h */
  return inner_product( derivative, this->GetLineSearchDirection() );

} // end DirectionalDerivative()


/**
 * ******************** PrintSelf **************************
 */

void
LineSearchOptimizer
::PrintSelf( std::ostream & os, Indent indent ) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf( os, indent );

  os << indent << "CurrentStepLength: "
     << this->m_CurrentStepLength << std::endl;
  os << indent << "MinimumStepLength: "
     << this->m_MinimumStepLength << std::endl;
  os << indent << "MaximumStepLength: "
     << this->m_MaximumStepLength << std::endl;
  os << indent << "InitialStepLengthEstimate: "
     << this->m_InitialStepLengthEstimate << std::endl;
  os << indent << "LineSearchDirection: "
     << this->m_LineSearchDirection << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // #ifndef __itkLineSearchOptimizer_cxx
