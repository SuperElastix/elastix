/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
