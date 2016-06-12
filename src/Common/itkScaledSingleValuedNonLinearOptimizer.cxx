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

#ifndef __itkScaledSingleValuedNonLinearOptimizer_cxx
#define __itkScaledSingleValuedNonLinearOptimizer_cxx

#include "itkScaledSingleValuedNonLinearOptimizer.h"

namespace itk
{

/**
 * ****************** Constructor *********************************
 */

ScaledSingleValuedNonLinearOptimizer
::ScaledSingleValuedNonLinearOptimizer()
{
  this->m_Maximize           = false;
  this->m_ScaledCostFunction = ScaledCostFunctionType::New();

} // end Constructor


/**
* ****************** InitializeScales ******************************
*/

void
ScaledSingleValuedNonLinearOptimizer
::InitializeScales( void )
{
  /** NB: we assume the scales entered by the user are meant
   * as squared scales (following the ITK convention)!
   */
  this->m_ScaledCostFunction->SetSquaredScales( this->GetScales() );
  this->Modified();

} // end InitializeScales()


/**
 * ****************** SetCostFunction ******************************
 */

void
ScaledSingleValuedNonLinearOptimizer
::SetCostFunction( CostFunctionType * costFunction )
{
  this->m_ScaledCostFunction->SetUnscaledCostFunction( costFunction );
  this->Superclass::SetCostFunction( costFunction );

} // end SetCostFunction()


/**
 * ********************* SetUseScales ******************************
 */

void
ScaledSingleValuedNonLinearOptimizer
::SetUseScales( bool arg )
{
  this->m_ScaledCostFunction->SetUseScales( arg );
  this->Modified();

} // end SetUseScales()


/**
 * ********************* GetUseScales ******************************
 */

bool
ScaledSingleValuedNonLinearOptimizer
::GetUseScales( void ) const
{
  return this->m_ScaledCostFunction->GetUseScales();

} // end GetUseScales()


/**
 * ********************* GetScaledValue *****************************
 */

ScaledSingleValuedNonLinearOptimizer::MeasureType
ScaledSingleValuedNonLinearOptimizer
::GetScaledValue( const ParametersType & parameters ) const
{
  return this->m_ScaledCostFunction->GetValue( parameters );

} // end GetScaledValue()


/**
 * ********************* GetScaledDerivative *****************************
 */

void
ScaledSingleValuedNonLinearOptimizer
::GetScaledDerivative(
  const ParametersType & parameters,
  DerivativeType & derivative ) const
{
  this->m_ScaledCostFunction->GetDerivative( parameters, derivative );

} // end GetScaledDerivative()


/**
 * ********************* GetScaledValueAndDerivative ***********************
 */

void
ScaledSingleValuedNonLinearOptimizer
::GetScaledValueAndDerivative(
  const ParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  this->m_ScaledCostFunction->
  GetValueAndDerivative( parameters, value, derivative );

} // end GetScaledValueAndDerivative()


/**
 * ********************* GetCurrentPosition ***********************
 */

const ScaledSingleValuedNonLinearOptimizer::ParametersType &
ScaledSingleValuedNonLinearOptimizer
::GetCurrentPosition( void ) const
{
  /** Get the current unscaled position. */
  const ParametersType & scaledCurrentPosition
    = this->GetScaledCurrentPosition();

  if( this->GetUseScales() )
  {
    /** Get the ScaledCurrentPosition and divide each
     * element through its scale. */
    this->m_UnscaledCurrentPosition = scaledCurrentPosition;
    this->m_ScaledCostFunction->
    ConvertScaledToUnscaledParameters( this->m_UnscaledCurrentPosition );

    return this->m_UnscaledCurrentPosition;
  }
  else
  {
    /** If no scaling is used, simply return the
     * ScaledCurrentPosition, since it is not scaled anyway
     */
    return scaledCurrentPosition;
  }

} // end GetCurrentPosition()


/**
 * ***************** SetScaledCurrentPosition *********************
 */

void
ScaledSingleValuedNonLinearOptimizer
::SetScaledCurrentPosition( const ParametersType & parameters )
{
  itkDebugMacro( "setting scaled current position to " << parameters );
  this->m_ScaledCurrentPosition = parameters; // slow copy
  this->Modified();

} // end SetScaledCurrentPosition()


/**
 * *********************** SetCurrentPosition *********************
 */

void
ScaledSingleValuedNonLinearOptimizer
::SetCurrentPosition( const ParametersType & param )
{
  /** Multiply the argument by the scales and set it as the
   * the ScaledCurrentPosition.
   */
  if( this->GetUseScales() )
  {
    ParametersType scaledParameters = param;
    this->m_ScaledCostFunction
    ->ConvertUnscaledToScaledParameters( scaledParameters );
    this->SetScaledCurrentPosition( scaledParameters );
  }
  else
  {
    this->SetScaledCurrentPosition( param );
  }

} // end SetCurrentPosition()


/**
 * ******************** SetMaximize *******************************
 */

void
ScaledSingleValuedNonLinearOptimizer
::SetMaximize( bool _arg )
{
  itkDebugMacro( "Setting Maximize to " << _arg );
  if( this->m_Maximize != _arg )
  {
    this->m_Maximize = _arg;
    this->m_ScaledCostFunction->SetNegateCostFunction( _arg );
    this->Modified();
  }
}  // end SetMaximize()


/**
 * ******************** PrintSelf *******************************
 */

void
ScaledSingleValuedNonLinearOptimizer
::PrintSelf( std::ostream & os, Indent indent ) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf( os, indent );

  os << indent << "ScaledCurrentPosition: "
     << this->m_ScaledCurrentPosition << std::endl;
  os << indent << "UnscaledCurrentPosition: "
     << this->m_UnscaledCurrentPosition << std::endl;
  os << indent << "ScaledCostFunction: "
     << this->m_ScaledCostFunction.GetPointer() << std::endl;
  os << indent << "Maximize: "
     << ( this->m_Maximize ? "true" : "false" ) << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // #ifndef __itkScaledSingleValuedNonLinearOptimizer_cxx
