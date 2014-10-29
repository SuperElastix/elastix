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

#ifndef __itkScaledSingleValuedCostFunction_cxx
#define __itkScaledSingleValuedCostFunction_cxx

#include "itkScaledSingleValuedCostFunction.h"
#include "vnl/vnl_math.h"

namespace itk
{

/**
 * **************** Constructor *****************************
 */

ScaledSingleValuedCostFunction
::ScaledSingleValuedCostFunction()
{
  this->m_UnscaledCostFunction = 0;
  this->m_UseScales            = false;
  this->m_NegateCostFunction   = false;

} // end Constructor


/**
 * ******************** GetValue *****************************
 */

ScaledSingleValuedCostFunction::MeasureType
ScaledSingleValuedCostFunction
::GetValue( const ParametersType & parameters ) const
{
  /** F(y)= f(y/s) */

  /** This function also checks if the UnscaledCostFunction has been set */
  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  if( parameters.GetSize() != numberOfParameters )
  {
    itkExceptionMacro( << "Number of parameters is not like the unscaled cost function expects." );
  }

  MeasureType returnvalue = NumericTraits< MeasureType >::Zero;

  if( this->m_UseScales )
  {
    ParametersType scaledParameters = parameters;
    this->ConvertScaledToUnscaledParameters( scaledParameters );
    returnvalue = this->m_UnscaledCostFunction->GetValue( scaledParameters );
  }
  else
  {
    returnvalue = this->m_UnscaledCostFunction->GetValue( parameters );
  }

  if( this->GetNegateCostFunction() )
  {
    return -returnvalue;
  }
  return returnvalue;

} // end GetValue()


/**
 * ******************** GetDerivative **************************
 */

void
ScaledSingleValuedCostFunction
::GetDerivative( const ParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** dF/dy(y)= 1/s * df/dx(y/s) */

  /** This function also checks if the UnscaledCostFunction has been set */
  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  if( parameters.GetSize() != numberOfParameters )
  {
    itkExceptionMacro( << "Number of parameters is not like the unscaled cost function expects." );
  }

  if( this->m_UseScales )
  {
    ParametersType scaledParameters = parameters;
    this->ConvertScaledToUnscaledParameters( scaledParameters );
    this->m_UnscaledCostFunction->GetDerivative( scaledParameters, derivative );

    const ScalesType & scales = this->GetScales();
    for( unsigned int i = 0; i < numberOfParameters; ++i )
    {
      derivative[ i ] /= scales[ i ];
    }
  }
  else
  {
    m_UnscaledCostFunction->GetDerivative( parameters, derivative );
  }

  if( this->GetNegateCostFunction() )
  {
    derivative = -derivative;
  }

} // end GetDerivative()


/**
 * **************** GetValueAndDerivative ************************
 */

void
ScaledSingleValuedCostFunction
::GetValueAndDerivative( const ParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  /** F(y)= f(y/s) */
  /** dF/dy(y)= 1/s * df/dx(y/s) */

  /** This function also checks if the UnscaledCostFunction has been set */
  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  if( parameters.GetSize() != numberOfParameters )
  {
    itkExceptionMacro( << "Number of parameters is not like the unscaled cost function expects." );
  }

  if( this->m_UseScales )
  {

    ParametersType scaledParameters = parameters;
    this->ConvertScaledToUnscaledParameters( scaledParameters );
    this->m_UnscaledCostFunction->GetValueAndDerivative( scaledParameters, value, derivative );

    const ScalesType & scales = this->GetScales();
    for( unsigned int i = 0; i < numberOfParameters; ++i )
    {
      derivative[ i ] /= scales[ i ];
    }
  }
  else
  {
    this->m_UnscaledCostFunction->GetValueAndDerivative( parameters, value, derivative );
  }

  if( this->GetNegateCostFunction() )
  {
    value      = -value;
    derivative = -derivative;
  }

} // end GetValueAndDerivative()


/**
 * **************** GetNumberOfParameters ************************
 */

unsigned int
ScaledSingleValuedCostFunction
::GetNumberOfParameters( void ) const
{
  if( this->m_UnscaledCostFunction.IsNull() )
  {
    itkExceptionMacro( << "UnscaledCostFunction has not been set!" );
  }
  return this->m_UnscaledCostFunction->GetNumberOfParameters();

} // end GetNumberOfParameters()


/**
 * **************** SetScales **********************************
 */

void
ScaledSingleValuedCostFunction
::SetScales( const ScalesType & scales )
{
  itkDebugMacro( "setting scales to " << scales );
  this->m_Scales = scales;
  this->m_SquaredScales.SetSize( scales.GetSize() );
  for( unsigned int i = 0; i < scales.Size(); ++i )
  {
    this->m_SquaredScales[ i ] = vnl_math_sqr( scales[ i ] );
  }
  this->Modified();

} // end SetScales()


/**
 * **************** SetSquaredScales *****************************
 */

void
ScaledSingleValuedCostFunction
::SetSquaredScales( const ScalesType & squaredScales )
{
  itkDebugMacro( "setting squared scales to " << squaredScales );
  this->m_SquaredScales = squaredScales;
  this->m_Scales.SetSize( squaredScales.GetSize() );
  for( unsigned int i = 0; i < squaredScales.Size(); ++i )
  {
    this->m_Scales[ i ] = vcl_sqrt( squaredScales[ i ] );
  }
  this->Modified();

} // end SetSquaredScales()


/**
 * *************** ConvertScaledToUnscaledParameters ********************
 */

void
ScaledSingleValuedCostFunction
::ConvertScaledToUnscaledParameters( ParametersType & parameters ) const
{
  if( this->m_UseScales )
  {
    const unsigned int numberOfParameters = parameters.GetSize();
    const ScalesType & scales             = this->GetScales();
    if( scales.GetSize() != numberOfParameters )
    {
      itkExceptionMacro( << "Number of scales is not correct." );
    }

    for( unsigned int i = 0; i < numberOfParameters; ++i )
    {
      parameters[ i ] /= scales[ i ];
    }

  } // end if use scales

} // end ConvertScaledToUnscaledParameters()


/**
 * *************** ConvertUnscaledToScaledParameters ********************
 */

void
ScaledSingleValuedCostFunction
::ConvertUnscaledToScaledParameters( ParametersType & parameters ) const
{
  if( this->m_UseScales )
  {
    const unsigned int numberOfParameters = parameters.GetSize();
    const ScalesType & scales             = this->GetScales();
    if( scales.GetSize() != numberOfParameters )
    {
      itkExceptionMacro( << "Number of scales is not correct." );
    }

    for( unsigned int i = 0; i < numberOfParameters; ++i )
    {
      parameters[ i ] *= scales[ i ];
    }

  } // end if use scales

} // end ConvertUnscaledToScaledParameters()


/**
 * *************** PrintSelf ********************
 */

void
ScaledSingleValuedCostFunction
::PrintSelf( std::ostream & os, Indent indent ) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf( os, indent );

  os << indent << "UseScales: "
     << ( this->m_UseScales ? "true" : "false" ) << std::endl;
  os << indent << "Scales: " << this->m_Scales << std::endl;
  os << indent << "SquaredScales: " << this->m_SquaredScales << std::endl;
  os << indent << "NegateCostFunction: "
     << ( this->m_NegateCostFunction ? "true" : "false" ) << std::endl;
  os << indent << "UnscaledCostFunction: "
     << this->m_UnscaledCostFunction.GetPointer() << std::endl;

} // end PrintSelf()


} //end namespace itk

#endif // #ifndef __itkScaledSingleValuedCostFunction_cxx
