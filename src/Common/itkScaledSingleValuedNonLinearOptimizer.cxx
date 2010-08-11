/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkScaledSingleValuedNonLinearOptimizer_cxx
#define __itkScaledSingleValuedNonLinearOptimizer_cxx

#include "itkScaledSingleValuedNonLinearOptimizer.h"

namespace itk
{

  /**
   * ****************** Constructor *********************************
   */

  ScaledSingleValuedNonLinearOptimizer::
    ScaledSingleValuedNonLinearOptimizer()
  {
    this->m_Maximize = false;
    this->m_ScaledCostFunction = ScaledCostFunctionType::New();

  } // end constructor


  /**
   * ****************** InitializeScales ******************************
   */

  void
    ScaledSingleValuedNonLinearOptimizer::InitializeScales()
  {
    /** NB: we assume the scales entered by the user are meant
     * as squared scales (following the ITK convention)! */
    this->m_ScaledCostFunction->SetSquaredScales( this->GetScales() );
    this->Modified();
  } // end InitializeScales


  /**
   * ****************** SetCostFunction ******************************
   */

  void
    ScaledSingleValuedNonLinearOptimizer::
    SetCostFunction(CostFunctionType * costFunction)
  {
    this->m_ScaledCostFunction->SetUnscaledCostFunction( costFunction );
    this->Superclass::SetCostFunction(costFunction);
  } //end SetCostFunction


  /**
   * ********************* SetUseScales ******************************
   */

  void
    ScaledSingleValuedNonLinearOptimizer::SetUseScales(bool arg)
  {
    this->m_ScaledCostFunction->SetUseScales(arg);
    this->Modified();
  } // end SetUseScales



  /**
   * ********************* GetUseScales ******************************
   */

  const bool
    ScaledSingleValuedNonLinearOptimizer::GetUseScales(void) const
  {
    return this->m_ScaledCostFunction->GetUseScales();
  } // end SetUseScales


  /**
   * ********************* GetScaledValue *****************************
   */

  ScaledSingleValuedNonLinearOptimizer::MeasureType
    ScaledSingleValuedNonLinearOptimizer::
    GetScaledValue( const ParametersType & parameters ) const
  {
    return this->m_ScaledCostFunction->GetValue(parameters);
  } // end GetScaledValue


  /**
   * ********************* GetScaledDerivative *****************************
   */

  void
    ScaledSingleValuedNonLinearOptimizer::
    GetScaledDerivative(
      const ParametersType & parameters,
      DerivativeType & derivative ) const
  {
    this->m_ScaledCostFunction->GetDerivative(parameters, derivative);
  } // end GetScaledDerivative


  /**
   * ********************* GetScaledValueAndDerivative ***********************
   */

  void
    ScaledSingleValuedNonLinearOptimizer::
    GetScaledValueAndDerivative(
      const ParametersType & parameters,
      MeasureType & value,
      DerivativeType & derivative ) const
  {
    this->m_ScaledCostFunction->
      GetValueAndDerivative(parameters, value, derivative);
  } // end GetScaledValueAndDerivative


  /**
   * ********************* GetCurrentPosition ***********************
   */

  const ScaledSingleValuedNonLinearOptimizer::ParametersType &
    ScaledSingleValuedNonLinearOptimizer::GetCurrentPosition() const
  {
    /** Get the current unscaled position */

    const ParametersType & scaledCurrentPosition =
        this->GetScaledCurrentPosition();

    if ( this->GetUseScales() )
    {
      /** Get the ScaledCurrentPosition and divide each
        * element through its scale. */

      m_UnscaledCurrentPosition = scaledCurrentPosition;
      this->m_ScaledCostFunction->
        ConvertScaledToUnscaledParameters(m_UnscaledCurrentPosition);

      return m_UnscaledCurrentPosition;

    }
    else
    {
      /** If no scaling is used, simply return the
       * ScaledCurrentPosition, since it is not scaled anyway
       */
      return scaledCurrentPosition;
    }

  } // end GetCurrentPosition


  /**
   * ***************** SetScaledCurrentPosition *********************
   */

  void
    ScaledSingleValuedNonLinearOptimizer::
    SetScaledCurrentPosition(const ParametersType & parameters)
  {
    itkDebugMacro("setting scaled current position to " << parameters);
    this->m_ScaledCurrentPosition = parameters;
    this->Modified();
  } // end SetScaledCurrentPosition


  /**
   * *********************** SetCurrentPosition *********************
   */

  void
    ScaledSingleValuedNonLinearOptimizer::
    SetCurrentPosition (const ParametersType &param)
  {
    /** Multiply the argument by the scales and set it as the
     * the ScaledCurrentPosition */

    if ( this->GetUseScales() )
    {
      ParametersType scaledParameters = param;
      this->m_ScaledCostFunction->
        ConvertUnscaledToScaledParameters(scaledParameters);
      this->SetScaledCurrentPosition(scaledParameters);
    }
    else
    {
      this->SetScaledCurrentPosition(param);
    }

  } // end SetCurrentPosition


  /**
   * ******************** SetMaximize *******************************
   */

  void
    ScaledSingleValuedNonLinearOptimizer::
    SetMaximize( bool _arg )
  {
    itkDebugMacro("Setting Maximize to " << _arg);
    if ( this->m_Maximize != _arg )
    {
      this->m_Maximize = _arg;
      this->m_ScaledCostFunction->SetNegateCostFunction(_arg);
      this->Modified();
    }
  }  // end SetMaximize



} // end namespace itk

#endif // #ifndef __itkScaledSingleValuedNonLinearOptimizer_cxx

