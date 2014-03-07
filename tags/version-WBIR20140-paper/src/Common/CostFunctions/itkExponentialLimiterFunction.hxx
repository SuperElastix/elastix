/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkExponentialLimiterFunction_hxx
#define __itkExponentialLimiterFunction_hxx

#include "itkExponentialLimiterFunction.h"
#include "vnl/vnl_math.h"

namespace itk
{

  /**
   * *************** Constructor ********************
   */

  template<class TInput, unsigned int NDimension>
    ExponentialLimiterFunction<TInput, NDimension>::
    ExponentialLimiterFunction()
  {
    this->ComputeLimiterSettings();
  } // end Constructor

  /**
   * **************** Initialize ***********************
   */

    template<class TInput, unsigned int NDimension>
    void
    ExponentialLimiterFunction<TInput, NDimension>::
    Initialize(void) throw ( ExceptionObject )
  {
    this->ComputeLimiterSettings();
  } // end Initialize


  /**
   * ******************** Evaluate ***********************
   * one argument
   */

  template<class TInput, unsigned int NDimension>
    typename ExponentialLimiterFunction<TInput, NDimension>::OutputType
    ExponentialLimiterFunction<TInput, NDimension>::Evaluate(const InputType & input) const
  {
    /** Apply a soft limit if the input is larger than the UpperThreshold */
    const double diffU = static_cast<double>( input - this->m_UpperThreshold );
    if ( diffU > 1e-10 )
    {
      return static_cast<OutputType>(
        this->m_UTminUB * vcl_exp( this->m_UTminUBinv * diffU ) + this->m_UpperBound ) ;
    }

    /** Apply a soft limit if the input is smaller than the LowerThreshold */
    const double diffL = static_cast<double>( input - this->m_LowerThreshold );
    if ( diffL < -1e-10 )
    {
      return static_cast<OutputType>(
        this->m_LTminLB * vcl_exp( this->m_LTminLBinv * diffL ) + this->m_LowerBound );
    }

    /** Leave the value as it is */
    return static_cast<OutputType>(input);
  } // end Evaluate (one argument)


  /**
   * *********************** Evaluate *************************
   * two arguments
   */

  template<class TInput, unsigned int NDimension>
    typename ExponentialLimiterFunction<TInput, NDimension>::OutputType
    ExponentialLimiterFunction<TInput, NDimension>::
    Evaluate(const InputType & input, DerivativeType & derivative) const
  {
    /** Apply a soft limit if the input is larger than the UpperThreshold */
    const double diffU = static_cast<double>( input - this->m_UpperThreshold );
    if ( diffU > 1e-10 )
    {
      const double temp = this->m_UTminUB * vcl_exp( this->m_UTminUBinv * diffU );
      const double gradientfactor = this->m_UTminUBinv * temp;
      for (unsigned int i = 0; i < Dimension; ++i)
      {
        derivative[i] = static_cast<DerivativeValueType>( derivative[i] * gradientfactor );
      }
      return static_cast<OutputType>( temp + this->m_UpperBound );
    }

    /** Apply a soft limit if the input is smaller than the LowerThreshold */
    const double diffL = static_cast<double>( input - this->m_LowerThreshold );
    if ( diffL < -1e-10 )
    {
      const double temp = this->m_LTminLB * vcl_exp( this->m_LTminLBinv * diffL );
      const double gradientfactor = this->m_LTminLBinv * temp;
      for (unsigned int i = 0; i < Dimension; ++i)
      {
        derivative[i] = static_cast<DerivativeValueType>( derivative[i] * gradientfactor );
      }
      return static_cast<OutputType>( temp + this->m_LowerBound );
    }

    /** Leave the value and derivative as they are */
    return static_cast<OutputType>(input);
  } // end Evaluate (two arguments)


  /**
   * ******************** ComputeLimiterSettings ********************
   */

  template<class TInput, unsigned int NDimension>
    void
    ExponentialLimiterFunction<TInput, NDimension>::
    ComputeLimiterSettings( void )
  {
    this->m_UTminUB = static_cast<double>( this->m_UpperThreshold ) - this->m_UpperBound;
    this->m_LTminLB = static_cast<double>( this->m_LowerThreshold ) - this->m_LowerBound;

    if ( this->m_UTminUB < -1e-10 )
    {
      this->m_UTminUBinv = 1.0 / this->m_UTminUB;
    }
    else
    {
      /** The result is a hard limiter */
      this->m_UTminUB = 0.0;
      this->m_UTminUBinv = 0.0;
    }
    if ( this->m_LTminLB > 1e-10 )
    {
      this->m_LTminLBinv = 1.0 / this->m_LTminLB;
    }
    else
    {
      /** The result is a hard limiter */
      this->m_LTminLB = 0.0;
      this->m_LTminLBinv = 0.0;
    }
  } // end ComputeLimiterSettings



} // end namespace itk

#endif
