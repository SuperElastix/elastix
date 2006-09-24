#ifndef __itkHardLimiterFunction_hxx
#define __itkHardLimiterFunction_hxx


#include "itkHardLimiterFunction.h"
#include "vnl/vnl_math.h"

namespace itk
{

  template <class TInput, unsigned int NDimension>
    typename HardLimiterFunction< TInput, NDimension >::OutputType
    HardLimiterFunction<TInput, NDimension>::Evaluate(const InputType & input) const
  {
    OutputType output = vnl_math_min( static_cast<OutputType>(input), this->m_UpperBound);
    return (vnl_math_max( output, this->m_LowerBound));
  } // end Evaluate


  template <class TInput, unsigned int NDimension>
    typename HardLimiterFunction<TInput, NDimension>::OutputType
    HardLimiterFunction<TInput, NDimension>::
    Evaluate(const InputType & input, DerivativeType & derivative) const
  {
    if ( input > this->m_UpperBound )
    {
      derivative.Fill( itk::NumericTraits<OutputType>::Zero );
      return (this->m_UpperBound);
    }
    if ( input < this->m_LowerBound )
    {
      derivative.Fill( itk::NumericTraits<OutputType>::Zero );
      return (this->m_LowerBound);
    }
    return (static_cast<OutputType>(input));
  } // end Evaluate


} // end namespace itk

#endif