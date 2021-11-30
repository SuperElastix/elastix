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
#ifndef itkHardLimiterFunction_hxx
#define itkHardLimiterFunction_hxx

#include "itkHardLimiterFunction.h"
#include <vnl/vnl_math.h>

namespace itk
{

template <class TInput, unsigned int NDimension>
auto
HardLimiterFunction<TInput, NDimension>::Evaluate(const InputType & input) const -> OutputType
{
  OutputType output = std::min(static_cast<OutputType>(input), this->m_UpperBound);
  return (std::max(output, this->m_LowerBound));
} // end Evaluate()


template <class TInput, unsigned int NDimension>
auto
HardLimiterFunction<TInput, NDimension>::Evaluate(const InputType & input, DerivativeType & derivative) const
  -> OutputType
{
  if (input > this->m_UpperBound)
  {
    derivative.Fill(itk::NumericTraits<OutputType>::ZeroValue());
    return (this->m_UpperBound);
  }
  if (input < this->m_LowerBound)
  {
    derivative.Fill(itk::NumericTraits<OutputType>::ZeroValue());
    return (this->m_LowerBound);
  }
  return (static_cast<OutputType>(input));
} // end Evaluate()


} // end namespace itk

#endif
