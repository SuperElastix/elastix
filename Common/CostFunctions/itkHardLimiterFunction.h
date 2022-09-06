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

#ifndef itkHardLimiterFunction_h
#define itkHardLimiterFunction_h

#include "itkLimiterFunctionBase.h"

namespace itk
{

/**
 * \class HardLimiterFunction
 * \brief A hard limiter
 *
 * If the input value exceeds the upper/lower bound the output is
 * set to the upper/lower bound and the derivative is filled with zeros.
 *
 * \ingroup Functions
 * \sa LimiterFunctionBase, ExponentialLimiterFunction
 *
 */
template <class TInput, unsigned int NDimension>
class ITK_TEMPLATE_EXPORT HardLimiterFunction : public LimiterFunctionBase<TInput, NDimension>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(HardLimiterFunction);

  /** Standard class typedefs. */
  using Self = HardLimiterFunction;
  using Superclass = LimiterFunctionBase<TInput, NDimension>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(HardLimiterFunction, LimiterFunctionBase);

  /** Define the New() function, for creation via the ObjectFactory */
  itkNewMacro(Self);

  /** Superclass' static consts */
  itkStaticConstMacro(Dimension, unsigned int, Superclass::Dimension);

  /** Superclass' typedefs */
  using typename Superclass::InputType;
  using typename Superclass::OutputType;
  using typename Superclass::DerivativeValueType;
  using typename Superclass::DerivativeType;

  /** Limit the input value */
  OutputType
  Evaluate(const InputType & input) const override;

  /** Limit the input value and change the input function derivative accordingly */
  OutputType
  Evaluate(const InputType & input, DerivativeType & derivative) const override;

protected:
  HardLimiterFunction() = default;
  ~HardLimiterFunction() override = default;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkHardLimiterFunction.hxx"
#endif

#endif
