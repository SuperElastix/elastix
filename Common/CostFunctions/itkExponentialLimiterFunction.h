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

#ifndef itkExponentialLimiterFunction_h
#define itkExponentialLimiterFunction_h

#include "itkLimiterFunctionBase.h"

namespace itk
{

/**
 * \class ExponentialLimiterFunction
 * \brief A soft limiter
 *
 * If the input value exceeds the upper/lower threshold the output is
 * diminished/increased, such that it never will exceed the UpperBound/LowerBound.
 * It does this in a smooth manner, with an exponential function.
 *
 * \f[ L(f(x)) = (T-B) e^{(f-T)/(T-B)} + B, \f]
 * where \f$B\f$ is the upper/lower bound and \f$T\f$ the upper/lower threshold
 *
 * \ingroup Functions
 * \sa LimiterFunctionBase, HardLimiterFunction
 *
 */
template <class TInput, unsigned int NDimension>
class ITK_TEMPLATE_EXPORT ExponentialLimiterFunction : public LimiterFunctionBase<TInput, NDimension>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ExponentialLimiterFunction);

  /** Standard class typedefs. */
  using Self = ExponentialLimiterFunction;
  using Superclass = LimiterFunctionBase<TInput, NDimension>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ExponentialLimiterFunction, LimiterFunctionBase);

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

  /** Initialize the limiter; calls the ComputeLimiterSettings() function */
  void
  Initialize() override;

protected:
  ExponentialLimiterFunction();
  ~ExponentialLimiterFunction() override = default;

  virtual void
  ComputeLimiterSettings();

  double m_UTminUB;
  double m_UTminUBinv;
  double m_LTminLB;
  double m_LTminLBinv;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkExponentialLimiterFunction.hxx"
#endif

#endif
