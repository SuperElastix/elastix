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
#ifndef itkLimiterFunctionBase_h
#define itkLimiterFunctionBase_h

#include "itkFunctionBase.h"
#include "itkMacro.h"

namespace itk
{

/**
 * \class LimiterFunctionBase
 * \brief Base class for all ITK limiter function objects
 *
 * LimiterFunctionBase is the base class for ITK limiter function objects.
 * The abstract method Evaluate() should limit a function, i.e.
 * it should make sure that its output is below a certain value.
 * The derivative of a function that is limited also changes.
 *
 * In formula:
 * \f[ L(y) = L(f(x)), \f]
 * where \f$f(x)\f$ is the original function. and \f$L(f(x))\f$ the limited version.
 * The derivative with respect to \f$x\f$ should satisfy:
 * \f[ dL/dx = \frac{dL}{df} \cdot \frac{df}{dx} \f]
 *
 * Subclasses must override Evaluate(value) and Evaluate(value, derivative) .
 *
 * This class is template over the input type and the dimension of \f$x\f$.
 *
 * \ingroup Functions
 *
 */
template <class TInput, unsigned int NDimension>
class ITK_TEMPLATE_EXPORT LimiterFunctionBase : public FunctionBase<TInput, typename NumericTraits<TInput>::RealType>
{
public:
  /** Standard class typedefs. */
  typedef LimiterFunctionBase                                            Self;
  typedef FunctionBase<TInput, typename NumericTraits<TInput>::RealType> Superclass;
  typedef SmartPointer<Self>                                             Pointer;
  typedef SmartPointer<const Self>                                       ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(LimiterFunctionBase, FunctionBase);

  itkStaticConstMacro(Dimension, unsigned int, NDimension);

  /** Superclass' typedefs */
  typedef typename Superclass::InputType  InputType;
  typedef typename Superclass::OutputType OutputType;

  typedef OutputType                                                              DerivativeValueType;
  typedef CovariantVector<DerivativeValueType, itkGetStaticConstMacro(Dimension)> DerivativeType;

  /** Limit the input value. */
  OutputType
  Evaluate(const InputType & input) const override = 0;

  /** Limit the input value and change the input function derivative accordingly */
  virtual OutputType
  Evaluate(const InputType & input, DerivativeType & derivative) const = 0;

  /** Set/Get the upper bound that the output should respect. Make sure it is higher
   * than the lower bound. */
  itkSetMacro(UpperBound, OutputType);
  itkGetConstMacro(UpperBound, OutputType);

  /** Set/Get the lower bound that the output should respect. Make sure it is lower
   * than the higher bound. */
  itkSetMacro(LowerBound, OutputType);
  itkGetConstMacro(LowerBound, OutputType);

  /** Set the point where the limiter starts to work. Only input values above this number
   * will possibly be affected. Make sure it is <= than the UpperBound. */
  itkSetMacro(UpperThreshold, InputType);
  itkGetConstMacro(UpperThreshold, InputType);

  /** Set the point where the limiter starts to work. Only input values below this number
   * will possibly be affected. Make sure it is >= than the LowerBound. */
  itkSetMacro(LowerThreshold, InputType);
  itkGetConstMacro(LowerThreshold, InputType);

  /** Initialize the limiter */
  virtual void
  Initialize(void)
  {}

protected:
  LimiterFunctionBase()
  {
    this->m_UpperBound = itk::NumericTraits<OutputType>::One + itk::NumericTraits<OutputType>::One;
    this->m_LowerBound = itk::NumericTraits<OutputType>::Zero;
    this->m_UpperThreshold = itk::NumericTraits<InputType>::One;
    this->m_LowerThreshold = itk::NumericTraits<InputType>::One;
  }


  ~LimiterFunctionBase() override = default;

  OutputType m_UpperBound;
  OutputType m_LowerBound;
  InputType  m_UpperThreshold;
  InputType  m_LowerThreshold;

private:
  LimiterFunctionBase(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#endif
