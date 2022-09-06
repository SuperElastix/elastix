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

#ifndef itkLineSearchOptimizer_h
#define itkLineSearchOptimizer_h

#include "itkSingleValuedNonLinearOptimizer.h"

#include "itkIntTypes.h" //tmp

namespace itk
{

/**
 * \class LineSearchOptimizer
 *
 * \brief A base class for LineSearch optimizers.
 *
 * Scales are expected to be handled by the main optimizer.
 */

class LineSearchOptimizer : public SingleValuedNonLinearOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(LineSearchOptimizer);

  using Self = LineSearchOptimizer;
  using Superclass = SingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  // itkNewMacro(Self); because this is an abstract base class.
  itkTypeMacro(LineSearchOptimizer, SingleValuedNonLinearOptimizer);

  using Superclass::MeasureType;
  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;

  /** Set/Get the LineSearchDirection */
  virtual void
  SetLineSearchDirection(const ParametersType & arg)
  {
    this->m_LineSearchDirection = arg;
    this->Modified();
  }


  itkGetConstReferenceMacro(LineSearchDirection, ParametersType);

  /** Inheriting classes may override these methods if they need
   * value/derivative information of the cost function at the
   * initial position.
   *
   * NB: It is not guaranteed that these methods are called.
   * If a main optimizer by chance has this information, it
   * should call these methods, to avoid possible unnecessary
   * computations.
   */
  virtual void
  SetInitialDerivative(const DerivativeType & itkNotUsed(derivative))
  {}
  virtual void
  SetInitialValue(MeasureType itkNotUsed(value))
  {}

  /** These methods must be implemented by inheriting classes. It
   * depends on the specific line search algorithm if it already computed
   * the value/derivative at the current position (in this case it
   * can just copy the cached data). If it did not
   * compute the value/derivative, it should call the cost function
   * and evaluate the value/derivative at the current position.
   *
   * These methods allow the main optimization algorithm to reuse
   * data that the LineSearch algorithm already computed.
   */
  virtual void
  GetCurrentValueAndDerivative(MeasureType & value, DerivativeType & derivative) const = 0;

  virtual void
  GetCurrentDerivative(DerivativeType & derivative) const = 0;

  virtual MeasureType
  GetCurrentValue() const = 0;

  /**
   * StepLength is a a scalar, defined as:
   * m_InitialPosition + StepLength * m_LineSearchDirection  =
   * m_CurrentPosition
   */
  itkGetConstMacro(CurrentStepLength, double);

  /** Settings: the maximum/minimum step length and the initial
   * estimate.
   * NOTE: Not all line search methods are guaranteed to
   * do something with this information.
   * However, if a certain optimizer (using a line search
   * optimizer) has any idea about the steplength it can
   * call these methods, 'in the hope' that the line search
   * optimizer does something sensible with it.
   */
  itkSetMacro(MinimumStepLength, double);
  itkGetConstMacro(MinimumStepLength, double);
  itkSetMacro(MaximumStepLength, double);
  itkGetConstMacro(MaximumStepLength, double);
  itkSetMacro(InitialStepLengthEstimate, double);
  itkGetConstMacro(InitialStepLengthEstimate, double);

protected:
  LineSearchOptimizer();
  ~LineSearchOptimizer() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  double m_CurrentStepLength;

  /** Set the current step length AND the current position, where
   * the current position is computed as:
   * m_CurrentPosition =
   * m_InitialPosition + StepLength * m_LineSearchDirection
   */
  virtual void
  SetCurrentStepLength(double step);

  /** Computes the inner product of the argument and the line search direction. */
  double
  DirectionalDerivative(const DerivativeType & derivative) const;

private:
  ParametersType m_LineSearchDirection;

  double m_MinimumStepLength;
  double m_MaximumStepLength;
  double m_InitialStepLengthEstimate;
};

} // end namespace itk

#endif // #ifndef itkLineSearchOptimizer_h
