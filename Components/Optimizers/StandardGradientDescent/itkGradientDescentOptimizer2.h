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
#ifndef itkGradientDescentOptimizer2_h
#define itkGradientDescentOptimizer2_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"


namespace itk
{

/** \class GradientDescentOptimizer2
 * \brief Implement a gradient descent optimizer
 *
 * GradientDescentOptimizer2 implements a simple gradient descent optimizer.
 * At each iteration the current position is updated according to
 *
 * \f[
 *        p_{n+1} = p_n
 *                + \mbox{learningRate}
                  \, \frac{\partial f(p_n) }{\partial p_n}
 * \f]
 *
 * The learning rate is a fixed scalar defined via SetLearningRate().
 * The optimizer steps through a user defined number of iterations;
 * no convergence checking is done.
 *
 * Additionally, user can scale each component of the \f$\partial f / \partial p\f$
 * but setting a scaling vector using method SetScale().
 *
 * The difference of this class with the itk::GradientDescentOptimizer
 * is that it's based on the ScaledSingleValuedNonLinearOptimizer
 *
 * \sa ScaledSingleValuedNonLinearOptimizer
 *
 * \ingroup Numerics Optimizers
 */

class GradientDescentOptimizer2 : public ScaledSingleValuedNonLinearOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GradientDescentOptimizer2);

  /** Standard class typedefs. */
  using Self = GradientDescentOptimizer2;
  using Superclass = ScaledSingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GradientDescentOptimizer2, ScaledSingleValuedNonLinearOptimizer);

  /** Typedefs inherited from the superclass. */
  using Superclass::MeasureType;
  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;
  using Superclass::ScalesType;
  using Superclass::ScaledCostFunctionType;
  using Superclass::ScaledCostFunctionPointer;

  /** Codes of stopping conditions
   * The MinimumStepSize stopcondition never occurs, but may
   * be implemented in inheriting classes */
  enum StopConditionType
  {
    MaximumNumberOfIterations,
    MetricError,
    MinimumStepSize
  };

  /** Advance one step following the gradient direction. */
  virtual void
  AdvanceOneStep();

  /** Start optimization. */
  void
  StartOptimization() override;

  /** Resume previously stopped optimization with current parameters
   * \sa StopOptimization. */
  virtual void
  ResumeOptimization();

  /** Stop optimization and pass on exception. */
  virtual void
  MetricErrorResponse(ExceptionObject & err);

  /** Stop optimization.
   * \sa ResumeOptimization */
  virtual void
  StopOptimization();

  /** Set the learning rate. */
  itkSetMacro(LearningRate, double);

  /** Get the learning rate. */
  itkGetConstReferenceMacro(LearningRate, double);

  /** Set the number of iterations. */
  itkSetMacro(NumberOfIterations, unsigned long);

  /** Get the number of iterations. */
  itkGetConstReferenceMacro(NumberOfIterations, unsigned long);

  /** Get the current iteration number. */
  itkGetConstMacro(CurrentIteration, unsigned int);

  /** Get the current value. */
  itkGetConstReferenceMacro(Value, double);

  /** Get Stop condition. */
  itkGetConstReferenceMacro(StopCondition, StopConditionType);

  /** Get current gradient. */
  itkGetConstReferenceMacro(Gradient, DerivativeType);

  /** Get current search direction */
  itkGetConstReferenceMacro(SearchDirection, DerivativeType);

  /** Set use OpenMP or not. */
  itkSetMacro(UseOpenMP, bool);

protected:
  GradientDescentOptimizer2();
  ~GradientDescentOptimizer2() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  // made protected so subclass can access
  DerivativeType    m_Gradient;
  DerivativeType    m_SearchDirection;
  StopConditionType m_StopCondition{ MaximumNumberOfIterations };

private:
  double        m_Value{ 0.0 };
  double        m_LearningRate{ 1.0 };
  bool          m_Stop{ false };
  unsigned long m_NumberOfIterations{ 100 };
  unsigned long m_CurrentIteration{ 0 };

  bool m_UseOpenMP;
};

} // end namespace itk

#endif
