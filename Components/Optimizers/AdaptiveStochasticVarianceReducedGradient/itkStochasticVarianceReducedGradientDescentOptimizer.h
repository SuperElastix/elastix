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
#ifndef itkStochasticVarianceReducedGradientDescentOptimizer_h
#define itkStochasticVarianceReducedGradientDescentOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "itkPlatformMultiThreader.h"

namespace itk
{
/** \class StochasticVarianceReducedGradientDescentOptimizer
 * \brief Implement a gradient descent optimizer
 *
 * StochasticVarianceReducedGradientDescentOptimizer implements a simple gradient descent optimizer.
 * At each iteration the current position is updated according to
 *
 * \f[
 *        p_{n+1} = p_n
 *                + \mbox{learningRate} \, \frac{\partial f(p_n) }{\partial p_n}
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

class StochasticVarianceReducedGradientDescentOptimizer : public ScaledSingleValuedNonLinearOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(StochasticVarianceReducedGradientDescentOptimizer);

  /** Standard class typedefs. */
  using Self = StochasticVarianceReducedGradientDescentOptimizer;
  using Superclass = ScaledSingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(StochasticVarianceReducedGradientDescentOptimizer, ScaledSingleValuedNonLinearOptimizer);

  /** Typedefs inherited from the superclass. */
  using Superclass::MeasureType;
  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;
  using Superclass::ScalesType;
  using Superclass::ScaledCostFunctionType;
  using Superclass::ScaledCostFunctionPointer;

  /** Codes of stopping conditions
   * The MinimumStepSize stop condition never occurs, but may
   * be implemented in inheriting classes */
  enum StopConditionType
  {
    MaximumNumberOfIterations,
    MetricError,
    MinimumStepSize,
    InvalidDiagonalMatrix,
    GradientMagnitudeTolerance,
    LineSearchError
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

  /** Get the inner LBFGSMemory. */
  itkGetConstMacro(LBFGSMemory, unsigned int);

  /** Get the number of iterations. */
  itkGetConstReferenceMacro(NumberOfIterations, unsigned long);

  /** Get the number of inner loop iterations. */
  itkGetConstReferenceMacro(NumberOfInnerIterations, unsigned long);

  /** Get the current iteration number. */
  itkGetConstMacro(CurrentIteration, unsigned int);

  /** Get the current inner iteration number. */
  itkGetConstMacro(CurrentInnerIteration, unsigned int);

  /** Get the current value. */
  itkGetConstReferenceMacro(Value, double);

  /** Get Stop condition. */
  itkGetConstReferenceMacro(StopCondition, StopConditionType);

  /** Get current gradient. */
  itkGetConstReferenceMacro(Gradient, DerivativeType);

  /** Get current search direction. */
  itkGetConstReferenceMacro(SearchDir, DerivativeType);

  /** Set the Previous Position. */
  itkSetMacro(PreviousPosition, ParametersType);

  /** Get the Previous Position. */
  itkGetConstReferenceMacro(PreviousPosition, ParametersType);

  /** Set the Previous gradient. */
  itkSetMacro(PreviousGradient, DerivativeType);

  /** Get the Previous gradient. */
  itkGetConstReferenceMacro(PreviousGradient, DerivativeType);

  /** Set the number of threads. */
  void
  SetNumberOfWorkUnits(ThreadIdType numberOfThreads)
  {
    this->m_Threader->SetNumberOfWorkUnits(numberOfThreads);
  }
  // itkGetConstReferenceMacro( NumberOfThreads, ThreadIdType );
  itkSetMacro(UseMultiThread, bool);

  itkSetMacro(UseOpenMP, bool);
  itkSetMacro(UseEigen, bool);

protected:
  StochasticVarianceReducedGradientDescentOptimizer();
  ~StochasticVarianceReducedGradientDescentOptimizer() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Typedefs for multi-threading. */
  using ThreaderType = itk::PlatformMultiThreader;
  using ThreadInfoType = ThreaderType::WorkUnitInfo;

  // made protected so subclass can access
  double         m_Value{ 0.0 };
  DerivativeType m_Gradient;
  ParametersType m_SearchDir;
  ParametersType m_PreviousSearchDir;
  // ParametersType                m_PrePreviousSearchDir;
  ParametersType    m_MeanSearchDir;
  double            m_LearningRate{ 1.0 };
  StopConditionType m_StopCondition{ MaximumNumberOfIterations };
  DerivativeType    m_PreviousGradient;
  // DerivativeType                m_PrePreviousGradient;
  ParametersType        m_PreviousPosition;
  ThreaderType::Pointer m_Threader{ ThreaderType::New() };

  bool          m_Stop{ false };
  unsigned long m_NumberOfIterations{ 100 };
  unsigned long m_NumberOfInnerIterations;
  unsigned long m_CurrentIteration{ 0 };
  unsigned long m_CurrentInnerIteration;
  unsigned long m_LBFGSMemory{ 0 };

private:
  // multi-threaded AdvanceOneStep:
  bool m_UseMultiThread{ false };
  struct MultiThreaderParameterType
  {
    ParametersType * t_NewPosition;
    Self *           t_Optimizer;
  };

  bool m_UseOpenMP{ false };
  bool m_UseEigen{ false };

  /** The callback function. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  AdvanceOneStepThreaderCallback(void * arg);

  /** The threaded implementation of AdvanceOneStep(). */
  inline void
  ThreadedAdvanceOneStep(ThreadIdType threadId, ParametersType & newPosition);
};

} // end namespace itk


#endif
