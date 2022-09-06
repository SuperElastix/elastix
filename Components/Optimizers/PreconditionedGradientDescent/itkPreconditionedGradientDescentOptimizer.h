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
#ifndef itkPreconditionedGradientDescentOptimizer_h
#define itkPreconditionedGradientDescentOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "itkArray2D.h"
#include <vnl/vnl_sparse_matrix.h>
#include "cholmod.h"

namespace itk
{
/** \class PreconditionedGradientDescentOptimizer
 * \brief Implement a gradient descent optimizer
 *
 * PreconditionedGradientDescentOptimizer implements a simple gradient descent optimizer.
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

class PreconditionedGradientDescentOptimizer : public ScaledSingleValuedNonLinearOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(PreconditionedGradientDescentOptimizer);

  /** Standard class typedefs. */
  using Self = PreconditionedGradientDescentOptimizer;
  using Superclass = ScaledSingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(PreconditionedGradientDescentOptimizer, ScaledSingleValuedNonLinearOptimizer);

  /** Typedefs inherited from the superclass. */
  using Superclass::MeasureType;
  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;
  using Superclass::ScalesType;
  using Superclass::ScaledCostFunctionType;
  using Superclass::ScaledCostFunctionPointer;

  /** Some typedefs for computing the SelfHessian */
  using PreconditionValueType = DerivativeType::ValueType;
  // typedef Array2D<PreconditionValueType>                  PreconditionType;
  // typedef vnl_symmetric_eigensystem<
  //  PreconditionValueType >                               EigenSystemType;
  using PreconditionType = vnl_sparse_matrix<PreconditionValueType>;

  /** Codes of stopping conditions
   * The MinimumStepSize stopcondition never occurs, but may
   * be implemented in inheriting classes.
   */
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
  virtual void
  StartOptimization();

  /** Resume previously stopped optimization with current parameters
   * \sa StopOptimization.
   */
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

  /** Set the preconditioning matrix, whose inverse actually will be used to precondition.
   * On setting the precondition matrix, an eigensystem is computed immediately, the
   * eigenvalues/vectors are modified and only the modified eigenvectors/values are stored
   * (in the EigenSystem).
   * NB: this function destroys the input matrix, to save memory.
   */
  virtual void
  SetPreconditionMatrix(PreconditionType & precondition);

  /** Temporary functions, for debugging */
  const cholmod_common *
  GetCholmodCommon() const
  {
    return this->m_CholmodCommon;
  }


  const cholmod_factor *
  GetCholmodFactor() const
  {
    return this->m_CholmodFactor;
  }

  /** P = P + diagonalWeight * max(eigenvalue) * Identity */
  itkSetMacro(DiagonalWeight, double);
  itkGetConstMacro(DiagonalWeight, double);

  /** Threshold for elements of cost function derivative; default 1e-10 */
  itkSetMacro(MinimumGradientElementMagnitude, double);
  itkGetConstMacro(MinimumGradientElementMagnitude, double);

  /** Get estimated condition number; only valid after calling
   * SetPreconditionMatrix */
  itkGetConstMacro(ConditionNumber, double);

  /** Get largestEigenValue; only valid after calling
   * SetPreconditionMatrix */
  itkGetConstMacro(LargestEigenValue, double);

  /** Get sparsity of selfhessian; only valid after calling
   * SetPreconditionMatrix; Takes into account that only upper half
   * of the matrix is stored. 1 = dense, 0 = all elements zero.
   */
  itkGetConstMacro(Sparsity, double);

protected:
  PreconditionedGradientDescentOptimizer();
  virtual ~PreconditionedGradientDescentOptimizer();

  void
  PrintSelf(std::ostream & os, Indent indent) const;

  /** Cholmod index type: define at central place */
  using cholmod_l = int CInt; // change to UF_long if using;

  // made protected so subclass can access
  DerivativeType    m_Gradient;
  double            m_LearningRate{ 1.0 };
  StopConditionType m_StopCondition{ MaximumNumberOfIterations };
  DerivativeType    m_SearchDirection;
  double            m_LargestEigenValue{ 1.0 };
  double            m_ConditionNumber{ 1.0 };
  double            m_Sparsity{ 1.0 };

  cholmod_common * m_CholmodCommon;
  cholmod_factor * m_CholmodFactor{ nullptr };
  cholmod_sparse * m_CholmodGradient{ nullptr };

  /** Solve Hx = g, using the Cholesky decomposition of the preconditioner.
   * Matlab notation: x = L'\(L\g) = Pg = searchDirection
   * The last argument can be used to also solve different systems, like L x = g.
   */
  virtual void
  CholmodSolve(const DerivativeType & gradient, DerivativeType & searchDirection, int solveType = CHOLMOD_A);

private:
  bool   m_Stop{ false };
  double m_Value{ 0.0 };

  unsigned long m_NumberOfIterations{ 100 };
  unsigned long m_CurrentIteration{ 0 };

  double m_DiagonalWeight{ 1e-6 };
  double m_MinimumGradientElementMagnitude{ 1e-10 };
};

} // end namespace itk


#endif
