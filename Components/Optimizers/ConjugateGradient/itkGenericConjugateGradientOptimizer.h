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

#ifndef itkGenericConjugateGradientOptimizer_h
#define itkGenericConjugateGradientOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "itkLineSearchOptimizer.h"
#include <vector>
#include <map>

namespace itk
{
/**
 * \class GenericConjugateGradientOptimizer
 * \brief A set of conjugate gradient algorithms.
 *
 * The steplength is determined at each iteration by means of a
 * line search routine. The itk::MoreThuenteLineSearchOptimizer works well.
 *
 *
 * \ingroup Numerics Optimizers
 */

class GenericConjugateGradientOptimizer : public ScaledSingleValuedNonLinearOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GenericConjugateGradientOptimizer);

  using Self = GenericConjugateGradientOptimizer;
  using Superclass = ScaledSingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkTypeMacro(GenericConjugateGradientOptimizer, ScaledSingleValuedNonLinearOptimizer);

  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;
  using Superclass::ScaledCostFunctionType;
  using Superclass::MeasureType;
  using Superclass::ScalesType;

  using LineSearchOptimizerType = LineSearchOptimizer;
  using LineSearchOptimizerPointer = LineSearchOptimizerType::Pointer;

  /** Typedef for a function that computes \f$\beta\f$, given the previousGradient,
   * the current gradient, and the previous search direction */
  using ComputeBetaFunctionType = double (Self::*)(const DerivativeType &,
                                                   const DerivativeType &,
                                                   const ParametersType &);
  using BetaDefinitionType = std::string;
  using BetaDefinitionMapType = std::map<BetaDefinitionType, ComputeBetaFunctionType>;

  enum StopConditionType
  {
    MetricError,
    LineSearchError,
    MaximumNumberOfIterations,
    GradientMagnitudeTolerance,
    ValueTolerance,
    InfiniteBeta,
    Unknown
  };

  void
  StartOptimization() override;

  virtual void
  ResumeOptimization();

  virtual void
  StopOptimization();

  /** Get information about optimization process: */
  itkGetConstMacro(CurrentIteration, unsigned long);
  itkGetConstMacro(CurrentValue, MeasureType);
  itkGetConstReferenceMacro(CurrentGradient, DerivativeType);
  itkGetConstMacro(InLineSearch, bool);
  itkGetConstReferenceMacro(StopCondition, StopConditionType);
  itkGetConstMacro(CurrentStepLength, double);

  /** Setting: the line search optimizer */
  itkSetObjectMacro(LineSearchOptimizer, LineSearchOptimizerType);
  itkGetModifiableObjectMacro(LineSearchOptimizer, LineSearchOptimizerType);

  /** Setting: the maximum number of iterations */
  itkGetConstMacro(MaximumNumberOfIterations, unsigned long);
  itkSetClampMacro(MaximumNumberOfIterations, unsigned long, 1, NumericTraits<unsigned long>::max());

  /** Setting: the mininum gradient magnitude. By default 1e-5.
   *
   * The optimizer stops when:
   * \f$ \|CurrentGradient\| <
   *   GradientMagnitudeTolerance * \max(1, \|CurrentPosition\| ) \f$
   */
  itkGetConstMacro(GradientMagnitudeTolerance, double);
  itkSetMacro(GradientMagnitudeTolerance, double)

    /** Setting: a stopping criterion, the value tolerance. By default 1e-5.
     *
     * The optimizer stops when
     * \f[ 2.0 * | f_k - f_{k-1} | \le
     *   ValueTolerance * ( |f_k| + |f_{k-1}| + 1e-20 ) \f]
     * is satisfied MaxNrOfItWithoutImprovement times in a row.
     */
    itkGetConstMacro(ValueTolerance, double);
  itkSetMacro(ValueTolerance, double);

  /** Setting: the maximum number of iterations in a row that
   * satisfy the value tolerance criterion. By default (if never set)
   * equal to the number of parameters. */
  virtual void
  SetMaxNrOfItWithoutImprovement(unsigned long arg);

  itkGetConstMacro(MaxNrOfItWithoutImprovement, unsigned long);

  /** Setting: the definition of \f$\beta\f$, by default "DaiYuanHestenesStiefel" */
  void
  SetBetaDefinition(const BetaDefinitionType & arg);

  itkGetConstReferenceMacro(BetaDefinition, BetaDefinitionType);

protected:
  GenericConjugateGradientOptimizer();
  ~GenericConjugateGradientOptimizer() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  DerivativeType    m_CurrentGradient;
  MeasureType       m_CurrentValue{ 0.0 };
  unsigned long     m_CurrentIteration{ 0 };
  StopConditionType m_StopCondition{ Unknown };
  bool              m_Stop{ false };
  double            m_CurrentStepLength{ 0.0 };

  /** Flag that is true as long as the method
   * SetMaxNrOfItWithoutImprovement is never called */
  bool m_UseDefaultMaxNrOfItWithoutImprovement{ true };

  /** Is true when the LineSearchOptimizer has been started. */
  bool m_InLineSearch{ false };
  itkSetMacro(InLineSearch, bool);

  /** Flag that says if the previous gradient and search direction are known.
   * Typically 'true' at the start of optimization, or when a stopped optimisation
   * is resumed (in the latter case the previous gradient and search direction
   * may of course still be valid, but to be safe it is assumed that they are not). */
  bool m_PreviousGradientAndSearchDirValid{ false };

  /** The name of the BetaDefinition */
  BetaDefinitionType m_BetaDefinition;

  /** A mapping that links the names of the BetaDefinitions to functions that
   * compute \f$\beta\f$. */
  BetaDefinitionMapType m_BetaDefinitionMap;

  /** Function to add a new beta definition. The first argument should be a name
   * via which a user can select this \f$\beta\f$ definition. The second argument is a
   * pointer to a method that computes \f$\beta\f$.
   * Called in the constructor of this class, and possibly by subclasses.
   */
  void
  AddBetaDefinition(const BetaDefinitionType & name, ComputeBetaFunctionType function);

  /**
   * Compute the search direction:
   *    \f[ d_{k} = - g_{k} + \beta_{k} d_{k-1} \f]
   *
   * In the first iteration the search direction is computed as:
   *    \f[ d_{0} = - g_{0} \f]
   *
   * On calling, searchDir should equal \f$d_{k-1}\f$. On return searchDir
   * equals \f$d_{k}\f$.
   */
  virtual void
  ComputeSearchDirection(const DerivativeType & previousGradient,
                         const DerivativeType & gradient,
                         ParametersType &       searchDir);

  /** Perform a line search along the search direction. On calling, \f$x, f\f$, and \f$g\f$ should
   * contain the current position, the cost function value at this position, and
   * the derivative. On return the step, \f$x\f$ (new position), \f$f\f$ (value at \f$x\f$), and \f$g\f$
   * (derivative at \f$x\f$) are updated. */
  virtual void
  LineSearch(const ParametersType searchDir, double & step, ParametersType & x, MeasureType & f, DerivativeType & g);

  /** Check if convergence has occured;
   * The firstLineSearchDone bool allows the implementation of TestConvergence to
   * decide to skip a few convergence checks when no line search has performed yet
   * (so, before the actual optimisation begins)  */
  virtual bool
  TestConvergence(bool firstLineSearchDone);

  /** Compute \f$\beta\f$ according to the user set \f$\beta\f$-definition */
  virtual double
  ComputeBeta(const DerivativeType & previousGradient,
              const DerivativeType & gradient,
              const ParametersType & previousSearchDir);

  /** Different definitions of \f$\beta\f$ */

  /** "SteepestDescent: beta=0 */
  double
  ComputeBetaSD(const DerivativeType & previousGradient,
                const DerivativeType & gradient,
                const ParametersType & previousSearchDir);

  /** "FletcherReeves" */
  double
  ComputeBetaFR(const DerivativeType & previousGradient,
                const DerivativeType & gradient,
                const ParametersType & previousSearchDir);

  /** "PolakRibiere" */
  double
  ComputeBetaPR(const DerivativeType & previousGradient,
                const DerivativeType & gradient,
                const ParametersType & previousSearchDir);

  /** "DaiYuan" */
  double
  ComputeBetaDY(const DerivativeType & previousGradient,
                const DerivativeType & gradient,
                const ParametersType & previousSearchDir);

  /** "HestenesStiefel" */
  double
  ComputeBetaHS(const DerivativeType & previousGradient,
                const DerivativeType & gradient,
                const ParametersType & previousSearchDir);

  /** "DaiYuanHestenesStiefel" */
  double
  ComputeBetaDYHS(const DerivativeType & previousGradient,
                  const DerivativeType & gradient,
                  const ParametersType & previousSearchDir);

private:
  unsigned long m_MaximumNumberOfIterations{ 100 };
  double        m_ValueTolerance{ 1e-5 };
  double        m_GradientMagnitudeTolerance{ 1e-5 };
  unsigned long m_MaxNrOfItWithoutImprovement{ 10 };

  LineSearchOptimizerPointer m_LineSearchOptimizer{ nullptr };
};

} // end namespace itk

#endif //#ifndef itkGenericConjugateGradientOptimizer_h
