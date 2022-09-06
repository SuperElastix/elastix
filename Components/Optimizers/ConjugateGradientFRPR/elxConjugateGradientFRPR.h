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
#ifndef elxConjugateGradientFRPR_h
#define elxConjugateGradientFRPR_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkFRPROptimizer.h"

namespace elastix
{

/**
 * \class ConjugateGradientFRPR
 * \brief The ConjugateGradientFRPR class.
 *
 * This component is based on the itkFRPROptimizer. This is a
 * Fletcher-Reeves conjugate gradient optimizer, in combination
 * with an exact (dBrent) line search, based on the description
 * in Numerical Recipes in C++
 *
 * This optimizer support the NewSamplesEveryIteration option. It requests
 * new samples upon every derivative evaluation, but
 * actually this makes no sense for a conjugate gradient optimizer.
 * So, think twice before using it.
 *
 * \note It prints out no stop conditions, since the itk superclass
 * does not generate them.
 * \note It considers line search iterations as elastix iterations.
 *
 * \parameter Optimizer: Select this optimizer as follows:\n
 *    <tt>(Optimizer "ConjugateGradientFRPR")</tt>\n
 * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
 *    example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
 *    Default value: 100.\n
 * \parameter MaximumNumberOfLineSearchIterations: The maximum number of iterations in each resolution. \n
 *    example: <tt>(MaximumNumberOfIterations 10 10 5)</tt> \n
 *    Default value: 10.\n
 * \parameter StepLength: Set the length of the initial step tried by the line seach,
 *    used to bracket the minimum.\n
 *    example: <tt>(StepLength 2.0 1.0 0.5)</tt> \n
 *    Default value: 1.0.\n
 * \parameter ValueTolerance: Convergence is declared if:
 *    \f[ 2.0 * | f_2 - f_1 | \le  ValueTolerance * ( | f_1 | + | f_2 | ) \f]
 *    example: <tt>(ValueTolerance 0.001 0.00001 0.000001)</tt> \n
 *    Default value: 0.00001.\n
 * \parameter LineSearchStepTolerance: Convergence of the line search is declared if:
 *    \f[ | x - x_m | \le tol * |x| - ( b - a ) / 2, \f]
 *    where:\n
 *    \f$x\f$ = current mininum of the gain\n
 *    \f$a, b\f$ = current brackets around the minimum\n
 *    \f$x_m = (a+b)/2 \f$\n
 *    example: <tt>(LineSearchStepTolerance 0.001 0.00001 0.000001)</tt> \n
 *    Default value: 0.00001.
 *
 * \ingroup Optimizers
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT ConjugateGradientFRPR
  : public itk::FRPROptimizer
  , public OptimizerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ConjugateGradientFRPR);

  /** Standard ITK.*/
  using Self = ConjugateGradientFRPR;
  using Superclass1 = itk::FRPROptimizer;
  using Superclass2 = OptimizerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConjugateGradientFRPR, itk::FRPROptimizer);

  /** Name of this class.*/
  elxClassNameMacro("ConjugateGradientFRPR");

  /** Typedef's inherited from Superclass1.*/
  using Superclass1::CostFunctionType;
  using Superclass1::CostFunctionPointer;
  // using Superclass1::StopConditionType; not implemented in this itkOptimizer
  using typename Superclass1::ParametersType;
  // not declared in Superclass, although it should be.
  using DerivativeType = SingleValuedNonLinearOptimizer::DerivativeType;

  /** Typedef's inherited from Elastix.*/
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;

  /** Methods to set parameters and print output at different stages
   * in the registration process.*/
  void
  BeforeRegistration() override;

  void
  BeforeEachResolution() override;

  void
  AfterEachResolution() override;

  void
  AfterEachIteration() override;

  void
  AfterRegistration() override;

  /** Override the SetInitialPosition.*/
  void
  SetInitialPosition(const ParametersType & param) override;

  /** Check if the optimizer is currently Bracketing the minimum, or is
   * optimizing along a line */
  itkGetConstMacro(LineOptimizing, bool);
  itkGetConstMacro(LineBracketing, bool);

  /** Return the magnitude of the cached derivative */
  itkGetConstReferenceMacro(CurrentDerivativeMagnitude, double);

  /** Get the current gain */
  itkGetConstReferenceMacro(CurrentStepLength, double);

  /** Get the magnitude of the line search direction */
  itkGetConstReferenceMacro(CurrentSearchDirectionMagnitude, double);

protected:
  ConjugateGradientFRPR();
  ~ConjugateGradientFRPR() override = default;

  /** To store the latest computed derivative's magnitude */
  double m_CurrentDerivativeMagnitude;

  /** Variable to store the line search direction magnitude */
  double m_CurrentSearchDirectionMagnitude;

  /** the current gain */
  double m_CurrentStepLength;

  /** Set if the optimizer is currently bracketing the minimum, or is
   * optimizing along a line */
  itkSetMacro(LineOptimizing, bool);
  itkSetMacro(LineBracketing, bool);

  /** Get the value of the n-dimensional cost function at this scalar step
   * distance along the current line direction from the current line origin.
   * Line origin and distances are set via SetLine.
   *
   * This implementation calls the Superclass' implementation and caches
   * the computed derivative's magnitude. Besides, it invokes the
   * SelectNewSamples method. */
  virtual void
  GetValueAndDerivative(ParametersType p, double * val, ParametersType * xi);

  /** The LineBracket routine from NRC. Uses current origin and line direction
   * (from SetLine) to find a triple of points (ax, bx, cx) that bracket the
   * extreme "near" the origin.  Search first considers the point StepLength
   * distance from ax.
   * IMPORTANT: The value of ax and the value of the function at ax (i.e., fa),
   * must both be provided to this function.
   *
   * This implementation sets the LineBracketing flag to 'true', calls the
   * superclass' implementation, stores bx as the current step length,
   * invokes an iteration event, and sets the LineBracketing flag to 'false' */
  void
  LineBracket(double * ax, double * bx, double * cx, double * fa, double * fb, double * fc) override;

  /** Given a bracketing triple of points and their function values, returns
   * a bounded extreme.  These values are in parameter space, along the
   * current line and wrt the current origin set via SetLine.   Optimization
   * terminates based on MaximumIteration, StepTolerance, or ValueTolerance.
   * Implemented as Brent line optimers from NRC.
   *
   * This implementation sets the LineOptimizing flag to 'true', calls the
   * the superclass's implementation, stores extX as the current step length,
   * and sets the LineOptimizing flag to 'false' again. */
  void
  BracketedLineOptimize(double   ax,
                        double   bx,
                        double   cx,
                        double   fa,
                        double   fb,
                        double   fc,
                        double * extX,
                        double * extVal) override;

  /**
   * store the line search direction's (xi) magnitude and call the superclass'
   * implementation.
   */
  virtual void
  LineOptimize(ParametersType * p, ParametersType xi, double * val);

private:
  elxOverrideGetSelfMacro;

  bool m_LineOptimizing;
  bool m_LineBracketing;

  const char *
  DeterminePhase() const;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxConjugateGradientFRPR.hxx"
#endif

#endif // end #ifndef elxConjugateGradientFRPR_h
