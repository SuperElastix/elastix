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

#ifndef itkMoreThuenteLineSearchOptimizer_h
#define itkMoreThuenteLineSearchOptimizer_h

#include "itkLineSearchOptimizer.h"

namespace itk
{
/**
 * \class MoreThuenteLineSearchOptimizer
 *
 * \brief ITK version of the MoreThuente line search algorithm.
 *
 * This class is an ITK version of the netlib function mcsrch_. It
 * gives exactly the same results.
 *
 * The purpose of this optimizer is to find a step which satisfies
 * a sufficient decrease condition and a curvature condition.
 *
 * At each stage the subroutine updates an interval of
 * uncertainty with endpoints stx and sty. The interval of
 * uncertainty is initially chosen so that it contains a
 * minimizer of the modified function
 *
 *   \f[ f(x+stp*s) - f(x) - ValueTolerance*stp*(gradf(x)'s). \f]
 *
 * If a step is obtained for which the modified function
 * has a nonpositive function value and nonnegative derivative,
 * then the interval of uncertainty is chosen so that it
 * contains a minimizer of \f$f(x+stp*s)\f$.
 *
 * The algorithm is designed to find a step which satisfies
 * the sufficient decrease condition
 *
 *   \f[ f(x+stp*s) <= f(x) + ValueTolerance*stp*(gradf(x)'s), \f]
 *
 * and the curvature condition
 *
 *   \f[ \| gradf(x+stp*s)'s) \| <= GradientTolerance * \| gradf(x)'s \|. \f]
 *
 * (together also called the Strong Wolfe Conditions)
 *
 * if the ValueTolerance is less than the GradientTolerance and if,
 * for example, the function is bounded below, then there is always
 * a step which satisfies both conditions. If no step can be found
 * which satisfies both conditions, then the algorithm usually stops
 * when rounding errors prevent further progress. In this case stp only
 * satisfies the sufficient decrease condition.
 *
 *
 * \ingroup Numerics Optimizers
 */

class MoreThuenteLineSearchOptimizer : public LineSearchOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MoreThuenteLineSearchOptimizer);

  using Self = MoreThuenteLineSearchOptimizer;
  using Superclass = LineSearchOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkTypeMacro(MoreThuenteLineSearchOptimizer, LineSearchOptimizer);

  using Superclass::MeasureType;
  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;

  enum StopConditionType
  {
    StrongWolfeConditionsSatisfied,
    MetricError,
    MaximumNumberOfIterations,
    StepTooSmall,
    StepTooLarge,
    IntervalTooSmall,
    RoundingError,
    AscentSearchDirection,
    Unknown
  };

  void
  StartOptimization() override;

  virtual void
  StopOptimization();

  /** If initial derivative and/or value are given we can save some
   * computation time!
   */
  void
  SetInitialDerivative(const DerivativeType & derivative) override;

  void
  SetInitialValue(MeasureType value) override;

  /** Progress information: value, derivative, and directional derivative
   * at the current position.
   */
  void
  GetCurrentValueAndDerivative(MeasureType & value, DerivativeType & derivative) const override;

  void
  GetCurrentDerivative(DerivativeType & derivative) const override;

  MeasureType
  GetCurrentValue() const override;

  virtual double
  GetCurrentDirectionalDerivative() const;

  /** Progress information: about the state of convergence */
  itkGetConstMacro(CurrentIteration, unsigned long);
  itkGetConstReferenceMacro(StopCondition, StopConditionType);
  itkGetConstMacro(SufficientDecreaseConditionSatisfied, bool);
  itkGetConstMacro(CurvatureConditionSatisfied, bool);

  /** Setting: the maximum number of iterations. 20 by default. */
  itkGetConstMacro(MaximumNumberOfIterations, unsigned long);
  itkSetClampMacro(MaximumNumberOfIterations, unsigned long, 1, NumericTraits<unsigned long>::max());

  /** Setting: the value tolerance. By default set to 1e-4.
   *
   * The line search tries to find a StepLength that satisfies
   * the sufficient decrease condition:
   * F(X + StepLength * s) <= F(X) + ValueTolerance * StepLength * dF/ds(X)
   * where s is the search direction
   *
   * It must be larger than 0.0, and smaller than the GradientTolerance.
   */
  itkSetClampMacro(ValueTolerance, double, 0.0, NumericTraits<double>::max());
  itkGetConstMacro(ValueTolerance, double);

  /** Setting: the gradient tolerance. By default set to 0.9.
   *
   * The line search tries to find a StepLength that satisfies
   * the curvature condition:
   * ABS(dF/ds(X + StepLength * s) <= GradientTolerance * ABS(dF/ds(X)
   *
   * The lower this value, the more accurate the line search. It must
   * be larger than the ValueTolerance.
   */
  itkSetClampMacro(GradientTolerance, double, 0.0, NumericTraits<double>::max());
  itkGetConstMacro(GradientTolerance, double);

  /** Setting: the interval tolerance. By default set to the
   * the machine precision.
   *
   * If value and gradient tolerance can not be satisfied
   * both, the algorithm stops when rounding errors prevent
   * further progress: when the interval of uncertainty is
   * smaller than the interval tolerance.
   */
  itkSetClampMacro(IntervalTolerance, double, 0.0, NumericTraits<double>::max());
  itkGetConstMacro(IntervalTolerance, double);

protected:
  MoreThuenteLineSearchOptimizer();
  ~MoreThuenteLineSearchOptimizer() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  unsigned long     m_CurrentIteration;
  bool              m_InitialDerivativeProvided;
  bool              m_InitialValueProvided;
  StopConditionType m_StopCondition;
  bool              m_Stop;
  bool              m_SufficientDecreaseConditionSatisfied;
  bool              m_CurvatureConditionSatisfied;

  /** Load the initial value and derivative into m_f and m_g. */
  virtual void
  GetInitialValueAndDerivative();

  /** Check the input settings for errors. */
  virtual int
  CheckSettings();

  /** Initialize the interval of uncertainty etc. */
  void
  InitializeLineSearch();

  /** Set the minimum and maximum steps to correspond to the
   * the present interval of uncertainty.
   */
  virtual void
  UpdateIntervalMinimumAndMaximum();

  /** Force a step to be within the bounds MinimumStepLength and MaximumStepLength */
  void
  BoundStep(double & step) const;

  /** Set m_step to the best step until now, if unusual termination is expected */
  virtual void
  PrepareForUnusualTermination();

  /** Ask the cost function to compute m_f and m_g at the current position. */
  virtual void
  ComputeCurrentValueAndDerivative();

  /** Check for convergence */
  virtual void
  TestConvergence(bool & stop);

  /** Update the interval of uncertainty and compute the new step */
  virtual void
  ComputeNewStepAndInterval();

  /** Force a sufficient decrease in the size of the interval of uncertainty */
  virtual void
  ForceSufficientDecreaseInIntervalWidth();

  /** Advance a step along the line search direction and update
   * the interval of uncertainty.
   */
  virtual int
  SafeGuardedStep(double &     stx,
                  double &     fx,
                  double &     dx,
                  double &     sty,
                  double &     fy,
                  double &     dy,
                  double &     stp,
                  const double fp,
                  const double dp,
                  bool &       brackt,
                  const double stpmin,
                  const double stpmax) const;

  double m_step;
  double m_stepx;
  double m_stepy;
  double m_stepmin;
  double m_stepmax;

  MeasureType m_f; // CurrentValue
  MeasureType m_fx;
  MeasureType m_fy;
  MeasureType m_finit;

  DerivativeType m_g;  // CurrentDerivative
  double         m_dg; // CurrentDirectionalDerivative
  double         m_dginit;
  double         m_dgx;
  double         m_dgy;
  double         m_dgtest;

  double m_width;
  double m_width1;

  bool m_brackt;
  bool m_stage1;
  bool m_SafeGuardedStepFailed;

private:
  unsigned long m_MaximumNumberOfIterations;
  double        m_ValueTolerance;
  double        m_GradientTolerance;
  double        m_IntervalTolerance;
};

} // end namespace itk

/** ***************** Original documentation ***********************************
 *
 * The implementation of this class is based on the netlib function mcsrch_.
 * The original documentation of this function is included below
 */

/*                     SUBROUTINE MCSRCH */

/*     A slight modification of the subroutine CSRCH of More' and Thuente. */
/*     The changes are to allow reverse communication, and do not affect */
/*     the performance of the routine. */

/*     THE PURPOSE OF MCSRCH IS TO FIND A STEP WHICH SATISFIES */
/*     A SUFFICIENT DECREASE CONDITION AND A CURVATURE CONDITION. */

/*     AT EACH STAGE THE SUBROUTINE UPDATES AN INTERVAL OF */
/*     UNCERTAINTY WITH ENDPOINTS STX AND STY. THE INTERVAL OF */
/*     UNCERTAINTY IS INITIALLY CHOSEN SO THAT IT CONTAINS A */
/*     MINIMIZER OF THE MODIFIED FUNCTION */

/*          F(X+STP*S) - F(X) - FTOL*STP*(GRADF(X)'S). */

/*     IF A STEP IS OBTAINED FOR WHICH THE MODIFIED FUNCTION */
/*     HAS A NONPOSITIVE FUNCTION VALUE AND NONNEGATIVE DERIVATIVE, */
/*     THEN THE INTERVAL OF UNCERTAINTY IS CHOSEN SO THAT IT */
/*     CONTAINS A MINIMIZER OF F(X+STP*S). */

/*     THE ALGORITHM IS DESIGNED TO FIND A STEP WHICH SATISFIES */
/*     THE SUFFICIENT DECREASE CONDITION */

/*           F(X+STP*S) .LE. F(X) + FTOL*STP*(GRADF(X)'S), */

/*     AND THE CURVATURE CONDITION */

/*           ABS(GRADF(X+STP*S)'S)) .LE. GTOL*ABS(GRADF(X)'S). */

/*     IF FTOL IS LESS THAN GTOL AND IF, FOR EXAMPLE, THE FUNCTION */
/*     IS BOUNDED BELOW, THEN THERE IS ALWAYS A STEP WHICH SATISFIES */
/*     BOTH CONDITIONS. IF NO STEP CAN BE FOUND WHICH SATISFIES BOTH */
/*     CONDITIONS, THEN THE ALGORITHM USUALLY STOPS WHEN ROUNDING */
/*     ERRORS PREVENT FURTHER PROGRESS. IN THIS CASE STP ONLY */
/*     SATISFIES THE SUFFICIENT DECREASE CONDITION. */

/*     THE SUBROUTINE STATEMENT IS */

/*        SUBROUTINE MCSRCH(N,X,F,G,S,STP,FTOL,XTOL, MAXFEV,INFO,NFEV,WA) */
/*     WHERE */

/*       N IS A POSITIVE INTEGER INPUT VARIABLE SET TO THE NUMBER */
/*         OF VARIABLES. */

/*       X IS AN ARRAY OF LENGTH N. ON INPUT IT MUST CONTAIN THE */
/*         BASE POINT FOR THE LINE SEARCH. ON OUTPUT IT CONTAINS */
/*         X + STP*S. */

/*       F IS A VARIABLE. ON INPUT IT MUST CONTAIN THE VALUE OF F */
/*         AT X. ON OUTPUT IT CONTAINS THE VALUE OF F AT X + STP*S. */

/*       G IS AN ARRAY OF LENGTH N. ON INPUT IT MUST CONTAIN THE */
/*         GRADIENT OF F AT X. ON OUTPUT IT CONTAINS THE GRADIENT */
/*         OF F AT X + STP*S. */

/*       S IS AN INPUT ARRAY OF LENGTH N WHICH SPECIFIES THE */
/*         SEARCH DIRECTION. */

/*       STP IS A NONNEGATIVE VARIABLE. ON INPUT STP CONTAINS AN */
/*         INITIAL ESTIMATE OF A SATISFACTORY STEP. ON OUTPUT */
/*         STP CONTAINS THE FINAL ESTIMATE. */

/*       FTOL AND GTOL ARE NONNEGATIVE INPUT VARIABLES. (In this reverse */
/*         communication implementation GTOL is defined in a COMMON */
/*         statement.) TERMINATION OCCURS WHEN THE SUFFICIENT DECREASE */
/*         CONDITION AND THE DIRECTIONAL DERIVATIVE CONDITION ARE */
/*         SATISFIED. */

/*       XTOL IS A NONNEGATIVE INPUT VARIABLE. TERMINATION OCCURS */
/*         WHEN THE RELATIVE WIDTH OF THE INTERVAL OF UNCERTAINTY */
/*         IS AT MOST XTOL. */

/*       STPMIN AND STPMAX ARE NONNEGATIVE INPUT VARIABLES WHICH */
/*         SPECIFY LOWER AND UPPER BOUNDS FOR THE STEP. (In this reverse */
/*         communication implementatin they are defined in a COMMON */
/*         statement). */

/*       MAXFEV IS A POSITIVE INTEGER INPUT VARIABLE. TERMINATION */
/*         OCCURS WHEN THE NUMBER OF CALLS TO FCN IS AT LEAST */
/*         MAXFEV BY THE END OF AN ITERATION. */

/*       INFO IS AN INTEGER OUTPUT VARIABLE SET AS FOLLOWS: */

/*         INFO = 0  IMPROPER INPUT PARAMETERS. */

/*        INFO =-1  A RETURN IS MADE TO COMPUTE THE FUNCTION AND GRADIENT.  */
/*       NFEV IS AN INTEGER OUTPUT VARIABLE SET TO THE NUMBER OF */
/*         CALLS TO FCN. */

/*       WA IS A WORK ARRAY OF LENGTH N. */

/*     SUBPROGRAMS CALLED */

/*       MCSTEP */

/*       FORTRAN-SUPPLIED...ABS,MAX,MIN */

/*     ARGONNE NATIONAL LABORATORY. MINPACK PROJECT. JUNE 1983 */
/*     JORGE J. MORE', DAVID J. THUENTE */

/*     ********** */

#endif // #ifndef itkMoreThuenteLineSearchOptimizer_h
