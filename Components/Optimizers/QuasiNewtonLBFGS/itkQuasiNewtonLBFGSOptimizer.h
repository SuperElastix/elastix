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

#ifndef itkQuasiNewtonLBFGSOptimizer_h
#define itkQuasiNewtonLBFGSOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "itkLineSearchOptimizer.h"
#include <vector>

namespace itk
{
/** \class QuasiNewtonLBFGSOptimizer
 * \brief ITK version of the lbfgs algorithm ...
 *
 * This class is an ITK version of the netlib lbfgs_ function.
 * It gives exactly the same results, if used in combination
 * with the itk::MoreThuenteLineSearchOptimizer.
 *
 * The optimizer solves the unconstrained minimization problem
 *
 *   \f[ \min F(x), \quad x = ( x_1,x_2,\ldots,x_N ), \f]
 *
 * using the limited memory BFGS method. The routine is especially
 * effective on problems involving a large number of variables. In
 * a typical iteration of this method an approximation \f$H_k\f$ to the
 * inverse of the Hessian is obtained by applying \f$M\f$ BFGS updates to
 * a diagonal matrix \f$H_0\f$, using information from the previous \f$M\f$ steps.
 * The user specifies the number \f$M\f$ (Memory), which determines the amount of
 * storage required by the routine.
 *
 * The algorithm is described in "On the limited memory BFGS method
 * for large scale optimization", by D. Liu and J. Nocedal,
 * Mathematical Programming B 45 (1989) 503-528.
 *
 * The steplength is determined at each iteration by means of a
 * line search routine. The itk::MoreThuenteLineSearchOptimizer works well.
 *
 *
 * \ingroup Numerics Optimizers
 */

class QuasiNewtonLBFGSOptimizer : public ScaledSingleValuedNonLinearOptimizer
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(QuasiNewtonLBFGSOptimizer);

  using Self = QuasiNewtonLBFGSOptimizer;
  using Superclass = ScaledSingleValuedNonLinearOptimizer;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkTypeMacro(QuasiNewtonLBFGSOptimizer, ScaledSingleValuedNonLinearOptimizer);

  using Superclass::ParametersType;
  using Superclass::DerivativeType;
  using Superclass::CostFunctionType;
  using Superclass::ScaledCostFunctionType;
  using Superclass::MeasureType;
  using Superclass::ScalesType;

  using RhoType = Array<double>;
  using SType = std::vector<ParametersType>;
  using YType = std::vector<DerivativeType>;
  using DiagonalMatrixType = Array<double>;
  using LineSearchOptimizerType = LineSearchOptimizer;

  using LineSearchOptimizerPointer = LineSearchOptimizerType::Pointer;

  enum StopConditionType
  {
    MetricError,
    LineSearchError,
    MaximumNumberOfIterations,
    InvalidDiagonalMatrix,
    GradientMagnitudeTolerance,
    ZeroStep,
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

  /** Setting: the mininum gradient magnitude.
   *
   * The optimizer stops when:
   * ||CurrentGradient|| < GradientMagnitudeTolerance * max(1, ||CurrentPosition||)
   */
  itkGetConstMacro(GradientMagnitudeTolerance, double);
  itkSetMacro(GradientMagnitudeTolerance, double);

  /** Setting: the memory. The number of iterations that are used
   * to estimate the Hessian. 5 by default. 0 results in (normalised) gradient
   * descent search directions */
  itkSetMacro(Memory, unsigned int);
  itkGetConstMacro(Memory, unsigned int);

protected:
  QuasiNewtonLBFGSOptimizer();
  ~QuasiNewtonLBFGSOptimizer() override = default;

  // \todo: should be implemented
  void
  PrintSelf(std::ostream & os, Indent indent) const override
  {}

  DerivativeType    m_CurrentGradient;
  MeasureType       m_CurrentValue{ 0.0 };
  unsigned long     m_CurrentIteration{ 0 };
  StopConditionType m_StopCondition{ Unknown };
  bool              m_Stop{ false };
  double            m_CurrentStepLength{ 0.0 };

  /** Is true when the LineSearchOptimizer has been started. */
  bool m_InLineSearch{ false };

  RhoType m_Rho;
  SType   m_S;
  YType   m_Y;

  unsigned int m_Point{ 0 };
  unsigned int m_PreviousPoint{ 0 };
  unsigned int m_Bound{ 0 };

  itkSetMacro(InLineSearch, bool);

  /** Compute H0
   *
   * Override this method if not satisfied with the default choice.
   */
  virtual void
  ComputeDiagonalMatrix(DiagonalMatrixType & diag_H0);

  /** Compute -Hg
   *
   *     COMPUTE -H*G USING THE FORMULA GIVEN IN: Nocedal, J. 1980,
   *     "Updating quasi-Newton matrices with limited storage",
   *     Mathematics of Computation, Vol.24, No.151, pp. 773-782.
   */
  virtual void
  ComputeSearchDirection(const DerivativeType & gradient, ParametersType & searchDir);

  /** Perform a line search along the search direction. On calling, x, f, and g should
   * contain the current position, the cost function value at this position, and
   * the derivative. On return the step, x (new position), f (value at x), and g
   * (derivative at x) are updated. */
  virtual void
  LineSearch(const ParametersType searchDir, double & step, ParametersType & x, MeasureType & f, DerivativeType & g);

  /** Store s = x_k - x_k-1 and y = g_k - g_k-1 in m_S and m_Y,
   * and store 1/(ys) in m_Rho. */
  virtual void
  StoreCurrentPoint(const ParametersType & step, const DerivativeType & grad_dif);

  /** Check if convergence has occured;
   * The firstLineSearchDone bool allows the implementation of TestConvergence to
   * decide to skip a few convergence checks when no line search has performed yet
   * (so, before the actual optimisation begins)  */
  virtual bool
  TestConvergence(bool firstLineSearchDone);

private:
  unsigned long              m_MaximumNumberOfIterations{ 100 };
  double                     m_GradientMagnitudeTolerance{ 1e-5 };
  LineSearchOptimizerPointer m_LineSearchOptimizer{ nullptr };
  unsigned int               m_Memory{ 5 };
};

} // end namespace itk

/** ********************* Original documentation **********************************
 *
 * The implementation of this class is based on the netlib function lbfgs_
 *
 * Below the original documentation can be found:
 */

/*        LIMITED MEMORY BFGS METHOD FOR LARGE SCALE OPTIMIZATION */
/*                          JORGE NOCEDAL */
/*                        *** July 1990 *** */

/*     This subroutine solves the unconstrained minimization problem */

/*                      min F(x),    x= (x1,x2,...,xN), */

/*      using the limited memory BFGS method. The routine is especially */
/*      effective on problems involving a large number of variables. In */
/*      a typical iteration of this method an approximation Hk to the */
/*      inverse of the Hessian is obtained by applying M BFGS updates to */
/*      a diagonal matrix Hk0, using information from the previous M steps.  */
/*      The user specifies the number M, which determines the amount of */
/*      storage required by the routine. The user may also provide the */
/*      diagonal matrices Hk0 if not satisfied with the default choice. */
/*      The algorithm is described in "On the limited memory BFGS method */
/*      for large scale optimization", by D. Liu and J. Nocedal, */
/*      Mathematical Programming B 45 (1989) 503-528. */

/*      The user is required to calculate the function value F and its */
/*      gradient G. In order to allow the user complete control over */
/*      these computations, reverse  communication is used. The routine */
/*      must be called repeatedly under the control of the parameter */
/*      IFLAG. */

/*      The steplength is determined at each iteration by means of the */
/*      line search routine MCVSRCH, which is a slight modification of */
/*      the routine CSRCH written by More' and Thuente. */

/*      The calling statement is */

/*          CALL LBFGS(N,M,X,F,G,DIAGCO,DIAG,IPRINT,EPS,XTOL,W,IFLAG) */

/*      where */

/*     N       is an INTEGER variable that must be set by the user to the */
/*             number of variables. It is not altered by the routine. */
/*             Restriction: N>0. */

/*     M       is an INTEGER variable that must be set by the user to */
/*             the number of corrections used in the BFGS update. It */
/*             is not altered by the routine. Values of M less than 3 are */
/*             not recommended; large values of M will result in excessive */
/*             computing time. 3<= M <=7 is recommended. Restriction: M>0.  */

/*     X       is a DOUBLE PRECISION array of length N. On initial entry */
/*             it must be set by the user to the values of the initial */
/*             estimate of the solution vector. On exit with IFLAG=0, it */
/*             contains the values of the variables at the best point */
/*             found (usually a solution). */

/*     F       is a DOUBLE PRECISION variable. Before initial entry and on */
/*             a re-entry with IFLAG=1, it must be set by the user to */
/*             contain the value of the function F at the point X. */

/*     G       is a DOUBLE PRECISION array of length N. Before initial */
/*             entry and on a re-entry with IFLAG=1, it must be set by */
/*             the user to contain the components of the gradient G at */
/*             the point X. */

/*     DIAGCO  is a LOGICAL variable that must be set to .TRUE. if the */
/*             user  wishes to provide the diagonal matrix Hk0 at each */
/*             iteration. Otherwise it should be set to .FALSE., in which */
/*             case  LBFGS will use a default value described below. If */
/*             DIAGCO is set to .TRUE. the routine will return at each */
/*             iteration of the algorithm with IFLAG=2, and the diagonal */
/*              matrix Hk0  must be provided in the array DIAG. */

/*     DIAG    is a DOUBLE PRECISION array of length N. If DIAGCO=.TRUE., */
/*             then on initial entry or on re-entry with IFLAG=2, DIAG */
/*             it must be set by the user to contain the values of the */
/*             diagonal matrix Hk0.  Restriction: all elements of DIAG */
/*             must be positive. */

/*     IPRINT  is an INTEGER array of length two which must be set by the */
/*             user. */

/*             IPRINT(1) specifies the frequency of the output: */
/*                IPRINT(1) < 0 : no output is generated, */
/*                IPRINT(1) = 0 : output only at first and last iteration, */
/*                IPRINT(1) > 0 : output every IPRINT(1) iterations. */

/*             IPRINT(2) specifies the type of output generated: */
/*                IPRINT(2) = 0 : iteration count, number of function */
/*                                evaluations, function value, norm of the */
/*                                gradient, and steplength, */
/*                IPRINT(2) = 1 : same as IPRINT(2)=0, plus vector of */
/*                                variables and  gradient vector at the */
/*                                initial point, */
/*                IPRINT(2) = 2 : same as IPRINT(2)=1, plus vector of */
/*                                variables, */
/*                IPRINT(2) = 3 : same as IPRINT(2)=2, plus gradient vector.*/

/*    EPS     is a positive DOUBLE PRECISION variable that must be set by */
/*            the user, and determines the accuracy with which the solution*/
/*            is to be found. The subroutine terminates when */

/*                         ||G|| < EPS max(1,||X||), */

/*            where ||.|| denotes the Euclidean norm. */

/*    XTOL    is a  positive DOUBLE PRECISION variable that must be set by */
/*            the user to an estimate of the machine precision (e.g. */
/*            10**(-16) on a SUN station 3/60). The line search routine will*/
/*            terminate if the relative width of the interval of uncertainty*/
/*            is less than XTOL. */

/*     W       is a DOUBLE PRECISION array of length N(2M+1)+2M used as */
/*             workspace for LBFGS. This array must not be altered by the */
/*             user. */

/*    IFLAG   is an INTEGER variable that must be set to 0 on initial entry*/
/*            to the subroutine. A return with IFLAG<0 indicates an error, */
/*            and IFLAG=0 indicates that the routine has terminated without*/
/*            detecting errors. On a return with IFLAG=1, the user must */
/*            evaluate the function F and gradient G. On a return with */
/*            IFLAG=2, the user must provide the diagonal matrix Hk0. */

/*            The following negative values of IFLAG, detecting an error, */
/*            are possible: */

/*              IFLAG=-1  The line search routine MCSRCH failed. The */
/*                        parameter INFO provides more detailed information */
/*                        (see also the documentation of MCSRCH): */

/*                       INFO = 0  IMPROPER INPUT PARAMETERS. */

/*                       INFO = 2  RELATIVE WIDTH OF THE INTERVAL OF */
/*                                 UNCERTAINTY IS AT MOST XTOL. */

/*                       INFO = 3  MORE THAN 20 FUNCTION EVALUATIONS WERE */
/*                                 REQUIRED AT THE PRESENT ITERATION. */

/*                       INFO = 4  THE STEP IS TOO SMALL. */

/*                       INFO = 5  THE STEP IS TOO LARGE. */

/*                       INFO = 6  ROUNDING ERRORS PREVENT FURTHER PROGRESS.*/
/*                                 THERE MAY NOT BE A STEP WHICH SATISFIES */
/*                                 THE SUFFICIENT DECREASE AND CURVATURE */
/*                                 CONDITIONS. TOLERANCES MAY BE TOO SMALL.  */

/*             IFLAG=-2  The i-th diagonal element of the diagonal inverse */
/*                       Hessian approximation, given in DIAG, is not */
/*                       positive. */

/*             IFLAG=-3  Improper input parameters for LBFGS (N or M are */
/*                       not positive). */

/*    ON THE DRIVER: */

/*    The program that calls LBFGS must contain the declaration: */

/*                       EXTERNAL LB2 */

/*    LB2 is a BLOCK DATA that defines the default values of several */
/*    parameters described in the COMMON section. */

/*    COMMON: */

/*     The subroutine contains one common area, which the user may wish to */
/*    reference: */

/* awf added stpawf */

/*    MP  is an INTEGER variable with default value 6. It is used as the */
/*        unit number for the printing of the monitoring information */
/*        controlled by IPRINT. */

/*    LP  is an INTEGER variable with default value 6. It is used as the */
/*        unit number for the printing of error messages. This printing */
/*        may be suppressed by setting LP to a non-positive value. */

/*    GTOL is a DOUBLE PRECISION variable with default value 0.9, which */
/*        controls the accuracy of the line search routine MCSRCH. If the */
/*        function and gradient evaluations are inexpensive with respect */
/*        to the cost of the iteration (which is sometimes the case when */
/*        solving very large problems) it may be advantageous to set GTOL */
/*        to a small value. A typical small value is 0.1.  Restriction: */
/*        GTOL should be greater than 1.D-04. */

/*    STPMIN and STPMAX are non-negative DOUBLE PRECISION variables which */
/*        specify lower and uper bounds for the step in the line search.  */
/*        Their default values are 1.D-20 and 1.D+20, respectively. These */
/*        values need not be modified unless the exponents are too large */
/*        for the machine being used, or unless the problem is extremely */
/*        badly scaled (in which case the exponents should be increased).  */

/*  MACHINE DEPENDENCIES */

/*        The only variables that are machine-dependent are XTOL, */
/*        STPMIN and STPMAX. */

/*  GENERAL INFORMATION */

/*    Other routines called directly:  DAXPY, DDOT, LB1, MCSRCH */

/*    Input/Output  :  No input; diagnostic messages on unit MP and */
/*                     error messages on unit LP. */

/*    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

#endif //#ifndef itkQuasiNewtonLBFGSOptimizer_h
