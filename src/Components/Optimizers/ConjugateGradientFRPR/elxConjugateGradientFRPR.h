/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxConjugateGradientFRPR_h
#define __elxConjugateGradientFRPR_h

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
    class ConjugateGradientFRPR :
    public
      itk::FRPROptimizer,
    public
      OptimizerBase<TElastix>
  {
  public:

    /** Standard ITK.*/
    typedef ConjugateGradientFRPR                    Self;
    typedef itk::FRPROptimizer                       Superclass1;
    typedef OptimizerBase<TElastix>                  Superclass2;
    typedef itk::SmartPointer<Self>                  Pointer;
    typedef itk::SmartPointer<const Self>            ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( ConjugateGradientFRPR, itk::FRPROptimizer );

    /** Name of this class.*/
    elxClassNameMacro( "ConjugateGradientFRPR" );

    /** Typedef's inherited from Superclass1.*/
    typedef Superclass1::CostFunctionType                   CostFunctionType;
    typedef Superclass1::CostFunctionPointer                CostFunctionPointer;
    //typedef Superclass1::StopConditionType                  StopConditionType; not implemented in this itkOptimizer
    typedef typename Superclass1::ParametersType            ParametersType;
    //not declared in Superclass, although it should be.
    typedef SingleValuedNonLinearOptimizer::DerivativeType  DerivativeType;

    /** Typedef's inherited from Elastix.*/
    typedef typename Superclass2::ElastixType           ElastixType;
    typedef typename Superclass2::ElastixPointer        ElastixPointer;
    typedef typename Superclass2::ConfigurationType     ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer  ConfigurationPointer;
    typedef typename Superclass2::RegistrationType      RegistrationType;
    typedef typename Superclass2::RegistrationPointer   RegistrationPointer;
    typedef typename Superclass2::ITKBaseType           ITKBaseType;

    /** Methods to set parameters and print output at different stages
     * in the registration process.*/
    virtual void BeforeRegistration(void);
    virtual void BeforeEachResolution(void);
    virtual void AfterEachResolution(void);
    virtual void AfterEachIteration(void);
    virtual void AfterRegistration(void);

    /** Override the SetInitialPosition.*/
    virtual void SetInitialPosition( const ParametersType & param );

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
      virtual ~ConjugateGradientFRPR() {};

      /** To store the latest computed derivative's magnitude */
      double          m_CurrentDerivativeMagnitude ;

      /** Variable to store the line search direction magnitude */
      double          m_CurrentSearchDirectionMagnitude;

      /** the current gain */
      double          m_CurrentStepLength;

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
      virtual void GetValueAndDerivative(ParametersType p, double * val,
        ParametersType * xi);

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
      virtual void   LineBracket(double *ax, double *bx, double *cx,
                                  double *fa, double *fb, double *fc);

      /** Given a bracketing triple of points and their function values, returns
       * a bounded extreme.  These values are in parameter space, along the
       * current line and wrt the current origin set via SetLine.   Optimization
       * terminates based on MaximumIteration, StepTolerance, or ValueTolerance.
       * Implemented as Brent line optimers from NRC.
       *
       * This implementation sets the LineOptimizing flag to 'true', calls the
       * the superclass's implementation, stores extX as the current step length,
       * and sets the LineOptimizing flag to 'false' again. */
      virtual void   BracketedLineOptimize(double ax, double bx, double cx,
                                            double fa, double fb, double fc,
                                            double * extX, double * extVal);

      /**
       * store the line search direction's (xi) magnitude and call the superclass'
       * implementation.
       */
      virtual void   LineOptimize(ParametersType * p, ParametersType xi,
                              double * val );


  private:

      ConjugateGradientFRPR( const Self& ); // purposely not implemented
      void operator=( const Self& );              // purposely not implemented

      bool m_LineOptimizing;
      bool m_LineBracketing;

      const char * DeterminePhase(void) const;


  };


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxConjugateGradientFRPR.hxx"
#endif

#endif // end #ifndef __elxConjugateGradientFRPR_h
