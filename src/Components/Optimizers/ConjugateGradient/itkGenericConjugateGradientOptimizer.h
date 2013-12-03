/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/


#ifndef __itkGenericConjugateGradientOptimizer_h
#define __itkGenericConjugateGradientOptimizer_h

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

  class GenericConjugateGradientOptimizer :
    public ScaledSingleValuedNonLinearOptimizer
  {
  public:

    typedef GenericConjugateGradientOptimizer     Self;
    typedef ScaledSingleValuedNonLinearOptimizer  Superclass;
    typedef SmartPointer<Self>                    Pointer;
    typedef SmartPointer<const Self>              ConstPointer;

    itkNewMacro(Self);
    itkTypeMacro(GenericConjugateGradientOptimizer,
      ScaledSingleValuedNonLinearOptimizer);

    typedef Superclass::ParametersType            ParametersType;
    typedef Superclass::DerivativeType            DerivativeType;
    typedef Superclass::CostFunctionType          CostFunctionType;
    typedef Superclass::ScaledCostFunctionType    ScaledCostFunctionType;
    typedef Superclass::MeasureType               MeasureType;
    typedef Superclass::ScalesType                ScalesType;

    typedef LineSearchOptimizer                   LineSearchOptimizerType;
    typedef LineSearchOptimizerType::Pointer      LineSearchOptimizerPointer;

    /** Typedef for a function that computes \f$\beta\f$, given the previousGradient,
     * the current gradient, and the previous search direction */
    typedef double (Self::*                       ComputeBetaFunctionType
      )( const DerivativeType & ,
         const DerivativeType & ,
         const ParametersType & );
    typedef std::string                           BetaDefinitionType;
    typedef std::map< BetaDefinitionType,
      ComputeBetaFunctionType >                   BetaDefinitionMapType;

    typedef enum {
      MetricError,
      LineSearchError,
      MaximumNumberOfIterations,
      GradientMagnitudeTolerance,
      ValueTolerance,
      InfiniteBeta,
      Unknown }                                   StopConditionType;

    virtual void StartOptimization(void);
    virtual void ResumeOptimization(void);
    virtual void StopOptimization(void);

    /** Get information about optimization process: */
    itkGetConstMacro(CurrentIteration, unsigned long);
    itkGetConstMacro(CurrentValue, MeasureType);
    itkGetConstReferenceMacro(CurrentGradient, DerivativeType);
    itkGetConstMacro(InLineSearch, bool);
    itkGetConstReferenceMacro(StopCondition, StopConditionType);
    itkGetConstMacro(CurrentStepLength, double);

    /** Setting: the line search optimizer */
    itkSetObjectMacro(LineSearchOptimizer, LineSearchOptimizerType);
    itkGetObjectMacro(LineSearchOptimizer, LineSearchOptimizerType);

    /** Setting: the maximum number of iterations */
    itkGetConstMacro(MaximumNumberOfIterations, unsigned long);
    itkSetClampMacro(MaximumNumberOfIterations, unsigned long,
      1, NumericTraits<unsigned long>::max());

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
    virtual void SetMaxNrOfItWithoutImprovement(unsigned long arg);
    itkGetConstMacro(MaxNrOfItWithoutImprovement, unsigned long);

    /** Setting: the definition of \f$\beta\f$, by default "DaiYuanHestenesStiefel" */
    virtual void SetBetaDefinition(const BetaDefinitionType & arg);
    itkGetConstReferenceMacro(BetaDefinition, BetaDefinitionType);

  protected:
    GenericConjugateGradientOptimizer();
    virtual ~GenericConjugateGradientOptimizer(){};

    void PrintSelf( std::ostream & os, Indent indent ) const;

    DerivativeType                m_CurrentGradient;
    MeasureType                   m_CurrentValue;
    unsigned long                 m_CurrentIteration;
    StopConditionType             m_StopCondition;
    bool                          m_Stop;
    double                        m_CurrentStepLength;

    /** Flag that is true as long as the method
     * SetMaxNrOfItWithoutImprovement is never called */
    bool                          m_UseDefaultMaxNrOfItWithoutImprovement;

    /** Is true when the LineSearchOptimizer has been started. */
    bool                          m_InLineSearch;
    itkSetMacro(InLineSearch, bool);

    /** Flag that says if the previous gradient and search direction are known.
     * Typically 'true' at the start of optimization, or when a stopped optimisation
     * is resumed (in the latter case the previous gradient and search direction
     * may of course still be valid, but to be safe it is assumed that they are not). */
    bool                          m_PreviousGradientAndSearchDirValid;

    /** The name of the BetaDefinition */
    BetaDefinitionType            m_BetaDefinition;

    /** A mapping that links the names of the BetaDefinitions to functions that
     * compute \f$\beta\f$. */
    BetaDefinitionMapType         m_BetaDefinitionMap;

    /** Function to add a new beta definition. The first argument should be a name
     * via which a user can select this \f$\beta\f$ definition. The second argument is a
     * pointer to a method that computes \f$\beta\f$.
     * Called in the constructor of this class, and possibly by subclasses.
     */
    virtual void AddBetaDefinition(
      const BetaDefinitionType & name,
      ComputeBetaFunctionType function);

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
    virtual void ComputeSearchDirection(
      const DerivativeType & previousGradient,
      const DerivativeType & gradient,
      ParametersType & searchDir);

    /** Perform a line search along the search direction. On calling, \f$x, f\f$, and \f$g\f$ should
     * contain the current position, the cost function value at this position, and
     * the derivative. On return the step, \f$x\f$ (new position), \f$f\f$ (value at \f$x\f$), and \f$g\f$
     * (derivative at \f$x\f$) are updated. */
    virtual void LineSearch(
      const ParametersType searchDir,
      double & step,
      ParametersType & x,
      MeasureType & f,
      DerivativeType & g );

    /** Check if convergence has occured;
     * The firstLineSearchDone bool allows the implementation of TestConvergence to
     * decide to skip a few convergence checks when no line search has performed yet
     * (so, before the actual optimisation begins)  */
    virtual bool TestConvergence(bool firstLineSearchDone);

    /** Compute \f$\beta\f$ according to the user set \f$\beta\f$-definition */
    virtual double ComputeBeta(
      const DerivativeType & previousGradient,
      const DerivativeType & gradient,
      const ParametersType & previousSearchDir);

    /** Different definitions of \f$\beta\f$ */

    /** "SteepestDescent: beta=0 */
    double ComputeBetaSD(
      const DerivativeType & previousGradient,
      const DerivativeType & gradient,
      const ParametersType & previousSearchDir);
    /** "FletcherReeves" */
    double ComputeBetaFR(
      const DerivativeType & previousGradient,
      const DerivativeType & gradient,
      const ParametersType & previousSearchDir);
    /** "PolakRibiere" */
    double ComputeBetaPR(
      const DerivativeType & previousGradient,
      const DerivativeType & gradient,
      const ParametersType & previousSearchDir);
    /** "DaiYuan" */
    double ComputeBetaDY(
      const DerivativeType & previousGradient,
      const DerivativeType & gradient,
      const ParametersType & previousSearchDir);
    /** "HestenesStiefel" */
    double ComputeBetaHS(
      const DerivativeType & previousGradient,
      const DerivativeType & gradient,
      const ParametersType & previousSearchDir);
    /** "DaiYuanHestenesStiefel" */
    double ComputeBetaDYHS(
      const DerivativeType & previousGradient,
      const DerivativeType & gradient,
      const ParametersType & previousSearchDir);


  private:
    GenericConjugateGradientOptimizer(const Self&); // purposely not implemented
    void operator=(const Self&); // purposely not implemented

    unsigned long                 m_MaximumNumberOfIterations;
    double                        m_ValueTolerance;
    double                        m_GradientMagnitudeTolerance;
    unsigned long                 m_MaxNrOfItWithoutImprovement;

    LineSearchOptimizerPointer    m_LineSearchOptimizer;

  };


} // end namespace itk


#endif //#ifndef __itkGenericConjugateGradientOptimizer_h

