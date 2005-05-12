
#ifndef __itkGenericConjugateGradientOptimizer_h
#define __itkGenericConjugateGradientOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "itkLineSearchOptimizer.h"
#include <vector>
#include <map>

namespace itk
{
  /** \class GenericConjugateGradientOptimizer
   * \brief A set of conjugate gradient algorithms ...
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
		
    /** Typedef for a function that computes \beta, given the previousGradient,
		 * the current gradient, and the previous search direction */
		typedef double (Self::*												ComputeBetaFunctionType
			)( const DerivativeType & ,
         const DerivativeType & ,
		     const ParametersType & );
		typedef std::string														BetaDefinitionType;
		typedef std::map< BetaDefinitionType,
			ComputeBetaFunctionType >										BetaDefinitionMapType;
		        
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
     * ||CurrentGradient|| < 
		 *   GradientMagnitudeTolerance * max(1, ||CurrentPosition||)
     */
    itkGetConstMacro(GradientMagnitudeTolerance, double);
    itkSetMacro(GradientMagnitudeTolerance, double)

    /** Setting: a stopping criterion, the value tolerance. By default 1e-5.
     * 
     * The optimizer stops when:
		 *
		 * 2.0 * abs( f_k - f_k-1 ) <= 
		 *   ValueTolerance * ( abs(f_k) + abs(f_k-1) + 1e-20 )
		 *
		 * is satisfied MaxNrOfItWithoutImprovement times in a row.
     */
    itkGetConstMacro(ValueTolerance, double);
    itkSetMacro(ValueTolerance, double);

		/** Setting: the maximum number of iterations in a row that 
		 * satisfy the value tolerance criterion. By default (if never set) 
		 * equal to the number of parameters. */
		virtual void SetMaxNrOfItWithoutImprovement(unsigned long arg);
		itkGetConstMacro(MaxNrOfItWithoutImprovement, unsigned long);

		/** Setting: the definition of \beta, by default "DaiYuanHestenesStiefel" */
		virtual void SetBetaDefinition(const BetaDefinitionType & arg);
		itkGetConstReferenceMacro(BetaDefinition, BetaDefinitionType);
		    
  protected:
    GenericConjugateGradientOptimizer();
    virtual ~GenericConjugateGradientOptimizer(){};

    void PrintSelf(std::ostream& os, Indent indent) const {};

    DerivativeType                m_CurrentGradient;
		MeasureType                   m_CurrentValue;
		unsigned long                 m_CurrentIteration;
    StopConditionType             m_StopCondition;
    bool                          m_Stop;
    double                        m_CurrentStepLength;

		/** Flag that is true as long as the method
		 * SetMaxNrOfItWithoutImprovement is never called */
		bool													m_UseDefaultMaxNrOfItWithoutImprovement;

    /** Is true when the LineSearchOptimizer has been started. */
    bool                          m_InLineSearch;
		itkSetMacro(InLineSearch, bool);

		/** Flag that says if the previous gradient and search direction are known.
		 * Typically 'true' at the start of optimization, or when a stopped optimisation
		 * is resumed (in the latter case the previous gradient and search direction
		 * may of course still be valid, but to be safe it is assumed that they are not). */
		bool													m_PreviousGradientAndSearchDirValid;

		/** The name of the BetaDefinition */
		BetaDefinitionType  					m_BetaDefinition;
		
		/** A mapping that links the names of the BetaDefinitions to functions that
		 * compute \beta. */
		BetaDefinitionMapType					m_BetaDefinitionMap;
		
		/** Function to add a new beta definition. The first argument should be a name 
		 * via which a user can select this beta definition. The second argument is a 
		 * pointer to a method that computes \beta.
		 * Called in the constructor of this class, and possibly by subclasses.
		 */
		virtual void AddBetaDefinition(
			const BetaDefinitionType & name,
			ComputeBetaFunctionType function);
   	    
    /** 
		 * Compute the search direction:
		 *  d_{k} = - g_{k} + \beta_{k} d_{k-1}
		 *
		 * In the first iteration the search direction is computed as:
		 *  d_{0} = - g_{0}
		 *
		 * On calling, searchDir should equal d_{k-1}. On return searchDir 
		 * equals d_{k}.
		 */
    virtual void ComputeSearchDirection(
			const DerivativeType & previousGradient,
      const DerivativeType & gradient,
			ParametersType & searchDir);

    /** Perform a line search along the search direction. On calling, x, f, and g should
     * contain the current position, the cost function value at this position, and 
     * the derivative. On return the step, x (new position), f (value at x), and g
     * (derivative at x) are updated. */
    virtual void LineSearch(
      const ParametersType searchDir,
      double & step,
      ParametersType & x,
      MeasureType & f,
      DerivativeType & g );

		/** Compute \beta according to the user set \beta-definition */
		virtual double ComputeBeta(
			const DerivativeType & previousGradient,
      const DerivativeType & gradient,
			const ParametersType & previousSearchDir);

		/** Different definitions of \beta */

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
    GenericConjugateGradientOptimizer(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    unsigned long                 m_MaximumNumberOfIterations;
    double                        m_ValueTolerance;
		double                        m_GradientMagnitudeTolerance;
		unsigned long									m_MaxNrOfItWithoutImprovement;
		
    LineSearchOptimizerPointer    m_LineSearchOptimizer;
    
    
	}; // end class GenericConjugateGradientOptimizer


} // end namespace itk


#endif //#ifndef __itkGenericConjugateGradientOptimizer_h

