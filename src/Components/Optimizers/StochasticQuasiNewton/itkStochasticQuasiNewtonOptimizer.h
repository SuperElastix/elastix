#ifndef __itkStochasticQuasiNewtonOptimizer_h
#define __itkStochasticQuasiNewtonOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "itkArray2D.h"

namespace itk
{
  /** \class StochasticQuasiNewtonOptimizer
   * \brief Experimental optimizer ...
   *
   * The optimizer solves the unconstrained minimization problem 
   *
   *                min F(x),    x= (x1,x2,...,xN), 
   *
   * 
   * \ingroup Numerics Optimizers
   */

  class StochasticQuasiNewtonOptimizer : public ScaledSingleValuedNonLinearOptimizer
  {
  public:

    typedef StochasticQuasiNewtonOptimizer             Self;
    typedef ScaledSingleValuedNonLinearOptimizer  Superclass;
    typedef SmartPointer<Self>                    Pointer;
    typedef SmartPointer<const Self>              ConstPointer;

    itkNewMacro(Self);
    itkTypeMacro(StochasticQuasiNewtonOptimizer, ScaledSingleValuedNonLinearOptimizer);

    typedef Superclass::ParametersType            ParametersType;
    typedef Superclass::DerivativeType            DerivativeType;
    typedef Superclass::CostFunctionType          CostFunctionType;
    typedef Superclass::ScaledCostFunctionType    ScaledCostFunctionType;
    typedef Superclass::MeasureType               MeasureType;
    typedef Superclass::ScalesType                ScalesType;

		typedef itk::Array2D<double>									HessianMatrixType;
		
    typedef enum {
      MetricError,
      MaximumNumberOfIterations,
      GradientMagnitudeTolerance,
			ZeroStep,
      Unknown }                                   StopConditionType;

    virtual void StartOptimization(void);
    virtual void ResumeOptimization(void);
    virtual void StopOptimization(void);
    
    /** Get information about optimization process: */
    itkGetConstMacro(CurrentIteration, unsigned long);
    itkGetConstMacro(CurrentValue, MeasureType);
    itkGetConstReferenceMacro(CurrentGradient, DerivativeType);
    itkGetConstReferenceMacro(StopCondition, StopConditionType);
    itkGetConstMacro(CurrentStepLength, double);
		itkGetConstMacro(GainFactor, double);
		itkGetConstMacro(UpdateFactor, double);
    
    /** Setting: the maximum number of iterations */
    itkGetConstMacro(MaximumNumberOfIterations, unsigned long);
    itkSetClampMacro(MaximumNumberOfIterations, unsigned long,
      1, NumericTraits<unsigned long>::max());

    /** Setting: the NumberOfInitializationSteps. The number of iterations
		 * that are used to estimate the initial Hessian. 5 by default.
		 *
		 * In these iterations the search directions are computed as:
		 * - gain *g / |g|.
		 * where the gain is varied between the InitialStepLengthEstimate
		 * and 1.0/InitialStepLengthEstimate.
		 */
    itkSetClampMacro(NumberOfInitializationSteps ,unsigned int,0,
										NumericTraits<unsigned int>::max());
    itkGetConstMacro(NumberOfInitializationSteps ,unsigned int);

		/** Setting: the initial step length estimate. In the 
		 * very first iteration the search direction is computed as:
		 * - steplength g / |g|
		 * Default value: 2.0 */
		itkSetMacro(InitialStepLengthEstimate, double);
		itkGetConstMacro(InitialStepLengthEstimate, double);

		itkSetMacro(BetaMax, double);
		itkSetMacro(DetMax, double);
		itkSetClampMacro(Decay_A, double, 0.0, NumericTraits<double>::max());
		itkSetMacro(Decay_alpha, double);

		itkGetConstMacro(BetaMax, double);
		itkGetConstMacro(DetMax, double);
		itkGetConstMacro(Decay_A, double);
		itkGetConstMacro(Decay_alpha, double);

		itkSetMacro(NormalizeInitialGradients, bool);
		itkGetConstMacro(NormalizeInitialGradients, bool);

		itkSetMacro(NumberOfGradientDescentIterations, unsigned int);
		itkGetConstMacro(NumberOfGradientDescentIterations, unsigned int);

		itkGetConstMacro(UseHessian, bool);


  protected:
    StochasticQuasiNewtonOptimizer();
    virtual ~StochasticQuasiNewtonOptimizer(){};

    void PrintSelf(std::ostream& os, Indent indent) const {};

    DerivativeType                m_CurrentGradient;
    MeasureType                   m_CurrentValue;
    unsigned long                 m_CurrentIteration;
    StopConditionType             m_StopCondition;
    bool                          m_Stop;
    double                        m_CurrentStepLength;

		HessianMatrixType							m_H;
		HessianMatrixType							m_B;
		ParametersType								m_Step;
		DerivativeType								m_GradientDifference;
		double												m_GainFactor;
		double												m_Diag;
		double												m_UpdateFactor;
		bool													m_UseHessian;

    /** Compute the new step.
     *
     */    
    virtual void ComputeSearchDirection(
      const DerivativeType & gradient,
      ParametersType & searchDir);

		virtual void ComputeInitialSearchDirection(
      const DerivativeType & gradient,
      ParametersType & searchDir);

		virtual void UpdateHessianMatrix(void);


  private:
    StochasticQuasiNewtonOptimizer(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    unsigned long                 m_MaximumNumberOfIterations;
    unsigned int                  m_NumberOfInitializationSteps;
		double												m_InitialStepLengthEstimate;
		double												m_BetaMax;
		double												m_DetMax;
		double												m_Decay_A;
		double												m_Decay_alpha;
		bool													m_NormalizeInitialGradients;
		unsigned int									m_NumberOfGradientDescentIterations;


  }; // end class StochasticQuasiNewtonOptimizer



} // end namespace itk



#endif //#ifndef __itkStochasticQuasiNewtonOptimizer_h

