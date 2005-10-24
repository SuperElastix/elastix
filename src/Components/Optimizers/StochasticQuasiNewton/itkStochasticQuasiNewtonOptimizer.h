#ifndef __itkStochasticQuasiNewtonOptimizer_h
#define __itkStochasticQuasiNewtonOptimizer_h

#include "itkScaledSingleValuedNonLinearOptimizer.h"
#include "itkArray2D.h"
#include <list>


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
    
    /** Setting: the maximum number of iterations */
    itkGetConstMacro(MaximumNumberOfIterations, unsigned long);
    itkSetClampMacro(MaximumNumberOfIterations, unsigned long,
      1, NumericTraits<unsigned long>::max());

    /** Setting: the minimum memory. The minimum number of iterations
		 * that are used to estimate the Hessian. 5 by default. */
    itkSetClampMacro(MinimumMemory,unsigned int,1,
										NumericTraits<unsigned int>::max());
    itkGetConstMacro(MinimumMemory,unsigned int);

		/** Setting: the initial step length estimate. If the 
		 * number of iterations is smaller than the mininum memory,
		 * the search direction is computed as: - steplength g / |g|
		 * Default value: 1.0 */
		itkSetMacro(InitialStepLengthEstimate, double);
		itkGetConstMacro(InitialStepLengthEstimate, double);


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
		HessianMatrixType							m_S;
		double												m_ss_ys;
		double												m_ys_yy;
		ParametersType								m_Step;
		DerivativeType								m_GradientDifference;
		unsigned long									m_NumberOfUpdates;
		double												m_GainFactor;

    /** Compute -Hg
     *
     */    
    virtual void ComputeSearchDirection(
      const DerivativeType & gradient,
      ParametersType & searchDir);


  private:
    StochasticQuasiNewtonOptimizer(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    unsigned long                 m_MaximumNumberOfIterations;
    unsigned int                  m_MinimumMemory;
		double												m_InitialStepLengthEstimate;

  }; // end class StochasticQuasiNewtonOptimizer



} // end namespace itk



#endif //#ifndef __itkStochasticQuasiNewtonOptimizer_h

