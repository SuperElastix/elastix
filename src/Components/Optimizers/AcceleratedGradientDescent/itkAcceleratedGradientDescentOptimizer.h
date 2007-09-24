#ifndef __itkStandardGradientDescentOptimizer_h
#define __itkStandardGradientDescentOptimizer_h

#include "itkGradientDescentOptimizer2.h"

namespace itk
{

	/**
	 * \class StandardGradientDescentOptimizer
	 * \brief This class implements a gradient descent optimizer with a decaying gain.
	 *
	 * If \f$C(x)\f$ is a costfunction that has to be minimised, the following iterative
	 * algorithm is used to find the optimal parameters \f$x\f$:
	 * 
	 *     \f[ x(k+1) = x(k) - a(k) dC/dx \f]
	 *
   * The gain \f$a(k)\f$ at each iteration \f$k\f$ is defined by:
	 *
	 *     \f[ a(k) =  a / (A + k + 1)^alpha \f].
	 * 
	 * It is very suitable to be used in combination with a stochastic estimate
	 * of the gradient \f$dC/dx\f$. For example, in image registration problems it is
	 * often advantageous to compute the metric derivative (\f$dC/dx\f$) on a new set 
	 * of randomly selected image samples in each iteration. You may set the parameter
	 * \c NewSamplesEveryIteration to \c "true" to achieve this effect.
	 * For more information on this strategy, you may have a look at:
	 *
	 * S. Klein, M. Staring, J.P.W. Pluim,
	 * "Comparison of gradient approximation techniques for optimisation of mutual information in nonrigid registration",
	 * in: SPIE Medical Imaging: Image Processing,
	 * Editor(s): J.M. Fitzpatrick, J.M. Reinhardt, SPIE press, 2005, vol. 5747, Proceedings of SPIE, pp. 192-203.
	 *   
	 * Note that the gain is computed in the same way as in the
	 * SimultaneousPerturbationOptimizer
	 * 
	 * \sa SimultaneousPerturbationOptimizer, StandardGradientDescent
	 * \ingroup Optimizers
	 */

	class StandardGradientDescentOptimizer :
		public GradientDescentOptimizer2
	{
	public:

		/** Standard ITK.*/
		typedef StandardGradientDescentOptimizer		Self;
		typedef GradientDescentOptimizer2						Superclass;
		
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( StandardGradientDescentOptimizer, GradientDescentOptimizer2 );
				
    /** Typedefs inherited from the superclass. */
    typedef Superclass::MeasureType               MeasureType;
    typedef Superclass::ParametersType            ParametersType;
    typedef Superclass::DerivativeType            DerivativeType;
    typedef Superclass::CostFunctionType          CostFunctionType;
    typedef Superclass::ScalesType                ScalesType;
    typedef Superclass::ScaledCostFunctionType    ScaledCostFunctionType;
    typedef Superclass::ScaledCostFunctionPointer ScaledCostFunctionPointer;
    typedef Superclass::StopConditionType			    StopConditionType;
	
		/** Set/Get a. */
		itkSetMacro( Param_a, double );
		itkGetConstMacro( Param_a, double );
		
		/** Set/Get A. */
		itkSetMacro( Param_A, double );
		itkGetConstMacro( Param_A, double );
		
		/** Set/Get alpha. */
		itkSetMacro( Param_alpha, double );
		itkGetConstMacro( Param_alpha, double );
		
		/** Sets a new LearningRate before calling the Superclass'
		 * implementation, and updates the current time. */
		virtual void AdvanceOneStep(void);

    /** Set current time to 0 and call superclass' implementation. */
		virtual void StartOptimization(void);

    /** Set/Get whether CruzzAcceleration is desired. Default: false */
    itkSetMacro( UseCruzAcceleration, bool );
    itkGetConstMacro( UseCruzAcceleration, bool );

    /** Set/Get the maximum of the sigmoid use by CruzAcceleration. 
     * Should be >0. Default: 1.0 */     
    itkSetMacro(SigmoidMax, double);
    itkGetConstMacro(SigmoidMax, double);

    /** Set/Get the maximum of the sigmoid use by CruzAcceleration. 
     * Should be <0. Default: -0.999 */     
    itkSetMacro(SigmoidMin, double);
    itkGetConstMacro(SigmoidMin, double);

    /** Set/Get the scaling of the sigmoid width. Large values 
     * cause a more wide sigmoid. Default: 1e-8. Should be >0. */     
    itkSetMacro(SigmoidScale, double);
    itkGetConstMacro(SigmoidScale, double);

    /** Set/Get the initial time. Should be >0. */     
    itkSetMacro(InitialTime, double);
    itkGetConstMacro(InitialTime, double);
    		
	protected:

		  StandardGradientDescentOptimizer();
			virtual ~StandardGradientDescentOptimizer() {};

		/** Function to compute the parameter at time/iteration k. */
		virtual double Compute_a( double k ) const;

    /** Function to update the current time
     * If UseCruzAcceleration is false this function just increments
     * the CurrentTime by 1. 
     * Else, the CurrentTime is updated according to:
     * time = max[0, time + sigmoid( -gradient*previousgradient) ]
     * In that case, also updates the previous gradient.
     */
    virtual void UpdateCurrentTime( void );

    /** The current time, which serves as input for Compute_a */
    double m_CurrentTime;

    /** The PreviousGradient, necessary for the CruzAcceleration */
    DerivativeType m_PreviousGradient;
			
	private:

		  StandardGradientDescentOptimizer( const Self& );	// purposely not implemented
			void operator=( const Self& );							// purposely not implemented

		/**Parameters, as described by Spall.*/
		double												m_Param_a;
		double												m_Param_A;
		double												m_Param_alpha;

    /** Settings */
    bool                          m_UseCruzAcceleration;
    double                        m_SigmoidMax;
    double                        m_SigmoidMin;
    double                        m_SigmoidScale;
    double                        m_InitialTime;
		
			
	}; // end class StandardGradientDescentOptimizer
	

} // end namespace itk


#endif // end #ifndef __itkStandardGradientDescentOptimizer_h


