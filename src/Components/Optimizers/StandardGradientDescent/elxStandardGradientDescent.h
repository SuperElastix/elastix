#ifndef __elxStandardGradientDescent_h
#define __elxStandardGradientDescent_h

#include "itkStandardGradientDescentOptimizer.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;


	/**
	 * \class StandardGradientDescent
	 * \brief A gradient descent optimizer with a decaying gain.
	 *
   * This class is a wrap around the StandardGradientDescentOptimizer class.
	 * It takes care of setting parameters and printing progress information.
	 * For more information about the optimisation method, please read the documentation
	 * of the StandardGradientDescentOptimizer class.
	 *
	 * The parameters used in this class are:
	 * \parameter Optimizer: Select this optimizer as follows:\n
	 *		<tt>(Optimizer "StandardGradientDescent")</tt>
	 * \parameter MaximumNumberOfIterations: The maximum number of iterations in each resolution. \n
	 *		example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
	 *    Default value: 100.
	 * \parameter SP_a: The gain \a a(k) at each iteration \a k is defined by \n
	 *   <em>a(k) =  SP_a / (SP_A + k + 1)^SP_alpha</em>. \n
	 *   SP_a can be defined for each resolution. \n
	 *   example: <tt>(SP_a 3200.0 3200.0 1600.0)</tt> \n
	 *   The default value is 400.0. Tuning this variable for you specific problem is recommended.
	 * \parameter SP_A: The gain \a a(k) at each iteration \a k is defined by \n
	 *   <em>a(k) =  SP_a / (SP_A + k + 1)^SP_alpha</em>. \n
	 *   SP_A can be defined for each resolution. \n
	 *   example: <tt>(SP_A 50.0 50.0 100.0)</tt> \n
	 *   The default/recommended value is 50.0.
   * \parameter SP_alpha: The gain \a a(k) at each iteration \a k is defined by \n
	 *   <em>a(k) =  SP_a / (SP_A + k + 1)^SP_alpha</em>. \n
	 *   SP_alpha can be defined for each resolution. \n
	 *   example: <tt>(SP_alpha 0.602 0.602 0.602)</tt> \n
	 *   The default/recommended value is 0.602.
	 *
	 * \sa StandardGradientDescentOptimizer
	 * \ingroup Optimizers
	 */

	template <class TElastix>
		class StandardGradientDescent :
		public
			itk::StandardGradientDescentOptimizer,
		public
			OptimizerBase<TElastix>
	{
	public:

		/** Standard ITK.*/
		typedef StandardGradientDescent							Self;
		typedef StandardGradientDescentOptimizer		Superclass1;
		typedef OptimizerBase<TElastix>							Superclass2;
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( StandardGradientDescent, StandardGradientDescentOptimizer );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific optimizer.
		 * example: <tt>(Optimizer "StandardGradientDescent")</tt>\n
		 */
		elxClassNameMacro( "StandardGradientDescent" );

		/** Typedef's inherited from Superclass1, the StandardGradientDescentOptimizer.*/
	  typedef Superclass1::CostFunctionType			CostFunctionType;
		typedef Superclass1::CostFunctionPointer	CostFunctionPointer;
		typedef Superclass1::StopConditionType		StopConditionType;
		
		/** Typedef's inherited from Superclass2, the elastix OptimizerBase .*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;
		
		/** Typedef for the ParametersType. */
		typedef typename Superclass1::ParametersType				ParametersType;

		/** Methods invoked by elastix, in which parameters can be set and 
		 * progress information can be printed. */
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		virtual void AfterEachResolution(void);
		virtual void AfterEachIteration(void);
		virtual void AfterRegistration(void);		
		
   /** Check if any scales are set, and set the UseScales flag on or off; 
		 * after that call the superclass' implementation */
		virtual void StartOptimization(void);

		/** Add SetCurrentPositionPublic, which calls the protected
		 * SetCurrentPosition of the itkStandardGradientDescentOptimizer class.
		 */
		virtual void SetCurrentPositionPublic( const ParametersType &param )
		{
			this->Superclass1::SetCurrentPosition( param );
		}
		
	protected:

    StandardGradientDescent(){};
		virtual ~StandardGradientDescent() {};
				
	private:

		StandardGradientDescent( const Self& );	// purposely not implemented
		void operator=( const Self& );							// purposely not implemented
			
	}; // end class StandardGradientDescent
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxStandardGradientDescent.hxx"
#endif

#endif // end #ifndef __elxStandardGradientDescent_h
