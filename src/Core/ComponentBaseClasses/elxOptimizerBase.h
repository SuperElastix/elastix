#ifndef __elxOptimizerBase_h
#define __elxOptimizerBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkOptimizer.h"


namespace elastix
{
using namespace itk;

	/**
	 * \class OptimizerBase
	 * \brief This class is the base for all Optimizers
	 *
	 * This class contains all the common functionality for Optimizers ...
	 *
	 * \ingroup Optimizers
	 * \ingroup ComponentBaseClasses
	 */

	template <class TElastix>
		class OptimizerBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard.*/
		typedef OptimizerBase								Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Typedefs inherited from Elastix.*/
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;

		/** ITKBaseType.*/
		typedef itk::Optimizer	ITKBaseType;

		/** Cast to ITKBaseType.*/
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}
		
	protected:

		OptimizerBase() {}
		virtual ~OptimizerBase() {}

	private:

		OptimizerBase( const Self& );		// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

	}; // end class OptimizerBase


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxOptimizerBase.hxx"
#endif

#endif // end #ifndef __elxOptimizerBase_h

