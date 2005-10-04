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
	 * \parameter NewSamplesEveryIteration: Flag that can set to "true" or "false". If "true" 
	 * some optimizers force the metric to (randomly) select a new set of spatial samples in
	 * every iteration. This, if used in combination with the correct optimizer (such as the
	 * StandardGradientDescent), allows for a very low number of spatial samples (around 2000),
	 * even with large images and transforms with a large number of parameters. \n
	 *   example: <tt> (NewSamplesEveryIteration "true" "true" "true") </tt> \n
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

		itkTypeMacro(OptimizerBase, BaseComponentSE);

		/** Typedefs inherited from Elastix.*/
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;

		/** ITKBaseType.*/
		typedef itk::Optimizer	ITKBaseType;

		/** Typedef needed for the SetCurrentPositionPublic function. */
		typedef typename ITKBaseType::ParametersType				ParametersType;

		/** Cast to ITKBaseType.*/
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}

		/** Add empty SetCurrentPositionPublic, so it is known everywhere. */
		virtual void SetCurrentPositionPublic( const ParametersType &param );

		/** Do some things before each resolution that make sense for each
		 * optimizer */
		virtual void BeforeEachResolutionBase();

	protected:

		OptimizerBase();
		virtual ~OptimizerBase() {}

		/**
		 * Force the metric to base its computation on a new subset of image samples.
		 * Not every metric may have implemented this.
		 */
		virtual void SelectNewSamples(void);

		/** Check whether the user asked to select new samples every iteration */
		virtual const bool GetNewSamplesEveryIteration(void) const;
		
	private:

		OptimizerBase( const Self& );		// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

		bool m_NewSamplesEveryIteration;

	}; // end class OptimizerBase


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxOptimizerBase.hxx"
#endif

#endif // end #ifndef __elxOptimizerBase_h

