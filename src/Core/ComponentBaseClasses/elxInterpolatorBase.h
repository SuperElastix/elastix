#ifndef __elxInterpolatorBase_h
#define __elxInterpolatorBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"

#include "itkInterpolateImageFunction.h"


namespace elastix
{
using namespace itk;

	/**
	 * \class InterpolatorBase
	 * \brief This class is the base for all Interpolators
	 *
	 * This class contains all the common functionality for Interpolators ...
	 *
	 * \ingroup Interpolators
	 * \ingroup ComponentBaseClasses
	 */

	template <class TElastix>
		class InterpolatorBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard.*/
		typedef InterpolatorBase Self;
		typedef BaseComponentSE<TElastix> Superclass;

		/** Typedefs inherited from Elastix.*/
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;

		/** Other typedef's.*/
		typedef typename ElastixType::MovingInternalImageType		InputImageType;
		typedef typename ElastixType::CoordRepType							CoordRepType;

		/** ITKBaseType.*/
		typedef InterpolateImageFunction< 
			InputImageType, CoordRepType>											ITKBaseType;

		/** Cast to ITKBaseType.*/
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}
		
	protected:

		InterpolatorBase() {}
		virtual ~InterpolatorBase() {}

	private:

		InterpolatorBase( const Self& );	// purposely not implemented
		void operator=( const Self& );		// purposely not implemented

	}; // end class InterpolatorBase


} // end namespace elastix



#ifndef ITK_MANUAL_INSTANTIATION
#include "elxInterpolatorBase.hxx"
#endif

#endif // end #ifndef __elxInterpolatorBase_h

