#ifndef __elxRegistrationBase_h
#define __elxRegistrationBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkMultiResolutionImageRegistrationMethod.h"


namespace elastix
{
using namespace itk;

	/**
	 * \class RegistrationBase
	 * \brief This class is the base for all Registrations
	 *
	 * This class contains all the common functionality for Registrations ...
	 *
	 * \ingroup Registrations
	 * \ingroup ComponentBaseClasses
	 */

	template <class TElastix>
		class RegistrationBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard ITK.*/
		typedef RegistrationBase						Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Typedef's from Elastix.*/
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;

		/** Useless here: */
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;
	
		/** the template parameters to be used: */
		typedef typename ElastixType::FixedInternalImageType		FixedImageType;
		typedef typename ElastixType::MovingInternalImageType		MovingImageType;

		/** Typedef for ITKBaseType.*/
		typedef itk::MultiResolutionImageRegistrationMethod<
			FixedImageType,	MovingImageType >				ITKBaseType;

		/** Cast to ITKBaseType.*/
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}
		
	protected:

		RegistrationBase() {}
		virtual ~RegistrationBase() {}

	private:

		RegistrationBase( const Self& );	// purposely not implemented
		void operator=( const Self& );		// purposely not implemented

	}; // end class RegistrationBase


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxRegistrationBase.hxx"
#endif

#endif // end #ifndef __elxRegistrationBase_h
