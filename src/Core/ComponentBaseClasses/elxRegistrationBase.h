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
	 * \brief This class is the base for all Registrations.
	 *
	 * This class contains all the common functionality for Registrations.
	 *
	 * \ingroup Registrations
	 * \ingroup ComponentBaseClasses
	 */

	template <class TElastix>
		class RegistrationBase : public BaseComponentSE<TElastix>
	{
	public:

		/** Standard ITK stuff. */
		typedef RegistrationBase						Self;
		typedef BaseComponentSE<TElastix>		Superclass;

		/** Run-time type information (and related methods). */
		itkTypeMacro( RegistrationBase, BaseComponentSE );

		/** Typedef's from Elastix. */
		typedef typename Superclass::ElastixType						ElastixType;
		typedef typename Superclass::ElastixPointer					ElastixPointer;
		typedef typename Superclass::ConfigurationType			ConfigurationType;
		typedef typename Superclass::ConfigurationPointer		ConfigurationPointer;
		typedef typename Superclass::RegistrationType				RegistrationType;
		typedef typename Superclass::RegistrationPointer		RegistrationPointer;
	
		/** Other typedef's. */
		typedef typename ElastixType::FixedInternalImageType		FixedImageType;
		typedef typename ElastixType::MovingInternalImageType		MovingImageType;

		/** Typedef for ITKBaseType. */
		typedef itk::MultiResolutionImageRegistrationMethod<
			FixedImageType,	MovingImageType >				ITKBaseType;

		/** Cast to ITKBaseType. */
		virtual ITKBaseType * GetAsITKBaseType(void)
		{
			return dynamic_cast<ITKBaseType *>(this);
		}
		
	protected:

		/** The constructor. */
		RegistrationBase() {}
		/** The destructor. */
		virtual ~RegistrationBase() {}

	private:

		/** The private constructor. */
		RegistrationBase( const Self& );	// purposely not implemented
		/** The private copy constructor. */
		void operator=( const Self& );		// purposely not implemented

	}; // end class RegistrationBase


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxRegistrationBase.hxx"
#endif

#endif // end #ifndef __elxRegistrationBase_h
