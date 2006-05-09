#ifndef __elxInstallFunctions_h
#define __elxInstallFunctions_h

#include "elxComponentDatabase.h"

namespace elastix
{


	/**
	 * \class InstallFunctions
	 *
	 * \brief The InstallFunctions class ....
	 */

	template<class TAnyItkObject>
	class InstallFunctions
	{
	public:

		/** Standard.*/
		typedef InstallFunctions			Self;
		typedef TAnyItkObject					AnyItkObjectType;

		/** Other typedef's.*/
		typedef ComponentDatabase::ObjectType									ObjectType;
		typedef ComponentDatabase::ObjectPointer							ObjectPointer;
		typedef ComponentDatabase::IndexType									IndexType; // unsigned int
		typedef ComponentDatabase::ComponentDescriptionType		ComponentDescriptionType;// std::string

		/** A wrap around the ::New() functions of itkObjects.*/
		static ObjectPointer Creator(void)
		{
			return dynamic_cast< ObjectType * >( AnyItkObjectType::New().GetPointer() );
		}

		/** This function places the address of the ::New() function
		 * of AnyItkObjectType in the ComponentDatabase.
		 */
		static int InstallComponent(
			const ComponentDescriptionType & name, 
			IndexType i, ComponentDatabase * cdb ) 
		{
			return cdb->SetCreator( name, i, Self::Creator );
		}

	}; // end class InstallFunctions


} // end namespace elastix


#endif // end #ifndef __elxInstallFunctions_h

