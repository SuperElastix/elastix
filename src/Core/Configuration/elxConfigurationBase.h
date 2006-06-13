#ifndef __elxConfigurationBase_h
#define __elxConfigurationBase_h

#include "elxBaseComponent.h"

namespace elastix
{
	
	/**
	 * \class ConfigurationBase
	 * \brief BaseClass for classes that deal with user given parameters/command line arguments.
	 *
	 * The ConfigurationBase class ....
	 * Base class for all Configuration classes. Currently it's a quite
	 * useless class, since there is only one class that inherits from it,
	 * but in future it might be used.
	 *
	 * \sa MyConfiguration
	 * \ingroup Configuration
	 */

	class ConfigurationBase : public BaseComponent
	{
	public:

		/** Standard.*/
		typedef ConfigurationBase		Self;
		typedef BaseComponent				Superclass;

	protected:

		ConfigurationBase() {}
		virtual ~ConfigurationBase() {}

	private:

		ConfigurationBase( const Self& );	// purposely not implemented
		void operator=( const Self& );		// purposely not implemented

	}; // end class ConfigurationBase


} // end namespace elastix



#endif // end #ifndef __elxConfigurationBase_h

