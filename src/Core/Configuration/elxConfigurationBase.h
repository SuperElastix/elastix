/**
* Base class for all Configuration classes. Currently it's a quite
* useless class, but in future it might be used.
*
*/

#ifndef __elxConfigurationBase_h
#define __elxConfigurationBase_h

#include "elxBaseComponent.h"

namespace elastix
{
	

	/**
	 * ********************** ConfigurationBase *********************
	 *
	 * The ConfigurationBase class ....
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

