#ifndef __elxBaseComponentSE_hxx
#define __elxBaseComponentSE_hxx

#include "elxBaseComponentSE.h"


namespace elastix
{
  using namespace itk;

	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
	BaseComponentSE<TElastix>::BaseComponentSE()
	{
		/** Initialize.*/
		m_Elastix = 0;
		m_Configuration = 0;
		m_Registration = 0;

	}


	/**
	 * *********************** SetElastix ***************************
	 */

	template <class TElastix>
	void BaseComponentSE<TElastix>::SetElastix( TElastix * _arg )
	{
		/** If m_Elastix is not set, then set it.*/
		if ( this->m_Elastix != _arg )
		{
			this->m_Elastix = _arg;

			m_Configuration = m_Elastix->GetConfiguration();
			m_Registration = dynamic_cast<RegistrationPointer>( m_Elastix->GetRegistration() );

			Object * thisasobject = dynamic_cast<Object *>(this);
			if ( thisasobject )
			{	
				thisasobject->Modified();
			}
		}
				
	} // end SetElastix


	/**
	 * *********************** SetConfiguration ***************************
	 *
	 * Added for transformix.
	 */

	template <class TElastix>
	void BaseComponentSE<TElastix>::SetConfiguration( ConfigurationType * _arg )
	{
		/** If m_Elastix is not set, then set it.*/
		if ( this->m_Configuration != _arg )
		{
			this->m_Configuration = _arg;

			Object * thisasobject = dynamic_cast<Object *>(this);
			if ( thisasobject )
			{	
				thisasobject->Modified();
			}
		}
				
	} // end SetConfiguration



} // end namespace elastix


#endif // end #ifndef __elxBaseComponentSE_hxx

