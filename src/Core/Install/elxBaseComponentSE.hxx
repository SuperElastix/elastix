#ifndef __elxBaseComponentSE_hxx
#define __elxBaseComponentSE_hxx

#include "elxBaseComponentSE.h"
#include "itkMultiThreader.h"


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
		this->m_Elastix = 0;
		this->m_Configuration = 0;
		this->m_Registration = 0;

	}


	/**
	 * *********************** SetElastix ***************************
	 */

	template <class TElastix>
	void BaseComponentSE<TElastix>::SetElastix( TElastix * _arg )
	{
		/** If this->m_Elastix is not set, then set it. */
		if ( this->m_Elastix != _arg )
		{
			this->m_Elastix = _arg;

			if ( this->m_Elastix.IsNotNull() )
			{
				this->m_Configuration = this->m_Elastix->GetConfiguration();
				this->m_Registration = dynamic_cast<RegistrationPointer>( 
          this->m_Elastix->GetElxRegistrationBase() );
			}

			Object * thisasobject = dynamic_cast<Object *>(this);
			if ( thisasobject )
			{	
				thisasobject->Modified();
			}
		}

    this->SetMaximumNumberOfThreads();
				
	} // end SetElastix


	/**
	 * *********************** SetConfiguration ***************************
	 *
	 * Added for transformix.
	 */

	template <class TElastix>
	void BaseComponentSE<TElastix>::SetConfiguration( ConfigurationType * _arg )
	{
		/** If this->m_Configuration is not set, then set it.*/
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


  /**
   * *********************** SetMaximumNumberOfThreads *************************
   */

  template <class TElastix>
	  void BaseComponentSE<TElastix>::SetMaximumNumberOfThreads(void) const
  {
    /** If wanted, set the priority of this process high or below normal. */
    if ( this->m_Configuration.IsNotNull() )
    {
      std::string maximumNumberOfThreadsString = "";
		  maximumNumberOfThreadsString = this->m_Configuration->GetCommandLineArgument( "-threads" );
  
      if ( maximumNumberOfThreadsString != "" )
      {
        const int maximumNumberOfThreads =
          atoi( maximumNumberOfThreadsString.c_str() );
        itk::MultiThreader::SetGlobalMaximumNumberOfThreads(
          maximumNumberOfThreads );
      }
    } // end if configuration != 0

  } // end SetMaximumNumberOfThreads

} // end namespace elastix


#endif // end #ifndef __elxBaseComponentSE_hxx

