#ifndef __xoutcell_hxx
#define __xoutcell_hxx

#include "xoutcell.h"

namespace xoutlibrary
{
	using namespace std;

	/**
	 * ************************ Constructor *************************
	 */

	template< class charT, class traits >
		xoutcell<charT, traits>::xoutcell()
	{		
		this->AddTargetCell( "InternalBuffer", &m_InternalBuffer );

	} // end Constructor


	/**
	 * ********************* Destructor *****************************
	 */

	template< class charT, class traits >
		xoutcell<charT, traits>::~xoutcell()
	{
		//nothing

	} // end Destructor


	/**
	 * ******************** WriteBufferedData ***********************
	 * 
	 * The buffered data is sent to the outputs.
	 */

	template< class charT, class traits >
		void xoutcell<charT, traits>::WriteBufferedData(void)
	{
		/** Make sure all data is written to the string */
		m_InternalBuffer << flush;
		
		const std::string & strbuf = m_InternalBuffer.str();

		const char * charbuf = strbuf.c_str();

		/** Send the string to the outputs */
		for ( CStreamMapIteratorType cit = m_COutputs.begin();
			cit != m_COutputs.end(); ++cit )
		{
			*(cit->second) << charbuf << flush;
		}
			
		/** Send the string to the outputs */
		for ( XStreamMapIteratorType xit = m_XOutputs.begin();
			xit != m_XOutputs.end(); ++xit )
		{
			*(xit->second) << charbuf;
			xit->second->WriteBufferedData();
		}

		/** Empty the internal buffer */
		m_InternalBuffer.str( string("") );

	} // end WriteBufferedData



} // end namespace xoutlibrary


#endif // end #ifndef __xoutcell_hxx

