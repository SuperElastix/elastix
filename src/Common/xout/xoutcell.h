#ifndef __xoutcell_h
#define __xoutcell_h

#include "xoutbase.h"
#include <sstream>

namespace xoutlibrary
{
	using namespace std;


	/*
	 *****************  xoutcell *********************
	 *
	 * An abstract base class, which defines the interface 
	 * for using xout.
	 */

	
	template<class charT, class traits = char_traits<charT> >
		class xoutcell : public xoutbase<charT, traits>
	{
	public:
		
		/** Typdef's.*/
		typedef xoutcell													Self;
		typedef xoutbase<charT, traits>						Superclass;

		typedef typename Superclass::traits_type		traits_type;
		typedef typename Superclass::char_type			char_type;
		typedef typename Superclass::int_type				int_type;
		typedef typename Superclass::pos_type				pos_type;
		typedef typename Superclass::off_type				off_type;
		typedef typename Superclass::ostream_type		ostream_type;
		typedef typename Superclass::ios_type				ios_type;
		
		typedef typename Superclass::CStreamMapType					CStreamMapType;
		typedef typename Superclass::XStreamMapType					XStreamMapType;
		typedef typename Superclass::CStreamMapIteratorType CStreamMapIteratorType;
		typedef typename Superclass::XStreamMapIteratorType XStreamMapIteratorType;
		typedef typename Superclass::CStreamMapEntryType		CStreamMapEntryType;
		typedef typename Superclass::XStreamMapEntryType		XStreamMapEntryType;
		
		typedef std::basic_ostringstream<charT, traits>			InternalBufferType;

		/** Constructors */
		xoutcell();

		/** Destructor */
		virtual ~xoutcell();

		/** Write the buffered cell data to the outputs */
		virtual void WriteBufferedData(void);

	protected:

		InternalBufferType m_InternalBuffer;
		

	}; // end class xoutcell


} // end namespace xoutlibrary


#include "xoutcell.hxx"


#endif // end #ifndef __xoutcell_h

