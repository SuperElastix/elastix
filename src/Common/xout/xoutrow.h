#ifndef __xoutrow_h
#define __xoutrow_h

#include "xoutbase.h"
#include "xoutcell.h"
#include <sstream>

namespace xoutlibrary
{
	using namespace std;


	/*
	 *****************  xoutrow *********************
	 *
	 * An abstract base class, which defines the interface 
	 * for using xout.
	 */

	
	template<class charT, class traits = char_traits<charT> >
		class xoutrow : public xoutbase<charT, traits>
	{
	public:
		
		typedef xoutrow														Self;
		typedef xoutbase<charT, traits>						Superclass;

		/** Typedefs of Superclass */
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
		
		/** Extra typedefs */		
		typedef xoutcell<charT, traits> XOutCellType;
		
		/** Constructors */
		xoutrow();			

		/** Destructor */
		virtual ~xoutrow();

		/** Write the buffered cell data in a row to the outputs, separated by tabs */
		virtual void WriteBufferedData(void);

		/** Writes the names of the target cells to the outputs;
		 * This method can also be executed by selecting the "WriteHeaders" target:
		 *		xout["WriteHeaders"] */
		virtual void WriteHeaders(void);

		/** This method adds an xoutcell to the map of Targets.  */
		virtual int AddTargetCell( const char * name );

		virtual int RemoveTargetCell( const char * name );

		/** Method to set all targets at once. The outputs of these targets
		 * are not set automatically, so make sure to do it yourself.
		 */ 
		virtual void SetTargetCells( const XStreamMapType & cellmap );

		/** Add/Remove an output stream (like cout, or an fstream, or an xout-object). 
		 * In addition to the behaviour of the Superclass's methods, these functions 
		 * set the outputs of the TargetCells as well; */
		virtual int AddOutput( const char * name, ostream_type * output );
		virtual int AddOutput( const char * name, Superclass * output );
		virtual int RemoveOutput( const char * name );

		virtual void SetOutputs( const CStreamMapType & outputmap );
		virtual void SetOutputs( const XStreamMapType & outputmap );

	protected:

		/** Returns a target cell.
		 * Extension: if input = "WriteHeaders" it calls
		 * this->WriteHeaders() and returns 'this' */
		virtual Superclass & SelectXCell( const char * name );

		XStreamMapType m_CellMap;
		
	}; // end class xoutrow


} // end namespace xoutlibrary

#include "xoutrow.hxx"

#endif // end #ifndef __xoutrow_h

