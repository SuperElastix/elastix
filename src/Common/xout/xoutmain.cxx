#ifndef __xoutmain_cxx
#define __xoutmain_cxx

#include "xoutmain.h"


namespace xoutlibrary
{


	
	static xoutbase_type * local_xout = 0;

	xoutbase_type & get_xout(void)
	{
		return *local_xout;
	}

	void set_xout(xoutbase_type * arg)
	{
		local_xout = arg;
	}


} // end namespace

#endif // end #ifndef __xoutmain_cxx

