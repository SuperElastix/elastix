#ifndef __elastix_h
#define __elastix_h

#include "elxElastixMain.h"
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include "itkObject.h"
#include "itkDataObject.h"

#include "elxTimer.h"

	/** Declare PrintHelp function.
	 *
	 * \commandlinearg --help: optional argument for elastix and transformix to call the help. \n
	 *		example: <tt>elastix --help</tt> \n
	 *		example: <tt>transformix --help</tt> \n
	 * \commandlinearg --version: optional argument for elastix and transformix to call
	 *		version information. \n
	 *		example: <tt>elastix --version</tt> \n
	 *		example: <tt>transformix --version</tt> \n
	 */
	void PrintHelp(void);

#endif

