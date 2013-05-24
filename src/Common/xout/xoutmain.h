/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __xoutmain_h
#define __xoutmain_h

/** Necessary includes for using xout. */
#include "xoutbase.h"
#include "xoutsimple.h"
#include "xoutrow.h"
#include "xoutcell.h"

/** Define a namespace alias. */
namespace xl = xoutlibrary;

#define xout get_xout()

/** Typedefs for the most common use of xout */
namespace xoutlibrary
{
  typedef xoutbase<char>    xoutbase_type;
  typedef xoutsimple<char>  xoutsimple_type;
  typedef xoutrow<char>     xoutrow_type;
  typedef xoutcell<char>    xoutcell_type;

  xoutbase_type & get_xout( void );
  void set_xout( xoutbase_type * arg );

} // end namespace xoutlibrary

#endif // end #ifndef __xoutmain_h

