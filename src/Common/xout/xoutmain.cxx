/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __xoutmain_cxx
#define __xoutmain_cxx

#include "xoutmain.h"


namespace xoutlibrary
{
static xoutbase_type * local_xout = 0;

xoutbase_type & get_xout( void )
{
  return *local_xout;
}

void set_xout( xoutbase_type * arg )
{
  local_xout = arg;
}


} // end namespace

#endif // end #ifndef __xoutmain_cxx

