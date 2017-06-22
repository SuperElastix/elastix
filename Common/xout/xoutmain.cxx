/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __xoutmain_cxx
#define __xoutmain_cxx

#include "xoutmain.h"

namespace xoutlibrary
{
static xoutbase_type * local_xout = 0;

xoutbase_type &
get_xout( void )
{
  return *local_xout;
}


void
set_xout( xoutbase_type * arg )
{
  local_xout = arg;
}


} // end namespace

#endif // end #ifndef __xoutmain_cxx
