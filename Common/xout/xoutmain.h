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
#ifndef xoutmain_h
#define xoutmain_h

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

/** The main xout class */
class xoutmain : public xoutbase
{};

xoutmain &
get_xout();

} // end namespace xoutlibrary

#endif // end #ifndef xoutmain_h
