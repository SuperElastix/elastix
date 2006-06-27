
#include "itksysDynamicLoaderGlobal.h"


// This file is actually 3 different implementations.
// 1. HP machines which uses shl_load
// 2. Mac OS X 10.2.x and earlier which uses NSLinkModule
// 3. Windows which uses LoadLibrary
// 4. Most unix systems (including Mac OS X 10.3 and later) which use dlopen
// (default) Each part of the ifdef contains a complete implementation for
// the static methods of DynamicLoader.

// ---------------------------------------------------------------
// 1. Implementation for HPUX  machines
#ifdef __hpux
  #define DYNAMICLOADER_DEFINED 1
#endif //__hpux


// ---------------------------------------------------------------
// 2. Implementation for Mac OS X 10.2.x and earlier
#ifdef __APPLE__
  #if MAC_OS_X_VERSION_MIN_REQUIRED < 1030
    #define DYNAMICLOADER_DEFINED 1
  #endif //MAC_OS_X_VERSION_MIN_REQUIRED < 1030
#endif // __APPLE__

// ---------------------------------------------------------------
// 3. Implementation for Windows win32 code
#ifdef _WIN32
  #define DYNAMICLOADER_DEFINED 1
#endif //_WIN32

// ---------------------------------------------------------------
// 4. Implementation for default UNIX machines.
// if nothing has been defined then use this
#ifndef DYNAMICLOADER_DEFINED
#define DYNAMICLOADER_DEFINED 1
#define __USE_RTLD_GLOBAL
// Setup for most unix machines
#include <dlfcn.h>
#endif // end 4

namespace itksys
{

DynamicLoaderGlobal::LibraryHandle DynamicLoaderGlobal::OpenLibraryGlobal(const char* libname )
{ 
  #ifdef __USE_RTLD_GLOBAL
    return dlopen(libname, RTLD_LAZY | RTLD_GLOBAL );
  #else
    return Superclass::OpenLibrary(libname);
  #endif

} // end OpenLibraryGlobal


} // end namespace itksys

