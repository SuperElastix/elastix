
#ifndef __itksysDynamicLoaderGlobal_h
#define __itksysDynamicLoaderGlobal_h

#include <itksys/DynamicLoader.hxx>

namespace itksys
{

  class DynamicLoaderGlobal : public DynamicLoader
  {
  public:

    /** Typedef's. */
    typedef DynamicLoader Superclass;
    typedef Superclass::LibraryHandle LibraryHandle;
    typedef Superclass::SymbolPointer SymbolPointer;

    /** Constructor and destructor. */
    DynamicLoaderGlobal(){};
    ~DynamicLoaderGlobal(){};

    /** Load a dynamic library into the current process
     * using the RTLD_GLOBAL option in linux like systems and 
     * the superclass's OpenLibrary function otherwise.
     * The returned LibraryHandle can be used to access the symbols in the
     * library. */
    static LibraryHandle OpenLibraryGlobal(const char*);

  }; // end class DynamicLoaderGlobal

} // end namespace itksys

#endif //#ifndef __itksysDynamicLoaderGlobal_h


