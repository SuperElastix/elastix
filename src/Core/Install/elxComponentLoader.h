
#ifndef __elxComponentLoader_h
#define __elxComponentLoader_h

#include "elxComponentDatabase.h"
#include "itksysDynamicLoaderGlobal.h"
#include "xoutmain.h"
#include <stack>

namespace elastix
{

	/**
	* \class ComponentLoader
	*
	* \brief Loads in all the desired libraries, which are elastix
	* components.
	*
	* This file defines the class elx::ComponentLoader. This class
	* loads .DLL's and stores pointers to the ::New() functions of
	* each component in the elx::ComponentDatabase.
	*
	* Each new component (a new metric for example should "make itself
	* known" by calling the elxInstallMacro, which is defined in
	* elxInstallFunctions.h .
	*/

	class ComponentLoader : public itk::Object
	{
	public:

		/** Standard ITK typedef's. */
		typedef ComponentLoader									Self;
		typedef itk::Object											Superclass;
		typedef itk::SmartPointer<Self>					Pointer;
		typedef itk::SmartPointer<const Self>		ConstPointer;
	
    /** Standard ITK stuff. */
		itkNewMacro(Self);
		itkTypeMacro(ComponentLoader, Object);
		
    /** Typedef's. */
		typedef ComponentDatabase								ComponentDatabaseType;
		typedef ComponentDatabaseType::Pointer	ComponentDatabasePointer;

    typedef itksys::DynamicLoaderGlobal			LibLoaderType;
    typedef LibLoaderType::SymbolPointer    LibSymbolPointer;
    typedef LibLoaderType::LibraryHandle		LibHandleType;
		typedef int (*InstallFunctionPointer)(ComponentDatabaseType *, xl::xoutbase_type *);
		typedef std::stack<LibHandleType>			  LibHandleContainerType;

    /** Set and get the ComponentDatabase. */
		itkSetObjectMacro( ComponentDatabase, ComponentDatabaseType);
		itkGetObjectMacro( ComponentDatabase, ComponentDatabaseType);

    /** Functions to (un)load libraries. */
		virtual int LoadComponents(const char * argv0);
		virtual void UnloadComponents(void);

	protected:
		/** Standard constructor and destructor. */
		ComponentLoader();
		virtual ~ComponentLoader();

		ComponentDatabasePointer  m_ComponentDatabase;
    LibLoaderType             m_LibLoader;
		LibHandleContainerType    m_LibHandleContainer;

		bool          m_ImageTypeSupportInstalled;
		virtual int   InstallSupportedImageTypes(void);

	private:
		/** Standard private (copy)constructor. */
		ComponentLoader( const Self& );	// purposely not implemented
		void operator=( const Self& );	// purposely not implemented
			
	}; // end class ComponentLoader

} //end namespace elastix


#endif // #ifndef __elxComponentLoader_h

