
#ifndef __elxComponentLoader_h
#define __elxComponentLoader_h

#include "elxComponentDatabase.h"
//#include "elxDynamicLoader.h"
#include <itksys/DynamicLoader.hxx>
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

		/** Standard.*/
		typedef ComponentLoader									Self;
		typedef itk::Object											Superclass;
		typedef itk::SmartPointer<Self>					Pointer;
		typedef itk::SmartPointer<const Self>		ConstPointer;
	
		itkNewMacro(Self);
		itkTypeMacro(ComponentLoader, Object);
		
		typedef ComponentDatabase								ComponentDatabaseType;
		typedef ComponentDatabaseType::Pointer	ComponentDatabasePointer;

		//typedef DynamicLoader							LibLoaderType;
    typedef itksys::DynamicLoader						LibLoaderType;
    typedef LibLoaderType::SymbolPointer    LibSymbolPointer;
		//typedef LibLoaderType::Pointer					LibLoaderPointer;
		//typedef LibHandle									LibHandleType;
    typedef LibLoaderType::LibraryHandle			LibHandleType;
		typedef int (*InstallFunctionPointer)(ComponentDatabaseType *, xl::xoutbase_type *);

		typedef std::stack<LibHandleType>			  LibHandleContainerType;
		
		itkSetObjectMacro( ComponentDatabase, ComponentDatabaseType);
		itkGetObjectMacro( ComponentDatabase, ComponentDatabaseType);

		virtual int LoadComponents(const char * argv0);
		virtual void UnloadComponents(void);

	protected:
		
		ComponentLoader();
		virtual ~ComponentLoader();

		ComponentDatabasePointer m_ComponentDatabase;

		//LibLoaderType::Pointer m_LibLoader;
    LibLoaderType   m_LibLoader;
		LibHandleContainerType m_LibHandleContainer;

		bool m_ImageTypeSupportInstalled;
		virtual int InstallSupportedImageTypes(void);

	private:
		
		ComponentLoader( const Self& );	// purposely not implemented
		void operator=( const Self& );	// purposely not implemented

			
	};

} //end namespace elastix


#endif // #ifndef __elxComponentLoader_h

