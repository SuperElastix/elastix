#ifndef __elxComponentLoader_cxx
#define __elxComponentLoader_cxx

#include "elxComponentLoader.h"
#include "itkDirectory.h"
#include "elxSupportedImageTypes.h"
#include "elxInstallFunctions.h"
#include "elxMacro.h"
#include <itksys/SystemTools.hxx>
#include <iostream>
#include <string>

namespace elastix
{
	using namespace xl;


	/**
	 * Definition of class template, needed in InstallSupportedImageTypes()
	 */

	/** Define a class<N> with a method DO(...) that calls class<N+1>::DO(...) */
	template < ComponentDatabase::IndexType VIndex> 
	class _installsupportedimagesrecursively 
	{ 
	public: 
		/** ElastixTypedef is defined in elxSupportedImageTypes.h, by means of the
			* the elxSupportedImageTypesMacro */
		typedef ElastixTypedef<VIndex> ET; 
		typedef typename ET::ElastixType ElastixType; 
		typedef ComponentDatabase::ComponentDescriptionType ComponentDescriptionType; 

		static int DO(const ComponentDescriptionType & name, ComponentDatabase * cdb) 
		{ 
			int dummy1 = InstallFunctions< ElastixType >::InstallComponent(name, VIndex, cdb); 
			int dummy2 = cdb->SetIndex( 
				ET::fPixelTypeAsString(),
				ET::fDim(),
				ET::mPixelTypeAsString(),
				ET::mDim(),
				VIndex  ); 
			if ( ElastixTypedef<VIndex+1>::Defined() ) 
				{	
					return _installsupportedimagesrecursively<VIndex+1>::DO(name, cdb); 
				} 
			return (dummy1 + dummy2);  
		} 
	}; // end template class

	/** To prevent an infinite loop, DO() does nothing in class<lastImageTypeCombination> */
	template <> 
		class _installsupportedimagesrecursively < NrOfSupportedImageTypes+1 > 
	{ 
	public: 
		typedef ComponentDatabase::ComponentDescriptionType ComponentDescriptionType; 
		static int DO(const ComponentDescriptionType & name, ComponentDatabase * cdb) 
			{ return 0; } 
	}; // end template class specialization




	/**
	 * ****************** Constructor ********************************
	 */
		
	ComponentLoader::ComponentLoader()
	{
		m_LibLoader = LibLoaderType::New();
		m_ImageTypeSupportInstalled = false;
	}

	
	/**
	 * ****************** Destructor *********************************
	 */

	ComponentLoader::~ComponentLoader()
	{
		this->UnloadComponents();
	}


	/**
	 * *************** InstallSupportedImageTypes ********************
	 */
	int ComponentLoader::InstallSupportedImageTypes(void)
	{
		/**
		* Method: A recursive template was defined at the top of this file, that
		* installs support for all combinations of ImageTypes defined in 
		* elxSupportedImageTypes.h
		*
    * Result: The VIndices are stored in the elx::ComponentDatabase::IndexMap.
		* The New() functions of ElastixTemplate<> in the
		* elx::ComponentDatabase::CreatorMap, with key "Elastix".
		*/

		/** Call class<1>::DO(...) */
		int _InstallDummy_SupportedImageTypes = 
			_installsupportedimagesrecursively<1>::DO( "Elastix", m_ComponentDatabase );

		if ( _InstallDummy_SupportedImageTypes==0 ) 
		{
			m_ImageTypeSupportInstalled = true;
		}

		return _InstallDummy_SupportedImageTypes;

	} // end function InstallSupportedImageTypes

	/**
	 * ****************** LoadComponents *****************************
	 */

	int ComponentLoader::LoadComponents(const char * argv0)
	{
		int installReturnCode = 0;

		if (!m_ImageTypeSupportInstalled)
		{
			installReturnCode =	this->InstallSupportedImageTypes();
			if (installReturnCode != 0)
			{
				xout["error"] 
					<< "ERROR: ImageTypeSupport installation failed. "
					<< std::endl;
				return installReturnCode;
			}
		} //end if !ImageTypeSupportInstalled

		/** Get the path to executable (which is assumed to be in the same dir
		 * as the libs
		 */
		std::string pathToExe("");
		std::string fullPathToExe("");
		std::string exeNotFound("");
		bool exeFound = itksys::SystemTools::FindProgramPath(argv0, fullPathToExe, exeNotFound, 0, 0);
		if (!exeFound)
		{
			xout["error"]
				<< "ERROR: Path to components could not be found\n" 
				<< exeNotFound << std::endl;
			return 1;
		}
		pathToExe = itksys::SystemTools::GetProgramPath(fullPathToExe.c_str());


		itk::Directory::Pointer componentDir = itk::Directory::New();
		bool validDir = componentDir->Load( pathToExe.c_str() );
		if (!validDir)
		{
			xout["error"]
				<< "ERROR: The assumed path to the components can not be opened: "
				<< pathToExe
				<< std::endl;
			return 1;
		}

		unsigned int nrOfFiles =
			static_cast<unsigned int>( componentDir->GetNumberOfFiles() );
		std::string libextension = itksys::SystemTools::LowerCase(
					m_LibLoader->LibExtension() );
		std::string libprefix = itksys::SystemTools::LowerCase(
					m_LibLoader->LibPrefix() );
		std::string currentLibName("");
		LibHandleType currentLib;
		void* addressOfInstallComponentFunction = 0;
		InstallFunctionPointer installer;
		bool fileIsLib;
		std::string fileName("");
		std::string fullFileName("");
		bool fileIsDir;
		std::string extension("");
		std::string elxCoreLibName(libprefix + "elxCore" + libextension);
		std::string elxCommonLibName(libprefix + "elxCommon" + libextension);

		for (unsigned int i = 0; i< nrOfFiles; i++)
		{
			fileIsLib = false;
			
			fileName = componentDir->GetFile(i);
			fullFileName = pathToExe + std::string("/") + fileName;
			
			fileIsDir =
				itksys::SystemTools::FileIsDirectory( fullFileName.c_str() );

			if (!fileIsDir)
			{
				extension = itksys::SystemTools::LowerCase(
					itksys::SystemTools::GetFilenameLastExtension( fullFileName.c_str() ) );
				
				if ( extension == libextension )
				{
					/** file may be lib, but check the prefix, to be sure: */
					// \todo : some smart string stuff, with string::compare(prefix::size() ofzo)

					if ( (fileName==elxCoreLibName) || (fileName==elxCommonLibName) )
					{
					  /** Not a component */
						fileIsLib = false;
					}
					else
					{
						fileIsLib = true;
						currentLibName = fullFileName;
					}
				}

			}
			
			/** Load the lib, check if it's a elxComponent, and install it. */
      if (fileIsLib)
			{			
		
				/** Open the lib */
				elxout
					<< "Loading library: "
					<< currentLibName
					<< std::endl;
				m_LibHandleContainer.push( m_LibLoader->OpenLibrary( currentLibName.c_str() ) );
				//currentLib = m_LibLoader->OpenLibrary( currentLibName.c_str() );
				currentLib = m_LibHandleContainer.top();
				/** Store the handle to the lib, because we need it for closing the lib later. */
				//LibHandleContainer.push(currentLib);
				
				/** Look for the InstallComponent function */
				addressOfInstallComponentFunction	=
					m_LibLoader->GetSymbolAddress(currentLib, "InstallComponent");
				
				/** If it exists, execute it */
				if (addressOfInstallComponentFunction)
				{
					/** Cast the void* to a function pointer */
					//this does not work in Linux:
					//InstallFunctionPointer fp = static_cast<InstallFunctionPointer>(adres);
					//but this does (with the C-style cast):
					installer =	(InstallFunctionPointer)(addressOfInstallComponentFunction);
					
					/** Execute it */
					/** \todo : How to check if the conversion went alright? */
					elxout
						<< "Installing component: "
						<< currentLibName
						<< std::endl;
					installReturnCode = installer( m_ComponentDatabase, & xout );

					if ( installReturnCode )
					{
						xout["warning"] 
							<< "WARNING: Installing the following component failed: "
							<< currentLibName
							<< std::endl;
					}

				} // end if (addressOfInstallComponentFunction!=0)
				else
				{
					elxout
						<< "No InstallComponent function found in: "
						<< currentLibName
						<< std::endl;
	
				}

			} //end if fileIsLib

		} // end for <loop over file-list>
	
		elxout << std::endl;

		return 0;


	} //end function LoadComponents

	
	
	/**
	 * ****************** UnloadComponents****************************
	 */

	void ComponentLoader::UnloadComponents()
	{
		/** 
		 * Close all libraries that were opened.
		 */

		LibHandleType currentLib;

		unsigned int nrOfLibs =
			static_cast<unsigned int>( m_LibHandleContainer.size() );

		for (unsigned int i = 0; i < nrOfLibs; i++)
		{
			
			currentLib = m_LibHandleContainer.top();
			m_LibLoader->CloseLibrary(currentLib);
			m_LibHandleContainer.pop();

		}
  
		//Not necessary I think:
		//m_ComponentDatabase = 0;

	}


} //end namespace elastix


#endif //#ifndef __elxComponentLoader_cxx


