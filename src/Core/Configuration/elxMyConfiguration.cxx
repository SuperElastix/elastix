#ifndef	__elxMyConfiguration_CXX__
#define __elxMyConfiguration_CXX__

#include "elxMyConfiguration.h"

namespace elastix
{
	
	using namespace itk;

	/**
	 * ********************* Constructor ****************************
	 */

	MyConfiguration::MyConfiguration()
	{
		/***/
		m_ParameterFileName = "";
		m_Initialized = false;

	} // end Constructor
	

	/**
	 * ************************ BeforeAll ***************************
	 *
	 * This function prints the ParameterFile to the log-file.
	 */
	
	int MyConfiguration::BeforeAll(void)
	{
		/** Open the ParameterFile.*/
		std::ifstream parfile( GetParameterFileName() );
		if ( parfile.is_open() )
		{
			/** Seperate clearly in log-file.*/
			xl::xout["logonly"] << std::endl << "=============== start of ParameterFile: "
				<< GetParameterFileName() << " ===============" << std::endl;
			/** Read and write.*/
			char inout;
			while ( !parfile.eof() )
			{				
				parfile.get( inout );
				xl::xout["logonly"] << inout;
			}

			/** Seperate clearly in log-file.*/
			xl::xout["logonly"] << std::endl << "=============== end of ParameterFile: "
				<< GetParameterFileName() << " ===============" << std::endl << std::endl;
		}
		else
		{
			xl::xout["warning"] << "WARNING: the file \"" << GetParameterFileName() <<
				"\" could not be opened!" << std::endl;
		}		

		/** Return a value.*/
		return 0;

	} // end BeforeAll

	/**
	 * ********************** Initialize ****************************
	 */

	int MyConfiguration::Initialize( ArgumentMapType & _arg )
	{
		m_ArgumentMap = _arg;

		/** This function can either be called by elastix or transformix.
		 * If called by elastix the command line argument "-p" has to be
		 * specified. If called by transformix the command line argument
		 * "-tp" has to be specified.
		 * NOTE: this implies that one can not use "-tp" for elastix and
		 * "-p" for transformix.
		 */

		std::string p = this->GetCommandLineArgument( "-p" );
		std::string tp = this->GetCommandLineArgument( "-tp" );

		if ( p != "" && tp == "" )
		{
			/** elastix called Initialize().*/
			this->SetParameterFileName( p.c_str() );
		}
		else if ( p == "" && tp != "" )
		{
			/** transformix called Initialize().*/
			this->SetParameterFileName( tp.c_str() );
		}
		else if ( p == "" && tp == "" )
		{
			xl::xout["error"] << "ERROR: No (Transform-)Parameter file has been entered" << std::endl;
			xl::xout["error"] << "for elastix: command line option \"-p\"" << std::endl;
			xl::xout["error"] << "for transformix: command line option \"-tp\"" << std::endl;
			return 1;
		}
		else
		{
			/** Both "p" and "tp" are used, which is prohibited.*/
			xl::xout["error"] << "ERROR: Both \"-p\" and \"-tp\" are used, which is prohibited."
				<< std::endl;
			return 1;
		}

		/** Open the ParameterFile.*/
		m_ParameterFile.Initialize( m_ParameterFileName.c_str() );

		m_Initialized = true;

		/** Return a value.*/
		return 0;

	} // end Initialize


	/**
	 * ********************** Initialized ***************************
	 *
	 * Check if Initialized.
	 */

	bool MyConfiguration::Initialized(void)
	{
		return m_Initialized;

	} // end Initialized


	/**
	 * ****************** GetCommandLineArgument ********************
	 */

	const char * MyConfiguration::GetCommandLineArgument( const char * key ) const
	{
		/** .*/
		if ( m_ArgumentMap.count( key ) == 0 )
		{
			return "";
		}
		else
		{
			return m_ArgumentMap[ key ].c_str();
		}

	} // end GetCommandLineArgument


	/**
	 * ****************** SetCommandLineArgument ********************
	 */

	void MyConfiguration::SetCommandLineArgument( const char * key, const char * value )
	{
		/** Remove all (!) entries with key 'key' and
		 * insert one entry ( key, value ).
		 */
		m_ArgumentMap.erase( key );
		m_ArgumentMap.insert( EntryType( key, value ) );

	} // end SetCommandLineArgument



} // end namespace elastix

#endif // end #ifndef	__elxMyConfiguration_CXX__

