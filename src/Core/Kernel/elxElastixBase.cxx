#ifndef __elxElastixBase_cxx
#define __elxElastixBase_cxx

#include "elxElastixBase.h"

namespace elastix
{

	/**
	 * ********************* Constructor ****************************
	 */

	ElastixBase::ElastixBase()
	{
		/** Initialize.*/
		this->m_Configuration = 0;
		this->m_CDB= 0;
		this->m_DBIndex = 0;

		/** The default output precision of elxout is set to 6 */
		this->m_DefaultOutputPrecision = 6;

	} // end Constructor




	/**
	 * ********************* SetConfiguration ***********************
	 */

	void ElastixBase::SetConfiguration( ConfigurationType * _arg )
	{
		/** If configuration is not set, set it.*/
		if ( this->m_Configuration != _arg )
		{
			this->m_Configuration = _arg;

			Object * thisasobject = dynamic_cast<Object *>(this);
			if ( thisasobject )
			{	
				thisasobject->Modified();
			}
		}

	} // end SetConfiguration


	/**
	 * ********************* SetDBIndex ***********************
	 */

	void ElastixBase::SetDBIndex( DBIndexType _arg )
	{
		/** If m_DBIndex is not set, set it.*/
		if ( this->m_DBIndex != _arg )
		{
			this->m_DBIndex = _arg;

			Object * thisasobject = dynamic_cast<Object *>(this);
			if ( thisasobject )
			{	
				thisasobject->Modified();
			}
		}

	} // end SetDBIndex


	/**
	 * ************************ BeforeAllBase ***************************
	 */
	
	int ElastixBase::BeforeAllBase(void)
	{
		/** Declare the return value and initialize it.*/
		int returndummy = 0;

		/** Set the default precision of floating values in the output */
		this->m_Configuration->ReadParameter(this->m_DefaultOutputPrecision, "DefaultOutputPrecision", 0, true);
		elxout << std::setprecision(this->m_DefaultOutputPrecision);

		/** Print to log file.*/
		elxout << std::setprecision(3);
		elxout << "ELASTIX version: " << __ELASTIX_VERSION << std::endl;
		elxout << std::setprecision( this->GetDefaultOutputPrecision() );

		/** Check Command line options and print them to the logfile.*/
		elxout << "Command line options from ElastixBase:" << std::endl;
		std::string check = "";

		/** Check for appearance of "-f".*/
		check = this->GetConfiguration()->GetCommandLineArgument( "-f" );
		if ( check == "" )
		{
			xl::xout["error"] << "ERROR: No CommandLine option \"-f\" given!" << std::endl;
			returndummy |= -1;
		}
		else
		{
			elxout << "-f\t\t" << check << std::endl;
		}
		
		/** Check for appearance of "-m".*/
		check = "";
		check = this->GetConfiguration()->GetCommandLineArgument( "-m" );
		if ( check == "" )
		{
			xl::xout["error"] << "ERROR: No CommandLine option \"-m\" given!" << std::endl;
			returndummy |= -1;
		}
		else
		{
			elxout << "-m\t\t" << check << std::endl;
		}

		/** Check for appearance of "-out".
		 * This check has allready been performed in elastix.cxx,
		 * so here we do it again.
		 */
		check = "";
		check = this->GetConfiguration()->GetCommandLineArgument( "-out" );
		if ( check == "" )
		{
			xl::xout["error"] << "ERROR: No CommandLine option \"-out\" given!" << std::endl;
			returndummy |= -1;
		}
		else
		{
			/** Make sure that last character of -out equals a '/'. */
			std::string folder( check );
			if ( folder.find_last_of( "/" ) != folder.size() - 1 )
			{
				folder.append( "/" );
				this->GetConfiguration()->SetCommandLineArgument( "-out", folder.c_str() );
			}
			elxout << "-out\t\t" << check << std::endl;
		}

		/** Print all "-p". */
		unsigned int i = 1;
		bool loop = true;
		while ( loop )
		{
			check = "";
			std::ostringstream tempPname("");
			tempPname << "-p(" << i << ")";
			std::string tempPName = tempPname.str();
			check = this->GetConfiguration()->GetCommandLineArgument( tempPName.c_str() );
			if ( check == "" ) loop = false;
			else elxout << "-p\t\t" << check << std::endl;
			i++;
		}

		/** Check for appearance of "-priority", if this is a Windows station. */
		#ifdef _WIN32
			check = "";
			check = this->GetConfiguration()->GetCommandLineArgument( "-priority" );
			if ( check == "" )
			{
				elxout << "-priority\tunspecified, so NORMAL process priority" << std::endl;
			}
			else
			{
				elxout << "-priority\t" << check << std::endl;
			}
		#endif

		/** Return a value.*/
		return returndummy;

	} // end BeforeAllBase


	/**
	 * ************************ BeforeRegistrationBase ******************
	 */
	
	void ElastixBase::BeforeRegistrationBase(void)
	{
		using namespace xl;

		/** Set up the "iteration" writing field.*/
		this->m_IterationInfo.SetOutputs( xout.GetCOutputs() );
		this->m_IterationInfo.SetOutputs( xout.GetXOutputs() );
	  
		xout.AddTargetCell( "iteration", &this->m_IterationInfo );

	} // end BeforeRegistrationBase


	/**
	 * **************** AfterRegistrationBase ***********************
	 */
	
	void ElastixBase::AfterRegistrationBase(void)
	{

		/** Remove the "iteration" writing field */
		xl::xout.RemoveTargetCell("iteration");

	} // end AfterRegistrationBase



} // end namespace elastix


#endif // end #ifndef __elxElastixBase_cxx

