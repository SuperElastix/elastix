#ifndef __elastix_cxx
#define __elastix_cxx


#include "elastix.h"

int main( int argc, char **argv )
{
	
	/** Check if "-help" was asked for.*/
	if ( argc == 1 )
	{
		std::cout << "Use \"elastix -help\" for information about elastix-usage." << std::endl;
		return 0;
	}
	else if ( argc == 2 )
	{
		std::string help( argv[ 1 ] );
		if ( help == "-help" || help == "--help" )
		{
			PrintHelp();
			return 0;
		}
		else
		{
			std::cout << "Use \"elastix -help\" for information about elastix-usage." << std::endl;
			return 0;
		}
	}

	/** Some typedef's.*/
	typedef elx::ElastixMain ElastixType;
	typedef ElastixType::Pointer ElastixPointer;
	typedef std::vector<ElastixPointer> ElastixVectorType;

	typedef ElastixType::ObjectType ObjectType;
	typedef ElastixType::DataObjectType DataObjectType;
	typedef ElastixType::ObjectPointer ObjectPointer;
	typedef ElastixType::DataObjectPointer DataObjectPointer;

	typedef ElastixType::ArgumentMapType ArgumentMapType;
	typedef ArgumentMapType::value_type ArgumentMapEntryType;

	typedef std::pair< std::string, std::string > ArgPairType;
	typedef std::queue< ArgPairType > ParameterFileListType;
	typedef ParameterFileListType::value_type ParameterFileListEntryType;
	
	/** Some declarations.*/
	ElastixVectorType elastices;
	
	ObjectPointer transform = 0;
	DataObjectPointer fixedImage = 0;
	DataObjectPointer movingImage = 0;
	DataObjectPointer fixedInternalImage = 0;
	DataObjectPointer movingInternalImage = 0;
	int returndummy = 0;
	unsigned long NrOfParameterFiles = 0;
	ArgumentMapType argMap;
	ParameterFileListType parameterFileList;
	bool outFolderPresent = false;
	std::string logFileName;

	/** Put command line parameters into parameterFileList.*/
	for ( unsigned int i = 1; i < ( argc - 1 ); i += 2 )
	{
		std::string key( argv[ i ] );
		std::string value( argv[ i + 1 ] );
		
		if ( key == "-p" )
		{
			/** Queue the ParameterFileNames */
			NrOfParameterFiles++;
			parameterFileList.push( 
				ParameterFileListEntryType( key.c_str(), value.c_str() ) );
			/** The different '-p' are stored in the argMap, with
			 * keys p(1), p(2), etc.*/
			std::ostringstream tempPname("");
			tempPname << "-p(" << NrOfParameterFiles << ")";
			std::string tempPName = tempPname.str();
			argMap.insert( ArgumentMapEntryType( tempPName.c_str(), value.c_str() ) );
		}
		else
		{			
			if ( key == "-out" )
			{
				outFolderPresent = true;

				/** Make sure that last character of the outputfolder equals a '/'.*/
				if ( value.find_last_of( "/" ) != value.size() - 1 )
				{
					value.append( "/" );
				} 
			} // end if key == "-out"
			
			/** Attempt to save the arguments in the ArgumentMap */
			if ( argMap.count( key.c_str() ) == 0 )
			{	
				argMap.insert( ArgumentMapEntryType( key.c_str(), value.c_str() ) );
			}
			else
			{
				/** duplicate argument */
				std::cerr << "WARNING!" << std::endl;
				std::cerr << "Argument "<< key.c_str() << "is only required once." << std::endl;
				std::cerr << "Arguments " << key.c_str() << " " << value.c_str() << "are ignored" << std::endl;
			}

		} // end else (so, if key does not equal "-p")

	} // end for loop
	
	/** The argv0 argument, required for finding the component.dll/so's. */
	argMap.insert( ArgumentMapEntryType( "-argv0", argv[0] )  );


	/** Check if at least once the option "-p" is given.*/
	if ( NrOfParameterFiles == 0 )
	{
		std::cerr << "ERROR: No CommandLine option \"-p\" given!" << std::endl;
		returndummy |= -1;
	}

	/** Check if the -out option is given. */
	if ( outFolderPresent )
	{
		/** Setup xout.*/
    logFileName = argMap[ "-out" ] + "elastix.log";
		int returndummy2 = elx::xoutSetup( logFileName.c_str() );
		if ( returndummy2 )
		{
			std::cerr << "ERROR while setting up xout." << std::endl;
		}
		returndummy |= returndummy2;
	}
	else
	{
		returndummy = -2;
		std::cerr << "ERROR: No CommandLine option \"-out\" given!" << std::endl;
	}

	/** Stop if some fatal errors occured */
	if ( returndummy )
	{
		return returndummy;
	}

	elxout << std::endl;

	/** Declare a timer, start it and print the start time.*/
	tmr::Timer::Pointer totaltimer = tmr::Timer::New();
	totaltimer->StartTimer();
	elxout << "Elastix is started at " << totaltimer->PrintStartTime() << ".\n" << std::endl;

	/**
	 * ********************* START REGISTRATION *********************
	 *
	 * Do the (possibly multiple) registration(s).
	 */

	for ( unsigned int i = 0; i < NrOfParameterFiles; i++ )
	{
		/** Create another instance of ElastixMain.*/
		elastices.push_back( elx::ElastixMain::New() );
		
		/** Set stuff we get from a former registration.*/
		elastices[ i ]->SetInitialTransform( transform );
		elastices[ i ]->SetFixedImage( fixedImage );
		elastices[ i ]->SetMovingImage( movingImage );
		elastices[ i ]->SetFixedInternalImage( fixedInternalImage );
		elastices[ i ]->SetMovingInternalImage( movingInternalImage );

		/** Set the current elastix-level.*/
		elastices[ i ]->SetElastixLevel( i );

		/** Delete the previous ParameterFileName.*/
		if ( argMap.count( "-p" ) )
		{
			argMap.erase( "-p" );
		}

		/** Read the first parameterFileName in the queue.*/
		ArgPairType argPair = parameterFileList.front();
		parameterFileList.pop();

		/** Put it in the ArgumentMap.*/
		argMap.insert( ArgumentMapEntryType( argPair.first, argPair.second ) );

		/** Print a start message.*/
		elxout << "-------------------------------------------------------------------------" << "\n" << std::endl;
		elxout << "Running Elastix with parameter file " << i
			<< ": \"" << argMap[ "-p" ] << "\".\n" << std::endl;

		/** Declare a timer, start it and print the start time.*/
		tmr::Timer::Pointer timer = tmr::Timer::New();
		timer->StartTimer();
		elxout << "Current time: " << timer->PrintStartTime() << "." << std::endl;

		/** Start registration.*/
		returndummy = elastices[ i ]->Run( argMap );
		
		/** Check for errors.*/
		if ( returndummy != 0 )
		{
			xl::xout["error"] << "Errors occured!" << std::endl;
			return returndummy;
		}
		
		/** Get the transform, the fixedImage and the movingImage
		 * in order to put it in the (possibly) next registration.
		 */
		transform						= elastices[ i ]->GetTransform();	
		fixedImage					= elastices[ i ]->GetFixedImage();
		movingImage					= elastices[ i ]->GetMovingImage();
		fixedInternalImage	= elastices[ i ]->GetFixedInternalImage();
		movingInternalImage = elastices[ i ]->GetMovingInternalImage();
		
		/** Print a finish message.*/
		elxout << "Running Elastix with parameter file " << i
			<< ": \"" << argMap[ "-p" ] << "\", has finished.\n" << std::endl;

		/** Stop timer and print it.*/
		timer->StopTimer();
		elxout << "\nCurrent time: " << timer->PrintStopTime() << "." << std::endl;
		elxout << "Time used for running Elastix with this parameter file: "
			<< timer->PrintElapsedTimeDHMS() << ".\n" << std::endl;

	} // end loop over registrations

	elxout << "-------------------------------------------------------------------------" << "\n" << std::endl;	

	/** Stop totaltimer and print it.*/
	totaltimer->StopTimer();
	elxout << "Total time elapsed: " << totaltimer->PrintElapsedTimeDHMS() << ".\n" << std::endl;


	/** 
	 * Make sure all the components that are defined in a Module (.DLL/.so) 
	 * are deleted before the modules are closed.
	 */

	for ( unsigned int i = 0; i < NrOfParameterFiles; i++ )
	{
		elastices[i] = 0;
	}	

	transform=0;
	fixedImage=0;
	movingImage=0;
	fixedInternalImage=0;
	movingInternalImage=0;
	
	/** Close the modules */
	ElastixType::UnloadComponents();
	
	/** Exit and return the error code.*/
	return returndummy;

} // end main




/**
 * *********************** PrintHelp ****************************
 */

void PrintHelp(void)
{
	std::cout << "*********** elastix help: ***********\n\n";

	/** What is elastix?*/
	std::cout << "Elastix registers a moving image to a fixed image." << std::endl;
	std::cout << "The registration-process is specified in the parameter file."
		<< std::endl << std::endl;

	/** Mandatory argments.*/
	std::cout << "Call elastix from the command line with mandatory arguments:" << std::endl;
	std::cout << "-f\t\tfixed image (.mhd)" << std::endl;
	std::cout << "-m\t\tmoving image (.mhd)" << std::endl;
	std::cout << "-out\t\toutput directory" << std::endl;
	std::cout << "-p\t\tparameter file, elastix handles 1 or more \"-p\"" << std::endl << std::endl;

	/** Optional arguments.*/
	std::cout << "Optional extra commands:" << std::endl;
	std::cout << "-fmask\t\tmask for fixed image (.mhd)" << std::endl;
	std::cout << "-mmask\t\tmask for moving image (.mhd)" << std::endl;
	std::cout << "-t0\t\tparameter file for initial transform" << std::endl;
	std::cout << "-priority\tset the process priority to high (Windows only)"
		<< std::endl << std::endl;
	
	/** The parameter file.*/
	std::cout << "The parameter-file must contain all the information necessary for elastix to run properly. That includes which metric to use, which optimizer, which transform, etc." << std::endl;
	std::cout << "It must also contain information specific for the metric, optimizer, transform,..." << std::endl;
	std::cout << "For a usable parameter-file, ask us." << std::endl << std::endl;

	std::cout << "Need further help? Ask Marius and/or Stefan. :-)" << std::endl;

} // end PrintHelp


#endif // end #ifndef __elastix_cxx

