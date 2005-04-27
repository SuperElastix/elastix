#ifndef __elxElastixMain_cxx
#define __elxElastixMain_cxx


/** If running on a Windows-system, include "windows.h".*/
#ifdef _WIN32
#include <windows.h>
#endif

#include "elxElastixMain.h"
#include "elxMacro.h"



namespace elastix
{
	using namespace xl;

	/**
	 * ******************* Global variables *************************
	 * 
	 * Some global variables (not part of the ElastixMain class, used
	 * by xoutSetup.	 
	 */
	
	/** \todo move to ElastixMain class, as static vars? */

	/** xout TargetCells. */
	xoutbase_type		g_xout;
	xoutsimple_type g_WarningXout;
	xoutsimple_type g_ErrorXout;
	xoutsimple_type g_StandardXout;
	xoutsimple_type g_CoutOnlyXout;
	xoutsimple_type g_LogOnlyXout;
	std::ofstream		g_LogFileStream;

	/**
	 * ********************* xoutSetup ******************************
	 * 
	 * NB: this function is a global function, not part of the ElastixMain
	 * class!!
	 */

	int xoutSetup(const char * logfilename)
	{
		/** the namespace of xout: */
		using namespace xl;

		int returndummy = 0;
		
		set_xout(&g_xout);

		/** Open the logfile for writing */
		g_LogFileStream.open( logfilename );
		if ( !g_LogFileStream.is_open() )
		{
			std::cout << "ERROR: LogFile cannot be opened!" << std::endl;
			return 1;
		}

		/** Set std::cout and the logfile as outputs of xout. */
		returndummy |= xout.AddOutput("log", &g_LogFileStream);
		returndummy |= xout.AddOutput("cout", &std::cout);

		/** Set outputs of LogOnly and CoutOnly.*/
		returndummy |= g_LogOnlyXout.AddOutput( "log", &g_LogFileStream );
		returndummy |= g_CoutOnlyXout.AddOutput( "cout", &std::cout );

		/** Copy the outputs to the warning-, error- and standard-xouts. */
		g_WarningXout.SetOutputs( xout.GetCOutputs() );
		g_ErrorXout.SetOutputs( xout.GetCOutputs() );
		g_StandardXout.SetOutputs( xout.GetCOutputs() );

		g_WarningXout.SetOutputs( xout.GetXOutputs() );
		g_ErrorXout.SetOutputs( xout.GetXOutputs() );
		g_StandardXout.SetOutputs( xout.GetXOutputs() );

		/** Link the warning-, error- and standard-xouts to xout. */	
		returndummy |= xout.AddTargetCell( "warning", &g_WarningXout );
		returndummy |= xout.AddTargetCell( "error", &g_ErrorXout );
		returndummy |= xout.AddTargetCell( "standard", &g_StandardXout );
		returndummy |= xout.AddTargetCell( "logonly", &g_LogOnlyXout );
		returndummy |= xout.AddTargetCell( "coutonly", &g_CoutOnlyXout );

		/** Format the output */
		xout["standard"] << std::fixed;
		xout["standard"] << std::showpoint;
		
		/** Return a value.*/
		return returndummy;

	} // end xoutSetup


	/**
	 * ********************* Constructor ****************************
	 */

	ElastixMain::ElastixMain()
	{
		/** Initialize the components.*/
		this->m_Configuration = ConfigurationType::New();

		this->m_Elastix = 0;
		this->m_elx_Elastix = 0;

		this->m_FixedImagePixelType = "";
		this->m_FixedImageDimension = 0;

		this->m_MovingImagePixelType = "";
		this->m_MovingImageDimension = 0;

		this->m_DBIndex = 0;

		this->m_FixedImage = 0;
		this->m_MovingImage = 0;
		this->m_FixedInternalImage = 0;
		this->m_MovingInternalImage = 0;

		this->m_Registration = 0;
		this->m_FixedImagePyramid = 0;
		this->m_MovingImagePyramid = 0;
		this->m_Interpolator = 0;
		this->m_Metric = 0;
		this->m_Optimizer = 0;
		this->m_Resampler = 0;
		this->m_ResampleInterpolator = 0;
		this->m_Transform = 0;

		this->m_InitialTransform = 0;

	} // end Constructor


	/**
	 * ****************** Initialization of static members *********
	 */
	ElastixMain::ComponentDatabasePointer ElastixMain::s_CDB = 0;
	ElastixMain::ComponentLoaderPointer ElastixMain::s_ComponentLoader = 0;

	/**
	 * ********************** Destructor ****************************
	 */

	ElastixMain::~ElastixMain()
	{
	 //nothing
	} // end Destructor


	/**
	 * *************** EnterCommandLineParameters *******************
	 */

	void ElastixMain
		::EnterCommandLineArguments( ArgumentMapType & argmap )
	{
		/** Initialize the configuration object with the 
		 * command line parameters entered by the user.
		 */		
		int dummy = this->m_Configuration->Initialize( argmap );
		if (dummy)
		{
			xout["error"] << "ERROR: Something went wrong during initialisation of the configuration object." << std::endl;
		}

	} // end EnterCommandLineParameters


	/**
	 * **************************** Run *****************************
	 *
	 * Assuming EnterCommandLineParameters has already been invoked.
	 * or that m_Configuration is initialised in another way.
	 */

	int ElastixMain::Run(void)
	{
		/** If wanted, set the priority of this process high.*/
		std::string processPriority = "";
		processPriority = this->m_Configuration->GetCommandLineArgument( "-priority" );
		if ( processPriority == "high" )
		{
			#ifdef _WIN32
			SetPriorityClass( GetCurrentProcess(), HIGH_PRIORITY_CLASS );
			#endif
		}

		/** Initialize database.*/		
		int ErrorCode = this->InitDBIndex();
		if ( ErrorCode != 0 )
		{
			return ErrorCode;
		}

		/** Get the different components.*/
		ComponentDescriptionType RegistrationName = "MultiResolutionRegistration";
		this->m_Configuration->ReadParameter( RegistrationName, "Registration", 0 );
		
		ComponentDescriptionType FixedImagePyramidName = "FixedRecursiveImagePyramid";
		this->m_Configuration->ReadParameter( FixedImagePyramidName, "FixedImagePyramid", 0 );

		ComponentDescriptionType MovingImagePyramidName = "MovingRecursiveImagePyramid";
		this->m_Configuration->ReadParameter( MovingImagePyramidName, "MovingImagePyramid", 0 );
		
		ComponentDescriptionType InterpolatorName = "BSplineInterpolator";
		this->m_Configuration->ReadParameter( InterpolatorName, "Interpolator", 0 );

		ComponentDescriptionType MetricName = "MattesMutualInformation";
		this->m_Configuration->ReadParameter( MetricName, "Metric", 0 );

		ComponentDescriptionType OptimizerName = "RegularStepGradientDescent";
		this->m_Configuration->ReadParameter( OptimizerName, "Optimizer", 0 );
		
		ComponentDescriptionType ResampleInterpolatorName = "FinalBSplineInterpolator";
		this->m_Configuration->ReadParameter( ResampleInterpolatorName, "ResampleInterpolator", 0 );

		ComponentDescriptionType ResamplerName = "DefaultResampler";
		this->m_Configuration->ReadParameter( ResamplerName, "Resampler", 0 );

		ComponentDescriptionType TransformName = "BSplineTransform";
		this->m_Configuration->ReadParameter( TransformName, "Transform", 0 );

		/** Create the components! */
		PtrToCreator testcreator;

		/** Key "Elastix", see elxComponentLoader::InstallSupportedImageTypes(). */
		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( "Elastix", this->m_DBIndex );
		this->m_Elastix	= testcreator ? testcreator() : NULL;
		
		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( RegistrationName,	this->m_DBIndex );
		this->m_Registration = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( FixedImagePyramidName, this->m_DBIndex );
		this->m_FixedImagePyramid = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( MovingImagePyramidName, this->m_DBIndex );
		this->m_MovingImagePyramid = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( InterpolatorName, this->m_DBIndex );
		this->m_Interpolator = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( MetricName, this->m_DBIndex );
		this->m_Metric = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( OptimizerName, this->m_DBIndex );
		this->m_Optimizer = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( ResamplerName, this->m_DBIndex );
		this->m_Resampler = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( ResampleInterpolatorName, this->m_DBIndex );
		this->m_ResampleInterpolator	= testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( TransformName, this->m_DBIndex );
		this->m_Transform = testcreator ? testcreator() : NULL;

		/** Check if all component could be created.*/
		if (	( this->m_Elastix.IsNull() ) |
					( this->m_Registration.IsNull() ) |
					( this->m_FixedImagePyramid.IsNull() ) |
					( this->m_MovingImagePyramid.IsNull() ) |
					( this->m_Interpolator.IsNull() ) |
					( this->m_Metric.IsNull() ) |
					( this->m_Optimizer.IsNull() ) |
					( this->m_Resampler.IsNull() ) |
					( this->m_ResampleInterpolator.IsNull() ) |
					( this->m_Transform.IsNull() ) )
		{
			xout["error"] << "ERROR:" << std::endl;
			xout["error"] << "One or more components could not be created." << std::endl;
			return 1;
		}
		
		/** Convert ElastixAsObject to a pointer to an ElastixBaseType. */
		this->m_elx_Elastix = dynamic_cast<ElastixBaseType *>( this->m_Elastix.GetPointer() );

		/** Set all components in the ElastixBase (so actually in
		* the appropriate ElastixTemplate) */
		this->m_elx_Elastix->SetConfiguration( this->m_Configuration );
		this->m_elx_Elastix->SetComponentDatabase(this->s_CDB);

		this->m_elx_Elastix->SetRegistration( this->m_Registration );
		this->m_elx_Elastix->SetFixedImagePyramid( this->m_FixedImagePyramid );
		this->m_elx_Elastix->SetMovingImagePyramid( this->m_MovingImagePyramid );
		this->m_elx_Elastix->SetInterpolator( this->m_Interpolator );
		this->m_elx_Elastix->SetMetric( this->m_Metric );
		this->m_elx_Elastix->SetOptimizer( this->m_Optimizer );
		this->m_elx_Elastix->SetResampler( this->m_Resampler );
		this->m_elx_Elastix->SetResampleInterpolator( this->m_ResampleInterpolator );
		this->m_elx_Elastix->SetTransform( this->m_Transform );

		this->m_elx_Elastix->SetDBIndex( this->m_DBIndex );

		/** Set the images. If not set by the user, it is not a problem.
		* ElastixTemplate!= will try to load them from disk.*/
		this->m_elx_Elastix->SetFixedImage( this->m_FixedImage );
		this->m_elx_Elastix->SetMovingImage( this->m_MovingImage );
		this->m_elx_Elastix->SetFixedInternalImage( this->m_FixedInternalImage );
		this->m_elx_Elastix->SetMovingInternalImage( this->m_MovingInternalImage );

		this->m_elx_Elastix->SetInitialTransform( this->m_InitialTransform );

		/** Run elastix! */
		try { ErrorCode = this->m_elx_Elastix->Run(); }
		catch( itk::ExceptionObject & excp )
		{
			/** We just print the exception and let the programm quit. */
			xl::xout["error"] << excp << std::endl;
			ErrorCode = 1;
		}

		/** Store the images in ElastixMain	*/
		this->SetFixedImage( this->m_elx_Elastix->GetFixedImage() );
		this->SetMovingImage( this->m_elx_Elastix->GetMovingImage() );
		this->SetFixedInternalImage( this->m_elx_Elastix->GetFixedInternalImage() );
		this->SetMovingInternalImage( this->m_elx_Elastix->GetMovingInternalImage() );
		
		/** Set processPriority to normal again.*/
		if ( processPriority == "high" )
		{
			#ifdef _WIN32
			SetPriorityClass( GetCurrentProcess(), NORMAL_PRIORITY_CLASS );
			#endif
		}

		return ErrorCode;

	} // end Run


	/**
	 * **************************** Run *****************************
	 *
	 * Calls EnterCommandLineParameters and then Run().
	 */

	int ElastixMain::Run( ArgumentMapType & argmap )
	{

		this->EnterCommandLineArguments( argmap );
		return this->Run();

	} // end Run

	/**
	 * ************************** InitDBIndex ***********************
	 *
	 * Checks if the configuration object has been initialised,
	 * determines the requested ImageTypes, and sets the m_DBIndex
	 * to the corresponding value (by asking the elx::ComponentDatabase).
	 */

	int ElastixMain::InitDBIndex(void)
	{
		/** .*/
		if ( this->m_Configuration->Initialized() )
		{			
			/** FixedImagePixelType */
			if ( this->m_FixedImagePixelType.empty() )
			{
				/** Try to read it from the parameterfile. */
				this->m_Configuration->ReadParameter( this->m_FixedImagePixelType,	"FixedImagePixelType", 0 );
				
				if ( this->m_FixedImagePixelType.empty() ) // not found in parameterfile
				{
					xout["error"] << "ERROR:" << std::endl;
					xout["error"] << "The FixedImagePixelType is not given." << std::endl;
					return 1;
				}
			}

			/** MovingImagePixelType */
			if ( this->m_MovingImagePixelType.empty() )
			{
				/** Try to read it from the parameterfile. */
				this->m_Configuration->ReadParameter( this->m_MovingImagePixelType, "MovingImagePixelType", 0 );

				if ( this->m_MovingImagePixelType.empty() )
				{
					xout["error"] << "ERROR:" << std::endl;
					xout["error"] << "The MovingImagePixelType is not given." << std::endl;
					return 1;
				}
			}

			/** FixedImageDimension */
			if ( this->m_FixedImageDimension == 0 )
			{
				/** Try to read it from the parameterfile. */
				this->m_Configuration->ReadParameter( this->m_FixedImageDimension, "FixedImageDimension", 0 );

				if ( this->m_FixedImageDimension == 0 )
				{
					xout["error"] << "ERROR:" << std::endl;
					xout["error"] << "The FixedImageDimension is not given." << std::endl;
					return 1;
				}
			}

			/** MovingImageDimension */
			if ( this->m_MovingImageDimension == 0 )
			{
				/** Try to read it from the parameterfile. */
				this->m_Configuration->ReadParameter( this->m_MovingImageDimension, "MovingImageDimension", 0 );

				if ( this->m_MovingImageDimension == 0 )
				{
					xout["error"] << "ERROR:" << std::endl;
					xout["error"] << "The MovingImageDimension is not given." << std::endl;
					return 1;
				}
			}
			
			/** Load the components */
			if (this->s_CDB.IsNull())
			{
				int loadReturnCode = this->LoadComponents();
				if (loadReturnCode !=0)
				{
					xout["error"] << "Loading components failed" << std::endl;
					return loadReturnCode;
				}
			}

			if (this->s_CDB.IsNotNull())
			{
				/** Get the DBIndex from the ComponentDatabase */
				this->m_DBIndex = this->s_CDB->GetIndex(
					this->m_FixedImagePixelType,
					this->m_FixedImageDimension,			
					this->m_MovingImagePixelType,
					this->m_MovingImageDimension );
				if ( this->m_DBIndex == 0 )
				{
					xout["error"] << "ERROR:" << std::endl;
					xout["error"] << "Something went wrong in the ComponentDatabase" << std::endl;
					return 1;
				}
			} //end if s_CDB!=0
	
		} // end if m_Configuration->Initialized();
		else
		{
			xout["error"] << "ERROR:" << std::endl;
			xout["error"] << "The configuration object has not been initialised." << std::endl;
			return 1;
		}

		/** Return an OK value.*/
		return 0;

	} // end InitDBIndex()


	/**
	 * ********************* SetElastixLevel ************************
	 */

	void ElastixMain::SetElastixLevel( unsigned int level )
	{
		/** Call SetElastixLevel from MyConfiguration.*/
		this->m_Configuration->SetElastixLevel( level );

	} // end SetElastixLevel


	/**
	 * ********************* GetElastixLevel ************************
	 */

	unsigned int ElastixMain::GetElastixLevel(void)
	{
		/** Call GetElastixLevel from MyConfiguration.*/
		return this->m_Configuration->GetElastixLevel();

	} // end GetElastixLevel


	/**
	 * ********************* LoadComponents **************************
	 */
	
	int ElastixMain::LoadComponents(void)
	{
		//look for dlls, load them, call the install function

		if (this->s_CDB.IsNull())
		{
			this->s_CDB = ComponentDatabaseType::New();
		}

		if (this->s_ComponentLoader.IsNull())
		{
			this->s_ComponentLoader = ComponentLoaderType::New();
			this->s_ComponentLoader->SetComponentDatabase(s_CDB);
		}


		const char * argv0 = this->m_Configuration->GetCommandLineArgument("-argv0");

		return this->s_ComponentLoader->LoadComponents(argv0);
		
	} // end LoadComponents


		
	/**
	 * ********************* UnloadComponents **************************
	 */
	
	void ElastixMain::UnloadComponents(void)
	{
				
		s_CDB = 0;
		s_ComponentLoader->SetComponentDatabase(0);

		if (s_ComponentLoader)	
		{
			s_ComponentLoader->UnloadComponents();
		}
	}
	
	
	
} // end namespace elastix

#endif // end #ifndef __elxElastixMain_cxx

