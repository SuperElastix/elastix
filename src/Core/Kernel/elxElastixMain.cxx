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
		m_Configuration = ConfigurationType::New();

		m_Elastix = 0;
		m_elx_Elastix = 0;

		m_FixedImagePixelType = "";
		m_FixedImageDimension = 0;

		m_MovingImagePixelType = "";
		m_MovingImageDimension = 0;

		m_DBIndex = 0;

		m_FixedImage = 0;
		m_MovingImage = 0;
		m_FixedInternalImage = 0;
		m_MovingInternalImage = 0;

		m_Registration = 0;
		m_FixedImagePyramid = 0;
		m_MovingImagePyramid = 0;
		m_Interpolator = 0;
		m_Metric = 0;
		m_Optimizer = 0;
		m_Resampler = 0;
		m_ResampleInterpolator = 0;
		m_Transform = 0;

		m_InitialTransform = 0;

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
		int dummy = m_Configuration->Initialize( argmap );
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
		processPriority = m_Configuration->GetCommandLineArgument( "-priority" );
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
		m_Configuration->ReadParameter( RegistrationName, "Registration", 0 );
		
		ComponentDescriptionType FixedImagePyramidName = "FixedRecursiveImagePyramid";
		m_Configuration->ReadParameter( FixedImagePyramidName, "FixedImagePyramid", 0 );

		ComponentDescriptionType MovingImagePyramidName = "MovingRecursiveImagePyramid";
		m_Configuration->ReadParameter( MovingImagePyramidName, "MovingImagePyramid", 0 );
		
		ComponentDescriptionType InterpolatorName = "BSplineInterpolator";
		m_Configuration->ReadParameter( InterpolatorName, "Interpolator", 0 );

		ComponentDescriptionType MetricName = "MattesMutualInformation";
		m_Configuration->ReadParameter( MetricName, "Metric", 0 );

		ComponentDescriptionType OptimizerName = "RegularStepGradientDescent";
		m_Configuration->ReadParameter( OptimizerName, "Optimizer", 0 );
		
		ComponentDescriptionType ResampleInterpolatorName = "FinalBSplineInterpolator";
		m_Configuration->ReadParameter( ResampleInterpolatorName, "ResampleInterpolator", 0 );

		ComponentDescriptionType ResamplerName = "DefaultResampler";
		m_Configuration->ReadParameter( ResamplerName, "Resampler", 0 );

		ComponentDescriptionType TransformName = "BSplineTransform";
		m_Configuration->ReadParameter( TransformName, "Transform", 0 );

		/** Create the components! */
		PtrToCreator testcreator;

		/** Key "Elastix", see elxComponentLoader::InstallSupportedImageTypes(). */
		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( "Elastix", m_DBIndex );
		m_Elastix	= testcreator ? testcreator() : NULL;
		
		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( RegistrationName,	m_DBIndex );
		m_Registration = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( FixedImagePyramidName, m_DBIndex );
		m_FixedImagePyramid = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( MovingImagePyramidName, m_DBIndex );
		m_MovingImagePyramid = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( InterpolatorName, m_DBIndex );
		m_Interpolator = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( MetricName, m_DBIndex );
		m_Metric = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( OptimizerName, m_DBIndex );
		m_Optimizer = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( ResamplerName, m_DBIndex );
		m_Resampler = testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( ResampleInterpolatorName, m_DBIndex );
		m_ResampleInterpolator	= testcreator ? testcreator() : NULL;

		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( TransformName, m_DBIndex );
		m_Transform = testcreator ? testcreator() : NULL;

		/** Check if all component could be created.*/
		if (	( m_Elastix == 0 ) |
					( m_Registration == 0 ) |
					( m_FixedImagePyramid == 0 ) |
					( m_MovingImagePyramid == 0 ) |
					( m_Interpolator == 0 ) |
					( m_Metric == 0 ) |
					( m_Optimizer == 0 ) |
					( m_Resampler == 0 ) |
					( m_ResampleInterpolator == 0 ) |
					( m_Transform == 0 ) )
		{
			xout["error"] << "ERROR:" << std::endl;
			xout["error"] << "One or more components could not be created." << std::endl;
			return 1;
		}
		
		/** Convert ElastixAsObject to a pointer to an ElastixBaseType. */
		m_elx_Elastix = dynamic_cast<ElastixBaseType *>( m_Elastix.GetPointer() );

		/** Set all components in the ElastixBase (so actually in
		* the appropriate ElastixTemplate) */
		m_elx_Elastix->SetConfiguration( m_Configuration );
		m_elx_Elastix->SetComponentDatabase(this->s_CDB);

		m_elx_Elastix->SetRegistration( m_Registration );
		m_elx_Elastix->SetFixedImagePyramid( m_FixedImagePyramid );
		m_elx_Elastix->SetMovingImagePyramid( m_MovingImagePyramid );
		m_elx_Elastix->SetInterpolator( m_Interpolator );
		m_elx_Elastix->SetMetric( m_Metric );
		m_elx_Elastix->SetOptimizer( m_Optimizer );
		m_elx_Elastix->SetResampler( m_Resampler );
		m_elx_Elastix->SetResampleInterpolator( m_ResampleInterpolator );
		m_elx_Elastix->SetTransform( m_Transform );

		m_elx_Elastix->SetDBIndex( m_DBIndex );

		/** Set the images. If not set by the user, it is not a problem.
		* ElastixTemplate!= will try to load them from disk.*/
		m_elx_Elastix->SetFixedImage( m_FixedImage );
		m_elx_Elastix->SetMovingImage( m_MovingImage );
		m_elx_Elastix->SetFixedInternalImage( m_FixedInternalImage );
		m_elx_Elastix->SetMovingInternalImage( m_MovingInternalImage );

		m_elx_Elastix->SetInitialTransform( m_InitialTransform );

		/** Run elastix! */
		try { ErrorCode = m_elx_Elastix->Run(); }
		catch( itk::ExceptionObject & excp )
		{
			/** We just print the exception and let the programm quit. */
			xl::xout["error"] << excp << std::endl;
			ErrorCode = 1;
		}

		/** Store the images in ElastixMain	*/
		this->SetFixedImage( m_elx_Elastix->GetFixedImage() );
		this->SetMovingImage( m_elx_Elastix->GetMovingImage() );
		this->SetFixedInternalImage( m_elx_Elastix->GetFixedInternalImage() );
		this->SetMovingInternalImage( m_elx_Elastix->GetMovingInternalImage() );
		
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
		if ( m_Configuration->Initialized() )
		{			
			/** FixedImagePixelType */
			if ( m_FixedImagePixelType.empty() )
			{
				/** Try to read it from the parameterfile. */
				m_Configuration->ReadParameter( m_FixedImagePixelType,	"FixedImagePixelType", 0 );
				
				if ( m_FixedImagePixelType.empty() ) // not found in parameterfile
				{
					xout["error"] << "ERROR:" << std::endl;
					xout["error"] << "The FixedImagePixelType is not given." << std::endl;
					return 1;
				}
			}

			/** MovingImagePixelType */
			if ( m_MovingImagePixelType.empty() )
			{
				/** Try to read it from the parameterfile. */
				m_Configuration->ReadParameter( m_MovingImagePixelType, "MovingImagePixelType", 0 );

				if ( m_MovingImagePixelType.empty() )
				{
					xout["error"] << "ERROR:" << std::endl;
					xout["error"] << "The MovingImagePixelType is not given." << std::endl;
					return 1;
				}
			}

			/** FixedImageDimension */
			if ( m_FixedImageDimension == 0 )
			{
				/** Try to read it from the parameterfile. */
				m_Configuration->ReadParameter( m_FixedImageDimension, "FixedImageDimension", 0 );

				if ( m_FixedImageDimension == 0 )
				{
					xout["error"] << "ERROR:" << std::endl;
					xout["error"] << "The FixedImageDimension is not given." << std::endl;
					return 1;
				}
			}

			/** MovingImageDimension */
			if ( m_MovingImageDimension == 0 )
			{
				/** Try to read it from the parameterfile. */
				m_Configuration->ReadParameter( m_MovingImageDimension, "MovingImageDimension", 0 );

				if ( m_MovingImageDimension == 0 )
				{
					xout["error"] << "ERROR:" << std::endl;
					xout["error"] << "The MovingImageDimension is not given." << std::endl;
					return 1;
				}
			}
			
			/** Load the components */
			if (this->s_CDB==0)
			{
				int loadReturnCode = this->LoadComponents();
				if (loadReturnCode !=0)
				{
					xout["error"] << "Loading components failed" << std::endl;
					return loadReturnCode;
				}
			}

			if (this->s_CDB!=0)
			{
				/** Get the DBIndex from the ComponentDatabase */
				m_DBIndex = this->s_CDB->GetIndex(
					m_FixedImagePixelType,
					m_FixedImageDimension,			
					m_MovingImagePixelType,
					m_MovingImageDimension );
				if ( m_DBIndex == 0 )
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
		m_Configuration->SetElastixLevel( level );

	} // end SetElastixLevel


	/**
	 * ********************* GetElastixLevel ************************
	 */

	unsigned int ElastixMain::GetElastixLevel(void)
	{
		/** Call GetElastixLevel from MyConfiguration.*/
		return m_Configuration->GetElastixLevel();

	} // end GetElastixLevel


	/**
	 * ********************* LoadComponents **************************
	 */
	
	int ElastixMain::LoadComponents(void)
	{
		//look for dlls, load them, call the install function

		if (this->s_CDB == 0)
		{
			this->s_CDB = ComponentDatabaseType::New();
		}

		if (this->s_ComponentLoader == 0)
		{
			this->s_ComponentLoader = ComponentLoaderType::New();
			this->s_ComponentLoader->SetComponentDatabase(s_CDB);
		}


		const char * argv0 = m_Configuration->GetCommandLineArgument("-argv0");

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

