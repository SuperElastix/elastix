#ifndef __elxTransformixMain_CXX_
#define __elxTransformixMain_CXX_

/** If running on a Windows-system, include "windows.h".*/
#ifdef _WIN32
#include <windows.h>
#endif


#include "elxTransformixMain.h"
#include "elxMacro.h"

namespace elastix
{
	
	
	/**
	 * **************************** Run *****************************
	 *
	 * Assuming EnterCommandLineParameters has already been invoked.
	 * or that m_Configuration is initialised in another way.
	 */
	
	int TransformixMain::Run(void)
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
		else if ( processPriority == "belownormal" )
		{
			#ifdef _WIN32
			SetPriorityClass( GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS );
			#endif
		}

		/** Initialize database.*/		
		int errorCode = this->InitDBIndex();
		if (errorCode != 0)
		{
			return errorCode;
		}

        /** Create the Elastix component */
    try 
    {
      /** Key "Elastix", see elxComponentLoader::InstallSupportedImageTypes(). */
      this->m_Elastix = this->CreateComponent( "Elastix" );
    }
		catch( itk::ExceptionObject & excp )
		{
			/** We just print the exception and let the programm quit. */
			xl::xout["error"] << excp << std::endl;
			errorCode = 1;
      return errorCode;
		}

    /** Set some information in the ElastixBase */
		this->GetElastixBase()->SetConfiguration( this->m_Configuration );
		this->GetElastixBase()->SetComponentDatabase(this->s_CDB);
		this->GetElastixBase()->SetDBIndex( this->m_DBIndex );

    /** Populate the component containers */
   this->GetElastixBase()->SetResampleInterpolatorContainer(
      this->CreateComponents( "ResampleInterpolator", "FinalBSplineInterpolator", errorCode) );
      
    this->GetElastixBase()->SetResamplerContainer(
      this->CreateComponents( "Resampler", "DefaultResampler", errorCode) );
      
    this->GetElastixBase()->SetTransformContainer(
      this->CreateComponents( "Transform", "TranslationTransform", errorCode) );
      
    /** Check if all component could be created. */
		if ( errorCode != 0 )
		{
      xl::xout["error"] << "ERROR:" << std::endl;
      xl::xout["error"] << "One or more components could not be created." << std::endl;
			return 1;
		}
			
		/** Set the images. If not set by the user, it is not a problem.
		 * ElastixTemplate will try to load them from disk. */
		this->GetElastixBase()->SetMovingImageContainer( this->GetMovingImageContainer() );

    /** Set the initial transform, if it happens to be there 
     * \todo: Does this make sense for transformix?
     */
		this->GetElastixBase()->SetInitialTransform( this->GetInitialTransform() );
		
		/** ApplyTransform! */
		try 
    {
      errorCode = this->GetElastixBase()->ApplyTransform();
    }
		catch( itk::ExceptionObject & excp )
		{
			/** We just print the exception and let the programm quit. */
			xl::xout["error"]	<< std::endl
				<< "--------------- Exception ---------------"
				<< std::endl << excp
				<< "-----------------------------------------" << std::endl;
			errorCode = 1;
		}

    /** Save the image container */
    this->SetMovingImageContainer( this->GetElastixBase()->GetMovingImageContainer() );
				
		/** Set processPriority to normal again. */
		if ( processPriority == "high" )
		{
			#ifdef _WIN32
			SetPriorityClass( GetCurrentProcess(), NORMAL_PRIORITY_CLASS );
			#endif
		}

		return errorCode;

	} // end Run


	/**
	 * **************************** Run *****************************
	 *
	 * Calls EnterCommandLineParameters and then Run().
	 */

	int TransformixMain::Run( ArgumentMapType & argmap )
	{

		this->EnterCommandLineArguments( argmap );
		return this->Run();

	} // end Run


	/**
	 * ********************* SetInputImage **************************
	 */
	
	void TransformixMain::SetInputImageContainer( DataObjectContainerType * inputImageContainer )
	{
		/** InputImage == MovingImage.*/
		this->SetMovingImageContainer( inputImageContainer );
		
	} // end SetInputImage
	

	/**
	 * ********************* InitDBIndex ****************************
	 */
	
	int TransformixMain::InitDBIndex(void)
	{
		/** Check if configuration object was already initialized.*/
		if ( m_Configuration->Initialized() )
		{			
			/** MovingImagePixelType */
			m_MovingImagePixelType = m_Configuration->GetCommandLineArgument( "-ipt" );

			if ( m_MovingImagePixelType.empty() )
			{
				/** Try to read it from the parameterfile. */
				m_MovingImagePixelType = "float";
				m_Configuration->ReadParameter( m_MovingImagePixelType,	"MovingInternalImagePixelType", 0 );
			}

			/** FixedImagePixelType */
			m_FixedImagePixelType = m_Configuration->GetCommandLineArgument( "-opt" );

			if ( m_FixedImagePixelType.empty() )
			{
				/** Try to read it from the parameterfile. */
        m_FixedImagePixelType = "float";
				m_Configuration->ReadParameter( m_FixedImagePixelType, "FixedInternalImagePixelType", 0 );
			}

			/** MovingImageDimension */
			if ( m_MovingImageDimension == 0 )
			{
				/** Try to read it from the parameterfile. */
				m_Configuration->ReadParameter( m_MovingImageDimension, "MovingImageDimension", 0 );

				if ( m_MovingImageDimension == 0 )
				{
					xl::xout["error"] << "ERROR:" << std::endl;
					xl::xout["error"] << "The MovingImageDimension is not given." << std::endl;
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
					xl::xout["error"] << "ERROR:" << std::endl;
					xl::xout["error"] << "The FixedImageDimension is not given." << std::endl;
					return 1;
				}
			}
			
			/** Load the components */
			if (this->s_CDB.IsNull())
			{
				int loadReturnCode = this->LoadComponents();
				if (loadReturnCode !=0)
				{
					xl::xout["error"] << "Loading components failed" << std::endl;
					return loadReturnCode;
				}
			}

			if (this->s_CDB.IsNotNull())
			{
				/** Get the DBIndex from the ComponentDatabase */
				m_DBIndex = this->s_CDB->GetIndex(
					m_FixedImagePixelType,
					m_FixedImageDimension,			
					m_MovingImagePixelType,
					m_MovingImageDimension );
				if ( m_DBIndex == 0 )
				{
					xl::xout["error"] << "ERROR:" << std::endl;
					xl::xout["error"] << "Something went wrong in the ComponentDatabase" << std::endl;
					return 1;
				}
			} //end if s_CDB!=0

		} // end if m_Configuration->Initialized();
		else
		{
			xl::xout["error"] << "ERROR:" << std::endl;
			xl::xout["error"] << "The configuration object has not been initialised." << std::endl;
			return 1;
		}

		/** Everything is OK! */
		return 0;
		
	} // end InitDBIndex
	

} // end namespace elastix


#endif // end #ifndef __elxTransformixMain_cxx

