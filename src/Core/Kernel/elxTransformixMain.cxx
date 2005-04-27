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
	 * ********************* Constructor ****************************
	 */
	
	TransformixMain::TransformixMain()
	{
		/** Initialize the components.*/
	} // end Constructor
	
	
	/**
	 * ********************** Destructor ****************************
	 */
	
	TransformixMain::~TransformixMain()
	{
	} // end Destructor
	
	
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

		/** Initialize database.*/		
		int ErrorCode = this->InitDBIndex();
		if (ErrorCode != 0)
		{
			return ErrorCode;
		}

		/** Get the different components.*/
		ComponentDescriptionType ResampleInterpolatorName = "FinalBSplineInterpolator";
		m_Configuration->ReadParameter( ResampleInterpolatorName, "ResampleInterpolator", 0 );
		
		ComponentDescriptionType ResamplerName = "DefaultResampler";
		m_Configuration->ReadParameter( ResamplerName, "Resampler", 0 );
		
		ComponentDescriptionType TransformName = "BSplineTransform";
		m_Configuration->ReadParameter( TransformName, "Transform", 0 );
		
		/** Create the components!*/
		PtrToCreator testcreator;
		
		/** Key "Elastix", see elxSupportedImageTypes.cxx.*/
		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( "Elastix", m_DBIndex );
		m_Elastix	= testcreator ? testcreator() : NULL;
		
		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( ResamplerName, m_DBIndex );
		m_Resampler = testcreator ? testcreator() : NULL;
		
		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( ResampleInterpolatorName, m_DBIndex );
		m_ResampleInterpolator	= testcreator ? testcreator() : NULL;
		
		testcreator = 0;
		testcreator = this->s_CDB->GetCreator( TransformName, m_DBIndex );
		m_Transform = testcreator ? testcreator() : NULL;
		
		/** Check if all components could be created.*/
		if (	( m_Elastix.IsNull() ) |
				( m_Resampler.IsNull() ) |
				( m_ResampleInterpolator.IsNull() ) |
				( m_Transform.IsNull() ) )
		{
			xl::xout["error"] << "ERROR:" << std::endl;
			xl::xout["error"] << "One or more components could not be created." << std::endl;
			return 1;
		}	
		
		/** Convert m_Elastix to a pointer to an ElastixBaseType.*/
		m_elx_Elastix = dynamic_cast<ElastixBaseType *>( m_Elastix.GetPointer() );
		
		/** Set all components in the ElastixBase (so actually in
		 * the appropriate ElastixTemplate).
		 */
		m_elx_Elastix->SetComponentDatabase(this->s_CDB);
		m_elx_Elastix->SetConfiguration( m_Configuration );
		m_elx_Elastix->SetResampler( m_Resampler );
		m_elx_Elastix->SetResampleInterpolator( m_ResampleInterpolator );
		m_elx_Elastix->SetTransform( m_Transform );

		m_elx_Elastix->SetDBIndex( m_DBIndex );
		
		/** Set the images. If not set by the user, it is not a problem:
		 * ElastixTemplate will try to load them from disk.
		 */
		m_elx_Elastix->SetMovingImage( this->GetMovingImage() );
		m_elx_Elastix->SetInitialTransform( m_InitialTransform );
		
		/** ApplyTransform! */
		try { ErrorCode = m_elx_Elastix->ApplyTransform(); }
		catch( itk::ExceptionObject & excp )
		{
			/** We just print the exception and let the programm quit. */
			xl::xout["error"]	<< std::endl
				<< "--------------- Exception ---------------"
				<< std::endl << excp
				<< "-----------------------------------------" << std::endl;
			ErrorCode = 1;
		}
		
		/** Set processPriority to normal again. */
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

	int TransformixMain::Run( ArgumentMapType & argmap )
	{

		this->Superclass::EnterCommandLineArguments( argmap );
		return this->Run();

	} // end Run


	/**
	 * ********************* SetInputImage **************************
	 */
	
	void TransformixMain::SetInputImage( DataObjectType * inputImage )
	{
		/** InputImage == MovingImage.*/
		this->Superclass::SetMovingImage( inputImage );
		
	} // end SetInputImage
	

	/**
	 * ********************* InitDBIndex ****************************
	 */
	
	int TransformixMain::InitDBIndex(void)
	{
		/** .*/
		if ( m_Configuration->Initialized() )
		{			
			/** InputImagePixelType */
			m_MovingImagePixelType = m_Configuration->GetCommandLineArgument( "-ipt" );

			if ( m_MovingImagePixelType.empty() )
			{
				/** Try to read it from the parameterfile. */
				//GetMovingImage();
				m_Configuration->ReadParameter( m_MovingImagePixelType,	"MovingImagePixelType", 0 );
				
				if ( m_MovingImagePixelType.empty() ) // not found in parameterfile
				{
					xl::xout["error"] << "ERROR:" << std::endl;
					xl::xout["error"] << "The MovingImagePixelType is not given." << std::endl;
					return 1;
				}
			}

			/** OutputImagePixelType */
			m_FixedImagePixelType = m_Configuration->GetCommandLineArgument( "-opt" );

			if ( m_FixedImagePixelType.empty() )
			{
				/** Try to read it from the parameterfile. */
				m_Configuration->ReadParameter( m_FixedImagePixelType, "FixedImagePixelType", 0 );

				if ( m_FixedImagePixelType.empty() )
				{
					xl::xout["error"] << "ERROR:" << std::endl;
					xl::xout["error"] << "The OutputImagePixelType is not given." << std::endl;
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

