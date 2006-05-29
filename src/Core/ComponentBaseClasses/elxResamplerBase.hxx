#ifndef __elxResamplerBase_hxx
#define __elxResamplerBase_hxx

#include "elxResamplerBase.h"

#include "elxTimer.h"

namespace elastix
{
	using namespace itk;


	/*
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		ResamplerBase<TElastix>::ResamplerBase()
	{

	} // end Constructor


	/*
	 * ******************* BeforeRegistrationBase *******************
	 */
	
	template<class TElastix>
		void ResamplerBase<TElastix>
		::BeforeRegistrationBase(void)
	{
		/** Connect the components. */
		this->SetComponents();
		
		/** Set the size of the image to be produced by the resampler. */
		
		/** Get a pointer to the fixedImage. */
		OutputImageType * fixedImage = dynamic_cast<OutputImageType *>(
			this->m_Elastix->GetFixedImage() );
		
		/** Set the region info to the same values as in the fixedImage. */
		this->GetAsITKBaseType()->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
		this->GetAsITKBaseType()->SetOutputStartIndex( fixedImage->GetLargestPossibleRegion().GetIndex() );
		this->GetAsITKBaseType()->SetOutputOrigin( fixedImage->GetOrigin() );
		this->GetAsITKBaseType()->SetOutputSpacing( fixedImage->GetSpacing() );
		
		/** Set the DefaultPixelValue (for pixels in the resampled image
		 * that come from outside the original (moving) image.
		 */
		OutputPixelType defaultPixelValue = NumericTraits<OutputPixelType>::Zero;
		this->m_Configuration->ReadParameter( defaultPixelValue, "DefaultPixelValue", 0 );
		
		/** Set the defaultPixelValue. */
		this->GetAsITKBaseType()->SetDefaultPixelValue( defaultPixelValue );

	} // end BeforeRegistrationBase


	/*
	 * ******************* AfterEachResolutionBase ********************
	 */
	
	template<class TElastix>
		void ResamplerBase<TElastix>
		::AfterEachResolutionBase(void)
	{
		/** Set the final transform parameters. */
		this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

		/** What is the current resolution level? */
		unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

		/** Decide whether or not to write the result image this resolution. */
		std::string writeResultImageThisResolution = "false";
		this->m_Configuration->ReadParameter(	
			writeResultImageThisResolution, "WriteResultImageAfterEachResolution", 0, true );
		this->m_Configuration->ReadParameter(	
			writeResultImageThisResolution, "WriteResultImageAfterEachResolution", level, true );

		/** Create a name for the final result. */
		std::string resultImageFormat = "mhd";
		this->m_Configuration->ReadParameter(	resultImageFormat, "ResultImageFormat", 0, true );
		std::ostringstream makeFileName( "" );
		makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
			<< "result." << this->m_Configuration->GetElastixLevel()
			<< ".R" << level
			<< "." << resultImageFormat;

		/** Writing result image. */
		if ( writeResultImageThisResolution == "true" )
		{
			/** Time the resampling. */
			typedef tmr::Timer TimerType;
			TimerType::Pointer timer = TimerType::New();
			timer->StartTimer();
			/** Apply the final transform, and save the result. */
			elxout << "Applying transform this resolution";
			/** Call WriteResultImage. */
			try
			{
				this->WriteResultImage( makeFileName.str().c_str() );
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << "Exception caught: " << std::endl;
				xl::xout["error"] << excp
					<< "Resuming elastix." << std::endl;
			}
			/** Print the elapsed time for the resampling. */
			timer->StopTimer();
			elxout << ", which took: "
				<< static_cast<long>( timer->GetElapsedClockSec() )
				<< " s." << std::endl;
		} // end if

	} // end AfterEachResolutionBase


	/*
	 * ******************* AfterRegistrationBase ********************
	 */
	
	template<class TElastix>
		void ResamplerBase<TElastix>
		::AfterRegistrationBase(void)
	{
		/** Set the final transform parameters. */
		this->GetElastix()->GetElxTransformBase()->SetFinalParameters();

		/** Decide whether or not to write the result image. */
		std::string writeResultImage = "true";
		this->m_Configuration->ReadParameter(	writeResultImage, "WriteResultImage", 0, true );

		/** Create a name for the final result. */
		std::string resultImageFormat = "mhd";
		this->m_Configuration->ReadParameter(	resultImageFormat, "ResultImageFormat", 0, true );
		std::ostringstream makeFileName( "" );
		makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
			<< "result." << this->m_Configuration->GetElastixLevel()
			<< "." << resultImageFormat;

		/** Writing result image. */
		if ( writeResultImage == "true" )
		{
			/** Time the resampling. */
			typedef tmr::Timer TimerType;
			TimerType::Pointer timer = TimerType::New();
			timer->StartTimer();
			/** Apply the final transform, and save the result. */
			elxout << std::endl << "Applying final transform";
			/** Call WriteResultImage. */
			try
			{
				this->WriteResultImage( makeFileName.str().c_str() );
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << "Exception caught: " << std::endl;
				xl::xout["error"] << excp
					<< "Resuming elastix." << std::endl;
			}
			/** Print the elapsed time for the resampling. */
			timer->StopTimer();
			elxout << ", which took: "
				<< static_cast<long>( timer->GetElapsedClockSec() )
				<< " s." << std::endl;
		}
		else
		{
			/** Do not apply the final transform. */
			elxout << std::endl
				<< "Skipping applying final transform, no resulting output image generated."
				<< std::endl;
		} // end if

	} // end AfterRegistrationBase


	/*
	 * *********************** SetComponents ************************
	 */
	
	template <class TElastix>
		void ResamplerBase<TElastix>
		::SetComponents(void)
	{
		/** Set the transform, the interpolator and the inputImage
		 * (which is the moving image).
		 */
		this->GetAsITKBaseType()->SetTransform( dynamic_cast<TransformType *>(
			this->m_Elastix->GetTransform() ) );
		
		this->GetAsITKBaseType()->SetInterpolator( dynamic_cast<InterpolatorType *>(
			this->m_Elastix->GetResampleInterpolator() ) );
		
		this->GetAsITKBaseType()->SetInput( dynamic_cast<InputImageType *>(
			this->m_Elastix->GetMovingImage() ) );
		
	} // end SetComponents


	/*
	 * ******************* WriteResultImage ********************
	 */
	
	template<class TElastix>
		void ResamplerBase<TElastix>
		::WriteResultImage( const char * filename )
	{
		/** Make sure the resampler is updated. */
		this->GetAsITKBaseType()->Modified();

		/** Create writer. */
		WriterPointer writer = WriterType::New();

		/** Setup the pipeline. */
		writer->SetInput( this->GetAsITKBaseType()->GetOutput() );

		/** Set the filename. */
		writer->SetFileName( filename );

		/** Do the writing. */
		try
		{
			writer->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			/** Add information to the exception. */
			excp.SetLocation( "ResamplerBase - AfterRegistrationBase()" );
			std::string err_str = excp.GetDescription();
			err_str += "\nError occured while writing resampled image.\n";
			excp.SetDescription( err_str );
			/** Pass the exception to an higher level. */
			throw excp;
		}

	} // WriteResultImage

	/*
	 * ************************* ReadFromFile ***********************
	 */
	
	template<class TElastix>
		void ResamplerBase<TElastix>
		::ReadFromFile(void)
	{
		/** Connect the components. */
		this->SetComponents();
		
		/** Get spacing, origin and size of the image to be produced by the resampler. */
		SpacingType			spacing;
		IndexType				index;
		OriginPointType	origin;
		SizeType				size;
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			/** No default size. Read size from the parameter file. */
			this->m_Configuration->ReadParameter(	size[ i ], "Size", i );

			/** Default index. Read index from the parameter file. */
			index[ i ] = 0;
			this->m_Configuration->ReadParameter(	index[ i ], "Index", i );

			/** Default spacing. Read spacing from the parameter file. */
			spacing[ i ] = 1.0;
			this->m_Configuration->ReadParameter(	spacing[ i ], "Spacing", i );

			/** Default origin. Read origin from the parameter file. */
			origin[ i ] = 0.0;
			this->m_Configuration->ReadParameter(	origin[ i ], "Origin", i );
		}

		/** Check for image size. */
		unsigned int sum = 0;
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			if ( size[ i ] == 0 ) sum++;
		}
		if ( sum > 0 )
		{
			xl::xout["error"] << "ERROR: One or more image sizes are 0!" << std::endl;
			/** \todo quit program nicely. */
		}
		
		/** Set the region info to the same values as in the fixedImage. */
		this->GetAsITKBaseType()->SetSize( size );
		this->GetAsITKBaseType()->SetOutputStartIndex( index );
		this->GetAsITKBaseType()->SetOutputOrigin( origin );
		this->GetAsITKBaseType()->SetOutputSpacing( spacing );
		
		/** Set the DefaultPixelValue (for pixels in the resampled image
		 * that come from outside the original (moving) image.
		 */
		int defaultPixelValue = 0;
		this->m_Configuration->ReadParameter( defaultPixelValue, "DefaultPixelValue", 0 );
		
		/** Set the defaultPixelValue in the Superclass. */
		this->GetAsITKBaseType()->SetDefaultPixelValue( defaultPixelValue );
		
	} // end ReadFromFile


	/**
	 * ******************* WriteToFile ******************************
	 */

	template <class TElastix>
		void ResamplerBase<TElastix>
		::WriteToFile(void)
	{
		/** Write Resampler specific things. */
		xl::xout["transpar"] << std::endl << "// Resampler specific" << std::endl;

		/** Write the name of the Resampler. */
		xl::xout["transpar"] << "(Resampler \""
			<< this->elxGetClassName() << "\")" << std::endl;

		/** Write the DefaultPixelValue. */
		xl::xout["transpar"] << "(DefaultPixelValue "
			<< this->GetAsITKBaseType()->GetDefaultPixelValue() << ")" << std::endl;

		/** Write the output image format. */
		std::string resultImageFormat = "mhd";
		this->m_Configuration->ReadParameter(	resultImageFormat, "ResultImageFormat", 0, true );
		xl::xout["transpar"] << "(ResultImageFormat \""
			<< resultImageFormat << "\")" << std::endl;

	} // end WriteToFile


} // end namespace elastix


#endif // end #ifndef __elxResamplerBase_hxx

