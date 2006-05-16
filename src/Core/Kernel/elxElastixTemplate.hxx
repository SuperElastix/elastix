#ifndef __elxElastixTemplate_hxx
#define __elxElastixTemplate_hxx

#include "elxElastixTemplate.h"

namespace elastix
{
	using namespace itk;
	using namespace xl;
	
	
	/**
	 * ********************* Constructor ****************************
	 */
	
	template <class TFixedImage, class TMovingImage>
	ElastixTemplate<TFixedImage, TMovingImage>
	::ElastixTemplate()
	{
		/** Initialize images. */
		this->m_FixedImage = 0;
		this->m_MovingImage = 0;
		this->m_FixedInternalImage = 0;
		this->m_MovingInternalImage = 0;
		
		/** Initialize the components as smartpointers to itkObjects. */
		this->m_FixedImagePyramid = 0;
		this->m_MovingImagePyramid = 0;
		this->m_Interpolator = 0;
		this->m_Metric = 0;
		this->m_Optimizer = 0;
		this->m_Registration = 0;
		this->m_Resampler = 0;
		this->m_ResampleInterpolator = 0;
		this->m_Transform = 0;
		
		/** Initialize the components as pointers to elx...Base objects. */
		this->m_elx_FixedImagePyramid = 0;
		this->m_elx_MovingImagePyramid = 0;
		this->m_elx_Interpolator = 0;
		this->m_elx_Metric = 0;
		this->m_elx_Optimizer = 0;
		this->m_elx_Registration = 0;
		this->m_elx_Resampler = 0;
		this->m_elx_ResampleInterpolator = 0;
		this->m_elx_Transform = 0;
		
		/** Initialize the Readers and Casters. */
		this->m_FixedImageReader = 0;
		this->m_MovingImageReader = 0;
		this->m_FixedImageCaster = 0;
		this->m_MovingImageCaster = 0;
		
		/** Initialize this->m_InitialTransform. */
		this->m_InitialTransform = 0;
		
		/** Initialize CallBack commands. */
		this->m_BeforeEachResolutionCommand = 0;
		this->m_AfterEachIterationCommand = 0;

		/** Create timers. */
		this->m_Timer0 = TimerType::New();
		this->m_IterationTimer = TimerType::New();
		this->m_ResolutionTimer = TimerType::New();

		/** Initialize the this->m_IterationCounter. */
		this->m_IterationCounter = 0;

		/** Initialize CurrentTransformParameterFileName. */
		this->m_CurrentTransformParameterFileName = "";
		
	} // end Constructor
	
	
	/**
	 * ********************** Destructor ****************************
	 */
	
	template <class TFixedImage, class TMovingImage>
	ElastixTemplate<TFixedImage, TMovingImage>
	::~ElastixTemplate()
	{
	} // end Destructor
	
	
	/**
	 * ********************** SetFixedImage *************************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::SetFixedImage( DataObjectType * _arg )
	{
		/** Cast DataObjectType to FixedImageType and assign to this->m_FixedImage. */
		if ( this->m_FixedImage != _arg )
		{
			this->m_FixedImage = dynamic_cast<FixedImageType *>( _arg );
			this->Modified();
		}
		
	} // end SetFixedImage
	
	
	/**
	 * ********************** SetMovingImage ************************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::SetMovingImage( DataObjectType * _arg )
	{
		/** Cast DataObjectType to MovingImageType and assign to this->m_MovingImage. */
		if ( this->m_MovingImage != _arg )
		{
			this->m_MovingImage = dynamic_cast<MovingImageType *>( _arg );
			this->Modified();
		}
		
	} // end SetMovingImage
	

		/**
	 * ********************** SetFixedInternalImage *************************
	 */
	
	template <class TFixedInternalImage, class TMovingInternalImage>
	void ElastixTemplate<TFixedInternalImage, TMovingInternalImage>
	::SetFixedInternalImage( DataObjectType * _arg )
	{
		/** Cast DataObjectType to FixedInternalImageType and assign to this->m_FixedInternalImage. */
		if ( this->m_FixedInternalImage != _arg )
		{
			this->m_FixedInternalImage = dynamic_cast<FixedInternalImageType *>( _arg );
			this->Modified();
		}
		
	} // end SetFixedInternalImage
	
	
	/**
	 * ********************** SetMovingInternalImage ************************
	 */
	
	template <class TFixedInternalImage, class TMovingInternalImage>
	void ElastixTemplate<TFixedInternalImage, TMovingInternalImage>
	::SetMovingInternalImage( DataObjectType * _arg )
	{
		/** Cast DataObjectType to MovingInternalImageType and assign to this->m_MovingInternalImage. */
		if ( this->m_MovingInternalImage != _arg )
		{
			this->m_MovingInternalImage = dynamic_cast<MovingInternalImageType *>( _arg );
			this->Modified();
		}
		
	} // end SetMovingInternalImage
	

	/**
	 * **************************** Run *****************************
	 */
	
	template <class TFixedImage, class TMovingImage>
	int ElastixTemplate<TFixedImage, TMovingImage>
	::Run(void)
	{
		/** Tell all components where to find the ElastixTemplate. */
		this->m_elx_Registration->SetElastix(this);
		this->m_elx_Transform->SetElastix(this);
		this->m_elx_Metric->SetElastix(this);
		this->m_elx_Interpolator->SetElastix(this);
		this->m_elx_Optimizer->SetElastix(this);
		this->m_elx_FixedImagePyramid->SetElastix(this);
		this->m_elx_MovingImagePyramid->SetElastix(this);
		this->m_elx_Resampler->SetElastix(this);
		this->m_elx_ResampleInterpolator->SetElastix(this);


		/** Call BeforeAll to do some checking. */
		int dummy = this->BeforeAll();
		if ( dummy != 0 ) return dummy;

		/** Setup Callbacks. This makes sure that the BeforeEachResolution()
		 * and AfterEachIteration() functions are called.
		 */
		this->m_BeforeEachResolutionCommand = BeforeEachResolutionCommandType::New();
		this->m_AfterEachResolutionCommand = AfterEachResolutionCommandType::New();
		this->m_AfterEachIterationCommand = AfterEachIterationCommandType::New();
		
		this->m_BeforeEachResolutionCommand->SetCallbackFunction( this, &Self::BeforeEachResolution );
		this->m_AfterEachResolutionCommand->SetCallbackFunction( this, &Self::AfterEachResolution );
		this->m_AfterEachIterationCommand->SetCallbackFunction( this, &Self::AfterEachIteration );
		
		this->m_Registration->AddObserver( itk::IterationEvent(), this->m_BeforeEachResolutionCommand );
		this->m_Optimizer->AddObserver( itk::IterationEvent(), this->m_AfterEachIterationCommand );
		this->m_Optimizer->AddObserver( itk::EndEvent(), this->m_AfterEachResolutionCommand );
	

		/** Start the timer for reading images. */
		this->m_Timer0->StartTimer();
		elxout << "\nReading images..." << std::endl;

		/** \todo Multithreaden? Reading the fixed and moving images could be two threads. */

		/** Set the fixedImage. */
		if ( !(this->m_FixedImage) )
		{
			this->m_FixedImageReader = FixedImageReaderType::New();
			this->m_FixedImageReader->SetFileName(
				this->GetConfiguration()->GetCommandLineArgument( "-f" )  );
			this->m_FixedImage = this->m_FixedImageReader->GetOutput();

			/** Do the reading. */
			try
			{
				this->m_FixedImage->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "ElastixTemplate - Run()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while reading fixed image.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
		}

		if ( !(this->m_FixedInternalImage) )
		{
			this->m_FixedImageCaster = FixedImageCasterType::New();
			this->m_FixedImageCaster->SetInput( this->m_FixedImage );
			this->m_FixedInternalImage = this->m_FixedImageCaster->GetOutput();

			/** Do the casting. */
			try
			{
				this->m_FixedInternalImage->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "ElastixTemplate - Run()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while casting fixed image to InternalImageType.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
		}

		/** Set the fixed image pointer in this registration. */
		this->m_elx_Registration->GetAsITKBaseType()->
			SetFixedImage( this->m_FixedInternalImage );		

		/** Set the movingImage. */
		if ( !(this->m_MovingImage) )
		{
			this->m_MovingImageReader = MovingImageReaderType::New();
			this->m_MovingImageReader->SetFileName(
				this->GetConfiguration()->GetCommandLineArgument( "-m" )  );
			this->m_MovingImage = this->m_MovingImageReader->GetOutput();

			/** Do the reading. */
			try
			{
				this->m_MovingImage->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "ElastixTemplate - Run()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while reading moving image.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
		}

		if ( !(this->m_MovingInternalImage) )
		{
			this->m_MovingImageCaster = MovingImageCasterType::New();
			this->m_MovingImageCaster->SetInput( this->m_MovingImage );
			this->m_MovingInternalImage = this->m_MovingImageCaster->GetOutput();

			/** Do the casting. */
			try
			{
				this->m_MovingInternalImage->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "ElastixTemplate - Run()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while casting moving image to InternalImageType.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
		}

		/** Set the moving image pointer in this registration. */
		this->m_elx_Registration->GetAsITKBaseType()->
			SetMovingImage( this->m_MovingInternalImage );
		
		/** Print the time spent on reading images. */
		this->m_Timer0->StopTimer();
		elxout << "Reading images took " <<	static_cast<unsigned long>(
			this->m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n" << std::endl;

		/** Give all components the opportunity to do some initialization. */
		this->BeforeRegistration();
	
		/** START! */
		try
		{
			( this->m_elx_Registration->GetAsITKBaseType() )->StartRegistration();
		}
		catch( itk::ExceptionObject & excp )
		{
			/** Add information to the exception. */
			excp.SetLocation( "ElastixTemplate - Run()" );
			std::string err_str = excp.GetDescription();
			err_str += "\nError occured during actual registration.\n";
			excp.SetDescription( err_str );
			/** Pass the exception to an higher level. */
			throw excp;
		}
		
		/** Save, show results etc. */
		this->AfterRegistration();
		
		/** Make sure that the transform has stored the final parameters. 
		 *
		 * The transform may be used as a transform in a next elastixLevel;
		 * We need to be sure that it has the final parameters set. 
		 * In the AfterRegistration-method of TransformBase, this method is
		 * already called, but some other component may change the parameters
		 * again in its AfterRegistration-method.
		 *
		 * For now we leave it commented, since there is only Resampler, which 
		 * already calls this method. Calling it again would just take time.
		 */
		//this->m_elx_Transform->SetFinalParameters();

		/** Decouple the components from Elastix. This increases the chance that
		 * some memory is released.
		 */				
		this->m_elx_Registration->SetElastix( 0 );
		this->m_elx_Transform->SetElastix( 0 );
		this->m_elx_Metric->SetElastix( 0 );
		this->m_elx_Interpolator->SetElastix( 0 );
		this->m_elx_Optimizer->SetElastix( 0 );
		this->m_elx_FixedImagePyramid->SetElastix( 0 );
		this->m_elx_MovingImagePyramid->SetElastix( 0 );
		this->m_elx_Resampler->SetElastix( 0 );
		this->m_elx_ResampleInterpolator->SetElastix( 0 );

		/** Return a value. */
		return 0;
		
	} // end Run
	
	
	/**
	 * ************************ ApplyTransform **********************
	 */
	
	template <class TFixedImage, class TMovingImage>
	int ElastixTemplate<TFixedImage, TMovingImage>
	::ApplyTransform(void)
	{
		/** Timer. */
		tmr::Timer::Pointer timer = tmr::Timer::New();

		/** Tell all components where to find the ElastixTemplate. */
		this->m_elx_Transform->SetElastix(this);
		this->m_elx_Resampler->SetElastix(this);
		this->m_elx_ResampleInterpolator->SetElastix(this);

		/** Call BeforeAllTransformix to do some checking. */
		int dummy = BeforeAllTransformix();
		if ( dummy != 0 ) return dummy;

		std::string inputImageFileName =
			this->GetConfiguration()->GetCommandLineArgument( "-in" );
		if ( inputImageFileName != "" )
		{
			/** Timer. */
			timer->StartTimer();

			/** Tell the user. */
			elxout << std::endl << "Reading input image ...";
			
			/** Set the inputImage == movingImage. */
			typename InputImageReaderType::Pointer inputImageReader;
			if ( !this->m_MovingImage )
			{
				inputImageReader = InputImageReaderType::New();
				inputImageReader->SetFileName(
					this->GetConfiguration()->GetCommandLineArgument( "-in" ) );
				this->m_MovingImage = inputImageReader->GetOutput();

				/** Do the reading. */
				try
				{
					this->m_MovingImage->Update();
				}
				catch( itk::ExceptionObject & excp )
				{
					/** Add information to the exception. */
					excp.SetLocation( "ElastixTemplate - ApplyTransform()" );
					std::string err_str = excp.GetDescription();
					err_str += "\nError occured while reading moving image.\n";
					excp.SetDescription( err_str );
					/** Pass the exception to an higher level. */
					throw excp;
				}
			} // end if !moving image

			/** Tell the user. */
			timer->StopTimer();
			elxout << "                   done, it took "
				<< timer->PrintElapsedTimeSec()
				<< " s" << std::endl;
		} // end if ! inputImageFileName

		/** Call all the ReadFromFile() functions. */
		timer->StartTimer();
		elxout << "Calling all ReadFromFile()'s ...";
		this->m_elx_ResampleInterpolator->ReadFromFile();		
		this->m_elx_Resampler->ReadFromFile();
		this->m_elx_Transform->ReadFromFile();

		/** Tell the user. */
		timer->StopTimer();
		elxout << "          done, it took "
			<< timer->PrintElapsedTimeSec()
			<< " s" << std::endl;
		timer->StartTimer();
		elxout << "Transforming points ..." << std::endl;

		/** Call TransformPoints. */
		try
		{
      this->m_elx_Transform->TransformPoints();
		}
		catch( itk::ExceptionObject & excp )
		{
			xout["error"] << excp << std::endl;
			xout["error"] << "However, transformix continues anyway with resampling." << std::endl;
		}
		timer->StopTimer();
		elxout << "  Transforming points done, it took "
			<< timer->PrintElapsedTimeSec()
			<< " s" << std::endl;

		/** Resample the image. */
		if ( inputImageFileName != "" )
		{
			timer->StartTimer();
			elxout << "Resampling image and writing to disk ...";
			
			/** Create a name for the final result. */
			std::string resultImageFormat = "mhd";
			this->m_Configuration->ReadParameter(	resultImageFormat, "ResultImageFormat", 0, true );
			std::ostringstream makeFileName("");
			makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
				<< "result." << resultImageFormat;
			
			/** Write the resampled image to disk. */
			this->m_elx_Resampler->WriteResultImage( makeFileName.str().c_str() );

			/** Tell the user. */
			timer->StopTimer();
			elxout << "  done, it took "
				<< timer->PrintElapsedTimeSec()
				<< " s" << std::endl;
		}

		/** Return a value. */
		return 0;

	} // end ApplyTransform
	

	/**
	 * ************************ BeforeAll ***************************
	 */
	
	template <class TFixedImage, class TMovingImage>
	int ElastixTemplate<TFixedImage, TMovingImage>
	::BeforeAll(void)
	{
		/** Declare the return value and initialize it. */
		int returndummy = 0;
		
		/** Call all the BeforeRegistration() functions. */
		returndummy |= this->BeforeAllBase();
		returndummy |= CallInEachComponentInt( &BaseComponentType::BeforeAllBase );
		returndummy |= CallInEachComponentInt( &BaseComponentType::BeforeAll );

		/** Return a value. */
		return returndummy;
		
	} // end BeforeAll


	/**
	 * ******************** BeforeAllTransformix ********************
	 */
	
	template <class TFixedImage, class TMovingImage>
	int ElastixTemplate<TFixedImage, TMovingImage>
	::BeforeAllTransformix(void)
	{
		/** Declare the return value and initialize it. */
		int returndummy = 0;

		/** Print to log file. */
		elxout << std::fixed;
		elxout << std::showpoint;
		elxout << std::setprecision(3);
		elxout << "ELASTIX version: " << __ELASTIX_VERSION << std::endl;
		elxout << std::setprecision( this->GetDefaultOutputPrecision() );

		/** Check Command line options and print them to the logfile. */
		elxout << "Command line options from ElastixTemplate:" << std::endl;
		std::string check = "";

		/** Check for appearance of "-in". */
		check = this->GetConfiguration()->GetCommandLineArgument( "-in" );
		if ( check == "" )
		{
			//xl::xout["error"] << "ERROR: No CommandLine option \"-in\" given!" << std::endl;
			//returndummy |= -1;
			elxout << "-in\t\tn.a." << std::endl;
		}
		else
		{
			elxout << "-in\t\t" << check << std::endl;
		}

		/** Check for appearance of "-out". */
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

		/** Print "-tp". */
		check = this->GetConfiguration()->GetCommandLineArgument( "-tp" );
		elxout << "-tp\t\t" << check << std::endl;
		
		/** Call all the BeforeAllTransformix() functions. */
		returndummy |= this->m_elx_ResampleInterpolator->BeforeAllTransformix();		
		returndummy |= this->m_elx_Resampler->BeforeAllTransformix();
		returndummy |= this->m_elx_Transform->BeforeAllTransformix();

		/** Print the Transform Parameter file. */
		returndummy |= this->GetConfiguration()->BeforeAll();

		/** Return a value. */
		return returndummy;

	} // end BeforeAllTransformix

	
	/**
	 * **************** BeforeRegistration Callback *****************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::BeforeRegistration(void)
	{
		/** Start timer for initializing all components. */
		this->m_Timer0->StartTimer();
		
		/** Call all the BeforeRegistration() functions. */
		this->BeforeRegistrationBase();
		CallInEachComponent( &BaseComponentType::BeforeRegistrationBase );
		CallInEachComponent( &BaseComponentType::BeforeRegistration );

		/** Add a column to iteration with the iteration number. */
		xout["iteration"].AddTargetCell("1:ItNr");

		/** Add a column to iteration with timing information. */
		xout["iteration"].AddTargetCell("Time[ms]");

		/** Print time for initializing. */
		this->m_Timer0->StopTimer();
		elxout << "Initialization of all components (before registration) took: "
			<< static_cast<unsigned long>( this->m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n";

		/** Start Timer0 here, to make it possible to measure the time needed for 
		 * preparation of the first resolution.
		 */
		this->m_Timer0->StartTimer();

	} // end BeforeRegistration Callback


	/**
	 * ************** BeforeEachResolution Callback *****************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::BeforeEachResolution(void)
	{
		/** Get current resolution level. */
		unsigned long level =
			this->m_elx_Registration->GetAsITKBaseType()->GetCurrentLevel();

		if ( level == 0 )
		{
			this->m_Timer0->StopTimer();
			elxout << "Preparation of the image pyramids took: "
				<< static_cast<unsigned long>( this->m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n";
			this->m_Timer0->StartTimer();
		}
		
		/** Reset the this->m_IterationCounter. */
		this->m_IterationCounter = 0;


		/** Print the current resolution. */
		elxout << "\nResolution: " <<	level	<< std::endl;

		this->OpenIterationInfoFile();

		/** Call all the BeforeEachResolution() functions. */
		this->BeforeEachResolutionBase();
		CallInEachComponent( &BaseComponentType::BeforeEachResolutionBase );
		CallInEachComponent( &BaseComponentType::BeforeEachResolution );
				
		/** Print the extra preparation time needed for this resolution. */
		this->m_Timer0->StopTimer();
		elxout << "Elastix initialization of all components (for this resolution) took: "
			<< static_cast<unsigned long>( this->m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n";

		/** Start ResolutionTimer, which measures the total iteration time in this resolution. */
		this->m_ResolutionTimer->StartTimer();

		/** Start IterationTimer here, to make it possible to measure the time
		 * of the first iteration.
		 */
		this->m_IterationTimer->StartTimer();


	} // end BeforeEachResolution Callback


	/**
	 * ************** AfterEachResolution Callback *****************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::AfterEachResolution(void)
	{
		/** Get current resolution level. */
		unsigned long level =
			this->m_elx_Registration->GetAsITKBaseType()->GetCurrentLevel();
		
		/** Print the total iteration time. */
		elxout << std::setprecision(3);
		this->m_ResolutionTimer->StopTimer();
		elxout 
			<< "Time spent in resolution " 
			<< ( level )
			<< " (ITK initialisation and iterating): "
			<< this->m_ResolutionTimer->GetElapsedClockSec()
			<< " s.\n";
		elxout << std::setprecision( this->GetDefaultOutputPrecision() );
		
		/** Call all the AfterEachResolution() functions. */
		this->AfterEachResolutionBase();
		CallInEachComponent( &BaseComponentType::AfterEachResolutionBase );
		CallInEachComponent( &BaseComponentType::AfterEachResolution );

		/** Create a TransformParameter-file for the current resolution. */
		std::string TranParOptionRes = "false";
		this->m_Configuration->ReadParameter( TranParOptionRes, "WriteTransformParametersEachResolution", 0, true );
		if ( TranParOptionRes == "true" )
		{
			/** Create the TransformParameters filename for this resolution. */
			std::ostringstream makeFileName("");
			makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
				<< "TransformParameters."
				<< this->m_Configuration->GetElastixLevel()
				<< ".R" << this->m_elx_Registration->GetAsITKBaseType()->GetCurrentLevel()
				<< ".txt";
			std::string FileName = makeFileName.str();
			
			/** Create a TransformParameterFile for this iteration. */
			this->CreateTransformParameterFile( FileName, false );
		}

		/** Start Timer0 here, to make it possible to measure the time needed for:
		 *    - executing the BeforeEachResolution methods (if this was not the last resolution)
		 *		- executing the AfterRegistration methods (if this was the last resolution)
		 */
		this->m_Timer0->StartTimer();
	
	} // end AfterEachResolution Callback
	

	/**
	 * ************** AfterEachIteration Callback *******************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::AfterEachIteration(void)
	{
		/** Write the headers of the colums that are printed each iteration. */
		if (this->m_IterationCounter==0)
		{
			xout["iteration"]["WriteHeaders"];
		}

		/** Call all the AfterEachIteration() functions. */
		this->AfterEachIterationBase();
		CallInEachComponent( &BaseComponentType::AfterEachIterationBase );
		CallInEachComponent( &BaseComponentType::AfterEachIteration );

		/** Write the iteration number to the table. */
		xout["iteration"]["1:ItNr"] << m_IterationCounter;

		/** Time in this iteration. */
		this->m_IterationTimer->StopTimer();
		xout["iteration"]["Time[ms]"]
			<< static_cast<unsigned long>( this->m_IterationTimer->GetElapsedClockSec() *1000 );

		/** Write the iteration info of this iteration. */
		xout["iteration"].WriteBufferedData();

		std::string TranParOption;
		this->m_Configuration->ReadParameter( TranParOption, "WriteTransformParametersEachIteration", 0, true );
		if ( TranParOption == "true" )
		{
			/** Add zeros to the number of iterations, to make sure 
			 * it always consists of 7 digits.
			 */
			std::ostringstream makeIterationString("");
			unsigned int border = 1000000;
			while (border > 1)
			{
				if (this->m_IterationCounter < border )
				{
					makeIterationString << "0";
					border /= 10;
				}
				else
				{
					/** Stop. */
					border = 1;
				}
			}
			makeIterationString << this->m_IterationCounter;

			/** Create the TransformParameters filename for this iteration. */
			std::ostringstream makeFileName("");
			makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
				<< "TransformParameters."
				<< this->m_Configuration->GetElastixLevel()
				<< ".R" << this->m_elx_Registration->GetAsITKBaseType()->GetCurrentLevel()
				<< ".It" << makeIterationString.str()
				<< ".txt";
			std::string FileName = makeFileName.str();
			
			/** Create a TransformParameterFile for this iteration. */
			this->CreateTransformParameterFile( FileName, false );
		}
		
		/** Count the number of iterations. */
		this->m_IterationCounter++;

		/** Start timer for next iteration. */
		this->m_IterationTimer->StartTimer();
		
	} // end AfterEachIteration Callback
	
	
	/**
	 * ************** AfterRegistration Callback *******************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::AfterRegistration(void)
	{
		/** No iteration info needed anymore. */
		xl::xout.RemoveTargetCell("iteration");
		
		/** Create the final TransformParameters filename. */
		std::ostringstream makeFileName("");
		makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
			<< "TransformParameters."
			<< this->m_Configuration->GetElastixLevel() << ".txt";
		std::string FileName = makeFileName.str();

		/** Create a final TransformParameterFile. */
		this->CreateTransformParameterFile( FileName, true );

		/** Call all the AfterRegistration() functions. */
		this->AfterRegistrationBase();
		CallInEachComponent( &BaseComponentType::AfterRegistrationBase );
		CallInEachComponent( &BaseComponentType::AfterRegistration );

		/** Print the time spent on things after the registration. */
		this->m_Timer0->StopTimer();
		elxout << "Time spent on saving the results, applying the final transform etc.: "
			<< static_cast<unsigned long>( this->m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n";
		
	} // end AfterRegistration Callback
	

	/**
	 * ************** CreateTransformParameterFile ******************
	 *
	 * Setup the xout transform parameter file, which will
	 * contain the final transform parameters.
	 */
	
	template <class TFixedImage, class TMovingImage>
		void ElastixTemplate<TFixedImage, TMovingImage>
		::CreateTransformParameterFile( std::string FileName, bool ToLog )
	{
		using namespace xl;

		/** Store CurrentTransformParameterFileName. */
		this->m_CurrentTransformParameterFileName = FileName;

		/** Create transformParameterFile and xout["transpar"]. */
		xoutsimple_type		transformationParameterInfo;
		std::ofstream			transformParameterFile;
		
		/** Set up the "TransformationParameters" writing field. */
		transformationParameterInfo.SetOutputs( xout.GetCOutputs() );
		transformationParameterInfo.SetOutputs( xout.GetXOutputs() );
		
		xout.AddTargetCell( "transpar", &transformationParameterInfo );
		
		/** Set it in the Transform, for later use. */
		this->m_elx_Transform->SetTransformParametersFileName( FileName.c_str() );
		
		/** Open the TransformParameter file. */
		transformParameterFile.open( FileName.c_str() );
		if ( !transformParameterFile.is_open() )
		{
			xout["error"] << "ERROR: File \"" << FileName << "\" could not be opened!" << std::endl;
		}
		
		/** This xout["transpar"] writes to the log and to the TransformParameter file. */
		transformationParameterInfo.RemoveOutput( "cout" );
		transformationParameterInfo.AddOutput( "tpf", &transformParameterFile );
		if ( !ToLog )
		{
			transformationParameterInfo.RemoveOutput( "log" );
		}
		
		/** Format specifiers of the transformation parameter file. */
		xout["transpar"] << std::showpoint;
		xout["transpar"] << std::fixed;
		xout["transpar"] << std::setprecision( this->GetDefaultOutputPrecision() );
		
		/** Separate clearly in log-file. */
		if ( ToLog )
		{
			xout["logonly"] <<
				"\n=============== start of TransformParameterFile ==============="	<< std::endl;
		}

		/** Call all the WriteToFile() functions. */
		this->m_elx_Transform->WriteToFile(
			this->m_elx_Optimizer->GetAsITKBaseType()->GetCurrentPosition() );
		this->m_elx_ResampleInterpolator->WriteToFile();
		this->m_elx_Resampler->WriteToFile();
		
		/** Separate clearly in log-file. */
		if ( ToLog )
		{
			xout["logonly"] << 
				"\n=============== end of TransformParameterFile ===============" << std::endl;
		}

		/** Remove the "transpar" writing field. */
		xout.RemoveTargetCell( "transpar" );
		
	} // end CreateTransformParameterFile


	/**
	 * ****************** CallInEachComponent ***********************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::CallInEachComponent( PtrToMemberFunction func )
	{
		/** Call the memberfunction 'func' of the this->m_elx_Components. */
		( (*(this->GetConfiguration())).*func )();
		( (*(this->m_elx_Registration)).*func )();
		( (*(this->m_elx_Transform)).*func )();
		( (*(this->m_elx_Metric)).*func )();
		( (*(this->m_elx_Interpolator)).*func )();
		( (*(this->m_elx_Optimizer)).*func )();
		( (*(this->m_elx_FixedImagePyramid)).*func )();
		( (*(this->m_elx_MovingImagePyramid)).*func )();
		( (*(this->m_elx_ResampleInterpolator)).*func )();		
		( (*(this->m_elx_Resampler)).*func )();
		
	} // end CallInEachComponent
	

	/**
	 * ****************** CallInEachComponentInt ********************
	 */
	
	template <class TFixedImage, class TMovingImage>
	int ElastixTemplate<TFixedImage, TMovingImage>
	::CallInEachComponentInt( PtrToMemberFunction2 func )
	{
		/** Declare the return value and initialize it. */
		int returndummy = 0;

		/** Call the memberfunction 'func' of the this->m_elx_Components. */
		returndummy |= ( (*( this->GetConfiguration())).*func )();		
		returndummy |= ( (*(this->m_elx_Registration)).*func )();
		returndummy |= ( (*(this->m_elx_Transform)).*func )();
		returndummy |= ( (*(this->m_elx_Metric)).*func )();
		returndummy |= ( (*(this->m_elx_Interpolator)).*func )();
		returndummy |= ( (*(this->m_elx_Optimizer)).*func )();
		returndummy |= ( (*(this->m_elx_FixedImagePyramid)).*func )();
		returndummy |= ( (*(this->m_elx_MovingImagePyramid)).*func )();
		returndummy |= ( (*(this->m_elx_ResampleInterpolator)).*func )();		
		returndummy |= ( (*(this->m_elx_Resampler)).*func )();

		/** Return a value. */
		return returndummy;
				
	} // end CallInEachComponent


	/**
	 * ************** OpenIterationInfoFile *************************
	 *
	 * Open a file called IterationInfo.<ElastixLevel>.R<Resolution>.txt,
	 * which will contain the iteration info table.
	 */
	
	template <class TFixedImage, class TMovingImage>
		void ElastixTemplate<TFixedImage, TMovingImage>
		::OpenIterationInfoFile( void )
	{
		using namespace xl;
		
		/** Remove the current iteration info output file, if any. */
		xout["iteration"].RemoveOutput( "IterationInfoFile" );
		
		if ( this->m_IterationInfoFile.is_open() )
		{
			this->m_IterationInfoFile.close();
		}

		/** Create the IterationInfo filename for this resolution. */
		std::ostringstream makeFileName("");
		makeFileName << this->m_Configuration->GetCommandLineArgument( "-out" )
			<< "IterationInfo."
			<< this->m_Configuration->GetElastixLevel()
			<< ".R" << this->m_elx_Registration->GetAsITKBaseType()->GetCurrentLevel()
			<< ".txt";
		std::string FileName = makeFileName.str();

		/** Open the IterationInfoFile. */
		this->m_IterationInfoFile.open( FileName.c_str() );
		if ( !(this->m_IterationInfoFile.is_open()) )
		{
			xout["error"] << "ERROR: File \"" << FileName << "\" could not be opened!" << std::endl;
		}
		else    
		{
			/** Add this file to the list of outputs of xout["iteration"]. */
			xout["iteration"].AddOutput( "IterationInfoFile", &(this->m_IterationInfoFile) );
		}

	} // end of function OpenIterationInfoFile

} // end namespace elastix


#endif // end #ifndef __elxElastixTemplate_hxx

