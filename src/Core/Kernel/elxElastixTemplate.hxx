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
		/** Initialize images.*/
		m_FixedImage = 0;
		m_MovingImage = 0;
		m_FixedInternalImage = 0;
		m_MovingInternalImage = 0;
		
		/** Initialize the components as smartpointers to itkObjects.*/
		m_FixedImagePyramid = 0;
		m_MovingImagePyramid = 0;
		m_Interpolator = 0;
		m_Metric = 0;
		m_Optimizer = 0;
		m_Registration = 0;
		m_Resampler = 0;
		m_ResampleInterpolator = 0;
		m_Transform = 0;
		
		/** Initialize the components as pointers to elx...Base objects.*/
		m_elx_FixedImagePyramid = 0;
		m_elx_MovingImagePyramid = 0;
		m_elx_Interpolator = 0;
		m_elx_Metric = 0;
		m_elx_Optimizer = 0;
		m_elx_Registration = 0;
		m_elx_Resampler = 0;
		m_elx_ResampleInterpolator = 0;
		m_elx_Transform = 0;
		
		/** Initialize the Readers and Casters.*/
		m_FixedImageReader = 0;
		m_MovingImageReader = 0;
		m_FixedImageCaster = 0;
		m_MovingImageCaster = 0;
		
		/** Initialize m_InitialTransform.*/
		m_InitialTransform = 0;
		
		/** Initialize CallBack commands.*/
		m_BeforeEachResolutionCommand = 0;
		m_AfterEachIterationCommand = 0;

		/** Create timers */
		m_Timer0 = TimerType::New();
		m_IterationTimer = TimerType::New();
		m_ResolutionTimer = TimerType::New();

		/** Initialize the m_IterationCounter.*/
		m_IterationCounter = 0;
		
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
		/** Cast DataObjectType to FixedImageType and assign to m_FixedImage.*/
		if ( m_FixedImage != _arg )
		{
			m_FixedImage = dynamic_cast<FixedImageType *>( _arg );
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
		/** Cast DataObjectType to MovingImageType and assign to m_MovingImage.*/
		if ( m_MovingImage != _arg )
		{
			m_MovingImage = dynamic_cast<MovingImageType *>( _arg );
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
		/** Cast DataObjectType to FixedInternalImageType and assign to m_FixedInternalImage.*/
		if ( m_FixedInternalImage != _arg )
		{
			m_FixedInternalImage = dynamic_cast<FixedInternalImageType *>( _arg );
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
		/** Cast DataObjectType to MovingInternalImageType and assign to m_MovingInternalImage.*/
		if ( m_MovingInternalImage != _arg )
		{
			m_MovingInternalImage = dynamic_cast<MovingInternalImageType *>( _arg );
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
		/** Tell all components where to find the ElastixTemplate.*/
		m_elx_Registration->SetElastix(this);
		m_elx_Transform->SetElastix(this);
		m_elx_Metric->SetElastix(this);
		m_elx_Interpolator->SetElastix(this);
		m_elx_Optimizer->SetElastix(this);
		m_elx_FixedImagePyramid->SetElastix(this);
		m_elx_MovingImagePyramid->SetElastix(this);
		m_elx_Resampler->SetElastix(this);
		m_elx_ResampleInterpolator->SetElastix(this);


		/** Call BeforeAll to do some checking.*/
		int dummy = this->BeforeAll();
		if ( dummy != 0 ) return dummy;

		/** Setup Callbacks. This makes sure that the BeforeEachResolution()
		 * and AfterEachIteration() functions are called.
		 */
		m_BeforeEachResolutionCommand = BeforeEachResolutionCommandType::New();
		m_AfterEachResolutionCommand = AfterEachResolutionCommandType::New();
		m_AfterEachIterationCommand = AfterEachIterationCommandType::New();
		
		m_BeforeEachResolutionCommand->SetCallbackFunction( this, &Self::BeforeEachResolution );
		m_AfterEachResolutionCommand->SetCallbackFunction( this, &Self::AfterEachResolution );
		m_AfterEachIterationCommand->SetCallbackFunction( this, &Self::AfterEachIteration );
		
		m_Registration->AddObserver( itk::IterationEvent(), m_BeforeEachResolutionCommand );
		m_Optimizer->AddObserver( itk::IterationEvent(), m_AfterEachIterationCommand );
		m_Optimizer->AddObserver( itk::EndEvent(), m_AfterEachResolutionCommand );
	

		/** Start the timer for reading images. */
		m_Timer0->StartTimer();
		elxout << "\nReading images..." << std::endl;

		/** \todo Multithreaden? Reading the fixed and moving images could be two threads. */

		/** Set the fixedImage.*/
		if ( !m_FixedImage )
		{
			m_FixedImageReader = FixedImageReaderType::New();
			m_FixedImageReader->SetFileName(
				this->GetConfiguration()->GetCommandLineArgument( "-f" )  );
			m_FixedImage = m_FixedImageReader->GetOutput();

			/** Do the reading. */
			try
			{
				m_FixedImage->Update();
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

		if ( !m_FixedInternalImage )
		{
			m_FixedImageCaster = FixedImageCasterType::New();
			m_FixedImageCaster->SetInput( m_FixedImage );
			m_FixedInternalImage = m_FixedImageCaster->GetOutput();
			/** Do the casting.*/
			try
			{
				m_FixedInternalImage->Update();
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

		m_elx_Registration->GetAsITKBaseType()->
			SetFixedImage( m_FixedInternalImage );		

		
		/** Set the movingImage.*/
		if ( !m_MovingImage )
		{
			m_MovingImageReader = MovingImageReaderType::New();
			m_MovingImageReader->SetFileName(
				this->GetConfiguration()->GetCommandLineArgument( "-m" )  );
			m_MovingImage = m_MovingImageReader->GetOutput();
			/** Do the reading.*/
			try
			{
				m_MovingImage->Update();
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

		if ( !m_MovingInternalImage )
		{
			m_MovingImageCaster = MovingImageCasterType::New();
			m_MovingImageCaster->SetInput( m_MovingImage );
			m_MovingInternalImage = m_MovingImageCaster->GetOutput();
			/** Do the casting.*/
			try
			{
				m_MovingInternalImage->Update();
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

		m_elx_Registration->GetAsITKBaseType()->
			SetMovingImage( m_MovingInternalImage );
		
		/** Print the time spent on reading images */
		m_Timer0->StopTimer();
		elxout << "Reading images took " <<	static_cast<unsigned long>(
			m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n" << std::endl;

		/** Give all components the opportunity to do some initialization. */
		this->BeforeRegistration();
	
		/** START! */
		try
		{
			( m_elx_Registration->GetAsITKBaseType() )->StartRegistration();
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
		
		/** Save, show results etc.*/
		this->AfterRegistration();
		
		return 0;
		
	} // end Run
	
	
	/**
	 * ************************ ApplyTransform **********************
	 */
	
	template <class TFixedImage, class TMovingImage>
	int ElastixTemplate<TFixedImage, TMovingImage>
	::ApplyTransform(void)
	{
		/** Tell all components where to find the ElastixTemplate.*/
		m_elx_Transform->SetElastix(this);
		m_elx_Resampler->SetElastix(this);
		m_elx_ResampleInterpolator->SetElastix(this);

		/** Call BeforeAllTransformix to do some checking.*/
		int dummy = BeforeAllTransformix();
		if ( dummy != 0 ) return dummy;

		std::string inputImageFileName =
			this->GetConfiguration()->GetCommandLineArgument( "-in" );
		if ( inputImageFileName != "")
		{
			/** Tell the user. */
			elxout << std::endl << "Reading input image ...";
			
			/** Set the inputImage == movingImage. */
			typename InputImageReaderType::Pointer	inputImageReader;
			if ( !m_MovingImage )
			{
				inputImageReader = InputImageReaderType::New();
				inputImageReader->SetFileName(
					this->GetConfiguration()->GetCommandLineArgument( "-in" ) );
				m_MovingImage = inputImageReader->GetOutput();

				/** Do the reading. */
				try
				{
					m_MovingImage->Update();
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

			/** Tell the user.*/
			elxout << "\t\t\t\tdone!" << std::endl;
		} // end if ! inputImageFileName

		/** Call all the ReadFromFile() functions. */
		elxout << "Calling all ReadFromFile()'s ...";
		m_elx_ResampleInterpolator->ReadFromFile();		
		m_elx_Resampler->ReadFromFile();
		m_elx_Transform->ReadFromFile();

		/** Tell the user. */
		elxout << "\t\tdone!" << std::endl;
		elxout << "Transforming points (if called for) ...";

		/** Call TransformPoints. */
		try
		{
      m_elx_Transform->TransformPoints();
		}
		catch( itk::ExceptionObject & excp )
		{
			xout["error"] << excp << std::endl;
			xout["error"] << "However, transformix continues anyway with resampling." << std::endl;
		}
		elxout << "\t\tdone!" << std::endl;

		/** Resample the image. */
		if (inputImageFileName != "")
		{
			elxout << "Resampling image and writing to disk ...";
			
			/** Create a name for the final result. */
			std::ostringstream makeFileName("");
			makeFileName << 
				m_Configuration->GetCommandLineArgument( "-out" ) << "result.mhd";
			
			/** Write the resampled image to disk. */
			typename OutputImageWriterType::Pointer writer = OutputImageWriterType::New();		
			writer->SetInput( m_elx_Resampler->GetAsITKBaseType()->GetOutput() );
			writer->SetFileName( makeFileName.str().c_str() );

			/** Do the writing. */
			try
			{
				writer->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "ElastixTemplate - ApplyTransform()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while writing resampled image.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
			
			/** Tell the user. */
			elxout << "\tdone!" << std::endl;
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
		/** Declare the return value and initialize it.*/
		int returndummy = 0;
		
		/** Call all the BeforeRegistration() functions.*/
		returndummy |= this->BeforeAllBase();
		returndummy |= CallInEachComponentInt( &BaseComponentType::BeforeAllBase );
		returndummy |= CallInEachComponentInt( &BaseComponentType::BeforeAll );

		/** Return a value.*/
		return returndummy;
		
	} // end BeforeAll


	/**
	 * ******************** BeforeAllTransformix ********************
	 */
	
	template <class TFixedImage, class TMovingImage>
	int ElastixTemplate<TFixedImage, TMovingImage>
	::BeforeAllTransformix(void)
	{
		/** Declare the return value and initialize it.*/
		int returndummy = 0;

		/** Print to log file.*/
		elxout << "ELASTIX version: " << __ELASTIX_VERSION << std::endl;

		/** Check Command line options and print them to the logfile.*/
		elxout << "Command line options:" << std::endl;
		std::string check = "";

		/** Check for appearance of "-in".*/
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

		/** Check for appearance of "-out".*/
		check = this->GetConfiguration()->GetCommandLineArgument( "-out" );
		if ( check == "" )
		{
			xl::xout["error"] << "ERROR: No CommandLine option \"-out\" given!" << std::endl;
			returndummy |= -1;
		}
		else
		{
			/** Make sure that last character of -out equals a '/'.*/
			std::string folder( check );
			if ( folder.find_last_of( "/" ) != folder.size() - 1 )
			{
				folder.append( "/" );
				this->GetConfiguration()->SetCommandLineArgument( "-out", folder.c_str() );
			}
			elxout << "-out\t\t" << check << std::endl;
		}		

		/** Print "-tp".*/
		check = this->GetConfiguration()->GetCommandLineArgument( "-tp" );
		elxout << "-tp\t\t" << check << std::endl;

		/** Call all the BeforeAllTransformix() functions.*/
		returndummy |= m_elx_ResampleInterpolator->BeforeAllTransformix();		
		returndummy |= m_elx_Resampler->BeforeAllTransformix();
		returndummy |= m_elx_Transform->BeforeAllTransformix();

		/** Return a value.*/
		return returndummy;

	} // end BeforeAllTransformix

	
	/**
	 * **************** BeforeRegistration Callback *****************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::BeforeRegistration(void)
	{
		/** Start timer for initializing all components.*/
		m_Timer0->StartTimer();
		
		/** Call all the BeforeRegistration() functions.*/
		this->BeforeRegistrationBase();
		CallInEachComponent( &BaseComponentType::BeforeRegistrationBase );
		CallInEachComponent( &BaseComponentType::BeforeRegistration );

		/** Add a column to iteration with timing information.*/
		xout["iteration"].AddTargetCell("Time[ms]");

		/** Print time for initializing.*/
		m_Timer0->StopTimer();
		elxout << "Initialization of all components (before registration) took: "
			<< static_cast<unsigned long>( m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n";

		/** Start Timer0 here, to make it possible to measure the time needed for 
		 * preparation of the first resolution.
		 */
		m_Timer0->StartTimer();

	} // end BeforeRegistration Callback


	/**
	 * ************** BeforeEachResolution Callback *****************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::BeforeEachResolution(void)
	{
		/** Get current resolution level.*/
		unsigned long level =
			m_elx_Registration->GetAsITKBaseType()->GetCurrentLevel();

		if ( level == 0 )
		{
			m_Timer0->StopTimer();
			elxout << "Preparation of the image pyramids took: "
				<< static_cast<unsigned long>( m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n";
			m_Timer0->StartTimer();
		}
		
		/** Reset the m_IterationCounter.*/
		m_IterationCounter = 0;


		/** Print the current resolution */
		elxout << "\nResolution: " <<	level	<< std::endl;

		this->OpenIterationInfoFile();

		/**
		 * Call all the BeforeEachResolution() functions.
		 */
		this->BeforeEachResolutionBase();
		CallInEachComponent( &BaseComponentType::BeforeEachResolutionBase );
		CallInEachComponent( &BaseComponentType::BeforeEachResolution );
				
		/** Print the extra preparation time needed for this resolution. */
		m_Timer0->StopTimer();
		elxout << "Elastix initialization of all components (for this resolution) took: "
			<< static_cast<unsigned long>( m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n";

		/** Start ResolutionTimer, which measures the total iteration time in this resolution */
		m_ResolutionTimer->StartTimer();

		/** Start IterationTimer here, to make it possible to measure the time
		 * of the first iteration */
		m_IterationTimer->StartTimer();


	} // end BeforeEachResolution Callback
	
	/**
	 * ************** AfterEachResolution Callback *****************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::AfterEachResolution(void)
	{

		/** Get current resolution level.*/
		unsigned long level =
			m_elx_Registration->GetAsITKBaseType()->GetCurrentLevel();
		
		/** Print the total iteration time */
		elxout << std::setprecision(3);
		m_ResolutionTimer->StopTimer();
		elxout 
			<< "Time spent in resolution " 
			<< ( level )
			<< " (ITK initialisation and iterating): "
			<< m_ResolutionTimer->GetElapsedClockSec()
			<< " s.\n";
		elxout << std::setprecision( this->GetDefaultOutputPrecision() );
		
		/**
		 * Call all the AfterEachResolution() functions.
		 */
		this->AfterEachResolutionBase();
		CallInEachComponent( &BaseComponentType::AfterEachResolutionBase );
		CallInEachComponent( &BaseComponentType::AfterEachResolution );

		/** Start Timer0 here, to make it possible to measure the time needed for:
		 *    - executing the BeforeEachResolution methods (if this was not the last resolution)
		 *		- executing the AfterRegistration methods (if this was the last resolution)
		 */
		m_Timer0->StartTimer();
	
	} // end AfterEachResolution Callback
	

	/**
	 * ************** AfterEachIteration Callback *******************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::AfterEachIteration(void)
	{
		/** Write the headers of the colums that are printed each iteration.*/
		if (m_IterationCounter==0)
		{
			xout["iteration"]["WriteHeaders"];
		}

		/** Call all the AfterEachIteration() functions.
		 */
		this->AfterEachIterationBase();
		CallInEachComponent( &BaseComponentType::AfterEachIterationBase );
		CallInEachComponent( &BaseComponentType::AfterEachIteration );

		/** Time in this iteration.*/
		m_IterationTimer->StopTimer();
		xout["iteration"]["Time[ms]"]
			<< static_cast<unsigned long>( m_IterationTimer->GetElapsedClockSec() *1000 );

		/** Write the iteration info of this iteration */
		xout["iteration"].WriteBufferedData();

		std::string TranParOption;
		m_Configuration->ReadParameter( TranParOption, "WriteTransformParametersEachIteration", 0, true );
		if ( TranParOption == "true" )
		{
			/** Add zeros to the number of iterations, to make sure 
			 * it always consists of 7 digits
			 */
			std::ostringstream makeIterationString("");
			unsigned int border = 1000000;
			while (border > 1)
			{
				if (m_IterationCounter < border )
				{
					makeIterationString << "0";
					border /= 10;
				}
				else
				{
					/** stop */
					border=1;
				}
			}
			makeIterationString << m_IterationCounter;

			/** Create the TransformParameters filename for this iteration.*/
			std::ostringstream makeFileName("");
			makeFileName << m_Configuration->GetCommandLineArgument( "-out" )
				<< "TransformParameters."
				<< m_Configuration->GetElastixLevel()
				<< ".R" << m_elx_Registration->GetAsITKBaseType()->GetCurrentLevel()
				<< ".It" << makeIterationString.str()
				<< ".txt";
			std::string FileName = makeFileName.str();
			
			/** Create a TransformParameterFile for this iteration.*/
			this->CreateTransformParameterFile( FileName, false );
		}
		
		/** Count the number of iterations */
		m_IterationCounter++;

		/** Start timer for next iteration*/
		m_IterationTimer->StartTimer();
		
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
		
		/** Create the final TransformParameters filename.*/
		std::ostringstream makeFileName("");
		makeFileName << m_Configuration->GetCommandLineArgument( "-out" )
			<< "TransformParameters." << 
			m_Configuration->GetElastixLevel() << ".txt";
		std::string FileName = makeFileName.str();

		/** Create a final TransformParameterFile.*/
		this->CreateTransformParameterFile( FileName, true );

		/**
		 * Call all the AfterRegistration() functions.
		 */
		this->AfterRegistrationBase();
		CallInEachComponent( &BaseComponentType::AfterRegistrationBase );
		CallInEachComponent( &BaseComponentType::AfterRegistration );
		
		/** Print the time spent on things after the registration. */
		m_Timer0->StopTimer();
		elxout << "Time spent on saving the results, applying the final transform etc.: "
			<< static_cast<unsigned long>( m_Timer0->GetElapsedClockSec() * 1000 ) << " ms.\n";
		
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

		/** Create transformParameterFile and xout["transpar"].*/
		xoutsimple_type		transformationParameterInfo;
		std::ofstream					transformParameterFile;
		
		/** Set up the "TransformationParameters" writing field.*/
		transformationParameterInfo.SetOutputs( xout.GetCOutputs() );
		transformationParameterInfo.SetOutputs( xout.GetXOutputs() );
		
		xout.AddTargetCell( "transpar", &transformationParameterInfo );
		
		/** Set it in the Transform, for later use.*/
		m_elx_Transform->SetTransformParametersFileName( FileName.c_str() );
		
		/** Open the TransformParameter file.*/
		transformParameterFile.open( FileName.c_str() );
		if ( !transformParameterFile.is_open() )
		{
			xout["error"] << "ERROR: File \"" << FileName << "\" could not be opened!" << std::endl;
		}
		
		/** This xout["transpar"] writes to the log and to the TransformParameter file.*/
		transformationParameterInfo.RemoveOutput( "cout" );
		transformationParameterInfo.AddOutput( "tpf", &transformParameterFile );
		if ( !ToLog )
		{
			transformationParameterInfo.RemoveOutput( "log" );
		}
		
		/** Format specifiers of the transformation parameter file */
		xout["transpar"] << std::showpoint;
		xout["transpar"] << std::fixed;
		xout["transpar"] << std::setprecision( this->GetDefaultOutputPrecision() );
		
		/** Separate clearly in log-file.*/
		if ( ToLog )
		{
			xout["logonly"] <<
				"\n=============== start of TransformParameterFile ==============="	<< std::endl;
		}

		/**
		* Call all the WriteToFile() functions.
		*/
		m_elx_Transform->WriteToFile(
			m_elx_Optimizer->GetAsITKBaseType()->GetCurrentPosition() );
		m_elx_ResampleInterpolator->WriteToFile();
		m_elx_Resampler->WriteToFile();
		
		/** Separate clearly in log-file.*/
		if ( ToLog )
		{
			xout["logonly"] << 
				"\n=============== end of TransformParameterFile ===============" << std::endl;
		}

		/** Remove the "transpar" writing field */
		xout.RemoveTargetCell( "transpar" );
		
	} // end CreateTransformParameterFile


	/**
	 * ****************** CallInEachComponent ***********************
	 */
	
	template <class TFixedImage, class TMovingImage>
	void ElastixTemplate<TFixedImage, TMovingImage>
	::CallInEachComponent( PtrToMemberFunction func )
	{
		/** Call the memberfunction 'func' of the m_elx_Components.*/
		( (    *( this->GetConfiguration() )    ).*func )();
		( (*m_elx_Registration).*func )();
		( (*m_elx_Transform).*func )();
		( (*m_elx_Metric).*func )();
		( (*m_elx_Interpolator).*func )();
		( (*m_elx_Optimizer).*func )();
		( (*m_elx_FixedImagePyramid).*func )();
		( (*m_elx_MovingImagePyramid).*func )();
		( (*m_elx_ResampleInterpolator).*func )();		
		( (*m_elx_Resampler).*func )();
		
	} // end CallInEachComponent
	

	/**
	 * ****************** CallInEachComponentInt ********************
	 */
	
	template <class TFixedImage, class TMovingImage>
	int ElastixTemplate<TFixedImage, TMovingImage>
	::CallInEachComponentInt( PtrToMemberFunction2 func )
	{
		/** Declare the return value and initialize it.*/
		int returndummy = 0;

		/** Call the memberfunction 'func' of the m_elx_Components.*/
		returndummy |= ( (    *( this->GetConfiguration() )    ).*func )();		
		returndummy |= ( (*m_elx_Registration).*func )();
		returndummy |= ( (*m_elx_Transform).*func )();
		returndummy |= ( (*m_elx_Metric).*func )();
		returndummy |= ( (*m_elx_Interpolator).*func )();
		returndummy |= ( (*m_elx_Optimizer).*func )();
		returndummy |= ( (*m_elx_FixedImagePyramid).*func )();
		returndummy |= ( (*m_elx_MovingImagePyramid).*func )();
		returndummy |= ( (*m_elx_ResampleInterpolator).*func )();		
		returndummy |= ( (*m_elx_Resampler).*func )();

		/** Return a value.*/
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
		
		/** Remove the current iteration info output file, if any */
		xout["iteration"].RemoveOutput( "IterationInfoFile" );
		
		if ( m_IterationInfoFile.is_open() )
		{
			m_IterationInfoFile.close();
		}

		/** Create the IterationInfo filename for this resolution.*/
			std::ostringstream makeFileName("");
			makeFileName << m_Configuration->GetCommandLineArgument( "-out" )
				<< "IterationInfo."
				<< m_Configuration->GetElastixLevel()
				<< ".R" << m_elx_Registration->GetAsITKBaseType()->GetCurrentLevel()
				<< ".txt";
			std::string FileName = makeFileName.str();

		/** Open the IterationInfoFile.*/
		m_IterationInfoFile.open( FileName.c_str() );
		if ( !m_IterationInfoFile.is_open() )
		{
			xout["error"] << "ERROR: File \"" << FileName << "\" could not be opened!" << std::endl;
		}
		else    
		{
			/** Add this file to the list of outputs of xout["iteration"] */
			xout["iteration"].AddOutput( "IterationInfoFile", &m_IterationInfoFile );
		}

	} //end of function penIterationInfoFile

} // end namespace elastix


#endif // end #ifndef __elxElastixTemplate_hxx

