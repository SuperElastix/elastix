#ifndef __elxTransformBase_hxx
#define __elxTransformBase_hxx

#include "elxTransformBase.h"

namespace elastix
{
	using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		TransformBase<TElastix>::TransformBase()
	{
		/** Initialize.*/
		m_TransformParametersPointer = 0;
		m_ConfigurationInitialTransform = 0;

	} // end Constructor


	/**
	 * ********************** Destructor ****************************
	 */

	template <class TElastix>
		TransformBase<TElastix>::~TransformBase()
	{
		/** Delete.*/
		delete m_TransformParametersPointer;

	} // end Destructor
	

	/**
	 * ******************** BeforeAllBase ***************************
	 */
	
	template <class TElastix>
		int TransformBase<TElastix>
		::BeforeAllBase(void)
	{
		/** Check Command line options and print them to the logfile.*/
		elxout << "Command line options:" << std::endl;
		std::string check("");
		
		/** Check for appearance of "-t0".*/
		check = m_Configuration->GetCommandLineArgument( "-t0" );
		if ( check.empty() )
		{
			elxout << "-t0\t\tunspecified, so no initial transform used" << std::endl;
		}
		else
		{
			elxout << "-t0\t\t" << check << std::endl;
		}

		return 0;

	} // end BeforeAllBase	


	/**
	 * ******************** BeforeAllTransformix ********************
	 */
	
	template <class TElastix>
		int TransformBase<TElastix>
		::BeforeAllTransformix(void)
	{
		/** Declare the return value and initialize it.*/
		int returndummy = 0;

		/** Declare check.*/
		std::string check = "";

		/** Check for appearance of "-ipp".*/
		check = m_Configuration->GetCommandLineArgument( "-ipp" );
		if ( check == "" )
		{
			elxout << "-ipp\t\tunspecified, so no inputpoints transformed" << std::endl;
		}
		else
		{
			elxout << "-ipp\t\t" << check << std::endl;
		}

		/** Return a value.*/
		return returndummy;

	} // end BeforeAllTransformix


	/**
	 * ******************* BeforeRegistrationBase *******************
	 */

	template <class TElastix>
		void TransformBase<TElastix>::BeforeRegistrationBase(void)
	{	
		/** Read from the configuration file how to combine the initial
		* transform with the current transform.
		*/
		std::string howToCombineTransforms = "Add"; //default
		m_Configuration->ReadParameter( howToCombineTransforms, "HowToCombineTransforms", 0, true );

		/***/
		TransformGrouperInterface * thisAsGrouper = 
			dynamic_cast< TransformGrouperInterface * >(this);
		if ( thisAsGrouper )
		{
			thisAsGrouper->SetGrouper( howToCombineTransforms );
		}

		/***/
		if ( m_Elastix->GetInitialTransform() )
		{
			this->SetInitialTransform( m_Elastix->GetInitialTransform() );
		}
		else
		{
			std::string fileName =  m_Configuration->GetCommandLineArgument( "-t0" );
			if ( !fileName.empty() )
			{
				this->ReadInitialTransformFromFile(	fileName.c_str() );
			}
		}
		
	} // end BeforeRegistrationBase


	/**
	 * ******************* GetInitialTransform **********************
	 */

	template <class TElastix>
		typename TransformBase<TElastix>::ObjectType * 
		TransformBase<TElastix>::GetInitialTransform(void)
	{
		/***/
		TransformGrouperInterface * thisAsGrouper = 
			dynamic_cast< TransformGrouperInterface * >(this);

		/***/
		if ( thisAsGrouper )
		{
			return thisAsGrouper->GetInitialTransform();
		}
		else
		{
			return 0;
		}

	} // end GetInitialTransform


	/**
	 * ******************* SetInitialTransform **********************
	 */

	template <class TElastix>
		void TransformBase<TElastix>::SetInitialTransform( ObjectType * _arg )
	{
		/***/
		TransformGrouperInterface * thisAsGrouper = 
			dynamic_cast< TransformGrouperInterface * >(this);

		/***/
		if ( thisAsGrouper )
		{
			thisAsGrouper->SetInitialTransform( _arg );
		}

	} // end SetInitialTransform


	/**
	 * ******************* AfterRegistrationBase ********************
	 */

	template <class TElastix>
		void TransformBase<TElastix>::AfterRegistrationBase(void)
	{
		/** Get the final Parameters.*/
		ParametersType finalParameters = m_Registration->GetAsITKBaseType()->GetLastTransformParameters();

		/** Set the final Parameters for the resampler.*/
		this->GetAsITKBaseType()->SetParameters( finalParameters );

	} // end AfterRegistrationBase


	/**
	 * ******************* ReadFromFile *****************************
	 */

	template <class TElastix>
		void TransformBase<TElastix>::ReadFromFile(void)
	{
		/** 
		 * This method assumes m_Configuration is initialised with a
		 * transformparameterfile, so not an elastix parameter file!!
		 */
						
		/** Task 1 - Read the parameters from file.*/

		/** Get the number of TransformParameters.*/
		unsigned int NumberOfParameters = 0;
		m_Configuration->ReadParameter( NumberOfParameters, "NumberOfParameters", 0 );

		/** Get the TransformParameters.*/
		if ( m_TransformParametersPointer ) delete m_TransformParametersPointer;
		m_TransformParametersPointer = new ParametersType( NumberOfParameters );
		/** If NumberOfParameters < 20, we read in the normal way.*/
		if ( NumberOfParameters < 20 )
		{			
			for ( unsigned int i = 0; i < NumberOfParameters; i++ )
			{
				m_Configuration->ReadParameter(
					(*m_TransformParametersPointer)[ i ], "TransformParameters", i );
			}
		}
		/** Else, do the reading more 'manually'.
		 * This is neccesary, because the ReadParameter can not handle
		 * many parameters.
		 */
		else
		{
			std::ifstream input( this->GetConfiguration()->GetCommandLineArgument( "-tp" ) );
			if ( input.is_open() )
			{
				/** p the getline->streamsize are set to have length 1000.
				 * BEWARE: if somewhere before TransformParameters, there
				 * exists a line with more then 1000 characters, the
				 * getline can not get beyond that line, and this will FAIL.
				 * It can be 'solved' by setting 1000 to a larger number.
				 */
				char p[ 1000 ];
				bool nextline = true;				
				while ( nextline )
				{
					/** Get a line, put it in a string, and compare it
					 * with "// (TransformParameters)".
					 */
					input.getline( p, 1000 );
					std::string ps( p );
					int pos = ps.find( "// (TransformParameters)" );
					/** If we have correspondence, we are at the right line.*/
					if ( pos == 0 )
					{
						/** First read the "// " into tmp.*/
						std::string tmp;
						input >> tmp;
						/** Now read the TransformParameters.*/
						for ( unsigned int i = 0; i < NumberOfParameters; i++ )
						{
							input >> (*m_TransformParametersPointer)[ i ];
						}
						/** We are done, so no need for reading more lines.*/
						nextline = false;
					} // end if at the right position
				} // end while reading lines
			} // end if input-file is open
		} // end else

		/** Set the parameters into this transform.*/
		this->GetAsITKBaseType()->SetParameters( *m_TransformParametersPointer );

		/** Task 2 - Get the InitialTransform.*/

		/** Get the InitialTransformName. */
		std::string fileName = "";
		m_Configuration->ReadParameter( fileName,
			"InitialTransformParametersFileName", 0 );
		
		/** Call the function ReadInitialTransformFromFile.*/
		if ( fileName != "NoInitialTransform" )
		{			
			this->ReadInitialTransformFromFile( fileName.c_str() );
		} 

		/** Task 3 - Read from the configuration file how to combine the
		 * initial transform with the current transform.
		 */
		std::string howToCombineTransforms = "Add"; // default
		m_Configuration->ReadParameter( howToCombineTransforms, "HowToCombineTransforms", 0, true );
		
		/** Convert 'this' to a pointer to a TransformGrouperInterface and set how
		 * to combine the current transform with the initial transform */
		TransformGrouperInterface * thisAsGrouper = 
			dynamic_cast< TransformGrouperInterface * >(this);
		if ( thisAsGrouper )
		{
			thisAsGrouper->SetGrouper( howToCombineTransforms );
		}		

		/** Task 4 - Remember the name of the TransformParametersFileName.
		 * This will be needed when another transform will use this transform
		 * as an initial transform (see the WriteToFile method)
		 */
		this->SetTransformParametersFileName(
			this->GetConfiguration()->GetCommandLineArgument( "-tp" ) );
		
	} // end ReadFromFile


	/**
	 * ******************* ReadInitialTransformFromFile *************
	 */

	template <class TElastix>
		void TransformBase<TElastix>::ReadInitialTransformFromFile(
			const char * transformParametersFileName)
	{
		/** Create a new configuration, which will be initialised with
		 * the transformParameterFileName.
		 */
		if ( !m_ConfigurationInitialTransform )
		{
			m_ConfigurationInitialTransform = ConfigurationType::New();
		}
		
		/** Create argmapInitialTransform.*/
		ArgumentMapType argmapInitialTransform;
		argmapInitialTransform.insert( ArgumentMapEntryType(
			"-tp", transformParametersFileName ) );
		
		int dummy = m_ConfigurationInitialTransform->Initialize( argmapInitialTransform );
		
		/** Read the InitialTransform name.*/
		ComponentDescriptionType InitialTransformName = "FixedCenterOfRotationAffineTransform";
		m_ConfigurationInitialTransform->ReadParameter( InitialTransformName, "Transform", 0 );
		
		/** Create an InitialTransform.*/
		ObjectType::Pointer initialTransform;
		
		PtrToCreator testcreator = 0;
		testcreator = this->GetElastix()->GetComponentDatabase()->
			GetCreator( InitialTransformName, m_Elastix->GetDBIndex() );
		initialTransform = testcreator ? testcreator() : NULL;
		
		Self * elx_initialTransform = dynamic_cast< Self * >( initialTransform.GetPointer() );			
		
		/** Call the ReadFromFile method of the initialTransform. */
		if ( elx_initialTransform )
		{
			//elx_initialTransform->SetTransformParametersFileName(transformParametersFileName);
			elx_initialTransform->SetElastix( this->GetElastix() );
			elx_initialTransform->SetConfiguration( m_ConfigurationInitialTransform );			
			elx_initialTransform->ReadFromFile();
		
			/** Set initial transform.*/
			this->SetInitialTransform( initialTransform );

		} // end if

	} // end ReadInitialTransformFromFile


	/**
	 * ******************* WriteToFile ******************************
	 */

	template <class TElastix>
		void TransformBase<TElastix>
		::WriteToFile(void)
	{
		/** Write the currently set parameters to file.*/
		this->WriteToFile( this->GetAsITKBaseType()->GetParameters() );
 
	} // end WriteToFile


	/**
	 * ******************* WriteToFile ******************************
	 */

	template <class TElastix>
		void TransformBase<TElastix>
		::WriteToFile( const ParametersType & param )
	{
		using namespace xl;

		/** Write the name of this transform.*/
		xout["transpar"] << "(Transform \""
			<< this->elxGetClassName() << "\")" << std::endl;

		/** Get the number of parameters of this transform.*/
		unsigned int nrP = m_Registration->GetAsITKBaseType()->GetTransform()
			->GetNumberOfParameters();

		/** Write the number of parameters of this transform.*/
		xout["transpar"] << "(NumberOfParameters "
			<< nrP << ")" << std::endl;

		/** Write the parameters of this transform.*/
		if ( nrP < 20 )
		{
			/** In this case, write in a normal way to the Parameterfile.*/
			xout["transpar"] << "(TransformParameters ";
			for ( unsigned int i = 0; i < nrP - 1; i++ )
			{
				xout["transpar"] << param[ i ] << " ";
			}
			xout["transpar"] << param[ nrP - 1 ] << ")" << std::endl;
		}
		else
		{
			/** Otherwise, write to Parameterfile with "// " in front of it.
			 * This is neccesary, because the ReadParameter can not handle
			 * many parameters.
			 */
			xout["transpar"] << "// (TransformParameters)" << std::endl << "// ";
			for ( unsigned int i = 0; i < nrP - 1; i++ )
			{
				xout["transpar"] << param[ i ] << " ";
			}
			xout["transpar"] << std::endl;
		}

		/** Write the name of the parameters-file of the initial transform.*/
		if ( this->GetInitialTransform() )
		{
			xout["transpar"] << "(InitialTransformParametersFileName \""
				<< (dynamic_cast<Self *>( this->GetInitialTransform() ))
				->GetTransformParametersFileName() << "\")" << std::endl;
		}
		else
		{
			xout["transpar"] << "(InitialTransformParametersFileName \"NoInitialTransform\")"
				<< std::endl;
		}
		
		/** Write the way Transforms are combined.*/
		xout["transpar"] << "(HowToCombineTransforms \""
			<< (dynamic_cast< TransformGrouperInterface * >(this))
			->GetNameOfDesiredGrouper() << "\")" << std::endl;

		/** Write image specific things.*/
		xout["transpar"] << std::endl << "// Image specific" << std::endl;

		/** Write image dimensions.*/
		unsigned int FixDim = FixedImageDimension;
		unsigned int MovDim = MovingImageDimension;
		xout["transpar"] << "(FixedImageDimension "
			<< FixDim << ")" << std::endl;
		xout["transpar"] << "(MovingImageDimension "
			<< MovDim << ")" << std::endl;

		/** Write image pixel types.*/
		std::string fixpix, movpix;
		m_Configuration->ReadParameter( fixpix, "FixedImagePixelType", 0 );
		m_Configuration->ReadParameter( movpix, "MovingImagePixelType", 0 );
		xout["transpar"] << "(FixedImagePixelType \""	<< fixpix << "\")" << std::endl;
		xout["transpar"] << "(MovingImagePixelType \""	<< movpix << "\")" << std::endl;

		/** Get the Size, Spacing and Origin of the fixed image.*/
		/** \todo we get it now from the resampler, but maybe from an inputimage?? */
		SizeType size = dynamic_cast<typename ElastixType::FixedImageType *>(
			m_Elastix->GetFixedImage() )->GetLargestPossibleRegion().GetSize();
		IndexType index = dynamic_cast<typename ElastixType::FixedImageType *>(
			m_Elastix->GetFixedImage() )->GetLargestPossibleRegion().GetIndex();
		SpacingType spacing = dynamic_cast<typename ElastixType::FixedImageType *>(
			m_Elastix->GetFixedImage() )->GetSpacing();
		OriginType origin = dynamic_cast<typename ElastixType::FixedImageType *>(
			m_Elastix->GetFixedImage() )->GetOrigin();

		/** Write image Size.*/
		xout["transpar"] << "(Size ";
		for ( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
		{
			xout["transpar"] << size[ i ] << " ";
		}
		xout["transpar"] << size[ FixedImageDimension - 1 ] << ")" << std::endl;

		/** Write image Index.*/
		xout["transpar"] << "(Index ";
		for ( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
		{
			xout["transpar"] << index[ i ] << " ";
		}
		xout["transpar"] << index[ FixedImageDimension - 1 ] << ")" << std::endl;

		/** Set the precision of cout to 2, because Spacing and
		 * Origin must have at least one digit precision.
		 */
		xout["transpar"] << std::setprecision(1);

		/** Write image Spacing.*/
		xout["transpar"] << "(Spacing ";
		for ( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
		{
			xout["transpar"] << spacing[ i ] << " ";
		}
		xout["transpar"] << spacing[ FixedImageDimension - 1 ] << ")" << std::endl;

		/** Write image Origin.*/
		xout["transpar"] << "(Origin ";
		for ( unsigned int i = 0; i < FixedImageDimension - 1; i++ )
		{
			xout["transpar"] << origin[ i ] << " ";
		}
		xout["transpar"] << origin[ FixedImageDimension - 1 ] << ")" << std::endl;

		/** Set the precision back to default value.*/
		xout["transpar"] << std::setprecision( m_Elastix->GetDefaultOutputPrecision() );

	} // end WriteToFile


	/**
	 * ******************* TransformPoints **************************
	 *
	 * This function reads points from a file (but only if requested)
	 * and transforms these fixed-image coordinates to moving-image
	 * coordinates.
	 */

	template <class TElastix>
		void TransformBase<TElastix>
		::TransformPoints(void)
	{
		/** If the optional command "-ipp" is given in the command
		 * line arguments, then and only then we continue.
		 */
		std::string ipp = this->GetConfiguration()->GetCommandLineArgument( "-ipp" );

		/** If there is an inputpoint-file?*/
		if ( ipp != "" && ipp != "all" )
		{
			this->TransformPointsSomePoints( ipp );
		}
		else if ( ipp == "all" )
		{
			this->TransformPointsAllPoints();
		}
		else
		{
			// nothing but maybe a message
		}

	} // end TransformPoints


	/**
	 * ************** TransformPointsSomePoints *********************
	 *
	 * This function reads points from a file and transforms
	 * these fixed-image coordinates to moving-image
	 * coordinates.
	 */

	template <class TElastix>
		void TransformBase<TElastix>
		::TransformPointsSomePoints( std::string filename )
	{
		/** Open the file containing the inputpoints.*/
		std::ifstream pointfile( filename.c_str() );
		unsigned int nrofpoints;
		std::vector< IndexType > inputvec;			
		if ( pointfile.is_open() )
		{
			/** Read the inputpoints from a text file.*/
			pointfile >> nrofpoints;
			inputvec.resize( nrofpoints );
			for ( unsigned int j = 0; j < nrofpoints; j++ )
			{
				for ( unsigned int i = 0; i < MovingImageDimension; i++ )
				{
					pointfile >> inputvec[ j ][ i ];
				}
			}
			
			/** Calculate the transformed position of the inputpoints.*/
			std::vector< InputPointType > inputpointvec( nrofpoints );
			std::vector< OutputPointType > outputpointvec( nrofpoints );
			std::vector< IndexType > outputvec( nrofpoints );
			std::vector< VectorType > deformationvec( nrofpoints );
			
			/** Make a temporary image with the right region info,
			* so that the TransformIndexToPhysicalPoint-functions will be right.
			*/
			typename DummyImageType::Pointer dummyImage = DummyImageType::New();
			RegionType region;
			
			region.SetIndex( m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputStartIndex() );
			region.SetSize( m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetSize() );
			dummyImage->SetRegions( region );
			dummyImage->SetOrigin( m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputOrigin() );
			dummyImage->SetSpacing( m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputSpacing() );
			
			/** Create filename and filestream.*/
			std::string outputPointsFileName = m_Configuration->GetCommandLineArgument( "-out" );
			outputPointsFileName += "outputpoints.txt";
			std::ofstream outputPointsFile( outputPointsFileName.c_str() );
			outputPointsFile << std::showpoint << std::fixed;
			
			elxout << "Printing the output points."  << std::endl;
			
			elxout.AddOutput( "opp", &outputPointsFile );
			
			/** Transform the coordinates of the inputpoints (indexes),
			 * so that we get the coordinates of the outputpoints
			 */
			for ( unsigned int j = 0; j < nrofpoints; j++ )
			{
				/** Transform the points to physical space.*/
				dummyImage->TransformIndexToPhysicalPoint( inputvec[ j ], inputpointvec[ j ] );
				/** Call TransformPoint.*/
				outputpointvec[ j ] = this->GetAsITKBaseType()->TransformPoint( inputpointvec[ j ] );
				/** Transform back to index.*/
				dummyImage->TransformPhysicalPointToIndex( outputpointvec[ j ], outputvec[ j ] );					
				deformationvec[ j ].CastFrom( outputpointvec[ j ] - inputpointvec[ j ] );

				/** Print the results.*/					

				//the input index
				elxout << "Point\t" << j << "\t; InputIndex = [ "; 
				for ( unsigned int i = 0; i < MovingImageDimension; i++ )
				{
					elxout << inputvec[ j ][ i ] << " ";
				}

				//the input point
				elxout << "]\t; InputPoint = [ "; 
				for ( unsigned int i = 0; i < MovingImageDimension; i++ )
				{
					elxout << inputpointvec[ j ][ i ] << " ";
				}

				//the output index
				elxout << "]\t; OutputIndex = [ "; 
				for ( unsigned int i = 0; i < MovingImageDimension; i++ )
				{
					elxout << outputvec[ j ][ i ] << " ";
				}
				
				//the output point
				elxout << "]\t; OutputPoint = [ "; 
				for ( unsigned int i = 0; i < MovingImageDimension; i++ )
				{
					elxout << outputpointvec[ j ][ i ] << " ";
				}
				
				//the output point
				elxout << "]\t; Deformation = [ "; 
				for ( unsigned int i = 0; i < MovingImageDimension; i++ )
				{
					elxout << deformationvec[ j ][ i ] << " ";
				}

				elxout << "]" << std::endl;
			}	// end for nrofpoints	
			
			elxout.RemoveOutput( "opp" );
			
		} // end if - file is open
		else
		{
			xl::xout["warning"] << "WARNING: the file \"" << filename <<
				"\" could not be opened!" << std::endl;
		}
		
	} // end TransformPointsSomePoints
	
	
	/**
	 * ************** TransformPointsAllPoints **********************
	 *
	 * This function transforms all indexes to a physical point.
	 * The difference vector (= the deformation at that index) is
	 * stored in an image of vectors (of floats).
	 */
	
	template <class TElastix>
		void TransformBase<TElastix>
		::TransformPointsAllPoints(void)
	{
		/** First, make a dummyImage with the right region info, so
		 * that the TransformIndexToPhysicalPoint-functions will be right.
		 */
		typename DummyImageType::Pointer dummyImage = DummyImageType::New();
		RegionType region;
		OriginType origin = m_Elastix->GetElxResamplerBase()
			->GetAsITKBaseType()->GetOutputOrigin();
		SpacingType spacing = m_Elastix->GetElxResamplerBase()
			->GetAsITKBaseType()->GetOutputSpacing();
		
		region.SetIndex( m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputStartIndex() );
		region.SetSize( m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetSize() );
		dummyImage->SetRegions( region );
		dummyImage->SetOrigin( origin );
		dummyImage->SetSpacing( spacing );
		
		/** Setup an outputImage of vectors and allocate memory(!).*/
		OutputImagePointer outputImage = OutputImageType::New();
		outputImage->SetRegions( region );
		outputImage->SetOrigin( origin );
		outputImage->SetSpacing( spacing );
		outputImage->Allocate();
		
		/** Setup an iterator over dummyImage and outputImage.*/
		DummyIteratorType iter( dummyImage, region );
		OutputImageIteratorType iterout( outputImage, region );
		
		/** Declare stuff.*/
		InputPointType inputPoint;
		OutputPointType outputPoint;
		VectorType diff_point;
		IndexType inputIndex;
		
		/** Calculate the TransformPoint of all voxels of the image.*/
		iter.Begin();
		iterout.Begin();
		while ( !iter.IsAtEnd() )
		{
			inputIndex = iter.GetIndex();
			/** Transform the points to physical space.*/
			dummyImage->TransformIndexToPhysicalPoint( inputIndex, inputPoint );
			/** Call TransformPoint.*/
			outputPoint = this->GetAsITKBaseType()->TransformPoint( inputPoint );
			/** Calculate the difference.*/
			for ( unsigned int i = 0; i < FixedImageDimension; i++ )
			{
				diff_point[ i ] = outputPoint[ i ] - inputPoint[ i ];
			}
			iterout.Set( diff_point );
			++iter;
			++iterout;
		}
		
		/** Create a name for the offsetImage file.*/
		std::ostringstream makeFileName( "" );
		makeFileName << 
			m_Configuration->GetCommandLineArgument( "-out" ) << "deformationField.mhd";
		
		/** Write outputImage to disk.*/
		typename OutputFileWriterType::Pointer outputWriter
			= OutputFileWriterType::New();
		outputWriter->SetInput( outputImage );
		outputWriter->SetFileName( makeFileName.str().c_str() );
		
		/** Do the writing.*/
		try
		{
			outputWriter->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			xl::xout["error"] << excp << std::endl;
		}
		
	} // end TransformPointsAllPoints
	
	
	/**
	 * ************** SetTransformParametersFileName ****************
	 */
	
	template <class TElastix>
		void TransformBase<TElastix>
		::SetTransformParametersFileName( const char * filename )
	{
		/** Copied from itkSetStringMacro.*/
		if ( filename && ( filename == this->m_TransformParametersFileName )  )
		{
			return;
		}
		if ( filename )
		{
			this->m_TransformParametersFileName = filename;
		}
		else
		{
			this->m_TransformParametersFileName = "";
		}
		ObjectType * thisAsObject = dynamic_cast<ObjectType *>(this);
		thisAsObject->Modified();

	} // end SetTransformParametersFileName



} // end namespace elastix


#endif // end #ifndef __elxTransformBase_hxx

