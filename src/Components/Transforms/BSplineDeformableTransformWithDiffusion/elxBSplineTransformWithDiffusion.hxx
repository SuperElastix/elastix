#ifndef __elxBSplineTransformWithDiffusion_HXX__
#define __elxBSplineTransformWithDiffusion_HXX__

#include "elxBSplineTransformWithDiffusion.h"
#include "math.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		BSplineTransformWithDiffusion<TElastix>
		::BSplineTransformWithDiffusion()
	{
		/** Initialize.*/
		m_Coeffs1 = ImageType::New();
		m_Upsampler = UpsamplerType::New();
		m_Upsampler->SetSplineOrder(SplineOrder); 

		m_Caster	= TransformCastFilterType::New();
		m_Writer	= TransformWriterType::New();

		/** Initialize things for diffusion. */
		m_Diffusion = 0;
		m_DeformationField = 0;
		m_DiffusedField = 0;
		m_GrayValueImage = 0;
		m_Resampler = 0;

		/** Initialize to false. */
		m_WriteDiffusionFiles = false;
		m_DeformationFieldWriter = 0;
		m_DiffusedFieldWriter = 0;
		m_GrayValueImageWriter = 0;

	} // end Constructor
	

	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void BSplineTransformWithDiffusion<TElastix>
		::BeforeRegistration(void)
	{
		/** Set initial transform parameters to a 1x1x1 grid, with deformation (0,0,0).
		 * In the method BeforeEachResolution() this will be replaced by the right grid size.
		 *
		 * This seems not logical, but it is required, since the registration
		 * class checks if the number of parameters in the transform is equal to
		 * the number of parameters in the registration class. This check is done
		 * before calling the BeforeEachResolution() methods.
		 */
		
		/** Task 1 - Set the Grid.*/

		/** Declarations.*/
		RegionType gridregion;
		SizeType gridsize;
		IndexType gridindex;
		SpacingType gridspacing;
		OriginType gridorigin;
		
		/** Fill everything with default values.*/
		gridsize.Fill(1);
		gridindex.Fill(0);
		gridspacing.Fill(1.0);
		gridorigin.Fill(0.0);
		
		/** Set it all.*/
		gridregion.SetIndex(gridindex);
		gridregion.SetSize(gridsize);
		this->SetGridRegion( gridregion );
		this->SetGridSpacing( gridspacing );
		this->SetGridOrigin( gridorigin );
    
		/** Task 2 - Give the registration an initial parameter-array.*/
		ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
		dummyInitialParameters.Fill(0.0);
		
		/** Put parameters in the registration.*/
		m_Registration->GetAsITKBaseType()->SetInitialTransformParameters( dummyInitialParameters );

		/** This registration uses a diffusion of the deformation field
		 * every n-th iteration, the diffusion filter must be created.
		 * Also allocate m_DeformationField and m_DiffusedField.
		 */
		
		/** Get diffusion information: radius. */
		unsigned int radius1D;
		m_Configuration->ReadParameter( radius1D, "Radius", 0 );
		RadiusType radius;
		for ( unsigned int i = 0; i < FixedImageDimension; i++ )
		{
			radius[ i ] = static_cast<long unsigned int>( radius1D );
		}

		/** Get diffusion information: Number of iterations. */
		unsigned int iterations = 0;
		m_Configuration->ReadParameter( iterations, "NumberOfDiffusionIterations", 0 );
		if ( iterations < 1 )
		{
			xout["warning"] << "WARNING: NumberOfDiffusionIterations == 0" << std::endl;
		}

		/** Get diffusion information: threshold information. */
		std::string thresholdbooltmp = "";
		bool thresholdBool = false;
		m_Configuration->ReadParameter( thresholdbooltmp, "ThresholdBool", 0 );
		if ( thresholdbooltmp == "true" ) thresholdBool = true;

		double threshold = 0.0;
		m_Configuration->ReadParameter( threshold, "Threshold", 0 );

		/** Get the appropriate image information. */
		m_DeformationOrigin = m_Elastix->GetElxResamplerBase()
			->GetAsITKBaseType()->GetOutputOrigin();
		m_DeformationSpacing = m_Elastix->GetElxResamplerBase()
			->GetAsITKBaseType()->GetOutputSpacing();
		m_DeformationRegion.SetIndex( m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetOutputStartIndex() );
		m_DeformationRegion.SetSize( m_Elastix->GetElxResamplerBase()->GetAsITKBaseType()->GetSize() );
		
		/** Set it in the DeformationFieldRegulizer class. */
		this->SetDeformationFieldRegion( m_DeformationRegion );
		this->SetDeformationFieldOrigin( m_DeformationOrigin );
		this->SetDeformationFieldSpacing( m_DeformationSpacing );

		/** Initialize the m_IntermediaryDeformationFieldTransform,
		 * which is in the DeformationFieldRegulizer class.
		 */
		this->InitializeDeformationFields();

		/** Create m_DeformationField and allocate memory. */
		m_DeformationField = OutputImageType::New();
		m_DeformationField->SetRegions( m_DeformationRegion );
		m_DeformationField->SetOrigin( m_DeformationOrigin );
		m_DeformationField->SetSpacing( m_DeformationSpacing );
		m_DeformationField->Allocate();

		/** Create m_DeformationField and allocate memory. */
		m_DiffusedField = OutputImageType::New();
		m_DiffusedField->SetRegions( m_DeformationRegion );
		m_DiffusedField->SetOrigin( m_DeformationOrigin );
		m_DiffusedField->SetSpacing( m_DeformationSpacing );
		m_DiffusedField->Allocate();

		/** Create m_GrayValueImage and allocate memory. */
		m_GrayValueImage = GrayValueImageType::New();
		m_GrayValueImage->SetRegions( m_DeformationRegion );
		m_GrayValueImage->SetOrigin( m_DeformationOrigin );
		m_GrayValueImage->SetSpacing( m_DeformationSpacing );
		m_GrayValueImage->Allocate();

		/** Create a resampler. */
		m_Resampler = ResamplerType::New();
		m_Resampler->SetTransform( this->GetIntermediaryDeformationFieldTransform() );
		//m_Resampler->SetInterpolator(); // default = LinearInterpolateImageFunction
		m_Resampler->SetInput( dynamic_cast<FixedImageELXType *>(
			m_Elastix->GetMovingImage() ) );
		unsigned int defaultPixelValue = 0;
		m_Configuration->ReadParameter( defaultPixelValue, "DefaultPixelValue", 0 );
		m_Resampler->SetDefaultPixelValue( defaultPixelValue );
		m_Resampler->SetSize( m_DeformationRegion.GetSize() );
		m_Resampler->SetOutputStartIndex( m_DeformationRegion.GetIndex() );
		m_Resampler->SetOutputOrigin( m_DeformationOrigin );
		m_Resampler->SetOutputSpacing( m_DeformationSpacing );

		/** Find out if the user wants to write the diffusion files:
		 * deformationField, GrayvalueIMage, diffusedField.
		 */
		std::string writetofile;
		m_Configuration->ReadParameter( writetofile, "WriteDiffusionFiles", 0 );
		if ( writetofile == "true" )
		{
			m_WriteDiffusionFiles = true;
			m_DeformationFieldWriter = DeformationFieldWriterType::New();
			m_DiffusedFieldWriter = DeformationFieldWriterType::New();
			m_GrayValueImageWriter = GrayValueImageWriterType::New();
		}

		/** Create m_Diffusion. */
		m_Diffusion = DiffusionFilterType::New();
		m_Diffusion->SetRadius( radius );
		m_Diffusion->SetNumberOfIterations( iterations );
		m_Diffusion->SetUseThreshold( thresholdBool );
		m_Diffusion->SetThreshold( threshold );
		m_Diffusion->SetGrayValueImage( m_GrayValueImage );
		m_Diffusion->SetInput( m_DeformationField );

	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void BSplineTransformWithDiffusion<TElastix>
		::BeforeEachResolution(void)
	{
		/** What is the current resolution level?*/
		unsigned int level = m_Registration->GetAsITKBaseType()->GetCurrentLevel();

		/** What is the UpsampleGridOption?
		 * This option defines the user's wish:
		 * - true: For lower resolution levels (i.e. smaller images),
		 *				 the GridSpacing is made larger, as a power of 2.
		 * - false: The GridSpacing remains equal for each resolution level.
		 */
		std::string upsampleBSplineGridOption( "true" );
		bool upsampleGridOption = true;
		m_Configuration->ReadParameter( upsampleBSplineGridOption, "UpsampleGridOption", 0, true );
		if ( upsampleBSplineGridOption == "true" ) upsampleGridOption = true;
		else if ( upsampleBSplineGridOption == "false" ) upsampleGridOption = false;
		
		/** Resample the grid.*/
		if ( level == 0 )
		{
			/** Set grid equal to lowest resolution fixed image.*/
			this->SetInitialGrid( upsampleGridOption );			
		}	
		else
		{
			/**	If wanted, we upsample the grid of control points.*/
			if ( upsampleGridOption ) this->IncreaseScale();
			/** Otherwise, nothing is done with the BSpline-Grid.*/
		} 
		
	} // end BeforeEachResolution
	

	/**
	 * ********************* AfterEachIteration *********************
	 */

	template <class TElastix>
		void BSplineTransformWithDiffusion<TElastix>
		::AfterEachIteration(void)
	{
		/** Find out after how many iterations a diffusion is wanted . */
		unsigned int CurrentIterationNumber = m_Elastix->GetIterationCounter();
		unsigned int DiffusionEachNIterations = 0;
		m_Configuration->ReadParameter( DiffusionEachNIterations, "DiffusionEachNIterations", 0 );

		/** Checking DiffusionEachNIterations. */
		if ( DiffusionEachNIterations < 1 )
		{
			xout["warning"] << "WARNING: DiffusionEachNIterations < 1" << std::endl;
			xout["warning"] << "\t\tDiffusionEachNIterations is set to 1" << std::endl;
			DiffusionEachNIterations = 1;
		}

		/** Get the MaximumNumberOfIterations of this resolution level. */
		unsigned int ResNr = m_Elastix->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel();
		unsigned int MaximumNumberOfIterations = 0;
		m_Configuration->ReadParameter( MaximumNumberOfIterations, "MaximumNumberOfIterations", ResNr );

		/** Determine if diffusion is wanted after this iteration:
		 * Do it every N iterations, but not at the first iteration
		 * of a resolution, and also at the last iteration.
		 */
		bool DiffusionNow = ( CurrentIterationNumber % DiffusionEachNIterations == 0 );
		DiffusionNow &= ( CurrentIterationNumber != 0 );
		DiffusionNow |= ( CurrentIterationNumber == ( MaximumNumberOfIterations - 1 ) );

		/** If wanted, do a diffusion. */
		if ( DiffusionNow )
		{
			this->DiffuseDeformationField();
		}

	} // end AfterEachIteration

	
	/**
	 * ********************* SetInitialGrid *************************
	 *
	 * Set grid equal to lowest resolution fixed image.
	 */

	template <class TElastix>
		void BSplineTransformWithDiffusion<TElastix>
		::SetInitialGrid( bool upsampleGridOption )
	{
		/** Declarations.*/
		RegionType	gridregion;
		SizeType		gridsize;
		IndexType		gridindex;
		SpacingType	gridspacing;
		OriginType	gridorigin;
		
		/** Get the fixed image.*/
		typename FixedImageType::Pointer fixedimage;
		fixedimage = const_cast< FixedImageType * >(
			m_Registration->GetAsITKBaseType()->GetFixedImage() );
		
		/** Get the size etc. of this image */
		gridregion	=	fixedimage->GetRequestedRegion();
		gridindex		=	gridregion.GetIndex();
		gridsize		=	gridregion.GetSize();
		gridspacing	=	fixedimage->GetSpacing();
		gridorigin	=	fixedimage->GetOrigin();
		
		/** Increase the size of the image, to take the B-Spline
		 * support region into account.
		 */
		double Offset = ( SplineOrder + 1.0 ) / 2.0;
		
		/** Read the desired grid spacing */
		double gridspacingfactor = 8.0;
		m_Configuration->ReadParameter( gridspacingfactor, "FinalGridSpacing", 0 );

		/** If multigrid, then start with a lower resolution grid */
		if (upsampleGridOption)
		{
			/** cast to int, otherwise gcc does not understand it... */
			int nrOfResolutions = static_cast<int>(
				this->GetRegistration()->GetAsITKBaseType()->GetNumberOfLevels()  );
			gridspacingfactor *= pow(2.0, (nrOfResolutions-1) );
		}

		/** Determine the correct grid size */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			gridspacing[ j ] = gridspacing[ j ] * gridspacingfactor;
			gridindex[ j ] -= 
				static_cast< typename RegionType::IndexValueType >( Offset );
			gridsize[j] = 
				static_cast< typename RegionType::SizeValueType> (
				ceil( gridsize[ j ] / gridspacingfactor ) + ceil( 2.0 * Offset ) );
		}
		
		/** Set the size data in the transform */
		gridregion.SetSize( gridsize );
		gridregion.SetIndex( gridindex );
		this->SetGridRegion( gridregion );
		this->SetGridSpacing( gridspacing );
		this->SetGridOrigin( gridorigin );
		
		/** Set initial parameters to 0.0 */
		m_Parameterspointer =
			new ParametersType( this->GetNumberOfParameters() );
		m_Parameterspointer->Fill(0.0);
		m_Registration->GetAsITKBaseType()->SetInitialTransformParametersOfNextLevel( *m_Parameterspointer );
		delete m_Parameterspointer;
		
	} // end SetInitialGrid()
	
	
	/**
	 * *********************** IncreaseScale ************************
	 *
	 * Upsample the grid of control points.
	 */

	template <class TElastix>
		void BSplineTransformWithDiffusion<TElastix>
		::IncreaseScale(void)
	{
		/**	Declarations.*/
		RegionType gridregion;
		SizeType gridsize;
		IndexType gridindex;
		
		/** Get the latest transform parameters */
		m_Parameterspointer =
			new ParametersType( this->GetNumberOfParameters() );
		*m_Parameterspointer = m_Registration->GetAsITKBaseType()->GetLastTransformParameters();
		
		/** Get the pointer to the data in *m_Parameterspointer */
		PixelType * dataPointer = static_cast<PixelType *>( m_Parameterspointer->data_block() );
		/** Get the number of pixels that should go into one coefficient image */
		unsigned int numberOfPixels = ( this->GetGridRegion() ).GetNumberOfPixels();
		
		/** Set the correct region/size info of the coeff image
		 * that will be filled with the parameters */
		m_Coeffs1->SetRegions( this->GetGridRegion() );
		m_Coeffs1->SetOrigin( (this->GetGridOrigin()).GetDataPointer() );
		m_Coeffs1->SetSpacing( (this->GetGridSpacing()).GetDataPointer() );
		
		/* Set this image as the input of the m_Upsampler filter */
		m_Upsampler->SetInput( m_Coeffs1 );
		
		/** initialise iterator in the parameterspointer_out */
		unsigned int i = 0; 
		
		/* Loop over dimension. */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			/** Fill the coeff image with parameter data (displacements
			 * of the control points in the direction of dimension j).
			 */		
			m_Coeffs1->GetPixelContainer()->
				SetImportPointer( dataPointer, numberOfPixels );
			dataPointer += numberOfPixels;
			
			/** Tell the m_Upsampler that the input has been changed */
			m_Upsampler->Modified();
			
			/** Do the upsampling.*/
			try
			{
				m_Upsampler->UpdateLargestPossibleRegion();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}

			/** Create an upsampled image.*/
			m_Coeffs2 = m_Upsampler->GetOutput();
			
			/** Needs to be done just once */
			if ( j == 0 ) 
			{
				/** Determine the gridsize after removal of the edge
				 * Necessary because upsampling also doubles the size
				 * of the edge (which was added because of the support
				 * region of a b-spline)
				 */
				gridregion	=	m_Coeffs2->GetLargestPossibleRegion();
				gridindex		=	gridregion.GetIndex();
				gridsize		=	gridregion.GetSize();
				double Offset = ( SplineOrder + 1.0 ) / 2.0;
				for ( unsigned int j2 = 0; j2 < SpaceDimension; j2++ )
				{
					gridindex[j2] += 
						static_cast< typename RegionType::IndexValueType >( Offset );
					gridsize[j2] -= 
						static_cast< typename RegionType::SizeValueType> ( 2 * Offset );
				}
				gridregion.SetIndex( gridindex );
				gridregion.SetSize( gridsize );
				
				/** Create the new vector of parameters, with the 
				 * correct size (which is now 2^dim as big as the
				 * size in the previous resolution!)
				 */
				m_Parameterspointer_out = new ParametersType(
					gridregion.GetNumberOfPixels() * SpaceDimension );
			} // end if j == 0
			
			/** Create an iterator on the new coefficient image*/
			IteratorType iterator( m_Coeffs2, gridregion );
			iterator.GoToBegin();
			while ( !iterator.IsAtEnd() )
			{
				/** Copy the contents of coeff2 in a ParametersType array*/
				(*m_Parameterspointer_out)[ i ] = iterator.Get();
				++iterator;
				++i;
			} // end while coeff2 iterator loop
			
		} // end for dimension loop
		
		/** Set the initial parameters for the next resolution level.*/
		this->SetGridRegion( gridregion );
		this->SetGridOrigin( m_Coeffs2->GetOrigin() );
		this->SetGridSpacing( m_Coeffs2->GetSpacing() );
		m_Registration->GetAsITKBaseType()
			->SetInitialTransformParametersOfNextLevel( *m_Parameterspointer_out );
		
		delete m_Parameterspointer;
		delete m_Parameterspointer_out;	
		
	}  // end IncreaseScale()
	

	/**
	 * ************************* ReadFromFile ************************
	 */

	template <class TElastix>
	void BSplineTransformWithDiffusion<TElastix>
	::ReadFromFile(void)
	{
		/** Read and Set the Grid: this is a BSplineTransform specific task.*/

		/** Declarations.*/
		RegionType	gridregion;
		SizeType		gridsize;
		IndexType		gridindex;
		SpacingType	gridspacing;
		OriginType	gridorigin;
		
		/** Fill everything with default values.*/
		gridsize.Fill(1);
		gridindex.Fill(0);
		gridspacing.Fill(1.0);
		gridorigin.Fill(0.0);

		/** Get GridSize, GridIndex, GridSpacing and GridOrigin.*/
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			m_Configuration->ReadParameter( gridsize[ i ], "GridSize", i );
			m_Configuration->ReadParameter( gridindex[ i ], "GridIndex", i );
			m_Configuration->ReadParameter( gridspacing[ i ], "GridSpacing", i );
			m_Configuration->ReadParameter( gridorigin[ i ], "GridOrigin", i );
		}
		
		/** Set it all.*/
		gridregion.SetIndex( gridindex );
		gridregion.SetSize( gridsize );
		this->SetGridRegion( gridregion );
		this->SetGridSpacing( gridspacing );
		this->SetGridOrigin( gridorigin );

		/** Call the ReadFromFile from the TransformBase.
		 * This must be done after setting the Grid, because later the
		 * ReadFromFile from TransformBase calls SetParameters, which
		 * checks the parameter-size, which is based on the GridSize.
		 */
		this->Superclass2::ReadFromFile();

	} // end ReadFromFile


	/**
	 * ************************* WriteToFile ************************
	 *
	 * Saves the TransformParameters as a vector and if wanted
	 * also as a deformation field.
	 */

	template <class TElastix>
		void BSplineTransformWithDiffusion<TElastix>
		::WriteToFile( const ParametersType & param )
	{
		/** Call the WriteToFile from the TransformBase.*/
		this->Superclass2::WriteToFile( param );

		/** Add some BSplineTransform specific lines.*/
		xout["transpar"] << std::endl << "// BSplineTransform specific" << std::endl;

		/** Get the GridSize, GridIndex, GridSpacing and
		 * GridOrigin of this transform.
		 */
		SizeType size = this->GetGridRegion().GetSize();
		IndexType index = this->GetGridRegion().GetIndex();
		SpacingType spacing = this->GetGridSpacing();
		OriginType origin = this->GetGridOrigin();

		/** Write the GridSize of this transform.*/
		xout["transpar"] << "(GridSize ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << size[ i ] << " ";
		}
		xout["transpar"] << size[ SpaceDimension - 1 ] << ")" << std::endl;
		
		/** Write the GridIndex of this transform.*/
		xout["transpar"] << "(GridIndex ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << index[ i ] << " ";
		}
		xout["transpar"] << index[ SpaceDimension - 1 ] << ")" << std::endl;
		
		/** Set the precision of cout to 2, because GridSpacing and
		 * GridOrigin must have at least one digit precision.
		 */
		xout["transpar"] << std::setprecision(1);

		/** Write the GridSpacing of this transform.*/
		xout["transpar"] << "(GridSpacing ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << spacing[ i ] << " ";
		}
		xout["transpar"] << spacing[ SpaceDimension - 1 ] << ")" << std::endl;

		/** Write the GridOrigin of this transform.*/
		xout["transpar"] << "(GridOrigin ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << origin[ i ] << " ";
		}
		xout["transpar"] << origin[ SpaceDimension - 1 ] << ")" << std::endl;

		/** Set the precision back to default value.*/
		xout["transpar"] << std::setprecision(6);

		/** If wanted, write the TransformParameters as deformation
		 * images to a file.
		 */
		if ( 0 )
		{
			/** Get the pointer to the data in 'param' */
			PixelType * dataPointer = const_cast<PixelType *>(
				static_cast<const PixelType *>( param.data_block() )		);
			unsigned int numberOfPixels =
				(this->GetGridRegion()).GetNumberOfPixels();
			
			/** Initialise the coeffs image */
			m_Coeffs1->SetRegions( this->GetGridRegion() );
			m_Coeffs1->SetOrigin( (this->GetGridOrigin()).GetDataPointer() );
			m_Coeffs1->SetSpacing( (this->GetGridSpacing()).GetDataPointer() );
			
			for ( unsigned int i = 0; i < SpaceDimension; i++ )
			{
				/** Get the set of parameters that represent the control point
				 * displacements in the i-th dimension.
				 */
				m_Coeffs1->GetPixelContainer()->
					SetImportPointer( dataPointer, numberOfPixels );
				dataPointer += numberOfPixels;
				m_Coeffs1->Modified();
				
				/** Create complete filename: <name>.<dimension>.mhd
				 * --> two files are created: a header (.mhd) and a data (.raw) file.
				 */
				std::ostringstream makeFileName( "" );
				makeFileName << m_Configuration->GetCommandLineArgument("-t")	<< "." << i << ".mhd";
				m_Writer->SetFileName( makeFileName.str().c_str() );
				
				/** Write the coefficient image (i-th dimension) to file. */
				m_Caster->SetInput( m_Coeffs1 );
				m_Writer->SetInput( m_Caster->GetOutput() );

				/** Do the writing.*/
				try
				{
					m_Writer->Update();
				}
				catch( itk::ExceptionObject & excp )
				{
					xl::xout["error"] << excp << std::endl;
				}
				
				/** Force the writer to make a new .raw file.*/
				m_Writer->SetImageIO(NULL);
				
			}  // end for i
			
		} // end if

	} // end WriteToFile


	/**
	 * ******************* DiffuseDeformationField ******************
	 */

	template <class TElastix>
		void BSplineTransformWithDiffusion<TElastix>
		::DiffuseDeformationField(void)
	{
		/** ------------- Create deformationField. ------------- */

		/** First, create a dummyImage with the right region info, so
		 * that the TransformIndexToPhysicalPoint-functions will be right.
		 */
		typename DummyImageType::Pointer dummyImage = DummyImageType::New();
		dummyImage->SetRegions( m_DeformationRegion );
		dummyImage->SetOrigin( m_DeformationOrigin );
		dummyImage->SetSpacing( m_DeformationSpacing );

		/** Setup an iterator over dummyImage and outputImage.*/
		DummyIteratorType				iter( dummyImage, m_DeformationRegion );
		OutputImageIteratorType	iterout( m_DeformationField, m_DeformationRegion );
		
		/** Declare stuff.*/
		InputPointType	inputPoint;
		OutputPointType	outputPoint;
		VectorType			diff_point;
		IndexType				inputIndex;
		
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

		/** Write deformation field to file if wanted. */
		if ( m_WriteDiffusionFiles )
		{
			std::ostringstream makeFileName( "" );
			makeFileName << m_Configuration->GetCommandLineArgument( "-out" )
				<< "deformationField"
				<< ".R" << m_Elastix->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel()
				<< ".It" << m_Elastix->GetIterationCounter()	<< ".mhd";
			m_DeformationFieldWriter->SetFileName( makeFileName.str().c_str() );
			m_DeformationFieldWriter->SetInput( m_DeformationField );
			/** Do the writing. */
			try
			{
				m_DeformationFieldWriter->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}

			/** Force the writer to make a new .raw file. */
			m_DeformationFieldWriter->SetImageIO(NULL);
		}

		/** ------------- Create GrayValueImage. ------------- */

		/** Update the deformationFieldTransform. */
		this->UpdateIntermediaryDeformationFieldTransformTemp( m_DeformationField );
		m_GrayValueImage = m_Resampler->GetOutput();

		/** Do the resampling. */
		try
		{
			m_GrayValueImage->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			xl::xout["error"] << excp << std::endl;
		}

		/** Write gray value image to file if wanted. */
		if ( m_WriteDiffusionFiles )
		{
			std::ostringstream makeFileName( "" );
			makeFileName << m_Configuration->GetCommandLineArgument( "-out" )
				<< "GrayValueImage"
				<< ".R" << m_Elastix->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel()
				<< ".It" << m_Elastix->GetIterationCounter()	<< ".mhd";
			m_GrayValueImageWriter->SetFileName( makeFileName.str().c_str() );
			m_GrayValueImageWriter->SetInput( m_GrayValueImage );
			/** Do the writing. */
			try
			{
				m_GrayValueImageWriter->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}

			/** Force the writer to make a new .raw file. */
			m_GrayValueImageWriter->SetImageIO(NULL);
		}

		/** ------------- Setup the diffusion. ------------- */

		m_Diffusion->SetGrayValueImage( m_GrayValueImage );
		m_Diffusion->SetInput( m_DeformationField );
		m_Diffusion->Modified();
		m_DiffusedField = m_Diffusion->GetOutput();

		/** Diffuse deformationField. */
		try
		{
			m_DiffusedField->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			xl::xout["error"] << excp << std::endl;
		}

		/** Write diffused field to file if wanted. */
		if ( m_WriteDiffusionFiles )
		{
			// reset m_DeformationFieldWriter??
			std::ostringstream makeFileName( "" );
			makeFileName << m_Configuration->GetCommandLineArgument( "-out" )
				<< "diffusedField"
				<< ".R" << m_Elastix->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel()
				<< ".It" << m_Elastix->GetIterationCounter()	<< ".mhd";
			m_DiffusedFieldWriter->SetFileName( makeFileName.str().c_str() );
			m_DiffusedFieldWriter->SetInput( m_DiffusedField );
			m_DiffusedFieldWriter->Modified();
			/** Do the writing. */
			try
			{
				m_DiffusedFieldWriter->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}

			/** Force the writer to make a new .raw file. */
			m_DiffusedFieldWriter->SetImageIO(NULL);
		}

		/** ------------- Update the intermediary transform. ------------- */

		this->UpdateIntermediaryDeformationFieldTransform( m_DiffusedField );

		/** ------------- Set the current B-spline transform parameters to zero. ------------- */

		ParametersType dummyParameters( this->GetNumberOfParameters() );
		dummyParameters.Fill( 0.0 );
		this->SetParameters( dummyParameters );
		this->m_Elastix->GetElxOptimizerBase()->SetCurrentPositionPublic( dummyParameters );

	} // end DiffuseDeformationField

	
} // end namespace elastix


#endif // end #ifndef __elxBSplineTransformWithDiffusion_HXX__

