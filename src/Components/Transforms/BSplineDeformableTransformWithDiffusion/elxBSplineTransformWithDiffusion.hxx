#ifndef __elxBSplineTransformWithDiffusion_HXX__
#define __elxBSplineTransformWithDiffusion_HXX__

#include "elxBSplineTransformWithDiffusion.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"
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
		m_Coeffs1 = 0;
	
		m_GridSpacingFactor = 8.0;

		m_Caster	= TransformCastFilterType::New();
		m_Writer	= TransformWriterType::New();

		/** Initialize things for diffusion. */
		m_Diffusion = 0;
		m_DeformationField = 0;
		m_DiffusedField = 0;
		m_GrayValueImage1 = 0;
		m_GrayValueImage2 = 0;
		m_Resampler = 0;

		/** Initialize to false. */
		m_WriteDiffusionFiles = false;

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
		m_DeformationField = VectorImageType::New();
		m_DeformationField->SetRegions( m_DeformationRegion );
		m_DeformationField->SetOrigin( m_DeformationOrigin );
		m_DeformationField->SetSpacing( m_DeformationSpacing );
		m_DeformationField->Allocate();

		/** Create m_DeformationField and allocate memory. */
		m_DiffusedField = VectorImageType::New();
		m_DiffusedField->SetRegions( m_DeformationRegion );
		m_DiffusedField->SetOrigin( m_DeformationOrigin );
		m_DiffusedField->SetSpacing( m_DeformationSpacing );
		m_DiffusedField->Allocate();

		/** Create m_GrayValueImage and allocate memory. */
		m_GrayValueImage1 = GrayValueImageType::New();
		m_GrayValueImage1->SetRegions( m_DeformationRegion );
		m_GrayValueImage1->SetOrigin( m_DeformationOrigin );
		m_GrayValueImage1->SetSpacing( m_DeformationSpacing );
		m_GrayValueImage1->Allocate();
		m_GrayValueImage2 = GrayValueImageType::New();

		/** Create a resampler. */
		m_Resampler = ResamplerType::New();
		m_Resampler->SetTransform( this->GetIntermediaryDeformationFieldTransform() );
		//m_Resampler->SetInterpolator(); // default = LinearInterpolateImageFunction
		m_Resampler->SetInput( dynamic_cast<MovingImageELXType *>(
			m_Elastix->GetMovingImage() ) );
		GrayValuePixelType defaultPixelValue = NumericTraits<GrayValuePixelType>::Zero;
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
		}

		/** Create m_Diffusion. */
		m_Diffusion = DiffusionFilterType::New();
		m_Diffusion->SetRadius( radius );
		m_Diffusion->SetNumberOfIterations( iterations );
		m_Diffusion->SetUseThreshold( thresholdBool );
		m_Diffusion->SetThreshold( threshold );
		m_Diffusion->SetGrayValueImage( m_GrayValueImage1 );
		m_Diffusion->SetInput( m_DeformationField );

	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void BSplineTransformWithDiffusion<TElastix>
		::BeforeEachResolution(void)
	{
		/** What is the current resolution level? */
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
		
		/** Resample the grid. */
		if ( level == 0 )
		{
			/** Set grid equal to lowest resolution fixed image. */
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
		/** Declare boolean. */
		bool DiffusionNow;

		/** Find out filterpattern. */
		unsigned int filterPattern = 1;
		m_Configuration->ReadParameter( filterPattern, "FilterPattern", 0 );
		if ( filterPattern != 1 && filterPattern != 2 )
		{
			filterPattern = 1;
			xout["warning"] << "WARNING: filterPattern set to 1" << std::endl;
		}

		/** Get the current iteration number. */
		unsigned int CurrentIterationNumber = m_Elastix->GetIterationCounter();

		/** Get the MaximumNumberOfIterations of this resolution level. */
		unsigned int ResNr = m_Elastix->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel();
		unsigned int MaximumNumberOfIterations = 0;
		m_Configuration->ReadParameter( MaximumNumberOfIterations, "MaximumNumberOfIterations", ResNr );

		/** Find out if we have to filter now. */
		if ( filterPattern == 1 )
		{
			/** Find out after how many iterations a diffusion is wanted . */
			unsigned int DiffusionEachNIterations = 0;
			m_Configuration->ReadParameter( DiffusionEachNIterations, "DiffusionEachNIterations", 0 );
      
			/** Checking DiffusionEachNIterations. */
			if ( DiffusionEachNIterations < 1 )
			{
				xout["warning"] << "WARNING: DiffusionEachNIterations < 1" << std::endl;
				xout["warning"] << "\t\tDiffusionEachNIterations is set to 1" << std::endl;
				DiffusionEachNIterations = 1;
			}

			/** Determine if diffusion is wanted after this iteration:
			 * Do it every N iterations, but not at the first iteration
			 * of a resolution, and also at the last iteration.
			 */
			DiffusionNow = ( ( CurrentIterationNumber + 1 ) % DiffusionEachNIterations == 0 );
			DiffusionNow &= ( CurrentIterationNumber != 0 );
			DiffusionNow |= ( CurrentIterationNumber == ( MaximumNumberOfIterations - 1 ) );
		}
		else if ( filterPattern == 2 )
		{
			/** Find out . */
			unsigned int afterIterations0, afterIterations1;
			m_Configuration->ReadParameter( afterIterations0, "AfterIterations", 0 );
			m_Configuration->ReadParameter( afterIterations1, "AfterIterations", 1 );

			/** Find out . */
			unsigned int howManyIterations0, howManyIterations1, howManyIterations2;
			m_Configuration->ReadParameter( howManyIterations0, "HowManyIterations", 0 );
			m_Configuration->ReadParameter( howManyIterations1, "HowManyIterations", 1 );
			m_Configuration->ReadParameter( howManyIterations2, "HowManyIterations", 2 );

			/** The first afterIterations0 the deformationField is filtered
			 * every howManyIterations0 iterations. Then, for iterations between
			 * afterIterations0 and afterIterations1 , the deformationField
			 * is filtered after every howManyIterations1 iterations. Finally,
			 * the deformationField is filtered every howManyIterations2 iterations.
			 */
			unsigned int diffusionEachNIterations;
			if ( CurrentIterationNumber < afterIterations0 )
			{
				diffusionEachNIterations = howManyIterations0;
			}
			else if ( CurrentIterationNumber >= afterIterations0 && CurrentIterationNumber < afterIterations1 )
			{
				diffusionEachNIterations = howManyIterations1;
			}
			else
			{
				diffusionEachNIterations = howManyIterations2;
			}

			/** This iteration filteren? Also filter after the last iteration. */
			DiffusionNow = ( ( CurrentIterationNumber + 1 ) % diffusionEachNIterations == 0 );
			//DiffusionNow &= ( CurrentIterationNumber != 0 );
			DiffusionNow |= ( CurrentIterationNumber == ( MaximumNumberOfIterations - 1 ) );

		} // else if filterpattern

		/** If wanted, do a diffusion. */
		if ( DiffusionNow )
		{
			this->DiffuseDeformationField();
		}

	} // end AfterEachIteration

	
	/**
	 * ********************* SetInitialGrid *************************
	 *
	 * Set the size of the initial control point grid.
	 *
	 * If multiresolution (UpsampleGridOption is "true") then the 
	 * grid size is equal to the size of the fixed image, divided by
	 * desired final gridspacing and a factor 2^(NrOfImageResolutions-1).
	 * Otherwise it's equal to the size of the fixed image, divided by
	 * the desired final gridspacing.
	 *	 
	 * In both cases some extra grid points are put at the edges,
	 * to take into account the support region of the b-splines.
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

		/** In elastix <=3.001: gridregion	=	fixedimage->GetRequestedRegion();  */
		/** later (because requested regions were not supported anyway consistently: */
		gridregion = fixedimage->GetLargestPossibleRegion();
		/** \todo: allow the user to enter a region of interest for the registration. 
		 * Especially the boundary conditions have to be dealt with carefully then. */
		gridindex		=	gridregion.GetIndex();
		/** \todo: always 0? doesn't a largestpossible region have an index 0 by definition? */
		gridsize		=	gridregion.GetSize();
		gridspacing	=	fixedimage->GetSpacing();
		gridorigin	=	fixedimage->GetOrigin();
		
		/** Read the desired grid spacing */
		m_GridSpacingFactor = 8.0;
		m_Configuration->ReadParameter( m_GridSpacingFactor, "FinalGridSpacing", 0 );

		/** If multigrid, then start with a lower resolution grid */
		if (upsampleGridOption)
		{
			/** cast to int, otherwise gcc does not understand it... */
			int nrOfResolutions = static_cast<int>(
				this->GetRegistration()->GetAsITKBaseType()->GetNumberOfLevels()  );
			m_GridSpacingFactor *= pow(2.0, (nrOfResolutions-1) );
		}

		/** Determine the correct grid size */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			gridspacing[ j ] = gridspacing[ j ] * m_GridSpacingFactor;
			gridorigin[j] -= gridspacing[j] *
				floor( static_cast<double>(SplineOrder) / 2.0 );
			gridindex[j] = 0; // isn't this always the case anyway?
			gridsize[j]= static_cast< typename RegionType::SizeValueType >
				( ceil( gridsize[j] / m_GridSpacingFactor ) + SplineOrder );
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
		/** Typedefs */
		typedef itk::ResampleImageFilter<ImageType, ImageType>
			UpsampleFilterType;
		typedef itk::IdentityTransform<CoordRepType, SpaceDimension>
			IdentityTransformType;
		typedef itk::BSplineResampleImageFunction<ImageType, CoordRepType> 
			CoefficientUpsampleFunctionType;
		typedef itk::BSplineDecompositionImageFilter<ImageType,ImageType>
			DecompositionFilterType;
		typedef ImageRegionConstIterator<ImageType>		IteratorType;

		/** The current region/spacing settings of the grid: */
		RegionType gridregionLow = this->GetGridRegion();
		SizeType gridsizeLow = gridregionLow.GetSize();
		IndexType gridindexLow = gridregionLow.GetIndex();
		SpacingType gridspacingLow = this->GetGridSpacing();
		OriginType gridoriginLow = this->GetGridOrigin();

		/** Get the fixed image.*/
		typename FixedImageType::Pointer fixedimage;
		fixedimage = const_cast< FixedImageType * >(
			m_Registration->GetAsITKBaseType()->GetFixedImage() );
		
		/** Set start values for computing the new grid size. */
		RegionType gridregionHigh	= fixedimage->GetLargestPossibleRegion();
		IndexType gridindexHigh		=	gridregionHigh.GetIndex();
		SizeType gridsizeHigh		=	gridregionHigh.GetSize();
		SpacingType gridspacingHigh	=	fixedimage->GetSpacing();
		OriginType gridoriginHigh	=	fixedimage->GetOrigin();
		
		/** A twice as dense grid: */
		m_GridSpacingFactor /= 2;

		/** Determine the correct grid size */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			gridspacingHigh[ j ] = gridspacingHigh[ j ] * m_GridSpacingFactor;
			gridoriginHigh[j] -= gridspacingHigh[j] *
				floor( static_cast<double>(SplineOrder) / 2.0 );
			gridindexHigh[j] = 0; // isn't this always the case anyway?
			gridsizeHigh[j]= static_cast< typename RegionType::SizeValueType >
				( ceil( gridsizeHigh[j] / m_GridSpacingFactor ) + SplineOrder );
		}
		gridregionHigh.SetSize(gridsizeHigh);
		gridregionHigh.SetIndex(gridindexHigh);

		/** Get the latest transform parameters */
		m_Parameterspointer =
			new ParametersType( this->GetNumberOfParameters() );
		*m_Parameterspointer = m_Registration->GetAsITKBaseType()->GetLastTransformParameters();
		
		/** Get the pointer to the data in *m_Parameterspointer */
		PixelType * dataPointer = static_cast<PixelType *>( m_Parameterspointer->data_block() );
		/** Get the number of pixels that should go into one coefficient image */
		unsigned int numberOfPixels = ( this->GetGridRegion() ).GetNumberOfPixels();
		
		/** Set the correct region/size info of the coeff image
		* that will be filled with the current parameters */
		m_Coeffs1 = ImageType::New();
		m_Coeffs1->SetRegions( this->GetGridRegion() );
		m_Coeffs1->SetOrigin( (this->GetGridOrigin()).GetDataPointer() );
		m_Coeffs1->SetSpacing( (this->GetGridSpacing()).GetDataPointer() );
		//m_Coeffs1->Allocate() not needed because the data is set by directly pointing
		// to an existing piece of memory.
		
		/** 
		 * Create the new vector of parameters, with the 
		 * correct size (which is now approx 2^dim as big as the
		 * size in the previous resolution!)
		 */
		m_Parameterspointer_out = new ParametersType(
			gridregionHigh.GetNumberOfPixels() * SpaceDimension );

		/** initialise iterator in the parameterspointer_out */
		unsigned int i = 0; 
		
		/** Loop over dimension. */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			/** Fill the coeff image with parameter data (displacements
			 * of the control points in the direction of dimension j).
			 */		
			m_Coeffs1->GetPixelContainer()->
				SetImportPointer( dataPointer, numberOfPixels );
			dataPointer += numberOfPixels;
				
			/*
			 * Set this image as the input of the upsampler filter. The 
			 * upsampler samples the deformation field at the locations
			 * of the new control points (note: it does not just interpolate
			 * the coefficient image, which would be wrong). The b-spline
			 * coefficients that describe the resulting image are computed
			 * by the decomposition filter.
			 * 
			 * This code is copied from the itk-example
			 * DeformableRegistration6.cxx .
			 */
			
			typename UpsampleFilterType::Pointer upsampler = UpsampleFilterType::New();
			typename IdentityTransformType::Pointer identity = IdentityTransformType::New();
			typename CoefficientUpsampleFunctionType::Pointer coeffUpsampleFunction =
				CoefficientUpsampleFunctionType::New();
			typename DecompositionFilterType::Pointer decompositionFilter = 
				DecompositionFilterType::New();

			upsampler->SetInterpolator(coeffUpsampleFunction);
			upsampler->SetTransform(identity);
      upsampler->SetSize(gridsizeHigh);
			upsampler->SetOutputStartIndex(gridindexHigh);
			upsampler->SetOutputSpacing(gridspacingHigh);
			upsampler->SetOutputOrigin(gridoriginHigh);
		  upsampler->SetInput( m_Coeffs1 );
						
			decompositionFilter->SetSplineOrder(SplineOrder);
			decompositionFilter->SetInput( upsampler->GetOutput() );

			/** Do the upsampling.*/
			try
			{
				decompositionFilter->UpdateLargestPossibleRegion();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}
			
			/** Create an upsampled image.*/
			m_Coeffs2 = decompositionFilter->GetOutput();
					
			/** Create an iterator on the new coefficient image*/
			IteratorType iterator( m_Coeffs2, gridregionHigh );
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
		this->SetGridRegion( gridregionHigh );
		this->SetGridOrigin( gridoriginHigh );
		this->SetGridSpacing( gridspacingHigh );
		m_Registration->GetAsITKBaseType()->
			SetInitialTransformParametersOfNextLevel( *m_Parameterspointer_out );
		
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
		/** Task 1 - Get and Set the DeformationField Image. */

		/** Read the name of the deformationFieldImage. */
		std::string fileName = "";
		m_Configuration->ReadParameter( fileName, "DeformationFieldFileName", 0 );

		/** Error checking ... */
		if ( fileName == "" )
		{
			xout["error"] << "ERROR: DeformationFieldFileName not specified."
				<< std::endl << "Unable to read and set the transform parameters." << std::endl;
			// \todo quit program nicely or throw an exception
		}

		/** Read in the deformationField image. */
		typename VectorReaderType::Pointer vectorReader = VectorReaderType::New();
		vectorReader->SetFileName( fileName.c_str() );

		/** Do the reading. */
		try
		{
			vectorReader->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			xl::xout["error"] << excp << std::endl;
		}

		/** Get image information and set it in the DeformationFieldTransform. */
		RegionType region = vectorReader->GetOutput()->GetLargestPossibleRegion();
		SpacingType spacing = vectorReader->GetOutput()->GetSpacing();
		OriginType origin = vectorReader->GetOutput()->GetOrigin();
		this->SetDeformationFieldRegion( region );
		this->SetDeformationFieldSpacing( spacing );
		this->SetDeformationFieldOrigin( origin );
		this->InitializeDeformationFields();

		/** Set the deformationFieldImage in the itkDeformationVectorFieldTransform. */
		this->UpdateIntermediaryDeformationFieldTransform( vectorReader->GetOutput() );

		/** Task 2 - Get and Set the B-spline part of this transform. */

		/** Declarations of the B-spline grid and fill everything with default values. */
		RegionType	gridregion;
		SizeType		gridsize;
		IndexType		gridindex;
		SpacingType	gridspacing;
		OriginType	gridorigin;
		gridsize.Fill(1);
		gridindex.Fill(0);
		gridspacing.Fill(1.0);
		gridorigin.Fill(0.0);

		/** Get GridSize, GridIndex, GridSpacing and GridOrigin. */
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			m_Configuration->ReadParameter( gridsize[ i ], "GridSize", i );
			m_Configuration->ReadParameter( gridindex[ i ], "GridIndex", i );
			m_Configuration->ReadParameter( gridspacing[ i ], "GridSpacing", i );
			m_Configuration->ReadParameter( gridorigin[ i ], "GridOrigin", i );
		}
		
		/** Set the B-spline grid. */
		gridregion.SetIndex( gridindex );
		gridregion.SetSize( gridsize );
		this->SetGridRegion( gridregion );
		this->SetGridSpacing( gridspacing );
		this->SetGridOrigin( gridorigin );

		/** Fill and set the B-spline parameters. */
		unsigned int nop = 0;
		m_Configuration->ReadParameter( nop, "NumberOfParameters", 0 );
		bsplineParameters.SetSize( nop );
		bsplineParameters.Fill(0.0);
		this->SetParameters( bsplineParameters );

		/** Do not call the ReadFromFile from the TransformBase,
		 * because that function tries to read parameters from the file,
		 * which is not necessary in this case, because the parameters are
		 * in the vectorImage.
		 * NOT: this->Superclass2::ReadFromFile();
		 * However, we do have to copy some of the functionality:
		 */

		/** Task 3 - Get and Set the Initial Transform. */

		/** Get the InitialTransformName. */
		fileName = "";
		m_Configuration->ReadParameter( fileName,
			"InitialTransformParametersFileName", 0 );
		
		/** Call the function ReadInitialTransformFromFile.*/
		if ( fileName != "NoInitialTransform" )
		{			
			this->Superclass2::ReadInitialTransformFromFile( fileName.c_str() );
		}

		/** Read from the configuration file how to combine the
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
		
		// \todo Test this ReadFromFile function.
		
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
		/** Make sure that the Transformbase::WriteToFile() does
		 * not write the transformParameters in the file.
		 */
		this->SetReadWriteTransformParameters( false );

		/** Call the WriteToFile from the TransformBase.*/
		this->Superclass2::WriteToFile( param );

		/** Add some BSplineTransformWithDiffusion specific lines.*/
		xout["transpar"] << std::endl << "// BSplineTransformWithDiffusion specific" << std::endl;

		/** Get the current iteration number and convert it to string. */
		unsigned int currentIterationNumber = m_Elastix->GetIterationCounter();
		std::ostringstream makeIterationString("");
		unsigned int border = 1000000;
		while (border > 1)
		{
			if ( currentIterationNumber < border )
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
		makeIterationString << currentIterationNumber;

		/** Write the filename of the deformationField image. */
		std::ostringstream makeFileName( "" );
		makeFileName << m_Configuration->GetCommandLineArgument( "-out" )
			<< "DeformationFieldImage."
			<< m_Configuration->GetElastixLevel()
			<< ".R" << this->GetRegistration()->GetAsITKBaseType()->GetCurrentLevel()
			<< ".It" << makeIterationString.str()
			<< ".mhd";
		xout["transpar"] << "(DeformationFieldFileName \""
			<< makeFileName.str() << "\")" << std::endl;

		/** Write the deformation field image. */
		typename DeformationFieldWriterType::Pointer writer
			= DeformationFieldWriterType::New();
		writer->SetFileName( makeFileName.str().c_str() );
		writer->SetInput( m_DiffusedField );
		/** Do the writing. */
		try
		{
			writer->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			xl::xout["error"] << excp << std::endl;
		}

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
		xout["transpar"] << std::setprecision( m_Elastix->GetDefaultOutputPrecision() );
		
	} // end WriteToFile


	/**
	 * ******************* DiffuseDeformationField ******************
	 */

	template <class TElastix>
		void BSplineTransformWithDiffusion<TElastix>
		::DiffuseDeformationField(void)
	{
		/** This function does:
		 * 1) Calculate current deformation field.
		 * 2) Update the intermediary deformationFieldtransform
		 *		with this deformation field.
		 * 3) Calculate the GrayValueImage with the resampler,
		 *		which is over the intermediary deformationFieldtransform.
		 * 4) Diffuse the current deformation field.
		 * 5) Update the intermediary deformationFieldtransform
		 *		with this diffused deformation field.
		 * 6) Reset the parameters of the BSplineTransform
		 *		and the optimizer. Reset the initial transform.
		 * 7) If wanted, write the deformationField, the 
		 *		GrayValueImage and the diffusedField.
		 */

		/** ------------- 1: Create deformationField. ------------- */

		/** First, create a dummyImage with the right region info, so
		 * that the TransformIndexToPhysicalPoint-functions will be right.
		 */
		typename DummyImageType::Pointer dummyImage = DummyImageType::New();
		dummyImage->SetRegions( m_DeformationRegion );
		dummyImage->SetOrigin( m_DeformationOrigin );
		dummyImage->SetSpacing( m_DeformationSpacing );

		/** Setup an iterator over dummyImage and outputImage. */
		DummyIteratorType				iter( dummyImage, m_DeformationRegion );
		VectorImageIteratorType	iterout( m_DeformationField, m_DeformationRegion );
		
		/** Declare stuff. */
		InputPointType	inputPoint;
		OutputPointType	outputPoint;
		VectorType			diff_point;
		IndexType				inputIndex;
		
		/** Calculate the TransformPoint of all voxels of the image. */
		iter.Begin();
		iterout.Begin();
		while ( !iter.IsAtEnd() )
		{
			inputIndex = iter.GetIndex();
			/** Transform the points to physical space. */
			dummyImage->TransformIndexToPhysicalPoint( inputIndex, inputPoint );
			/** Call TransformPoint. */
			outputPoint = this->TransformPoint( inputPoint );
			/** Calculate the difference. */
			for ( unsigned int i = 0; i < FixedImageDimension; i++ )
			{
				diff_point[ i ] = outputPoint[ i ] - inputPoint[ i ];
			}
			iterout.Set( diff_point );
			++iter;
			++iterout;
		}

		/** ------------- 2: Update the intermediary deformationFieldTransform. ------------- */

		this->UpdateIntermediaryDeformationFieldTransform( m_DeformationField );

		/** ------------- 3: Create GrayValueImage. ------------- */

		m_Resampler->Modified();
		m_GrayValueImage1 = m_Resampler->GetOutput();

		/** Do the resampling. */
		try
		{
			m_GrayValueImage1->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			xl::xout["error"] << excp << std::endl;
		}

		/** If wanted also take the fixed image into account
		 * for the derivation of the GrayValueImage.
		 */
		/** Check if wanted. */
		std::string alsoFixed = "true";
		m_Configuration->ReadParameter( alsoFixed, "GrayValueImageAlsoBasedOnFixedImage", 0 );
		typename MaximumImageFilterType::Pointer maximumImageFilter
			= MaximumImageFilterType::New();
		if( alsoFixed == "true" )
		{
			maximumImageFilter->SetInput( 0, m_GrayValueImage1 );
			maximumImageFilter->SetInput( 1, dynamic_cast<FixedImageELXType *>(
			m_Elastix->GetFixedImage() ) );
			m_GrayValueImage2 = maximumImageFilter->GetOutput();

			/** Do the maximum (OR filter). */
			try
			{
				m_GrayValueImage2->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}
		} // end if

		/** Get diffusion information: threshold information. */
		std::string thresholdbooltmp = "";
		bool thresholdBool = false;
		m_Configuration->ReadParameter( thresholdbooltmp, "ThresholdBool", 0 );
		if ( thresholdbooltmp == "true" ) thresholdBool = true;

		int threshold = 150;
		m_Configuration->ReadParameter( threshold, "ThresholdHU", 0 );

		if ( thresholdBool )
		{
			/** Setup iterator. */
			GrayValueImageIteratorType it( m_GrayValueImage2, m_GrayValueImage2->GetLargestPossibleRegion() );
			it.GoToBegin();
			while ( !it.IsAtEnd() )
			{
				/** Threshold or just make sure everything is between 0 and 1. */
				if ( it.Get() < threshold ) it.Set( 0 );
				if ( it.Get() >= threshold ) it.Set( 100 );
				/** Update iterator. */
				++it;
			} // end while
		} // end if

		/** ------------- 4: Setup the diffusion. ------------- */

		if( alsoFixed == "true" )
		{
			m_Diffusion->SetGrayValueImage( m_GrayValueImage2 );
		}
		else
		{
			m_Diffusion->SetGrayValueImage( m_GrayValueImage1 );
		}
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

		/** ------------- 5: Update the intermediary transform. ------------- */

		this->UpdateIntermediaryDeformationFieldTransform( m_DiffusedField );

		/** ------------- 6: Reset the current transform parameters. ------------- */

		ParametersType dummyParameters( this->GetNumberOfParameters() );
		dummyParameters.Fill( 0.0 );
		/** Reset the BSpline. */
		this->SetParameters( dummyParameters );
		/** Reset the optimizer. */
		this->m_Elastix->GetElxOptimizerBase()->SetCurrentPositionPublic( dummyParameters );
		/** Get rid of the initial transform, because this is now captured
		 * within the DeformationFieldTransform.
		 */
		this->SetGrouper( "NoInitialTransform" );
		this->Superclass1::SetInitialTransform( 0 );

		/** ------------- 7: Write images. ------------- */

		/** If wanted, write the deformationField, the GrayValueImage and the diffusedField. */
		if ( m_WriteDiffusionFiles )
		{
			/** Filename. */
			std::ostringstream makeFileName1( "" ), begin(""), end("");
			begin	<< m_Configuration->GetCommandLineArgument( "-out" );
			end		<< ".R" << m_Elastix->GetElxRegistrationBase()->GetAsITKBaseType()->GetCurrentLevel()
				<< ".It" << m_Elastix->GetIterationCounter()	<< ".mhd";

			/** Write the deformationFieldImage. */
			makeFileName1 << begin.str() << "deformationField" << end.str();
			typename DeformationFieldWriterType::Pointer deformationFieldWriter
				= DeformationFieldWriterType::New();
			deformationFieldWriter->SetFileName( makeFileName1.str().c_str() );
			deformationFieldWriter->SetInput( m_DeformationField );
			/** Do the writing. */
			try
			{
				deformationFieldWriter->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}

			/** Write the GrayValueImage. */
			std::ostringstream makeFileName2( "" );
			makeFileName2 << begin.str() << "GrayValueImage" << end.str();
			typename GrayValueImageWriterType::Pointer grayValueImageWriter
				= GrayValueImageWriterType::New();
			grayValueImageWriter->SetFileName( makeFileName2.str().c_str() );
			if( alsoFixed == "true" )
			{
				grayValueImageWriter->SetInput( m_GrayValueImage2 );
			}
			else
			{
				grayValueImageWriter->SetInput( m_GrayValueImage1 );
			}
			
			/** Do the writing. */
			try
			{
				grayValueImageWriter->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}

			/** Write the diffusedFieldImage. */
			std::ostringstream makeFileName3( "" );
			makeFileName3 << begin.str() << "diffusedField" << end.str();
			typename DeformationFieldWriterType::Pointer diffusedFieldWriter
				= DeformationFieldWriterType::New();
			diffusedFieldWriter->SetFileName( makeFileName3.str().c_str() );
			diffusedFieldWriter->SetInput( m_DiffusedField );
			/** Do the writing. */
			try
			{
				diffusedFieldWriter->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}

		} // end if m_WriteDiffusionFiles

	} // end DiffuseDeformationField

	
} // end namespace elastix


#endif // end #ifndef __elxBSplineTransformWithDiffusion_HXX__

