#ifndef __elxBSplineTransform_hxx
#define __elxBSplineTransform_hxx

#include "elxBSplineTransform.h"

#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionExclusionConstIteratorWithIndex.h"
#include "vnl/vnl_math.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		BSplineTransform<TElastix>::
		BSplineTransform()
	{
		this->m_BSplineTransform = BSplineTransformType::New();
		this->SetCurrentTransform( this->m_BSplineTransform );

		/** Initialize. */
    this->m_GridScheduleComputer = GridScheduleComputerType::New();
    this->m_GridScheduleComputer->SetBSplineOrder( SplineOrder );
		//this->m_GridSpacingFactor.Fill( 8.0 );
		//this->m_UpsampleBSplineGridOption.push_back( true );

	} // end Constructor
	

	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>
		::BeforeRegistration( void )
	{
		/** Set initial transform parameters to a 1x1x1 grid, with deformation (0,0,0).
		 * In the method BeforeEachResolution() this will be replaced by the right grid size.
		 * This seems not logical, but it is required, since the registration
		 * class checks if the number of parameters in the transform is equal to
		 * the number of parameters in the registration class. This check is done
		 * before calling the BeforeEachResolution() methods.
		 */
		
		/** Task 1 - Set the Grid. */

		/** Declarations. */
		RegionType gridregion;
		SizeType gridsize;
		IndexType gridindex;
		SpacingType gridspacing;
		OriginType gridorigin;
		
		/** Fill everything with default values. */
		gridsize.Fill(1);
		gridindex.Fill(0);
		gridspacing.Fill(1.0);
		gridorigin.Fill(0.0);
		
		/** Set it all. */
		gridregion.SetIndex( gridindex );
		gridregion.SetSize( gridsize );
		this->m_BSplineTransform->SetGridRegion( gridregion );
		this->m_BSplineTransform->SetGridSpacing( gridspacing );
		this->m_BSplineTransform->SetGridOrigin( gridorigin );
    
		/** Task 2 - Give the registration an initial parameter-array. */
		ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
		dummyInitialParameters.Fill( 0.0 );
		
		/** Put parameters in the registration. */
		this->m_Registration->GetAsITKBaseType()
			->SetInitialTransformParameters( dummyInitialParameters );

    /** Precompute the B-spline grid regions. */
    this->PreComputeGridInformation();
    
	} // end BeforeRegistration()
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>
		::BeforeEachResolution( void )
	{
		/** What is the current resolution level? */
		unsigned int level = 
			this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

		/** Define the grid. */
		if ( level == 0 )
		{
			this->InitializeTransform();			
		}	
		else
		{
			/** Check if the BSpline grid should be upsampled now. */
      if ( this->m_GridScheduleComputer->GetDoUpsampling( level - 1 ) )
			{
				this->IncreaseScale();
			}
			/** Otherwise, nothing is done with the BSpline-Grid. */
		}

    /** Get the PassiveEdgeWidth and use it to set the OptimizerScales. */
    unsigned int passiveEdgeWidth = 0;
    this->GetConfiguration()->ReadParameter( passiveEdgeWidth,
      "PassiveEdgeWidth", this->GetComponentLabel(), level, 0 );
		this->SetOptimizerScales( passiveEdgeWidth );
	
	} // end BeforeEachResolution()
	

  /**
	 * ******************** PreComputeGridInformation ***********************
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>::
		PreComputeGridInformation( void )
	{
    /** The old deprecated way. */
    if ( this->ComputeInitialGridSpacing_Deprecated() ) return;

    /** Get the total number of resolution levels. */
    unsigned int nrOfResolutions = 
			this->m_Registration->GetAsITKBaseType()->GetNumberOfLevels();

    /** Get the grid spacing schedule from the parameter file.
     * Way 1: The user suplies the "GridSpacingScheduleFull".
     *        - When nrOfResolutions spacings are specified, then it is
     *          assumed that an isotropic grid spacing is desired.
     *        - When nrOfResolutions * SpaceDimension spacings are
     *          specified, then the grid spacing is fully specified.
     *        - When another number of spacing is specified, then it is
     *          assumed that an error is made.
     * Way 2: The user supplies the "GridSpacingSchedule" and the
     *        "GridSpacingUpsampleFactor".
     * One of these two options is required; way 1 overrides way 2.
     */

    /** Way 1: the complete schedule. */
    float dummy = 1.0;
    unsigned int count = this->m_Configuration
      ->CountNumberOfParameters( dummy, "GridSpacingScheduleFull" );
    bool invalidCount = false;

    GridScheduleType schedule( nrOfResolutions );
    if ( count == nrOfResolutions )
    {
      for ( unsigned int i = 0; i < nrOfResolutions; ++i )
      {
        float spacing;
        this->m_Configuration->ReadParameter(
          spacing, "GridSpacingScheduleFull", i );
        schedule[ i ].Fill( spacing );
      }
    }
    else if ( count == SpaceDimension * nrOfResolutions )
    {
      for ( unsigned int i = 0; i < nrOfResolutions; ++i )
      {
        for ( unsigned int j = 0; j < SpaceDimension; ++j )
        {
          this->m_Configuration->ReadParameter(
            schedule[ i ][ j ], "GridSpacingScheduleFull", i * SpaceDimension + j  );
        }
      }
    }
    else
    {
      invalidCount = true;
    }

    /** Check validity of this option. */
    int way1;
    if ( count > 0 )
    {
      if ( invalidCount )
      {
        itkExceptionMacro( << "ERROR: \"GridSpacingScheduleFull\" not fully specified." );
      }
      else
      {
        this->m_GridScheduleComputer->SetGridSpacingSchedule( schedule );
      }
    }
    else
    {
      /** Way 2: the final grid spacing. */
      SpacingType finalGridSpacing;
      finalGridSpacing[ 0 ] = 8.0;
      way1 = this->m_Configuration->ReadParameter(
        finalGridSpacing[ 0 ], "GridSpacingSchedule", 0, true );
      finalGridSpacing.Fill( finalGridSpacing[ 0 ] );
      for ( unsigned int i = 1; i < SpaceDimension; ++i )
      {
        this->m_Configuration->ReadParameter(
          finalGridSpacing[ i ], "GridSpacingSchedule", i, true );
      }

      /** Way 2: the upsample factor. */
      float upsampleFactor = 2.0;
      this->m_Configuration->ReadParameter(
        upsampleFactor, "GridSpacingUpsampleFactor", 0, true );

      /** If this option is used. */
      if ( way1 == 0 )
      {
        this->m_GridScheduleComputer
          ->SetDefaultGridSpacingSchedule( nrOfResolutions, finalGridSpacing, upsampleFactor );
      }
    }

    /** Which option was selected by the user? */
    if ( way1 == 1 && count == 0 )
    {
      itkExceptionMacro(
        << "ERROR: You should specify "
        << "EITHER \"GridSpacingSchedule\" and \"GridSpacingUpsampleFactor\""
        << "OR \"GridSpacingScheduleFull\"."
        );
    }

    /** Set other required information. */
    this->m_GridScheduleComputer->SetOrigin(
      this->GetElastix()->GetFixedImage()->GetOrigin() );
    this->m_GridScheduleComputer->SetSpacing(
      this->GetElastix()->GetFixedImage()->GetSpacing() );
    this->m_GridScheduleComputer->SetRegion(
      this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion() );

    /** Compute the necessary information. */
    this->m_GridScheduleComputer->ComputeBSplineGrid();
		
  } // end PreComputeGridInformation()

	
	/**
	 * ******************** InitializeTransform ***********************
	 *
	 * Set the size of the initial control point grid and initialize
	 * the parameters to 0.
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>::
		InitializeTransform( void )
	{
		/** Compute for each dimension the grid spacing factor and 
		 * store it in the array m_GridSpacingFactor.
     *
		this->ComputeInitialGridSpacing_Deprecated();

		/** Compute the B-spline grid region, origin, and spacing. */
		RegionType gridRegion;
		OriginType gridOrigin;
		SpacingType gridSpacing;
		//this->DefineGrid( gridregion, gridorigin, gridspacing );
    this->m_GridScheduleComputer->GetBSplineGrid( 0,
      gridRegion, gridSpacing, gridOrigin );

		/** Set it in the BSplineTransform. */
		this->m_BSplineTransform->SetGridRegion( gridRegion );
		this->m_BSplineTransform->SetGridSpacing( gridSpacing );
		this->m_BSplineTransform->SetGridOrigin( gridOrigin );

		/** Set initial parameters for the first resolution to 0.0. */
		ParametersType initialParameters( this->GetNumberOfParameters() );
		initialParameters.Fill( 0.0 );
		this->m_Registration->GetAsITKBaseType()->
			SetInitialTransformParametersOfNextLevel( initialParameters );
		
	} // end InitializeTransform()
	
	
	/**
	 * *********************** IncreaseScale ************************
	 *
	 * Upsample the grid of control points.
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>::
		IncreaseScale( void )
	{
		/** Typedefs. */
		typedef itk::ResampleImageFilter<
      ImageType, ImageType >                      UpsampleFilterType;
		typedef itk::IdentityTransform<
      CoordRepType, SpaceDimension >   			      IdentityTransformType;
		typedef itk::BSplineResampleImageFunction<
      ImageType, CoordRepType >  			            CoefficientUpsampleFunctionType;
		typedef itk::BSplineDecompositionImageFilter<
      ImageType, ImageType >			                DecompositionFilterType;
		typedef ImageRegionConstIterator<
      ImageType >		                              IteratorType;
		
    /** What is the current resolution level? */
		unsigned int level = 
			this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

		/** The current grid. */
		RegionType  gridRegionLow  = this->m_BSplineTransform->GetGridRegion();
		SpacingType gridSpacingLow = this->m_BSplineTransform->GetGridSpacing();
		OriginType  gridOriginLow  = this->m_BSplineTransform->GetGridOrigin();

		/** We want a twice as dense grid as the current grid: */
		//this->m_GridSpacingFactor /= 2.0;
		
		/** The new grid. */
		RegionType  gridRegionHigh;
		OriginType  gridOriginHigh;
		SpacingType gridSpacingHigh;
		//this->DefineGrid( gridregionHigh, gridoriginHigh, gridspacingHigh );
    this->m_GridScheduleComputer->GetBSplineGrid( level,
      gridRegionHigh, gridSpacingHigh, gridOriginHigh );
		IndexType gridIndexHigh	=	gridRegionHigh.GetIndex();
		SizeType  gridSizeHigh	=	gridRegionHigh.GetSize();

		/** Get the latest transform parameters. */
		ParametersType latestParameters =
			this->m_Registration->GetAsITKBaseType()->GetLastTransformParameters();
		
		/** Get the pointer to the data in latestParameters. */
		PixelType * dataPointer = static_cast<PixelType *>( latestParameters.data_block() );
		/** Get the number of pixels that should go into one coefficient image. */
		unsigned int numberOfPixels = 
			( this->m_BSplineTransform->GetGridRegion() ).GetNumberOfPixels();
		
		/** Set the correct region/size info of the coefficient image
		 * that will be filled with the current parameters.	 */
		ImagePointer coeffs1 = ImageType::New();
		coeffs1->SetRegions( gridRegionLow );
		coeffs1->SetOrigin(  gridOriginLow );
		coeffs1->SetSpacing( gridSpacingLow );
		//coeffs1->Allocate() not needed because the data is set by directly pointing
		// to an existing piece of memory.
		
		/** Create the new vector of parameters, with the 
		 * correct size (which is now approx 2^dim as big as the
		 * size in the previous resolution!). */
		ParametersType parameters_out(
			gridRegionHigh.GetNumberOfPixels() * SpaceDimension );

		/** initialise iterator in the parameters_out. */
		unsigned int i = 0; 
		
		/** Loop over dimension. */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			/** Fill the coeff image with parameter data (displacements
			 * of the control points in the direction of dimension j).
			 */		
			coeffs1->GetPixelContainer()->
				SetImportPointer( dataPointer, numberOfPixels );
			dataPointer += numberOfPixels;
				
			/*
			 * Set this image as the input of the upsampler filter. The 
			 * upsampler samples the deformation field at the locations
			 * of the new control points, given the current coefficients
			 * (note: it does not just interpolate the coefficient image,
			 * which would be wrong). The b-spline coefficients that
			 * describe the resulting image are computed by the 
			 * decomposition filter.
			 * 
			 * This code is copied from the itk-example
			 * DeformableRegistration6.cxx .
			 */
			
			typename UpsampleFilterType::Pointer upsampler
				= UpsampleFilterType::New();
			typename IdentityTransformType::Pointer identity
				= IdentityTransformType::New();
			typename CoefficientUpsampleFunctionType::Pointer coeffUpsampleFunction
				= CoefficientUpsampleFunctionType::New();
			typename DecompositionFilterType::Pointer decompositionFilter
				= DecompositionFilterType::New();

			upsampler->SetInterpolator( coeffUpsampleFunction );
			upsampler->SetTransform( identity );
      upsampler->SetSize( gridSizeHigh );
			upsampler->SetOutputStartIndex( gridIndexHigh );
			upsampler->SetOutputSpacing( gridSpacingHigh );
			upsampler->SetOutputOrigin( gridOriginHigh );
		  upsampler->SetInput( coeffs1 );
						
			decompositionFilter->SetSplineOrder( SplineOrder );
			decompositionFilter->SetInput( upsampler->GetOutput() );

			/** Do the upsampling. */
			try
			{
				decompositionFilter->UpdateLargestPossibleRegion();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "BSplineTransform - IncreaseScale()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError occured while using decompositionFilter.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}
			
			/** Create an upsampled image. */
			ImagePointer coeffs2 = decompositionFilter->GetOutput();
					
			/** Create an iterator on the new coefficient image. */
			IteratorType iterator( coeffs2, gridRegionHigh );
			iterator.GoToBegin();
			while ( !iterator.IsAtEnd() )
			{
				/** Copy the contents of coeffs2 in a ParametersType array. */
				parameters_out[ i ] = iterator.Get();
				++iterator;
				++i;
			} // end while coeff2 iterator loop
			
		} // end for dimension loop
		
		/** Set the new grid definition in the BSplineTransform. */
		this->m_BSplineTransform->SetGridRegion( gridRegionHigh );
		this->m_BSplineTransform->SetGridOrigin( gridOriginHigh );
		this->m_BSplineTransform->SetGridSpacing( gridSpacingHigh );

    /** Set the initial parameters for the next level */
		this->m_Registration->GetAsITKBaseType()->
			SetInitialTransformParametersOfNextLevel( parameters_out );

    /** Set the parameters in the BsplineTransform */
    this->m_BSplineTransform->SetParameters(
      this->m_Registration->GetAsITKBaseType()->
      GetInitialTransformParametersOfNextLevel() );
	
	}  // end IncreaseScale()
	

	/**
	 * ************************* ReadFromFile ************************
	 */

	template <class TElastix>
	void BSplineTransform<TElastix>::
		ReadFromFile( void )
	{
		/** Read and Set the Grid: this is a BSplineTransform specific task. */

		/** Declarations. */
		RegionType	gridregion;
		SizeType		gridsize;
		IndexType		gridindex;
		SpacingType	gridspacing;
		OriginType	gridorigin;
		
		/** Fill everything with default values. */
		gridsize.Fill(1);
		gridindex.Fill(0);
		gridspacing.Fill(1.0);
		gridorigin.Fill(0.0);

		/** Get GridSize, GridIndex, GridSpacing and GridOrigin. */
		for ( unsigned int i = 0; i < SpaceDimension; i++ )
		{
			this->m_Configuration->ReadParameter( gridsize[ i ], "GridSize", i );
			this->m_Configuration->ReadParameter( gridindex[ i ], "GridIndex", i );
			this->m_Configuration->ReadParameter( gridspacing[ i ], "GridSpacing", i );
			this->m_Configuration->ReadParameter( gridorigin[ i ], "GridOrigin", i );
		}
		
		/** Set it all. */
		gridregion.SetIndex( gridindex );
		gridregion.SetSize( gridsize );
		this->m_BSplineTransform->SetGridRegion( gridregion );
		this->m_BSplineTransform->SetGridSpacing( gridspacing );
		this->m_BSplineTransform->SetGridOrigin( gridorigin );

		/** Call the ReadFromFile from the TransformBase.
		 * This must be done after setting the Grid, because later the
		 * ReadFromFile from TransformBase calls SetParameters, which
		 * checks the parameter-size, which is based on the GridSize.
		 */
		this->Superclass2::ReadFromFile();

	} // end ReadFromFile()


	/**
	 * ************************* WriteToFile ************************
	 *
	 * Saves the TransformParameters as a vector and if wanted
	 * also as a deformation field.
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>::
		WriteToFile( const ParametersType & param )
	{
		/** Call the WriteToFile from the TransformBase. */
		this->Superclass2::WriteToFile( param );

		/** Add some BSplineTransform specific lines. */
		xout["transpar"] << std::endl << "// BSplineTransform specific" << std::endl;

		/** Get the GridSize, GridIndex, GridSpacing and
		 * GridOrigin of this transform.
		 */
		SizeType size = this->m_BSplineTransform->GetGridRegion().GetSize();
		IndexType index = this->m_BSplineTransform->GetGridRegion().GetIndex();
		SpacingType spacing = this->m_BSplineTransform->GetGridSpacing();
		OriginType origin = this->m_BSplineTransform->GetGridOrigin();

		/** Write the GridSize of this transform. */
		xout["transpar"] << "(GridSize ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << size[ i ] << " ";
		}
		xout["transpar"] << size[ SpaceDimension - 1 ] << ")" << std::endl;
		
		/** Write the GridIndex of this transform. */
		xout["transpar"] << "(GridIndex ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << index[ i ] << " ";
		}
		xout["transpar"] << index[ SpaceDimension - 1 ] << ")" << std::endl;
		
		/** Set the precision of cout to 2, because GridSpacing and
		 * GridOrigin must have at least one digit precision.
		 */
		xout["transpar"] << std::setprecision(10);

		/** Write the GridSpacing of this transform. */
		xout["transpar"] << "(GridSpacing ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << spacing[ i ] << " ";
		}
		xout["transpar"] << spacing[ SpaceDimension - 1 ] << ")" << std::endl;

		/** Write the GridOrigin of this transform. */
		xout["transpar"] << "(GridOrigin ";
		for ( unsigned int i = 0; i < SpaceDimension - 1; i++ )
		{
			xout["transpar"] << origin[ i ] << " ";
		}
		xout["transpar"] << origin[ SpaceDimension - 1 ] << ")" << std::endl;

		/** Set the precision back to default value. */
		xout["transpar"] << std::setprecision(
			this->m_Elastix->GetDefaultOutputPrecision() );

	} // end WriteToFile()

	
	/**
	 * ******************** ComputeInitialGridSpacing_Deprecated *********************
	 *
	 * Computes m_GridSpacingFactor for the first resolution.
	 */

	template <class TElastix>
		bool
    BSplineTransform<TElastix>
    ::ComputeInitialGridSpacing_Deprecated( void )
	{
		/** Read the desired grid spacing for each dimension for the final resolution.
		 * If only one gridspacing factor is given, that one is used for each dimension.
		 */
    SpacingType finalGridSpacing;
    finalGridSpacing[ 0 ] = 8.0;
		int ret = this->m_Configuration->ReadParameter(
      finalGridSpacing[ 0 ], "FinalGridSpacing", 0 );
    finalGridSpacing.Fill( finalGridSpacing[ 0 ] );
		for ( unsigned int i = 1; i < SpaceDimension; ++i )
		{
      this->m_Configuration->ReadParameter(
        finalGridSpacing[ i ], "FinalGridSpacing", i );
		}

    /** If ret is 1, then the parameter file did NOT contain "FinalGridSpacing". */
    if ( ret == 1 )
    {
      /** This deprecated option is not used, and we let the caller know. */
      return false;
    }
    else
    {
      /** This option is deprecated, issue a warning. */
      elxout["warning"] << "WARNING: the option \"FinalGridSpacing\" is deprecated." << std::endl;
      elxout["warning"] << "Use EITHER \"GridSpacingSchedule\" together with \"GridSpacingUpsampleFactor\"" << std::endl;
      elxout["warning"] << "OR \"GridSpacingScheduleFull\" instead." << std::endl;
    }
    
		/** If multigrid, then start with a lower resolution grid.
		 * First, we have to find out what the resolution is for the initial grid,
		 * i.e. the grid in the first resolution. This depends on the number of times
		 * the grid has to be upsampled. The user can specify this amount with the
		 * option "UpsampleGridOption".
		 * - In case the user specifies only one such option
		 * it is assumed that between all resolutions upsampling is needed or not.
		 * This is also in line with former API (backwards compatability):
		 *    (UpsampleGridOption "true") or (UpsampleGridOption "false")
		 * In this case the factor is multiplied with 2^(nrOfResolutions - 1).
		 * - In the other case the user has to specify after each resolution
		 * whether or not the grid should be upsampled, i.e. (nrOfResolutions - 1)
		 * times. For 4 resolutions this is done as:
		 *    (UpsampleGridOption "true" "false" "true")
		 * In this case the factor is multiplied with 2^(# of true's).
		 * - In case nothing is given in the parameter-file, by default the
		 * option (UpsampleGridOption "true") is assumed.
		 */

		/** Get the number of resolutions. */
		int nrOfResolutions = static_cast<int>(
			this->GetRegistration()->GetAsITKBaseType()->GetNumberOfLevels() );
    unsigned int size = vnl_math_max( 1, nrOfResolutions - 1 );

		/** Fill upsampleBSplineGridOption. */
		bool tmp = true;
		this->GetConfiguration()->ReadParameter( tmp, "UpsampleGridOption", 0 );
		std::vector< bool > upsampleGridOption( size, tmp );
		for ( unsigned int i = 1; i < nrOfResolutions - 1; ++i )
		{
      tmp = upsampleGridOption[ i ];
      this->m_Configuration->ReadParameter(
        tmp, "UpsampleGridOption", i );
      upsampleGridOption[ i ] = tmp;
      // strangely the following does not compile??
      //this->m_Configuration->ReadParameter( upsampleGridOption[ i ], "UpsampleGridOption", i );
		}

    /** Create a B-spline grid schedule. */
    GridScheduleType schedule( nrOfResolutions, finalGridSpacing );
    float factor = 2.0;
    unsigned int j = 0;
    for ( int i = nrOfResolutions - 2; i > -1; --i )
    {
      if ( upsampleGridOption[ j ] )
      {
        schedule[ i ] *= factor;
        factor *= factor;
      }
      j++;
    }

    /** Set the grid spacing schedule. */
    this->m_GridScheduleComputer->SetGridSpacingSchedule( schedule );

    /** Set other required information. */
    this->m_GridScheduleComputer->SetOrigin(
      this->GetElastix()->GetFixedImage()->GetOrigin() );
    this->m_GridScheduleComputer->SetSpacing(
      this->GetElastix()->GetFixedImage()->GetSpacing() );
    this->m_GridScheduleComputer->SetRegion(
      this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion() );

    /** Compute the necessary information. */
    this->m_GridScheduleComputer->ComputeBSplineGrid();

    /** Return true. */
    return true;

	} // end ComputeInitialGridSpacing_Deprecated()

	
	/**
	 * *********************** DefineGrid ************************
	 *
	 * Defines the grid region, origin and spacing.
	 * 
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>::
		DefineGrid(RegionType & gridregion,
			OriginType & gridorigin, SpacingType & gridspacing ) const
	{
		/** Typedefs. *
		typedef ImageRegionExclusionConstIteratorWithIndex<FixedImageType>
			BoundaryIteratorType;
		typedef typename Superclass1::InitialTransformConstPointer 
			InitialTransformConstPointer;
				
		/** Declarations. *
		RegionType  fixedImageRegion;
		SizeType    fixedImageSize;
		IndexType   fixedImageIndex;
		OriginType  fixedImageOrigin;
		SpacingType fixedImageSpacing;
		SizeType		gridsize;
		IndexType		gridindex;
		typename FixedImageType::ConstPointer fixedImage;
				
		/** Get the fixed image. *
		fixedImage = this->m_Registration->GetAsITKBaseType()->GetFixedImage();
		
		/** Get the region (size and index), spacing, and origin  of this image. */

		/** In elastix <=3.001: fixedImageRegion	=	fixedimage->GetRequestedRegion();  */
		/** later (because requested regions were not supported anyway consistently: *
		fixedImageRegion = fixedImage->GetLargestPossibleRegion();
		/** \todo: allow the user to enter a region of interest for the registration. 
		 * Especially the boundary conditions have to be dealt with carefully then. *
		fixedImageIndex		=	fixedImageRegion.GetIndex();
		/** \todo: always 0? doesn't a largestpossible region have an index 0 by definition? *
		fixedImageSize		=	fixedImageRegion.GetSize();
		fixedImageSpacing	=	fixedImage->GetSpacing();
		fixedImageOrigin	=	fixedImage->GetOrigin();
		
		/** Take into account the initial transform, if composition is used to 
		 * combine it with the current (bspline) transform *
		if ( (this->GetUseComposition()) && (this->Superclass1::GetInitialTransform() != 0) )
		{
			/** We have to determine a bounding box around the fixed image after
			 * applying the initial transform. This is done by iterating over the
			 * the boundary of the fixed image, evaluating the initial transform
			 * at those points, and keeping track of the minimum/maximum transformed
			 * coordinate in each dimension 
			 */

			/** Make a temporary copy; who knows, maybe some speedup can be achieved... *
			InitialTransformConstPointer initialTransform = 
				this->Superclass1::GetInitialTransform();
			/** The points that define the bounding box *
			InputPointType maxPoint;
			InputPointType minPoint;
			maxPoint.Fill( NumericTraits< CoordRepType >::NonpositiveMin() );
			minPoint.Fill( NumericTraits< CoordRepType >::max() );
			/** An iterator over the boundary of the fixed image *
			BoundaryIteratorType bit(fixedImage, fixedImageRegion);
			bit.SetExclusionRegionToInsetRegion();
			bit.GoToBegin();
			/** start loop over boundary; determines minPoint and maxPoint *
			while ( !bit.IsAtEnd() )
			{
				/** Get index, transform to physical point, apply initial transform
				 * NB: the OutputPointType of the initial transform by definition equals
				 * the InputPointType of this transform. *
        IndexType inputIndex = bit.GetIndex();
				InputPointType inputPoint;
				fixedImage->TransformIndexToPhysicalPoint(inputIndex, inputPoint);
				InputPointType outputPoint = 
					initialTransform->TransformPoint(	inputPoint );
				/** update minPoint and maxPoint *
				for ( unsigned int i = 0; i < SpaceDimension; i++ )
				{
					CoordRepType & outi = outputPoint[i];
					CoordRepType & maxi = maxPoint[i];
					CoordRepType & mini = minPoint[i];
					if ( outi > maxi )
					{
						maxi = outi;
					}
					if ( outi < mini )
					{
						mini = outi;
					}
				}
				/** step to next voxel *
				++bit;
			} //end while loop over fixed image boundary

			/** Set minPoint as new "fixedImageOrigin" (between quotes, since it
			 * is not really the origin of the fixedImage anymore) *
			fixedImageOrigin = minPoint;

			/** Compute the new "fixedImageSpacing" in each dimension *
			const double smallnumber = NumericTraits<double>::epsilon();
			for ( unsigned int i = 0; i < SpaceDimension; i++ )
			{
				/** Compute the length of the fixed image (in mm) for dimension i *
				double oldLength_i = 
					fixedImageSpacing[i] * static_cast<double>( fixedImageSize[i] - 1 );
				/** Compute the length of the bounding box (in mm) for dimension i *
        double newLength_i = static_cast<double>( maxPoint[i] - minPoint[i] );
				/** Scale the fixedImageSpacing by their ratio. *
				if (oldLength_i > smallnumber)
				{
					fixedImageSpacing[i] *= ( newLength_i / oldLength_i );
				}				
			}

		  /** We have now adapted the fixedImageOrigin and fixedImageSpacing.
			 * This makes sure that the BSpline grid is located at the position
			 * of the fixed image after undergoing the initial transform.  *
		     		  
		} // end if UseComposition && InitialTransform!=0

		/** Determine the grid region (size and index), origin and spacing.
		 * \li The fixed image spacing is multiplied by the m_GridSpacingFactor
		 *     to compute the gridspacing.
		 * \li Some extra grid points are put at the edges, to take into account 
		 *     the support region of the B-splines.
		 *
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			gridspacing[ j ] = fixedImageSpacing[ j ] * this->m_GridSpacingFactor[ j ];
			gridorigin[ j ]  = fixedImageOrigin[ j ] - 
				gridspacing[ j ] * vcl_floor( static_cast<double>( SplineOrder ) / 2.0 );
			gridindex[ j ]   = 0; // \todo: isn't this always the case anyway?

      /** The grid size without the extra grid points at the edges. *
      const unsigned int bareGridSize = static_cast<unsigned int>( 
        vcl_ceil( fixedImageSize[ j ] / this->m_GridSpacingFactor[ j ] ) );
			gridsize[ j ] = static_cast< typename RegionType::SizeValueType >(
				bareGridSize + SplineOrder );

      /** Shift the origin a little to the left, to place the grid
       * symmetrically on the image.
       *
      gridorigin[ j ] -= ( gridspacing[ j ] * bareGridSize
        - fixedImageSpacing[ j ] * ( fixedImageSize[ j ] - 1 ) ) / 2.0;
		}
		gridregion.SetSize( gridsize );
		gridregion.SetIndex( gridindex );
				*/
	} // end DefineGrid()


  /**
	 * *********************** SetOptimizerScales ***********************
	 * Set the optimizer scales of the edge coefficients to infinity.
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>::
		SetOptimizerScales( unsigned int edgeWidth )
	{
    typedef ImageRegionExclusionConstIteratorWithIndex<ImageType>		IteratorType;
    typedef typename RegistrationType::ITKBaseType					ITKRegistrationType;
		typedef typename ITKRegistrationType::OptimizerType			OptimizerType;
		typedef typename OptimizerType::ScalesType							ScalesType;
    typedef typename ScalesType::ValueType                  ScalesValueType;

    /** Define new scales */
    const unsigned long numberOfParameters
      = this->m_BSplineTransform->GetNumberOfParameters();
    const unsigned long offset = numberOfParameters / SpaceDimension;
    ScalesType newScales( numberOfParameters );
    newScales.Fill( NumericTraits<ScalesValueType>::One );
    const ScalesValueType infScale = 10000.0;
    
    if ( edgeWidth == 0 )
    { 
      /** Just set the unit scales into the optimizer. */
		  this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newScales );
      return;
    }

		/** Get the grid region information and create a fake coefficient image. */
    RegionType gridregion = this->m_BSplineTransform->GetGridRegion();
    SizeType gridsize = gridregion.GetSize();
    IndexType gridindex = gridregion.GetIndex();
    ImagePointer coeff = ImageType::New();
    coeff->SetRegions( gridregion );
    coeff->Allocate();
    
    /** Determine inset region. (so, the region with active parameters). */
    RegionType insetgridregion;
    SizeType insetgridsize;
    IndexType insetgridindex;
    for ( unsigned int i = 0; i < SpaceDimension; ++i )
    {
      insetgridsize[ i ] = static_cast<unsigned int>( vnl_math_max( 0, 
        static_cast<int>( gridsize[ i ] - 2 * edgeWidth ) ) );
      if ( insetgridsize[ i ] == 0 )
      {
        xl::xout["error"] 
          << "ERROR: you specified a PassiveEdgeWidth of "
          << edgeWidth
          << " while the total grid size in dimension " 
          << i
          << " is only "
          << gridsize[ i ] << "." << std::endl;
        itkExceptionMacro( << "ERROR: the PassiveEdgeWidth is too large!" );
      }
      insetgridindex[ i ] = gridindex[ i ] + edgeWidth;
    }
    insetgridregion.SetSize( insetgridsize );
    insetgridregion.SetIndex( insetgridindex );

    /** Set up iterator over the coefficient image. */
    IteratorType cIt( coeff, coeff->GetLargestPossibleRegion() );
    cIt.SetExclusionRegion( insetgridregion );
    cIt.GoToBegin();   
    
    /** Set the scales to infinity that correspond to edge coefficients
     * This (hopefully) makes sure they are not optimised during registration.
     */
    while ( !cIt.IsAtEnd() )
    {
      const IndexType & index = cIt.GetIndex();
      const unsigned long baseOffset = coeff->ComputeOffset( index );
      for ( unsigned int i = 0; i < SpaceDimension; ++i )
      {
        const unsigned int scalesIndex = static_cast<unsigned int>(
          baseOffset + i * offset );
        newScales[ scalesIndex ] = infScale;
      }
      ++cIt;
    }

    /** Set the scales into the optimizer. */
		this->m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newScales );

  } // end SetOptimizerScales()

	
} // end namespace elastix


#endif // end #ifndef __elxBSplineTransform_hxx

