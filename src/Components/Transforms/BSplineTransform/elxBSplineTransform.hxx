#ifndef __elxBSplineTransform_hxx
#define __elxBSplineTransform_hxx

#include "elxBSplineTransform.h"

#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkImageRegionConstIterator.h"
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
		/** Initialize. */
		this->m_GridSpacingFactor.Fill( 8.0 );
		this->m_UpsampleBSplineGridOption.push_back( true );

	} // end Constructor
	

	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>
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
		this->SetGridRegion( gridregion );
		this->SetGridSpacing( gridspacing );
		this->SetGridOrigin( gridorigin );
    
		/** Task 2 - Give the registration an initial parameter-array. */
		ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
		dummyInitialParameters.Fill( 0.0 );
		
		/** Put parameters in the registration. */
		this->m_Registration->GetAsITKBaseType()
			->SetInitialTransformParameters( dummyInitialParameters );
		
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>
		::BeforeEachResolution(void)
	{
		/** What is the current resolution level? */
		unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

		/** What is the UpsampleGridOption? See SetInitialGrid(). */
		
		/** Resample the grid. */
		if ( level == 0 )
		{
			/** Set grid equal to lowest resolution fixed image. */
			this->SetInitialGrid();			
		}	
		else
		{
			/** Check if the BSpline grid should be upsampled now. */
			if ( this->m_UpsampleBSplineGridOption[ level - 1 ] ) this->IncreaseScale();
			/** Otherwise, nothing is done with the BSpline-Grid. */
		}
	
	} // end BeforeEachResolution
	
	
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
	 * \todo This is not up to date anymore.
	 *
	 * In both cases some extra grid points are put at the edges,
	 * to take into account the support region of the B-splines.
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>::
		SetInitialGrid()
	{
		/** Declarations. */
		RegionType	gridregion;
		SizeType		gridsize;
		IndexType		gridindex;
		SpacingType	gridspacing;
		OriginType	gridorigin;
		
		/** Get the fixed image. */
		typename FixedImageType::Pointer fixedimage;
		fixedimage = const_cast< FixedImageType * >(
			this->m_Registration->GetAsITKBaseType()->GetFixedImage() );
		
		/** Get the size etc. of this image. */

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
		
		/** Read the desired grid spacing for each dimension for the final resolution.
		 * If only one gridspacing factor is given, that one is used for each dimension.
		 */
		this->m_GridSpacingFactor[ 0 ] = 8.0;
		this->m_Configuration->ReadParameter( this->m_GridSpacingFactor[ 0 ], "FinalGridSpacing", 0 );
    this->m_GridSpacingFactor.Fill( this->m_GridSpacingFactor[ 0 ] );
		for ( unsigned int j = 1; j < SpaceDimension; j++ )
		{
      this->m_Configuration->ReadParameter( this->m_GridSpacingFactor[ j ], "FinalGridSpacing", j );
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
		 *
		 * \todo maybe replace this to beforeregistration.
		 */

		/** Get the number of resolutions. */
		int nrOfResolutions = static_cast<int>(
			this->GetRegistration()->GetAsITKBaseType()->GetNumberOfLevels() );

		/** Fill m_UpsampleBSplineGridOption. */
		std::string tmp = "true";
		this->GetConfiguration()->ReadParameter( tmp, "UpsampleGridOption", 0 );
		unsigned int size = vnl_math_max( 1, nrOfResolutions - 1 );
		std::vector< std::string > upsampleGridOptionVector( size, tmp );
		if ( tmp == "false" ) this->m_UpsampleBSplineGridOption[ 0 ] = false;
		this->m_UpsampleBSplineGridOption.resize( size );
		for ( unsigned int i = 1; i < nrOfResolutions - 1; i++ )
		{
      this->m_Configuration->ReadParameter( upsampleGridOptionVector[ i ], "UpsampleGridOption", i );
			if ( upsampleGridOptionVector[ i ] == "true" ) this->m_UpsampleBSplineGridOption[ i ] = true;
			else this->m_UpsampleBSplineGridOption[ i ] = false;
		}

		/** Multiply with the m_GridSpacingFactor. */
		if ( nrOfResolutions != 1 )
		{
			/** Upsample only if true for this level. */
			double numberOfTrues = 0.0;
			for ( unsigned int i = 0; i < nrOfResolutions - 1; i++ )
			{
				if ( this->m_UpsampleBSplineGridOption[ i ] ) numberOfTrues++;
			}
			this->m_GridSpacingFactor *= vcl_pow( 2.0, numberOfTrues );
		}

		/** Determine the correct grid size. */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			gridspacing[ j ] = gridspacing[ j ] * this->m_GridSpacingFactor[ j ];
			gridorigin[ j ] -= gridspacing[ j ] *
				vcl_floor( static_cast<double>( SplineOrder ) / 2.0 );
			gridindex[ j ] = 0; // isn't this always the case anyway?
			gridsize[ j ]= static_cast< typename RegionType::SizeValueType >
				( vcl_ceil( gridsize[ j ] / this->m_GridSpacingFactor[ j ] ) + SplineOrder );
		}
		
		/** Set the size data in the transform. */
		gridregion.SetSize( gridsize );
		gridregion.SetIndex( gridindex );
		this->SetGridRegion( gridregion );
		this->SetGridSpacing( gridspacing );
		this->SetGridOrigin( gridorigin );
		
		/** Set initial parameters to 0.0. */
		ParametersType initialParameters( this->GetNumberOfParameters() );
		initialParameters.Fill( 0.0 );
		this->m_Registration->GetAsITKBaseType()
			->SetInitialTransformParametersOfNextLevel( initialParameters );
		
	} // end SetInitialGrid
	
	
	/**
	 * *********************** IncreaseScale ************************
	 *
	 * Upsample the grid of control points.
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>::
		IncreaseScale(void)
	{
		/** Typedefs. */
		typedef itk::ResampleImageFilter<ImageType, ImageType>
			UpsampleFilterType;
		typedef itk::IdentityTransform<CoordRepType, SpaceDimension>
			IdentityTransformType;
		typedef itk::BSplineResampleImageFunction<ImageType, CoordRepType> 
			CoefficientUpsampleFunctionType;
		typedef itk::BSplineDecompositionImageFilter<ImageType,ImageType>
			DecompositionFilterType;
		typedef ImageRegionConstIterator<ImageType>		IteratorType;

		/** The current region/spacing settings of the grid. */
		RegionType gridregionLow = this->GetGridRegion();
		SizeType gridsizeLow = gridregionLow.GetSize();
		IndexType gridindexLow = gridregionLow.GetIndex();
		SpacingType gridspacingLow = this->GetGridSpacing();
		OriginType gridoriginLow = this->GetGridOrigin();

		/** Get the fixed image. */
		typename FixedImageType::Pointer fixedimage;
		fixedimage = const_cast< FixedImageType * >(
			this->m_Registration->GetAsITKBaseType()->GetFixedImage() );
		
		/** Set start values for computing the new grid size. */
		RegionType gridregionHigh	= fixedimage->GetLargestPossibleRegion();
		IndexType gridindexHigh		=	gridregionHigh.GetIndex();
		SizeType gridsizeHigh		=	gridregionHigh.GetSize();
		SpacingType gridspacingHigh	=	fixedimage->GetSpacing();
		OriginType gridoriginHigh	=	fixedimage->GetOrigin();
		
		/** A twice as dense grid: */
		this->m_GridSpacingFactor /= 2;

		/** Determine the correct grid size. */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			gridspacingHigh[ j ] = gridspacingHigh[ j ] * this->m_GridSpacingFactor[ j ];
			gridoriginHigh[ j ] -= gridspacingHigh[ j ] *
				vcl_floor( static_cast<double>( SplineOrder ) / 2.0 );
			gridindexHigh[ j ] = 0; // isn't this always the case anyway?
			gridsizeHigh[ j ]= static_cast< typename RegionType::SizeValueType >
				( vcl_ceil( gridsizeHigh[ j ] / this->m_GridSpacingFactor[ j ] ) + SplineOrder );
		}
		gridregionHigh.SetSize(gridsizeHigh);
		gridregionHigh.SetIndex(gridindexHigh);

		/** Get the latest transform parameters. */
		ParametersType latestParameters =
			this->m_Registration->GetAsITKBaseType()->GetLastTransformParameters();
		
		/** Get the pointer to the data in latestParameters. */
		PixelType * dataPointer = static_cast<PixelType *>( latestParameters.data_block() );
		/** Get the number of pixels that should go into one coefficient image. */
		unsigned int numberOfPixels = ( this->GetGridRegion() ).GetNumberOfPixels();
		
		/** Set the correct region/size info of the coefficient image
		 * that will be filled with the current parameters.
		 */
		ImagePointer coeffs1 = ImageType::New();
		coeffs1->SetRegions( this->GetGridRegion() );
		coeffs1->SetOrigin( (this->GetGridOrigin()).GetDataPointer() );
		coeffs1->SetSpacing( (this->GetGridSpacing()).GetDataPointer() );
		//coeffs1->Allocate() not needed because the data is set by directly pointing
		// to an existing piece of memory.
		
		/** 
		 * Create the new vector of parameters, with the 
		 * correct size (which is now approx 2^dim as big as the
		 * size in the previous resolution!).
		 */
		ParametersType parameters_out(
			gridregionHigh.GetNumberOfPixels() * SpaceDimension );

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
			 * of the new control points (note: it does not just interpolate
			 * the coefficient image, which would be wrong). The b-spline
			 * coefficients that describe the resulting image are computed
			 * by the decomposition filter.
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
      upsampler->SetSize( gridsizeHigh );
			upsampler->SetOutputStartIndex( gridindexHigh );
			upsampler->SetOutputSpacing( gridspacingHigh );
			upsampler->SetOutputOrigin( gridoriginHigh );
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
			IteratorType iterator( coeffs2, gridregionHigh );
			iterator.GoToBegin();
			while ( !iterator.IsAtEnd() )
			{
				/** Copy the contents of coeffs2 in a ParametersType array. */
				parameters_out[ i ] = iterator.Get();
				++iterator;
				++i;
			} // end while coeff2 iterator loop
			
		} // end for dimension loop
		
		/** Set the initial parameters for the next resolution level. */
		this->SetGridRegion( gridregionHigh );
		this->SetGridOrigin( gridoriginHigh );
		this->SetGridSpacing( gridspacingHigh );
		this->m_Registration->GetAsITKBaseType()->
			SetInitialTransformParametersOfNextLevel( parameters_out );
	
	}  // end IncreaseScale
	

	/**
	 * ************************* ReadFromFile ************************
	 */

	template <class TElastix>
	void BSplineTransform<TElastix>::
		ReadFromFile(void)
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
			this->m_Configuration->ReadParameter( gridsize[ i ], "GridSize", i );
			this->m_Configuration->ReadParameter( gridindex[ i ], "GridIndex", i );
			this->m_Configuration->ReadParameter( gridspacing[ i ], "GridSpacing", i );
			this->m_Configuration->ReadParameter( gridorigin[ i ], "GridOrigin", i );
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
		SizeType size = this->GetGridRegion().GetSize();
		IndexType index = this->GetGridRegion().GetIndex();
		SpacingType spacing = this->GetGridSpacing();
		OriginType origin = this->GetGridOrigin();

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
		xout["transpar"] << std::setprecision( this->m_Elastix->GetDefaultOutputPrecision() );

	} // end WriteToFile


	
} // end namespace elastix


#endif // end #ifndef __elxBSplineTransform_hxx

