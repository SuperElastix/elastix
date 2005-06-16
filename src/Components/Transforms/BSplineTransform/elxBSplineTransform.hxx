#ifndef __elxBSplineTransform_hxx
#define __elxBSplineTransform_hxx

#include "elxBSplineTransform.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "math.h"

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
		/** Initialize.*/		
		this->m_Coeffs1 = 0;
		
		this->m_GridSpacingFactor.Fill(8.0);

		this->m_Caster	= TransformCastFilterType::New();
		this->m_Writer	= TransformWriterType::New();
		
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
		this->m_Registration->GetAsITKBaseType()->SetInitialTransformParameters( dummyInitialParameters );
		
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>
		::BeforeEachResolution(void)
	{
		/** What is the current resolution level?*/
		unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

		/** What is the UpsampleGridOption?
		 * This option defines the user's wish:
		 * - true: For lower resolution levels (i.e. smaller images),
		 *				 the GridSpacing is made larger, as a power of 2.
		 * - false: The GridSpacing remains equal for each resolution level.
		 */
		std::string upsampleBSplineGridOption( "true" );
		bool upsampleGridOption = true;
		this->m_Configuration->ReadParameter( upsampleBSplineGridOption, "UpsampleGridOption", 0 );
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
		void BSplineTransform<TElastix>::
		SetInitialGrid( bool upsampleGridOption )
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
			this->m_Registration->GetAsITKBaseType()->GetFixedImage() );
		
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
		
		/** Read the desired grid spacing for each dimension. If only one gridspacing factor
		 * is given, that one is used for each dimension. */
		this->m_GridSpacingFactor[0]=8.0;
		this->m_Configuration->ReadParameter( this->m_GridSpacingFactor[0], "FinalGridSpacing", 0);
    this->m_GridSpacingFactor.Fill( this->m_GridSpacingFactor[0] );
		for ( unsigned int j = 1; j < SpaceDimension; j++ )
		{
      this->m_Configuration->ReadParameter( this->m_GridSpacingFactor[j], "FinalGridSpacing", j);
		}

		/** If multigrid, then start with a lower resolution grid */
		if (upsampleGridOption)
		{
			/** cast to int, otherwise gcc does not understand it... */
			int nrOfResolutions = static_cast<int>(
				this->GetRegistration()->GetAsITKBaseType()->GetNumberOfLevels()  );
			this->m_GridSpacingFactor *= pow(2.0, (nrOfResolutions-1) );
		}

		/** Determine the correct grid size */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			gridspacing[ j ] = gridspacing[ j ] * this->m_GridSpacingFactor[j];
			gridorigin[j] -= gridspacing[j] *
				floor( static_cast<double>(SplineOrder) / 2.0 );
			gridindex[j] = 0; // isn't this always the case anyway?
			gridsize[j]= static_cast< typename RegionType::SizeValueType >
				( ceil( gridsize[j] / this->m_GridSpacingFactor[j] ) + SplineOrder );
		}
		
		/** Set the size data in the transform */
		gridregion.SetSize( gridsize );
		gridregion.SetIndex( gridindex );
		this->SetGridRegion( gridregion );
		this->SetGridSpacing( gridspacing );
		this->SetGridOrigin( gridorigin );
		
		/** Set initial parameters to 0.0 */
		this->m_Parameterspointer =
			new ParametersType( this->GetNumberOfParameters() );
		this->m_Parameterspointer->Fill(0.0);
		this->m_Registration->GetAsITKBaseType()->
			SetInitialTransformParametersOfNextLevel( *(this->m_Parameterspointer) );
		delete this->m_Parameterspointer;
		
	} // end SetInitialGrid()
	
	
	/**
	 * *********************** IncreaseScale ************************
	 *
	 * Upsample the grid of control points.
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>::
		IncreaseScale(void)
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
			this->m_Registration->GetAsITKBaseType()->GetFixedImage() );
		
		/** Set start values for computing the new grid size. */
		RegionType gridregionHigh	= fixedimage->GetLargestPossibleRegion();
		IndexType gridindexHigh		=	gridregionHigh.GetIndex();
		SizeType gridsizeHigh		=	gridregionHigh.GetSize();
		SpacingType gridspacingHigh	=	fixedimage->GetSpacing();
		OriginType gridoriginHigh	=	fixedimage->GetOrigin();
		
		/** A twice as dense grid: */
		this->m_GridSpacingFactor /= 2;

		/** Determine the correct grid size */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			gridspacingHigh[ j ] = gridspacingHigh[ j ] * this->m_GridSpacingFactor[j];
			gridoriginHigh[j] -= gridspacingHigh[j] *
				floor( static_cast<double>(SplineOrder) / 2.0 );
			gridindexHigh[j] = 0; // isn't this always the case anyway?
			gridsizeHigh[j]= static_cast< typename RegionType::SizeValueType >
				( ceil( gridsizeHigh[j] / this->m_GridSpacingFactor[j] ) + SplineOrder );
		}
		gridregionHigh.SetSize(gridsizeHigh);
		gridregionHigh.SetIndex(gridindexHigh);

		/** Get the latest transform parameters */
		this->m_Parameterspointer =
			new ParametersType( this->GetNumberOfParameters() );
		*(this->m_Parameterspointer) =
			this->m_Registration->GetAsITKBaseType()->GetLastTransformParameters();
		
		/** Get the pointer to the data in *(this->m_Parameterspointer) */
		PixelType * dataPointer = static_cast<PixelType *>( this->m_Parameterspointer->data_block() );
		/** Get the number of pixels that should go into one coefficient image */
		unsigned int numberOfPixels = ( this->GetGridRegion() ).GetNumberOfPixels();
		
		/** Set the correct region/size info of the coeff image
		* that will be filled with the current parameters */
		this->m_Coeffs1 = ImageType::New();
		this->m_Coeffs1->SetRegions( this->GetGridRegion() );
		this->m_Coeffs1->SetOrigin( (this->GetGridOrigin()).GetDataPointer() );
		this->m_Coeffs1->SetSpacing( (this->GetGridSpacing()).GetDataPointer() );
		//this->m_Coeffs1->Allocate() not needed because the data is set by directly pointing
		// to an existing piece of memory.
		
		/** 
		 * Create the new vector of parameters, with the 
		 * correct size (which is now approx 2^dim as big as the
		 * size in the previous resolution!)
		 */
		this->m_Parameterspointer_out = new ParametersType(
			gridregionHigh.GetNumberOfPixels() * SpaceDimension );

		/** initialise iterator in the parameterspointer_out */
		unsigned int i = 0; 
		
		/** Loop over dimension. */
		for ( unsigned int j = 0; j < SpaceDimension; j++ )
		{
			/** Fill the coeff image with parameter data (displacements
			 * of the control points in the direction of dimension j).
			 */		
			this->m_Coeffs1->GetPixelContainer()->
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
		  upsampler->SetInput( this->m_Coeffs1 );
						
			decompositionFilter->SetSplineOrder(SplineOrder);
			decompositionFilter->SetInput( upsampler->GetOutput() );

			/** Do the upsampling.*/
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
			
			/** Create an upsampled image.*/
			this->m_Coeffs2 = decompositionFilter->GetOutput();
					
			/** Create an iterator on the new coefficient image*/
			IteratorType iterator( this->m_Coeffs2, gridregionHigh );
			iterator.GoToBegin();
			while ( !iterator.IsAtEnd() )
			{
				/** Copy the contents of coeff2 in a ParametersType array*/
				(*(this->m_Parameterspointer_out))[ i ] = iterator.Get();
				++iterator;
				++i;
			} // end while coeff2 iterator loop
			
		} // end for dimension loop
		
		/** Set the initial parameters for the next resolution level.*/
		this->SetGridRegion( gridregionHigh );
		this->SetGridOrigin( gridoriginHigh );
		this->SetGridSpacing( gridspacingHigh );
		this->m_Registration->GetAsITKBaseType()->
			SetInitialTransformParametersOfNextLevel( *(this->m_Parameterspointer_out) );
		
		delete this->m_Parameterspointer;
		delete this->m_Parameterspointer_out;	
		
	}  // end IncreaseScale()
	

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
		xout["transpar"] << std::setprecision(10);

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
		xout["transpar"] << std::setprecision( this->m_Elastix->GetDefaultOutputPrecision() );

		/** If wanted, write the TransformParameters as deformation
		 * images to a file.
		 */
		if ( 0 )
		{
			//??????? is dit nodig:
			//this->SetParameters( param );nee

			/** Get the pointer to the data in 'param' */
			PixelType * dataPointer = const_cast<PixelType *>(
				static_cast<const PixelType *>( param.data_block() )		);
			unsigned int numberOfPixels =
				(this->GetGridRegion()).GetNumberOfPixels();
			
			/** Initialise the coeffs image */
			this->m_Coeffs1->SetRegions( this->GetGridRegion() );
			this->m_Coeffs1->SetOrigin( (this->GetGridOrigin()).GetDataPointer() );
			this->m_Coeffs1->SetSpacing( (this->GetGridSpacing()).GetDataPointer() );
			
			for ( unsigned int i = 0; i < SpaceDimension; i++ )
			{
				/** Get the set of parameters that represent the control point
				 * displacements in the i-th dimension.
				 */
				this->m_Coeffs1->GetPixelContainer()->
					SetImportPointer( dataPointer, numberOfPixels );
				dataPointer += numberOfPixels;
				this->m_Coeffs1->Modified();
				
				/** Create complete filename: "name"."dimension".mhd
				 * --> two files are created: a header (.mhd) and a data (.raw) file.
				 */
				std::ostringstream makeFileName( "" );
				makeFileName << this->m_Configuration->GetCommandLineArgument("-t")	<< "." << i << ".mhd";
				this->m_Writer->SetFileName( makeFileName.str().c_str() );
				

				/** Write the coefficient image (i-th dimension) to file. */
				this->m_Caster->SetInput( this->m_Coeffs1 );
				this->m_Writer->SetInput( this->m_Caster->GetOutput() );

				/** Do the writing.*/
				try
				{
					this->m_Writer->Update();
				}
				catch( itk::ExceptionObject & excp )
				{
					/** Add information to the exception. */
					excp.SetLocation( "BSplineTransform - WriteToFile()" );
					std::string err_str = excp.GetDescription();
					err_str += "\nError occured while writing B-spline coefficient image.\n";
					excp.SetDescription( err_str );
					/** Print the exception. */
					xl::xout["error"] << excp << std::endl;
				}
				
				/** Force the writer to make a new .raw file.*/
				this->m_Writer->SetImageIO(NULL);
				
			}  // end for i
			
		} // end if

	} // end WriteToFile


	
} // end namespace elastix


#endif // end #ifndef __elxBSplineTransform_hxx

