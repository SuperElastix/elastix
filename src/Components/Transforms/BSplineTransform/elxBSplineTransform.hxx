#ifndef __elxBSplineTransform_hxx
#define __elxBSplineTransform_hxx

#include "elxBSplineTransform.h"
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
		m_Coeffs1 = ImageType::New();
		m_Upsampler = UpsamplerType::New();
		m_Upsampler->SetSplineOrder(SplineOrder); 

		m_Caster	= TransformCastFilterType::New();
		m_Writer	= TransformWriterType::New();
		
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
		m_Registration->GetAsITKBaseType()->SetInitialTransformParameters( dummyInitialParameters );
		
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void BSplineTransform<TElastix>
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
	 * ********************* SetInitialGrid *************************
	 *
	 * Set grid equal to lowest resolution fixed image.
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
		void BSplineTransform<TElastix>::
		IncreaseScale(void)
	{
		/** \todo Implement the method as in the example DeformableRegistration6.cxx */
		
	
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
		
		/** Loop over dimension. */
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
			//??????? is dit nodig:
			//this->SetParameters( param );nee

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
				
				/** Create complete filename: "name"."dimension".mhd
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


	
} // end namespace elastix


#endif // end #ifndef __elxBSplineTransform_hxx

