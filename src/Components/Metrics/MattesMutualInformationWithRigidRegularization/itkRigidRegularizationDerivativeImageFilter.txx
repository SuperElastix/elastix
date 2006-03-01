/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkRigidRegularizationDerivativeImageFilter_txx
#define _itkRigidRegularizationDerivativeImageFilter_txx

#include "itkRigidRegularizationDerivativeImageFilter.h"

/** Include splitter and combiner of vector images. */
#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkScalarToArrayCastImageFilter.h"

#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "itkProgressAccumulator.h"

//tmp
#include <iostream>
#include <iomanip>
#include "itkCastImageFilter.h"
#include "itkVectorCastImageFilter.h"

namespace itk
{

	/**
	 * ************************ Constructor **************************
	 */

	template< class TInputImage, class TOutputImage >
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::RigidRegularizationDerivativeImageFilter()
	{
		this->m_UseImageSpacing = true;
		this->m_SecondOrderWeight = 1.0;
		this->m_OrthonormalityWeight = 1.0;
		this->m_PropernessWeight = 1.0;
		this->m_RigidRegulizerValue = NumericTraits<InputVectorValueType>::Zero;
		this->m_GenerateDataCalled = false;
		this->m_RigidityImage = 0;
		
	} // end Constructor


	/**
	 * ************************ PrintSelf ***************************
	 */
  
	template< class TInputImage, class TOutputImage >
		void
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::PrintSelf( std::ostream& os, Indent indent ) const  
	{
		/** Call the superclass' PrintSelf. */
		Superclass::PrintSelf( os, indent );

		/** Add debugging information. */
		os << indent << "UseImageSpacing = "
			<< this->m_UseImageSpacing << std::endl;
		os << indent << "ImageSpacingUsed = ";
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			os << this->m_ImageSpacingUsed[ i ] << " ";
		}
		os << std::endl;
		os << indent << "SecondOrderWeight: "
			<< this->m_SecondOrderWeight << std::endl;
		os << indent << "OrthonormalityWeight: "
			<< this->m_OrthonormalityWeight << std::endl;
		os << indent << "PropernessWeight: "
			<< this->m_PropernessWeight << std::endl;
		os << indent << "RigidityImage: "
			<< this->m_RigidityImage << std::endl;
		os << indent << "OutputDirectoryName: "
			<< this->m_OutputDirectoryName << std::endl;
		os << indent << "RigidRegulizerValue: "
			<< this->m_RigidRegulizerValue << std::endl;
		os << indent << "GenerateDataCalled: "
			<< this->m_GenerateDataCalled << std::endl;

	} // end PrintSelf


	/**
	 * **************** GenerateInputRequestedRegion ****************
	 */
	
	template <class TInputImage, class TOutputImage>
		void 
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::GenerateInputRequestedRegion() throw( InvalidRequestedRegionError )
	{
		/** Call the superclass' implementation of this method. This should
		 * copy the output requested region to the input requested region.
		 */
		Superclass::GenerateInputRequestedRegion();

		/** Get pointers to the input and output. */
		InputVectorImagePointer  inputPtr =
			const_cast< InputVectorImageType * > ( this->GetInput() );

		if ( !inputPtr )
		{
			return;
		}

		/** Build an operator so that we can determine the kernel size. */
		//SOOperatorType oper;

		/** Get a copy of the input requested region, which should
		 * equal the output requested region. */
		typename InputVectorImageType::RegionType inputRequestedRegion;
		inputRequestedRegion = inputPtr->GetRequestedRegion();

		/** Pad the input requested region by the operator radius. */
		//inputRequestedRegion.PadByRadius( oper.GetRadius() );
		RadiusInputType radius;
		radius.Fill( 1 );
		inputRequestedRegion.PadByRadius( radius );

		/** Crop the input requested region at the input's largest possible region. */
		if ( inputRequestedRegion.Crop( inputPtr->GetLargestPossibleRegion() ) )
		{
			inputPtr->SetRequestedRegion( inputRequestedRegion );
			return;
		}
		else
		{
			/** Couldn't crop the region (requested region is outside the largest
			 * possible region). Throw an exception.
			 */

			/** Store what we tried to request (prior to trying to crop). */
			inputPtr->SetRequestedRegion( inputRequestedRegion );

			/** Build an exception. */
			InvalidRequestedRegionError e(__FILE__, __LINE__);
			OStringStream msg;
			msg << static_cast<const char *>( this->GetNameOfClass() )
				<< "::GenerateInputRequestedRegion()";
			e.SetLocation( msg.str().c_str() );
			e.SetDescription( "Requested region is (at least partially) outside the largest possible region." );
			e.SetDataObject( inputPtr );
			throw e;
		}

	} // end GenerateInputRequestedRegion


	/**
	 * ************************** SetImageSpacingUsed ***********************
	 */

	template< class TInputImage, class TOutputImage >
		void
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::SetImageSpacingUsed()
	{
		/** Check the image spacing and get the SpacingScaling s. */
		double s[ ImageDimension ];
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			if ( this->GetInput( 0 )->GetSpacing()[ i ] == 0.0 )
			{
				itkExceptionMacro( << "Image spacing cannot be zero" );
			}
			else
			{
				s[ i ] = this->GetInput( 0 )->GetSpacing()[ i ];
			}
		}

		/** Set m_ImageSpacingUsed to the input image spacing if
		 * requested and to 1.0 otherwise.
		 */
		if ( this->GetUseImageSpacing() )
		{
			for ( unsigned int i = 0; i < ImageDimension; i++ ) m_ImageSpacingUsed[ i ] = s[ i ];
		}
		else
		{
			for ( unsigned int i = 0; i < ImageDimension; i++ ) m_ImageSpacingUsed[ i ] = 1.0;
		}

	} // end SetImageSpacingUsed


	/**
	 * ************************** GenerateData ***********************
	 */

	template< class TInputImage, class TOutputImage >
		void
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::GenerateData()
	{
		/** Sanity check. */
		if ( ImageDimension != 2 && ImageDimension != 3 )
		{
			itkExceptionMacro( << "ERROR: This filter is only implemented for dimension 2 and 3." );
		}

		/** Some typedef's. */
		typedef VectorIndexSelectionCastImageFilter<
			InputVectorImageType, InputScalarImageType >			VectorImageSplitterType;
		typedef typename VectorImageSplitterType::Pointer		VectorImageSplitterPointer;
		typedef ScalarToArrayCastImageFilter<
			OutputScalarImageType, OutputVectorImageType >		ScalarImageCombineType;
		typedef typename ScalarImageCombineType::Pointer		ScalarImageCombinePointer;

		/** Create stuff. */

		/** Create 1D operators. */
		std::vector< NeighborhoodType > Operators_A( ImageDimension ),
			Operators_B( ImageDimension ), Operators_C( ImageDimension );
		std::vector< NeighborhoodType > Operators_D( ImageDimension ),
			Operators_E( ImageDimension ), Operators_F( ImageDimension ),
			Operators_G( ImageDimension ), Operators_H( ImageDimension ),
			Operators_I( ImageDimension );
		/** Create scalar images that are filtered once. */
		std::vector< InputScalarImagePointer > ui_FA( ImageDimension ),
			ui_FB( ImageDimension ), ui_FC( ImageDimension );
		std::vector< InputScalarImagePointer > ui_FD( ImageDimension ),
			ui_FE( ImageDimension ), ui_FF( ImageDimension ),
			ui_FG( ImageDimension ), ui_FH( ImageDimension ),
			ui_FI( ImageDimension );
		/** Create iterators over ui_F. */
		std::vector< OutputScalarImageIteratorType > itA( ImageDimension ),
			itB( ImageDimension ), itC( ImageDimension );
		/** Create neighborhood iterators over ui_F. */
		std::vector< NeighborhoodIteratorOutputType >
			nit1_FD( ImageDimension ), nit1_FE( ImageDimension ),
			nit1_FF( ImageDimension ), nit1_FG( ImageDimension ),
			nit1_FH( ImageDimension ), nit1_FI( ImageDimension );
		/** Create first order parts. */
		std::vector< OutputScalarImagePointer > FOParts( ImageDimension );
		/** Create iterators over the first and second order parts and over the output. */
		std::vector< OutputScalarImageIteratorType > foit( ImageDimension ),
			soit( ImageDimension ), oit( ImageDimension );
		/** Create a neigborhood iterator over the rigidity image. */
		RadiusInputType radiusIn;
		radiusIn.Fill( 1 );
		NeighborhoodIteratorInputType nit_RI( radiusIn, this->m_RigidityImage,
			this->m_RigidityImage->GetLargestPossibleRegion() );
		/** Create ND operators. */
		NeighborhoodType Operator_A, Operator_B, Operator_C;
		NeighborhoodType Operator_D, Operator_E, Operator_F,
			Operator_G, Operator_H, Operator_I;
		/** Create second order parts. */
		std::vector< OutputScalarImagePointer >		SOParts( ImageDimension );


		/** TASK 0:
		 * Get the input image, split it over its dimensions, and create an output image.
		 ************************************************************************* */

		/** Get a handle to the input vector image. */
		InputVectorImagePointer inputImage = const_cast< InputVectorImageType * >( this->GetInput() );

		//tmp
		if(1)
		{
			typedef Vector< float, ImageDimension >		FloatVecType;
			typedef Image<FloatVecType, ImageDimension >	FloatVecImageType;
			typedef ImageFileWriter<FloatVecImageType>	FloatVeccWriterType;
			typedef VectorCastImageFilter<InputVectorImageType,FloatVecImageType> CasterrType;
			typename CasterrType::Pointer castere = CasterrType::New();
			//typedef ImageFileWriter<InputVectorImageType>	VeccWriterType;
			//typename VeccWriterType::Pointer writere = VeccWriterType::New();
			typename FloatVeccWriterType::Pointer writere = FloatVeccWriterType::New();
			castere->SetInput( inputImage );
			writere->SetFileName( "parameters.mhd" );
			writere->SetInput( castere->GetOutput() );
			//writere->SetInput( inputImage );
			writere->Update();

			typedef Image< float, ImageDimension > FloatImage;
			typedef CastImageFilter< InputScalarImageType, FloatImage > CasterrrrType;
			typedef ImageFileWriter<FloatImage>	ScallWriterType;
			typename CasterrrrType::Pointer caster = CasterrrrType::New();
			typename ScallWriterType::Pointer writersc = ScallWriterType::New();
			caster->SetInput( m_RigidityImage );
			writersc->SetFileName( "rigidity.mhd" );
			writersc->SetInput( caster->GetOutput() );
			writersc->Update();
		}

		/** Split the input over the dimensions. */
		std::vector< InputScalarImagePointer >	inputImages( ImageDimension );
		std::vector< VectorImageSplitterPointer >	splitters( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			//inputImages[ i ] = InputScalarImageType::New();
			splitters[ i ] = VectorImageSplitterType::New();
			splitters[ i ]->SetInput( inputImage );
			splitters[ i ]->SetIndex( i );
			inputImages[ i ] = splitters[ i ]->GetOutput();

			/** Split the image. */
			try
			{
        inputImages[ i ]->Update();
			}
			catch( ExceptionObject & err )
			{
				std::cerr << "ExceptionObject caught !" << std::endl;
				std::cerr << err << std::endl;
			}
		}

		/** Create and allocate the output vector image. */
		OutputVectorImagePointer outputImage = this->GetOutput();
		outputImage->SetRegions( inputImage->GetLargestPossibleRegion() );
		outputImage->Allocate();

		/** Error. */
		if ( !inputImage || !outputImage )
    {
    return;
    }

		/** Set the image spacing that is used. */
		this->SetImageSpacingUsed();

		/** Create scalar output images, each holding a component of the vector field. */
		std::vector< OutputScalarImagePointer > outputImages( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			outputImages[ i ] = OutputScalarImageType::New();
			outputImages[ i ]->SetRegions( inputImages[ i ]->GetLargestPossibleRegion() );
			outputImages[ i ]->Allocate();
		}

		// tmp
		if (1)
		{
			typedef Image< float, ImageDimension > FloatImage;
			typedef ImageFileWriter< FloatImage >		FloatWriterType;
			typedef	typename FloatWriterType::Pointer		FloatWriterPointer;
			typedef CastImageFilter< InputScalarImageType, FloatImage > CasterrrrType;
			typedef typename CasterrrrType::Pointer			CasterrrrPointer;

			// inputimages
			std::vector< FloatWriterPointer > inputWriter( ImageDimension );
			std::vector< CasterrrrPointer > inputCaster( ImageDimension );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				inputWriter[ i ] = FloatWriterType::New();
				inputCaster[ i ] = CasterrrrType::New();
			}
			inputWriter[ 0 ]->SetFileName( "input1.mhd" );
			inputWriter[ 1 ]->SetFileName( "input2.mhd" );
			inputWriter[ 2 ]->SetFileName( "input3.mhd" );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				inputCaster[ i ]->SetInput( inputImages[ i ] );
				inputWriter[ i ]->SetInput( inputCaster[ i ]->GetOutput() );
				inputWriter[ i ]->Update();
			}
		}


		/** TASK 1:
		 * Create the one time filtered images ui_FA, ui_FB and ui_FC.
		 ************************************************************************* */

		/** Create operators A, B (and C in 3D). *
		std::vector< NeighborhoodType > Operators_A( ImageDimension ),
			Operators_B( ImageDimension ), Operators_C( ImageDimension );

		/** Create scalar images that are filtered once. *
		std::vector< InputScalarImagePointer > ui_FA( ImageDimension ),
			ui_FB( ImageDimension ), ui_FC( ImageDimension );

		/** For all dimensions ... */
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			NeighborhoodType tmp_oper;
			/** ... create the filtered images ... */
			ui_FA[ i ] = InputScalarImageType::New();
			ui_FB[ i ] = InputScalarImageType::New();
			if ( ImageDimension == 3 ) ui_FC[ i ] = InputScalarImageType::New();
			/** ... and fill the apropiate operators. */
			//this->Create1DOperator( Operators_A[ i ], "FA_xi", i + 1 );
			this->Create1DOperator( tmp_oper, "FA_xi", i + 1 );
			Operators_A[ i ] = tmp_oper;
			this->Create1DOperator( Operators_B[ i ], "FB_xi", i + 1 );
			if ( ImageDimension == 3 ) this->Create1DOperator( Operators_C[ i ], "FC_xi", i + 1 );
		}

		/** Filter the inputImages. */
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			ui_FA[ i ] = this->FilterSeparable( inputImages[ i ], Operators_A );
			ui_FB[ i ] = this->FilterSeparable( inputImages[ i ], Operators_B );
			if ( ImageDimension == 3 ) ui_FC[ i ] = this->FilterSeparable( inputImages[ i ], Operators_C );
		}

		// tmp
		if (1)
		{
			typedef Image< float, ImageDimension > FloatImage;
			typedef ImageFileWriter< FloatImage >		FloatWriterType;
			typedef	typename FloatWriterType::Pointer		FloatWriterPointer;
			typedef CastImageFilter< InputScalarImageType, FloatImage > CasterrrrType;
			typedef typename CasterrrrType::Pointer			CasterrrrPointer;

			// ui_FA
			std::vector< FloatWriterPointer > ui_FAWriter( ImageDimension );
			std::vector< CasterrrrPointer > ui_FACaster( ImageDimension );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				ui_FAWriter[ i ] = FloatWriterType::New();
				ui_FACaster[ i ] = CasterrrrType::New();
			}
			ui_FAWriter[ 0 ]->SetFileName( "u1_FA.mhd" );
			ui_FAWriter[ 1 ]->SetFileName( "u2_FA.mhd" );
			ui_FAWriter[ 2 ]->SetFileName( "u3_FA.mhd" );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				ui_FACaster[ i ]->SetInput( ui_FA[ i ] );
				ui_FAWriter[ i ]->SetInput( ui_FACaster[ i ]->GetOutput() );
				ui_FAWriter[ i ]->Update();
			}

			// ui_FB
			std::vector< FloatWriterPointer > ui_FBWriter( ImageDimension );
			std::vector< CasterrrrPointer > ui_FBCaster( ImageDimension );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				ui_FBWriter[ i ] = FloatWriterType::New();
				ui_FBCaster[ i ] = CasterrrrType::New();
			}
			ui_FBWriter[ 0 ]->SetFileName( "u1_FB.mhd" );
			ui_FBWriter[ 1 ]->SetFileName( "u2_FB.mhd" );
			ui_FBWriter[ 2 ]->SetFileName( "u3_FB.mhd" );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				ui_FBCaster[ i ]->SetInput( ui_FB[ i ] );
				ui_FBWriter[ i ]->SetInput( ui_FBCaster[ i ]->GetOutput() );
				ui_FBWriter[ i ]->Update();			}
		}


		/** TASK 2:
		 * Create the subparts, which are combinations of ui_FA, ui_FB and ui_FC.
		 ************************************************************************* */

		/** Calculate all subparts of Sdmu_ij: the derivative to mu_{k,i}
		 * of the rigid smoother S.
		 * The number of these parts are equal to the ImageDimension.
		 * The first dimension of Sdmu_ij contains the derivative to mu_i;
		 * the second dimension the two or three parts.
		 */
		std::vector < std::vector< OutputScalarImagePointer > > Sdmu_ij( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			/** Resize the vector. */
			Sdmu_ij[ i ].resize( ImageDimension );
			/** Create parts images of Sdmu_ij. */
			for ( unsigned int j = 0; j < ImageDimension; j++ )
			{
				Sdmu_ij[ i ][ j ] = OutputScalarImageType::New();
				Sdmu_ij[ i ][ j ]->SetRegions( inputImages[ 0 ]->GetLargestPossibleRegion() );
				Sdmu_ij[ i ][ j ]->Allocate();
			}
		}

		/** Create iterators over the subparts and one time filtered images. */
		std::vector< std::vector< OutputScalarImageIteratorType > > it_mu_ij( ImageDimension );
		//std::vector< OutputScalarImageIteratorType > itA( ImageDimension ),
		//	itB( ImageDimension ), itC( ImageDimension );

		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			/** Create iterators over one time filtered images. */
			itA[ i ] = OutputScalarImageIteratorType( ui_FA[ i ], ui_FA[ i ]->GetLargestPossibleRegion() );
			itB[ i ] = OutputScalarImageIteratorType( ui_FB[ i ], ui_FB[ i ]->GetLargestPossibleRegion() );
			if ( ImageDimension == 3 ) itC[ i ] = OutputScalarImageIteratorType( ui_FC[ i ], ui_FC[ i ]->GetLargestPossibleRegion() );
			/** Reset iterators. */
			itA[ i ].GoToBegin(); itB[ i ].GoToBegin();
			if ( ImageDimension == 3 ) itC[ i ].GoToBegin();
			/** Resize the vector with the subparts. */
			it_mu_ij[ i ].resize( ImageDimension );
			/** Create iterators over subparts. */
			for ( unsigned int j = 0; j < ImageDimension; j++ )
			{
				it_mu_ij[ i ][ j ] = OutputScalarImageIteratorType( Sdmu_ij[ i ][ j ],
					Sdmu_ij[ i ][ j ]->GetLargestPossibleRegion() );					
				/** Reset iterators. */
				it_mu_ij[ i ][ j ].GoToBegin();
			}
		}

		/** Create the subparts. */
		while ( !it_mu_ij[ 0 ][ 0 ].IsAtEnd() )
		{
			/** Fill values with the neccesary information. */
			std::vector<OutputVectorValueType> values( ImageDimension * ImageDimension );
			/*values = {{ itA[ 0 ].Get(), itA[ 1 ].Get(), itA[ 2 ].Get(),
			itB[ 0 ].Get(), itB[ 1 ].Get(), itB[ 2 ].Get(),
			itC[ 0 ].Get(), itC[ 1 ].Get(), itC[ 2 ].Get() }};*/
			if ( ImageDimension == 2 )
			{
				values[ 0 ] = itA[ 0 ].Get();
				values[ 1 ] = itA[ 1 ].Get();
				values[ 2 ] = itB[ 0 ].Get();
				values[ 3 ] = itB[ 1 ].Get();
			}
			else if ( ImageDimension == 3 )
			{
				values[ 0 ] = itA[ 0 ].Get();
				values[ 1 ] = itA[ 1 ].Get();
				values[ 2 ] = itA[ 2 ].Get();
				values[ 3 ] = itB[ 0 ].Get();
				values[ 4 ] = itB[ 1 ].Get();
				values[ 5 ] = itB[ 2 ].Get();
				values[ 6 ] = itC[ 0 ].Get();
				values[ 7 ] = itC[ 1 ].Get();
				values[ 8 ] = itC[ 2 ].Get();
			}
			/** Resize the vector. */
			std::vector< std::vector< double > > answer( ImageDimension );
			/** Calculate all subparts. */
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				answer[ i ].resize( ImageDimension );
				for ( unsigned int j = 0; j < ImageDimension; j++ )
				{
					answer[ i ][ j ] = this->CalculateSubPart( i, j, values );
					it_mu_ij[ i ][ j ].Set( answer[ i ][ j ] );
				}
			}
			/** Increase all iterators. */
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				++itA[ i ];++itB[ i ];
				if ( ImageDimension == 3 ) ++itC[ i ];
				for ( unsigned int j = 0; j < ImageDimension; j++ )
				{
					++it_mu_ij[ i ][ j ];
				}
			}
		} // end while

		// tmp
		if (1)
		{
			typedef Image< float, ImageDimension > FloatImage;
			typedef ImageFileWriter< FloatImage >		FloatWriterType;
			typedef	typename FloatWriterType::Pointer		FloatWriterPointer;
			typedef CastImageFilter< OutputScalarImageType, FloatImage > CasterrrrType;
			typedef typename CasterrrrType::Pointer			CasterrrrPointer;

			// subparts
			unsigned int tot = ImageDimension * ImageDimension;
			std::vector< FloatWriterPointer > SdmuWriter( ImageDimension * ImageDimension );
			std::vector< CasterrrrPointer > SdmuCaster( ImageDimension * ImageDimension );
			for ( unsigned int i = 0; i < tot; i++ )
			{
				SdmuWriter[ i ] = FloatWriterType::New();
				SdmuCaster[ i ] = CasterrrrType::New();
			}
			SdmuWriter[ 0 ]->SetFileName( "Sdmu_00.mhd" );
			SdmuWriter[ 1 ]->SetFileName( "Sdmu_01.mhd" );
			SdmuWriter[ 2 ]->SetFileName( "Sdmu_02.mhd" );
			SdmuWriter[ 3 ]->SetFileName( "Sdmu_10.mhd" );
			SdmuWriter[ 4 ]->SetFileName( "Sdmu_11.mhd" );
			SdmuWriter[ 5 ]->SetFileName( "Sdmu_12.mhd" );
			SdmuWriter[ 6 ]->SetFileName( "Sdmu_20.mhd" );
			SdmuWriter[ 7 ]->SetFileName( "Sdmu_21.mhd" );
			SdmuWriter[ 8 ]->SetFileName( "Sdmu_22.mhd" );
			SdmuCaster[ 0 ]->SetInput( Sdmu_ij[ 0 ][ 0 ] );
			SdmuCaster[ 1 ]->SetInput( Sdmu_ij[ 0 ][ 1 ] );
			SdmuCaster[ 2 ]->SetInput( Sdmu_ij[ 0 ][ 2 ] );
			SdmuCaster[ 3 ]->SetInput( Sdmu_ij[ 1 ][ 0 ] );
			SdmuCaster[ 4 ]->SetInput( Sdmu_ij[ 1 ][ 1 ] );
			SdmuCaster[ 5 ]->SetInput( Sdmu_ij[ 1 ][ 2 ] );
			SdmuCaster[ 6 ]->SetInput( Sdmu_ij[ 2 ][ 0 ] );
			SdmuCaster[ 7 ]->SetInput( Sdmu_ij[ 2 ][ 1 ] );
			SdmuCaster[ 8 ]->SetInput( Sdmu_ij[ 2 ][ 2 ] );
			for ( unsigned int i = 0; i < tot; i++ )
			{
				SdmuWriter[ i ]->SetInput( SdmuCaster[ i ]->GetOutput() );
				SdmuWriter[ i ]->Update();
			}
		}

		/** TASK 3:
		 * Create filtered versions of the subparts,
		 * which are the first order parts. They are F_A * {subpart_0} + F_B * {subpart_1},
		 * and (for 3D) + F_C * {subpart_2}, for all dimensions.
		 ************************************************************************* */

		/** Create the first order parts. */
		//std::vector< OutputScalarImagePointer > FOParts( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			/** Resize the vector. */
			FOParts[ i ] = OutputScalarImageType::New();
			FOParts[ i ]->SetRegions( inputImages[ i ]->GetLargestPossibleRegion() );
			FOParts[ i ]->Allocate();
		}

		/** Create (neighborhood) iterators over the first order parts
		 * and over the subparts Sdmu_ij. Also reset them. */
		//std::vector< OutputScalarImageIteratorType > foit( ImageDimension );
		std::vector< std::vector< NeighborhoodIteratorOutputType > >	nit1_sp( ImageDimension );
		RadiusOutputType radiusOut;
		radiusOut.Fill( 1 );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			foit[ i ] = OutputScalarImageIteratorType( FOParts[ i ],
				FOParts[ i ]->GetLargestPossibleRegion() );
			nit1_sp[ i ].resize( ImageDimension );
			for ( unsigned int j = 0; j < ImageDimension; j++ )
			{
				nit1_sp[ i ][ j ] = NeighborhoodIteratorOutputType( radiusOut,
					Sdmu_ij[ i ][ j ], Sdmu_ij[ i ][ j ]->GetLargestPossibleRegion() );

				/** Reset iterators. */
				nit1_sp[ i ][ j ].GoToBegin();
			}
			/** Reset iterators. */
			foit[ i ].GoToBegin();
		}

		/** Create neighborhood iterator over m_RigidityImage and reset it. */
		//RadiusInputType radiusIn;
		//radiusIn.Fill( 1 );
		//NeighborhoodIteratorInputType nit_RI( radiusIn, m_RigidityImage, m_RigidityImage->GetLargestPossibleRegion() );
		nit_RI.GoToBegin();
		unsigned int neighborhoodSize = nit_RI.Size();

		/** Create the ND filters A, B and C. */
		//NeighborhoodType Operator_A, Operator_B, Operator_C;
		this->CreateNDOperator( Operator_A, "FA" );
		this->CreateNDOperator( Operator_B, "FB" );
		if ( ImageDimension == 3 ) this->CreateNDOperator( Operator_C, "FC" );
		
		/** Loop over all pixels. */
		while ( !foit[ 0 ].IsAtEnd() )
		{
			/** Create and reset tmp with zeros. */
			std::vector<double> tmp( ImageDimension,  0.0 );

			/** Loop over all dimensions. */
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
        /** Loop over the neighborhood. */
				for ( unsigned int k = 0; k < neighborhoodSize; ++k )
				{
					/*typedef typename NeighborhoodIteratorOutputType::IndexType NIndexType;
					NIndexType ind = nit1_sp[ i ][ 0 ].GetIndex();
					double opA, opB, sp0, sp1, ck;
					if ( ind[0]==60 && ind[1]==60 )
					{
						opA = Operator_A.GetElement( k );
						opB = Operator_B.GetElement( k );
						sp0 = nit1_sp[ i ][ 0 ].GetPixel( k );
						sp1 = nit1_sp[ i ][ 1 ].GetPixel( k );
						ck = nit_RI.GetPixel( k );
					}
					double ck = nit_RI.GetPixel( k );
					if ( ck < 0.0 || ck > 1.0 )
					{
						std::cerr << "ERROR: rigidity smaller than 0.0 or larger than 1.0" << std::endl;
					}*/
					/** Calculation of the inner product. */
					tmp[ i ] += Operator_A.GetElement( k ) *		// FA *
						nit1_sp[ i ][ 0 ].GetPixel( k ) *					// subpart[ i ][ 0 ]
						nit_RI.GetPixel( k );											// c(k)
					tmp[ i ] += Operator_B.GetElement( k ) *		// FB *
						nit1_sp[ i ][ 1 ].GetPixel( k ) *					// subpart[ i ][ 1 ]
						nit_RI.GetPixel( k );											// c(k)
					if ( ImageDimension == 3 )
					{
						tmp[ i ] += Operator_C.GetElement( k ) *	// FC *
							nit1_sp[ i ][ 2 ].GetPixel( k ) *				// subpart[ i ][ 2 ]
							nit_RI.GetPixel( k );										// c(k)
					}
				} // end loop over neighborhood

				/** Set the result in the majorpart. */
				foit[ i ].Set( 2.0 * tmp[ i ] );

			} // end loop over dimension i

			/** Increase all iterators. */
			++nit_RI;
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				++foit[ i ];
				for ( unsigned int j = 0; j < ImageDimension; j++ )
				{
					++nit1_sp[ i ][ j ];
				}
			}

		} // end while

		// tmp
		if(1)
		{
			typedef Image< float, ImageDimension > FloatImage;
			typedef ImageFileWriter< FloatImage >		FloatWriterType;
			typedef	typename FloatWriterType::Pointer		FloatWriterPointer;
			typedef CastImageFilter< OutputScalarImageType, FloatImage > CasterrrrType;
			typedef typename CasterrrrType::Pointer			CasterrrrPointer;

			// FOparts
			std::vector< FloatWriterPointer > FOWriter( ImageDimension );
			std::vector< CasterrrrPointer > FOCaster( ImageDimension );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				FOWriter[ i ] = FloatWriterType::New();
				FOCaster[ i ] = CasterrrrType::New();
			}
			FOWriter[ 0 ]->SetFileName( "FO_0.mhd" );
			FOWriter[ 1 ]->SetFileName( "FO_1.mhd" );
			FOWriter[ 2 ]->SetFileName( "FO_2.mhd" );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				FOCaster[ i ]->SetInput( FOParts[ i ] );
				FOWriter[ i ]->SetInput( FOCaster[ i ]->GetOutput() );
				FOWriter[ i ]->Update();
			}
		}

		/** Filter the subparts. *
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			majorParts[ i ][ 0 ] = this->FilterSeparable( Sdmu_ij[ i ][ 0 ], Operators_A );
			majorParts[ i ][ 1 ] = this->FilterSeparable( Sdmu_ij[ i ][ 1 ], Operators_B );
			if ( ImageDimension == 3 )
			{
				majorParts[ i ][ 2 ] = this->FilterSeparable( Sdmu_ij[ i ][ 2 ], Operators_C );
			}
		}

		/** TASK 4:
		 * Create the first order parts Sdmu_i,
		 * which is an addition of the major parts over j.
		 ************************************************************************* *

		/** Create the first order parts Sdmu_i. *
		std::vector< OutputScalarImagePointer > Sdmu_i( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			Sdmu_i[ i ] = OutputScalarImageType::New();
			Sdmu_i[ i ]->SetRegions( inputImages[ 0 ]->GetLargestPossibleRegion() );
			Sdmu_i[ i ]->Allocate();
		}

		/** Create iterators over the majorParts and Sdmu_i. *
		std::vector< std::vector< OutputScalarImageIteratorType > > it_majorP_ij( ImageDimension );
		std::vector< OutputScalarImageIteratorType > foit( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			/** Create iterators over the first order parts Sdmu_i images. *
			foit[ i ] = OutputScalarImageIteratorType( Sdmu_i[ i ],
				Sdmu_i[ i ]->GetLargestPossibleRegion() );
			/** Reset iterators. *
			foit[ i ].GoToBegin();
			/** Resize the vector with iterators over the majorParts. *
			it_majorP_ij[ i ].resize( ImageDimension );
			/** Create iterators over the major parts. *
			for ( unsigned int j = 0; j < ImageDimension; j++ )
			{
				it_majorP_ij[ i ][ j ] = OutputScalarImageIteratorType( majorParts[ i ][ j ],
					majorParts[ i ][ j ]->GetLargestPossibleRegion() );					
				/** Reset iterators. *
				it_majorP_ij[ i ][ j ].GoToBegin();
			}
		}

		/** Create the first order parts Sdmu_i. *
		while ( !it_majorP_ij[ 0 ][ 0 ].IsAtEnd() )
		{
			/** Add all major parts to form the first order parts. *
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				OutputVectorValueType temp = 0.0;
				for ( unsigned int j = 0; j < ImageDimension; j++ )
				{
					temp += it_majorP_ij[ i ][ j ].Get();
					/** Increase iterator. *
					++it_majorP_ij[ i ][ j ];
				}
				foit[ i ].Set( -2.0 * temp );
				/** Increase iterator. *
				++foit[ i ];
			}
		} // end while

		/** TASK 5:
		* Create the SecondOrder part.
		************************************************************************* */

		/** For the 2D case we use a non-separable filter; for the 3D
		* case a separable one.
		*/
		
		/** Create the SecondOrderRegularizationNonSeparableOperator operator,
		 * which is implemented in another class. *
		std::vector< SOOperatorType>							SOoper( ImageDimension );

		/** Create second order part images. */
		//std::vector< OutputScalarImagePointer >		SOParts( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			SOParts[ i ] = OutputScalarImageType::New();
			SOParts[ i ]->SetRegions( inputImages[ i ]->GetLargestPossibleRegion() );
			SOParts[ i ]->Allocate();
		}

		/** Create operators D, E, F, G, H and I. *
		std::vector< NeighborhoodType > Operators_D( ImageDimension ),
			Operators_E( ImageDimension ), Operators_F( ImageDimension ),
			Operators_G( ImageDimension ), Operators_H( ImageDimension ),
			Operators_I( ImageDimension );

		/** Create scalar images that are filtered once. *
		std::vector< InputScalarImagePointer > ui_FD( ImageDimension ),
			ui_FE( ImageDimension ), ui_FF( ImageDimension ),
			ui_FG( ImageDimension ), ui_FH( ImageDimension ),
			ui_FI( ImageDimension );

		/** D, E and G correspond in 2D to C, D and E from the paper. */

		/** For all dimensions ... */
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			/** ... create the filtered images ... */
			ui_FD[ i ] = InputScalarImageType::New();
			ui_FE[ i ] = InputScalarImageType::New();
			ui_FG[ i ] = InputScalarImageType::New();
			if ( ImageDimension == 3 )
			{
				ui_FF[ i ] = InputScalarImageType::New();
				ui_FH[ i ] = InputScalarImageType::New();
				ui_FI[ i ] = InputScalarImageType::New();
			}
			/** ... and the apropiate operators. */
			this->Create1DOperator( Operators_D[ i ], "FD_xi", i + 1 );
			this->Create1DOperator( Operators_E[ i ], "FE_xi", i + 1 );
			this->Create1DOperator( Operators_G[ i ], "FG_xi", i + 1 );
			if ( ImageDimension == 3 )
			{
				this->Create1DOperator( Operators_F[ i ], "FF_xi", i + 1 );
				this->Create1DOperator( Operators_H[ i ], "FH_xi", i + 1 );
				this->Create1DOperator( Operators_I[ i ], "FI_xi", i + 1 );
			}
		}

		/** Filter the inputImages once. */
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			ui_FD[ i ] = this->FilterSeparable( inputImages[ i ], Operators_D );
			ui_FE[ i ] = this->FilterSeparable( inputImages[ i ], Operators_E );
			ui_FG[ i ] = this->FilterSeparable( inputImages[ i ], Operators_G );
			if ( ImageDimension == 3 )
			{
				ui_FF[ i ] = this->FilterSeparable( inputImages[ i ], Operators_F );
				ui_FH[ i ] = this->FilterSeparable( inputImages[ i ], Operators_H );
				ui_FI[ i ] = this->FilterSeparable( inputImages[ i ], Operators_I );
			}
		}

		/** Create iterators over the second order parts. *
		std::vector< OutputScalarImageIteratorType > soit( ImageDimension );
		/** Initialize them. */
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			soit[ i ] = OutputScalarImageIteratorType( SOParts[ i ],
				SOParts[ i ]->GetLargestPossibleRegion() );
			soit[ i ].GoToBegin();
		}

		/** Create neighborhood iterators over the second order subparts. *
		std::vector< NeighborhoodIteratorOutputType >
			nit1_FD( ImageDimension ), nit1_FE( ImageDimension ),
			nit1_FF( ImageDimension ), nit1_FG( ImageDimension ),
			nit1_FH( ImageDimension ), nit1_FI( ImageDimension );
		/** Initialize them. */
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			nit1_FD[ i ] = NeighborhoodIteratorOutputType( radiusOut,
				ui_FD[ i ], ui_FD[ i ]->GetLargestPossibleRegion() );
			nit1_FD[ i ].GoToBegin();
			nit1_FE[ i ] = NeighborhoodIteratorOutputType( radiusOut,
				ui_FE[ i ], ui_FE[ i ]->GetLargestPossibleRegion() );
			nit1_FE[ i ].GoToBegin();
			nit1_FG[ i ] = NeighborhoodIteratorOutputType( radiusOut,
				ui_FG[ i ], ui_FG[ i ]->GetLargestPossibleRegion() );
			nit1_FG[ i ].GoToBegin();
			if ( ImageDimension == 3 )
			{
				nit1_FF[ i ] = NeighborhoodIteratorOutputType( radiusOut,
					ui_FF[ i ], ui_FF[ i ]->GetLargestPossibleRegion() );
				nit1_FF[ i ].GoToBegin();
				nit1_FH[ i ] = NeighborhoodIteratorOutputType( radiusOut,
					ui_FH[ i ], ui_FH[ i ]->GetLargestPossibleRegion() );
				nit1_FH[ i ].GoToBegin();
				nit1_FI[ i ] = NeighborhoodIteratorOutputType( radiusOut,
					ui_FI[ i ], ui_FI[ i ]->GetLargestPossibleRegion() );
				nit1_FI[ i ].GoToBegin();
			}
		}

		/** Reset neighborhood iterator over m_RigidityImage. */
		nit_RI.GoToBegin();

		/** Create the ND filters D - I. */
		//NeighborhoodType Operator_D, Operator_E, Operator_F,
			//Operator_G, Operator_H, Operator_I;
		this->CreateNDOperator( Operator_D, "FD" );
		this->CreateNDOperator( Operator_E, "FE" );
		this->CreateNDOperator( Operator_G, "FG" );
		if ( ImageDimension == 3 )
		{
			this->CreateNDOperator( Operator_F, "FF" );
			this->CreateNDOperator( Operator_H, "FH" );
			this->CreateNDOperator( Operator_I, "FI" );
		}
		
		/** Loop over all pixels. */
		while ( !soit[ 0 ].IsAtEnd() )
		{
			/** Reset tmp with zeros. */
			std::vector<double> tmp( ImageDimension, 0.0 );

			/** Loop over all dimensions. */
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
        /** Loop over the neighborhood. */
				for ( unsigned int k = 0; k < neighborhoodSize; ++k )
				{
					/** Calculation of the inner product. */
					tmp[ i ] += Operator_D.GetElement( k ) *	// FD *
						nit1_FD[ i ].GetPixel( k ) *						// mu_iFD[ i ]
						nit_RI.GetPixel( k );										// * c(k)
					tmp[ i ] += Operator_E.GetElement( k ) *	// FE *
						nit1_FE[ i ].GetPixel( k ) *						// mu_iFE[ i ]
						nit_RI.GetPixel( k );										// * c(k)
					tmp[ i ] += Operator_G.GetElement( k ) *	// FG *
						nit1_FG[ i ].GetPixel( k ) *						// mu_iFG[ i ]
						nit_RI.GetPixel( k );										// * c(k)
					if ( ImageDimension == 3 )
					{
						tmp[ i ] += Operator_F.GetElement( k ) *	// FF *
							nit1_FF[ i ].GetPixel( k ) *						// mu_iFF[ i ]
							nit_RI.GetPixel( k );										// * c(k)
						tmp[ i ] += Operator_H.GetElement( k ) *	// FH *
							nit1_FH[ i ].GetPixel( k ) *						// mu_iFH[ i ]
							nit_RI.GetPixel( k );										// * c(k)
						tmp[ i ] += Operator_I.GetElement( k ) *	// FI *
							nit1_FI[ i ].GetPixel( k ) *						// mu_iFI[ i ]
							nit_RI.GetPixel( k );										// * c(k)
					}
				} // end loop over neighborhood

				/** Set the result in the majorpart. */
				soit[ i ].Set( 2.0 * tmp[ i ] );

			} // end loop over dimension i

			/** Increase all iterators. */
			++nit_RI;
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				++soit[ i ];
				++nit1_FD[ i ];++nit1_FE[ i ];++nit1_FG[ i ];
				if ( ImageDimension == 3 ) ++nit1_FF[ i ];++nit1_FH[ i ];++nit1_FI[ i ];
			}

		} // end while

		// tmp
		if(1)
		{
			typedef Image< float, ImageDimension > FloatImage;
			typedef ImageFileWriter< FloatImage >		FloatWriterType;
			typedef	typename FloatWriterType::Pointer		FloatWriterPointer;
			typedef CastImageFilter< OutputScalarImageType, FloatImage > CasterrrrType;
			typedef typename CasterrrrType::Pointer			CasterrrrPointer;

			// FOparts
			std::vector< FloatWriterPointer > SOWriter( ImageDimension );
			std::vector< CasterrrrPointer > SOCaster( ImageDimension );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				SOWriter[ i ] = FloatWriterType::New();
				SOCaster[ i ] = CasterrrrType::New();
			}
			SOWriter[ 0 ]->SetFileName( "SO_0.mhd" );
			SOWriter[ 1 ]->SetFileName( "SO_1.mhd" );
			SOWriter[ 2 ]->SetFileName( "SO_2.mhd" );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				SOCaster[ i ]->SetInput( SOParts[ i ] );
				SOWriter[ i ]->SetInput( SOCaster[ i ]->GetOutput() );
				SOWriter[ i ]->Update();
			}
		}

		/** Create an iterator over the second order image. *
		std::vector< OutputScalarImageIteratorType >	soit( ImageDimension );

		/** The 2D case: create the operator and do the filtering. *
		if ( ImageDimension == 2 )
		{
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				SOoper[ i ].SetSpacingScalings( this->GetImageSpacingUsed() );
				SOoper[ i ].CreateOperator();
				SOpart[ i ] = this->FilterNonSeparable( inputImages[ i ], SOoper[ i ] );
			}
		}
		/** The 3D case: the filtering is performed separable. *
		else if ( ImageDimension == 3 )
		{
			/** Create operators D, E, F, G, H and I. *
			std::vector< NeighborhoodType > Operators_D( ImageDimension ),
				Operators_E( ImageDimension ), Operators_F( ImageDimension ),
				Operators_G( ImageDimension ), Operators_H( ImageDimension ),
				Operators_I( ImageDimension );

			/** Create scalar images that are filtered once. *
			std::vector< InputScalarImagePointer > ui_FD( ImageDimension ),
				ui_FE( ImageDimension ), ui_FF( ImageDimension ),
				ui_FG( ImageDimension ), ui_FH( ImageDimension ),
				ui_FI( ImageDimension );

			/** Create scalar images that are filtered twice. *
			std::vector< InputScalarImagePointer > ui_FD_FD( ImageDimension ),
				ui_FE_FE( ImageDimension ), ui_FF_FF( ImageDimension ),
				ui_FG_FG( ImageDimension ), ui_FH_FH( ImageDimension ),
				ui_FI_FI( ImageDimension );

			/** For all dimensions ... *
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				/** ... create the filtered images ... *
				ui_FD[ i ] = InputScalarImageType::New();
				ui_FD_FD[ i ] = InputScalarImageType::New();
				ui_FE[ i ] = InputScalarImageType::New();
				ui_FE_FE[ i ] = InputScalarImageType::New();
				ui_FF[ i ] = InputScalarImageType::New();
				ui_FF_FF[ i ] = InputScalarImageType::New();
				ui_FG[ i ] = InputScalarImageType::New();
				ui_FG_FG[ i ] = InputScalarImageType::New();
				ui_FH[ i ] = InputScalarImageType::New();
				ui_FH_FH[ i ] = InputScalarImageType::New();
				ui_FI[ i ] = InputScalarImageType::New();
				ui_FI_FI[ i ] = InputScalarImageType::New();
				/** ... and the apropiate operators. *
				this->Create1DOperator( Operators_D[ i ], "FD_xi", i + 1 );
				this->Create1DOperator( Operators_E[ i ], "FE_xi", i + 1 );
				this->Create1DOperator( Operators_F[ i ], "FF_xi", i + 1 );
				this->Create1DOperator( Operators_G[ i ], "FG_xi", i + 1 );
				this->Create1DOperator( Operators_H[ i ], "FH_xi", i + 1 );
				this->Create1DOperator( Operators_I[ i ], "FI_xi", i + 1 );
			}

			/** Filter the inputImages twice. *
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				ui_FD[ i ] = this->FilterSeparable( inputImages[ i ], Operators_D );
				ui_FD_FD[ i ] = this->FilterSeparable( ui_FD[ i ], Operators_D );
				ui_FE[ i ] = this->FilterSeparable( inputImages[ i ], Operators_E );
				ui_FE_FE[ i ] = this->FilterSeparable( ui_FE[ i ], Operators_E );
				ui_FF[ i ] = this->FilterSeparable( inputImages[ i ], Operators_F );
				ui_FF_FF[ i ] = this->FilterSeparable( ui_FF[ i ], Operators_F );
				ui_FG[ i ] = this->FilterSeparable( inputImages[ i ], Operators_G );
				ui_FG_FG[ i ] = this->FilterSeparable( ui_FG[ i ], Operators_G );
				ui_FH[ i ] = this->FilterSeparable( inputImages[ i ], Operators_H );
				ui_FH_FH[ i ] = this->FilterSeparable( ui_FH[ i ], Operators_H );
				ui_FI[ i ] = this->FilterSeparable( inputImages[ i ], Operators_I );
				ui_FI_FI[ i ] = this->FilterSeparable( ui_FI[ i ], Operators_I );
			}

			/** Allocate memory for second order part images. *
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				SOpart[ i ]->SetRegions( inputImages[ i ]->GetLargestPossibleRegion() );
				SOpart[ i ]->Allocate();
			}

			/** Create iterators. *
			std::vector< OutputScalarImageIteratorType >
				itD( ImageDimension ), itE( ImageDimension ), itF( ImageDimension ),
				itG( ImageDimension ), itH( ImageDimension ), itI( ImageDimension );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				/** Create iterators. *
				soit[ i ] = OutputScalarImageIteratorType( SOpart[ i ], SOpart[ i ]->GetLargestPossibleRegion() );
				itD[ i ] = OutputScalarImageIteratorType( ui_FD_FD[ i ], ui_FD_FD[ i ]->GetLargestPossibleRegion() );
				itE[ i ] = OutputScalarImageIteratorType( ui_FE_FE[ i ], ui_FE_FE[ i ]->GetLargestPossibleRegion() );
				itF[ i ] = OutputScalarImageIteratorType( ui_FF_FF[ i ], ui_FF_FF[ i ]->GetLargestPossibleRegion() );
				itG[ i ] = OutputScalarImageIteratorType( ui_FG_FG[ i ], ui_FG_FG[ i ]->GetLargestPossibleRegion() );
				itH[ i ] = OutputScalarImageIteratorType( ui_FH_FH[ i ], ui_FH_FH[ i ]->GetLargestPossibleRegion() );
				itI[ i ] = OutputScalarImageIteratorType( ui_FI_FI[ i ], ui_FI_FI[ i ]->GetLargestPossibleRegion() );
				/** Reset iterators. *
				soit[ i ].GoToBegin();
				itD[ i ].GoToBegin(); itE[ i ].GoToBegin(); itF[ i ].GoToBegin();
				itG[ i ].GoToBegin(); itH[ i ].GoToBegin(); itI[ i ].GoToBegin();
			}

			/** Do the addition. *
			while ( !soit[ 0 ].IsAtEnd() )
			{
				for ( unsigned int i = 0; i < ImageDimension; i++ )
				{
					/** Add all parts. *
					soit[ i ].Set( -2.0 * ( itD[ i ].Get() + itE[ i ].Get()
						+ itF[ i ].Get() + itG[ i ].Get() + itH[ i ].Get() + itI[ i ].Get() ) );
					/** Increase all iterators. *
					++soit[ i ];
					++itD[ i ];++itE[ i ];++itF[ i ];++itG[ i ];++itH[ i ];++itI[ i ];
				}
			} // end while
		} // end if the 3D case

		/** TASK 6:
		 * Add it all to create the final outputImages.
		 ************************************************************************* */

		/** Create iterators. */
		//std::vector< OutputScalarImageIteratorType >	oit( ImageDimension );
		//std::vector< OutputScalarImageIteratorType >	foit( ImageDimension );
		//std::vector< OutputScalarImageIteratorType >	soit( ImageDimension );

		/** Create the outputimages. */
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			/** Create iterators. */
			oit[ i ] = OutputScalarImageIteratorType( outputImages[ i ],
				outputImages[ i ]->GetLargestPossibleRegion() );
			//foit[ i ] = OutputScalarImageIteratorType( FOParts[ i ],
//				FOParts[ i ]->GetLargestPossibleRegion() );
	//		soit[ i ] = OutputScalarImageIteratorType( SOParts[ i ],
		//		SOParts[ i ]->GetLargestPossibleRegion() );
			/** Reset output iterator, first order iterators and second order iterators. */
			oit[ i ].GoToBegin();
			foit[ i ].GoToBegin();
			soit[ i ].GoToBegin();
		}

		/** Do the addition. */
		while ( !oit[ 0 ].IsAtEnd() )
		{
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				/** Add all parts. */
				// add it all the right way
				oit[ i ].Set( foit[ i ].Get() + this->GetSecondOrderWeight() * soit[ i ].Get() );
				// take only the first order part
				//oit[ i ].Set( foit[ i ].Get() );
				// take only the second order part
				//oit[ i ].Set( soit[ i ].Get() );
				/** Increase all iterators. */
				++oit[ i ];++foit[ i ];++soit[ i ];
			}
		} // end while

		/** TASK 7:
		 * Combine the final outputImages to get an output vector image.
		 ************************************************************************* */

		/** Combine all outputImages to one outputImage. */
		ScalarImageCombinePointer combiner = ScalarImageCombineType::New();
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			combiner->SetInput( i, outputImages[ i ] );
		}
		combiner->Update();
		
		/** Graft the output of the mini-pipeline back onto the filter's output.
		 * This copies back the region ivars and meta-data.
		 */
    this->GraftOutput( combiner->GetOutput() );

		/** TASK 8:
		 * Calculate the VALUE of the rigid regulizer.
		 * \todo seperate this from the calculation of the derivative, so that
		 * it is not necessary to do that heavy calculation stuff if just the
		 * value is required.
		 ************************************************************************* */

		if ( ImageDimension == 2 )
		{
			/** Create a penalty-image. */
			OutputScalarImagePointer pim = OutputScalarImageType::New();
			pim->SetRegions( inputImages[ 0 ]->GetLargestPossibleRegion() );
			pim->Allocate();
			OutputScalarImageIteratorType pimit( pim, pim->GetLargestPossibleRegion() );
			pimit.GoToBegin();

			/** Create operators C, D and E. */
			std::vector< NeighborhoodType > Operators_C( ImageDimension ),
				Operators_D( ImageDimension ), Operators_E( ImageDimension );

			/** Create scalar images that are filtered once. */
			std::vector< InputScalarImagePointer > ui_FC( ImageDimension ),
				ui_FD( ImageDimension ), ui_FE( ImageDimension );

			/** For all dimensions ... */
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				/** ... create the filtered images ... */
				ui_FC[ i ] = InputScalarImageType::New();
				ui_FD[ i ] = InputScalarImageType::New();
				ui_FE[ i ] = InputScalarImageType::New();
				/** ... and the apropiate operators.
				 * The operators C, D and E from the paper are here created
				 * by Create1DOperator D, E and G, because of the 3D case and history.
				 */
				this->Create1DOperator( Operators_C[ i ], "FD_xi", i + 1 );
				this->Create1DOperator( Operators_D[ i ], "FE_xi", i + 1 );
				this->Create1DOperator( Operators_E[ i ], "FG_xi", i + 1 );
			}

			/** Filter the inputImages. */
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				ui_FC[ i ] = this->FilterSeparable( inputImages[ i ], Operators_C );
				ui_FD[ i ] = this->FilterSeparable( inputImages[ i ], Operators_D );
				ui_FE[ i ] = this->FilterSeparable( inputImages[ i ], Operators_E );
			}

			/** Create iterators. */
			std::vector< OutputScalarImageIteratorType >
				itC( ImageDimension ), itD( ImageDimension ), itE( ImageDimension );
			for ( unsigned int i = 0; i < ImageDimension; i++ )
			{
				/** Create iterators. */
				itC[ i ] = OutputScalarImageIteratorType( ui_FC[ i ], ui_FC[ i ]->GetLargestPossibleRegion() );
				itD[ i ] = OutputScalarImageIteratorType( ui_FD[ i ], ui_FD[ i ]->GetLargestPossibleRegion() );
				itE[ i ] = OutputScalarImageIteratorType( ui_FE[ i ], ui_FE[ i ]->GetLargestPossibleRegion() );
				/** Reset iterators. */
				itA[ i ].GoToBegin(); itB[ i ].GoToBegin(); 
				itC[ i ].GoToBegin(); itD[ i ].GoToBegin(); itE[ i ].GoToBegin();
			}

			/** Create iterator over coeficient image. */
			InputScalarImageIteratorType it_Coef( m_RigidityImage,
				m_RigidityImage->GetLargestPossibleRegion() );
			it_Coef.GoToBegin();

			/** Do the addition. */
			while ( !itA[ 0 ].IsAtEnd() )
			{
				/** First order part. */
				InputVectorValueType tmp = 0.0;
				tmp = vcl_pow(
						+ vcl_pow( static_cast<double>( 1.0 + itA[ 0 ].Get() ), 2.0 )
						+ vcl_pow( static_cast<double>( itA[ 1 ].Get() ), 2.0 )
						- 1.0
					, 2.0 )
					+ vcl_pow(
						+ vcl_pow( static_cast<double>( itB[ 0 ].Get() ), 2.0 )
						+ vcl_pow( static_cast<double>( 1.0 + itB[ 1 ].Get() ), 2.0 )
						- 1.0
					, 2.0 )
					+ vcl_pow(
						+ ( 1.0 + itA[ 0 ].Get() ) * ( itB[ 0 ].Get() )
						+ ( itA[ 1 ].Get() ) * ( 1.0 + itB[ 1 ].Get() )
					, 2.0 )
          + vcl_pow(
						+ ( 1.0 + itA[ 0 ].Get() ) * ( 1.0 + itB[ 1 ].Get() )
						- ( itA[ 1 ].Get() ) * ( itB[ 0 ].Get() )
						- 1.0
					, 2.0 )
					/** Second order part. */
					+ this->m_SecondOrderWeight * (
						+ vcl_pow( static_cast<double>( itC[ 0 ].Get() ), 2.0 )
						+ vcl_pow( static_cast<double>( itC[ 1 ].Get() ), 2.0 )
						+ vcl_pow( static_cast<double>( itD[ 0 ].Get() ), 2.0 )
						+ vcl_pow( static_cast<double>( itD[ 1 ].Get() ), 2.0 )
						+ vcl_pow( static_cast<double>( itE[ 0 ].Get() ), 2.0 )
						+ vcl_pow( static_cast<double>( itE[ 1 ].Get() ), 2.0 )
					);

				this->m_RigidRegulizerValue += it_Coef.Get() * tmp;

				/** Fill the penalty image. */
				pimit.Set( it_Coef.Get() * tmp );
				++pimit;

				/** Increase all iterators. */
				for ( unsigned int i = 0; i < ImageDimension; i++ )
				{
					++itA[ i ];++itB[ i ];++itC[ i ];++itD[ i ];++itE[ i ];
				}
				++it_Coef;

			} // end while

			/** Write the penalty image to file. */
			typedef ImageFileWriter< OutputScalarImageType >		PenaltyWriterType;
			typename PenaltyWriterType::Pointer penaltywriter = PenaltyWriterType::New();
			std::string filename1 = this->m_OutputDirectoryName + "penaltyImage.mhd";
			penaltywriter->SetFileName( filename1.c_str() );
			penaltywriter->SetInput( pim );
			penaltywriter->Update();

			/** Write the penalty image to file. */
			typedef ImageFileWriter< InputScalarImageType >			RigidityWriterType;
			typename RigidityWriterType::Pointer rigiditywriter = RigidityWriterType::New();
			std::string filename2 = this->m_OutputDirectoryName + "rigidityImage.mhd";
			rigiditywriter->SetFileName( filename2.c_str() );
			rigiditywriter->SetInput( this->m_RigidityImage );
			rigiditywriter->Update();

		} // end if the 2D case
		else if ( ImageDimension == 3 )
		{
			this->m_RigidRegulizerValue = NumericTraits<InputVectorValueType>::Zero;
		} // end if the 3D case

		/** Check if this function is called. */
		this->m_GenerateDataCalled = true;
	
	} // end GenerateData


	/**
	 * ************************ Create1DOperator *********************
	 */

	template< class TInputImage, class TOutputImage >
		void
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::Create1DOperator( NeighborhoodType & F, std::string WhichF, unsigned int WhichDimension )
	{
		/** Sanity check. */
		if ( WhichDimension > ImageDimension + 1 )
		{
			itkExceptionMacro( << "ERROR: You are trying to filter in dimension " << WhichDimension + 1 );
		}

		/** Create an operator size and set it in the operator. */
		SizeType r;
		r.Fill( NumericTraits<unsigned int>::Zero );
		r[ WhichDimension - 1 ] = 1;
		F.SetRadius( r );

		/** Get the image spacing factors that we are going to use. */
		std::vector< double > s( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ ) s[ i ] = this->GetImageSpacingUsed()[ i ];
		
		/** Create the required operator (neighborhood), depending on
		 * WhichF. The operator is either 3x1 or 1x3 in 2D and
		 * either 3x1x1 or 1x3x1 or 1x1x3 in 3D.
		 */
		if ( WhichF == "FA_xi" && WhichDimension == 1 )
		{
			/** This case refers to the vector
			 * [ B2(3/2)-B2(1/2), B2(1/2)-B2(-1/2), B2(-1/2)-B2(-3/2) ],
			 * which is something like 1/2 * [-1 0 1].
			 */

			/** Fill the operator. */
			F[ 0 ] = -0.5 / s[ 0 ]; F[ 1 ] = 0.0; F[ 2 ] = 0.5 / s[ 0 ];
		}
		else if ( WhichF == "FA_xi" && WhichDimension == 2 )
		{
			/** This case refers to the vector
			 * [ B3(-1), B3(0), B3(1) ],
			 * which is something like 1/6 * [1 4 1].
			 */

			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FA_xi" && WhichDimension == 3 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FB_xi" && WhichDimension == 1 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FB_xi" && WhichDimension == 2 )
		{
			/** Fill the operator. */
			F[ 0 ] = -0.5 / s[ 1 ]; F[ 1 ] = 0.0; F[ 2 ] = 0.5 / s[ 1 ];
		}
		else if ( WhichF == "FB_xi" && WhichDimension == 3 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FC_xi" && WhichDimension == 1 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FC_xi" && WhichDimension == 2 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FC_xi" && WhichDimension == 3 )
		{
			/** Fill the operator. */
			F[ 0 ] = -0.5 / s[ 2 ]; F[ 1 ] = 0.0; F[ 2 ] = 0.5 / s[ 2 ];
		}
		else if ( WhichF == "FD_xi" && WhichDimension == 1 )
		{
			/** This case refers to the vector
			 * [ B1(0), -2*B1(0), B1(0)],
			 * which is something like 1/2 * [1 -2 1].
			 */

			/** Fill the operator. */
			F[ 0 ] = 0.5 / ( s[ 0 ] * s[ 0 ] );
			F[ 1 ] = -1.0 / ( s[ 0 ] * s[ 0 ] );
			F[ 2 ] = 0.5 / ( s[ 0 ] * s[ 0 ] );
		}
		else if ( WhichF == "FD_xi" && WhichDimension == 2 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FD_xi" && WhichDimension == 3 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FE_xi" && WhichDimension == 1 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FE_xi" && WhichDimension == 2 )
		{
			/** Fill the operator. */
			F[ 0 ] = 0.5 / ( s[ 1 ] * s[ 1 ] );
			F[ 1 ] = -1.0 / ( s[ 1 ] * s[ 1 ] );
			F[ 2 ] = 0.5 / ( s[ 1 ] * s[ 1 ] );
		}
		else if ( WhichF == "FE_xi" && WhichDimension == 3 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FF_xi" && WhichDimension == 1 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FF_xi" && WhichDimension == 2 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FF_xi" && WhichDimension == 3 )
		{
			/** Fill the operator. */
			F[ 0 ] = 0.5 / ( s[ 2 ] * s[ 2 ] );
			F[ 1 ] = -1.0 / ( s[ 2 ] * s[ 2 ] );
			F[ 2 ] = 0.5 / ( s[ 2 ] * s[ 2 ] );
		}
		else if ( WhichF == "FG_xi" && WhichDimension == 1 )
		{
			/** Fill the operator. */
			F[ 0 ] = -0.5 / ( s[ 0 ] * s[ 1 ] );
			F[ 1 ] = 0.0;
			F[ 2 ] = 0.5 / ( s[ 0 ] * s[ 1 ] );
		}
		else if ( WhichF == "FG_xi" && WhichDimension == 2 )
		{
			/** Fill the operator. */
			F[ 0 ] = -0.5 / ( s[ 0 ] * s[ 1 ] );
			F[ 1 ] = 0.0;
			F[ 2 ] = 0.5 / ( s[ 0 ] * s[ 1 ] );
		}
		else if ( WhichF == "FG_xi" && WhichDimension == 3 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FH_xi" && WhichDimension == 1 )
		{
			/** Fill the operator. */
			F[ 0 ] = -0.5 / ( s[ 0 ] * s[ 2 ] );
			F[ 1 ] = 0.0;
			F[ 2 ] = 0.5 / ( s[ 0 ] * s[ 2 ] );
		}
		else if ( WhichF == "FH_xi" && WhichDimension == 2 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FH_xi" && WhichDimension == 3 )
		{
			/** Fill the operator. */
			F[ 0 ] = -0.5 / ( s[ 0 ] * s[ 2 ] );
			F[ 1 ] = 0.0;
			F[ 2 ] = 0.5 / ( s[ 0 ] * s[ 2 ] );
		}
		else if ( WhichF == "FI_xi" && WhichDimension == 1 )
		{
			/** Fill the operator. */
			F[ 0 ] = 1.0 / 6.0; F[ 1 ] = 4.0 / 6.0; F[ 2 ] = 1.0 / 6.0;
		}
		else if ( WhichF == "FI_xi" && WhichDimension == 2 )
		{
			/** Fill the operator. */
			F[ 0 ] = -0.5 / ( s[ 1 ] * s[ 2 ] );
			F[ 1 ] = 0.0;
			F[ 2 ] = 0.5 / ( s[ 1 ] * s[ 2 ] );
		}
		else if ( WhichF == "FI_xi" && WhichDimension == 3 )
		{
			/** Fill the operator. */
			F[ 0 ] = -0.5 / ( s[ 1 ] * s[ 2 ] );
			F[ 1 ] = 0.0;
			F[ 2 ] = 0.5 / ( s[ 1 ] * s[ 2 ] );
		}
		else
		{
			/** Throw an exception. */
			itkExceptionMacro( << "Can not create this type of operator." );
		}

	} // end Create1DOperator


	/**
	 * ************************ CreateNDOperator *********************
	 */

	template< class TInputImage, class TOutputImage >
		void
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::CreateNDOperator( NeighborhoodType & F, std::string WhichF )
	{
		/** Create an operator size and set it in the operator. */
		SizeType r;
		r.Fill( 1 );
		F.SetRadius( r );

		/** Get the image spacing factors that we are going to use. */
		std::vector< double > s( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ ) s[ i ] = this->GetImageSpacingUsed()[ i ];
		
		/** Create the required operator (neighborhood), depending on
		 * WhichF. The operator is either 3x3 in 2D or 3x3x3 in 3D.
		 */
		if ( WhichF == "FA" )
		{
			if ( ImageDimension == 2 )
			{
				/** Fill the operator. */
				F[ 0 ] = 1.0 / 12.0 / s[ 0 ]; F[ 1 ] = 0.0;	F[ 2 ] = -1.0 / 12.0 / s[ 0 ];
				F[ 3 ] = 1.0 /  3.0 / s[ 0 ]; F[ 4 ] = 0.0;	F[ 5 ] = -1.0 / 3.0 / s[ 0 ];
				F[ 6 ] = 1.0 / 12.0 / s[ 0 ]; F[ 7 ] = 0.0;	F[ 8 ] = -1.0 / 12.0 / s[ 0 ];
			}
			else if ( ImageDimension == 3 )
			{
				/** Fill the operator. First slice. */
				F[ 0 ] = 1.0 / 72.0 / s[ 0 ];	F[ 1 ] = 0.0;	F[ 2 ] = -1.0 / 72.0 / s[ 0 ];
				F[ 3 ] = 1.0 / 18.0 / s[ 0 ];	F[ 4 ] = 0.0;	F[ 5 ] = -1.0 / 18.0 / s[ 0 ];
				F[ 6 ] = 1.0 / 72.0 / s[ 0 ];	F[ 7 ] = 0.0;	F[ 8 ] = -1.0 / 72.0 / s[ 0 ];
				/** Second slice. */
				F[  9 ] = 1.0 / 18.0 / s[ 0 ];	F[ 10 ] = 0.0; F[ 11 ] = -1.0 / 18.0 / s[ 0 ];
				F[ 12 ] = 2.0 /  9.0 / s[ 0 ];	F[ 13 ] = 0.0; F[ 14 ] = -2.0 /  9.0 / s[ 0 ];
				F[ 15 ] = 1.0 / 18.0 / s[ 0 ];	F[ 16 ] = 0.0; F[ 17 ] = -1.0 / 18.0 / s[ 0 ];
				/** Third slice. */
				F[ 18 ] = 1.0 / 72.0 / s[ 0 ];	F[ 19 ] = 0.0;	F[ 20 ] = -1.0 / 72.0 / s[ 0 ];
				F[ 21 ] = 1.0 / 18.0 / s[ 0 ];	F[ 22 ] = 0.0;	F[ 23 ] = -1.0 / 18.0 / s[ 0 ];
				F[ 24 ] = 1.0 / 72.0 / s[ 0 ];	F[ 25 ] = 0.0;	F[ 26 ] = -1.0 / 72.0 / s[ 0 ];
			}
		}
		else if ( WhichF == "FB" )
		{
			if ( ImageDimension == 2 )
			{
				/** Fill the operator. */
				F[ 0 ] =  1.0 / 12.0 / s[ 1 ];	F[ 1 ] =  1.0 / 3.0 / s[ 1 ];		F[ 2 ] =  1.0 / 12.0 / s[ 1 ];
				F[ 3 ] =  0.0;									F[ 4 ] =  0.0;									F[ 5 ] =  0.0;
				F[ 6 ] = -1.0 / 12.0 / s[ 1 ];	F[ 7 ] = -1.0 / 3.0 / s[ 1 ];		F[ 8 ] = -1.0 / 12.0 / s[ 1 ];
			}
			else if ( ImageDimension == 3 )
			{
				/** Fill the operator. First slice. */
				F[ 0 ] =  1.0 / 72.0 / s[ 1 ];	F[ 1 ] =  1.0 / 18.0 / s[ 1 ];	F[ 2 ] =  1.0 / 72.0 / s[ 1 ];
				F[ 3 ] =  0.0;									F[ 4 ] =  0.0;									F[ 5 ] =  0.0;
				F[ 6 ] = -1.0 / 72.0 / s[ 1 ];	F[ 7 ] = -1.0 / 18.0 / s[ 1 ];	F[ 8 ] = -1.0 / 72.0 / s[ 1 ];
				/** Second slice. */
				F[  9 ] =  1.0 / 18.0 / s[ 1 ];	F[ 10 ] =  2.0 / 9.0 / s[ 1 ];	F[ 11 ] =  1.0 / 18.0 / s[ 1 ];
				F[ 12 ] =  0.0;									F[ 13 ] =  0.0;									F[ 14 ] =  0.0;
				F[ 15 ] = -1.0 / 18.0 / s[ 1 ];	F[ 16 ] = -2.0 / 9.0 / s[ 1 ];	F[ 17 ] = -1.0 / 18.0 / s[ 1 ];
				/** Third slice. */
				F[ 18 ] =  1.0 / 72.0 / s[ 1 ];	F[ 19 ] =  1.0 / 18.0 / s[ 1 ];	F[ 20 ] =  1.0 / 72.0 / s[ 1 ];
				F[ 21 ] =  0.0;									F[ 22 ] =  0.0;									F[ 23 ] =  0.0;
				F[ 24 ] = -1.0 / 72.0 / s[ 1 ];	F[ 25 ] = -1.0 / 18.0 / s[ 1 ];	F[ 26 ] = -1.0 / 72.0 / s[ 1 ];
			}
		}
		else if ( WhichF == "FC" )
		{
			if ( ImageDimension == 2 )
			{
				/** Not appropriate. Throw an exception. */
				itkExceptionMacro( << "This type of operator (FC) is not appropriate in 2D." );
			}
			else if ( ImageDimension == 3 )
			{
				/** Fill the operator. First slice. */
				F[ 0 ] = 1.0 / 72.0 / s[ 2 ];	F[ 1 ] = 1.0 / 18.0 / s[ 2 ];	F[ 2 ] = 1.0 / 72.0 / s[ 2 ];
				F[ 3 ] = 1.0 / 18.0 / s[ 2 ];	F[ 4 ] = 2.0 /  9.0 / s[ 2 ];	F[ 5 ] = 1.0 / 18.0 / s[ 2 ];
				F[ 6 ] = 1.0 / 72.0 / s[ 2 ];	F[ 7 ] = 1.0 / 18.0 / s[ 2 ];	F[ 8 ] = 1.0 / 72.0 / s[ 2 ];
				/** Second slice. */
				F[  9 ] = 0.0; F[ 10 ] = 0.0; F[ 11 ] = 0.0;
				F[ 12 ] = 0.0; F[ 13 ] = 0.0; F[ 14 ] = 0.0;
				F[ 15 ] = 0.0; F[ 16 ] = 0.0; F[ 17 ] = 0.0;
				/** Third slice. */
				F[ 18 ] = -1.0 / 72.0 / s[ 2 ]; F[ 19 ] = -1.0 / 18.0 / s[ 2 ];	F[ 20 ] = -1.0 / 72.0 / s[ 2 ];
				F[ 21 ] = -1.0 / 18.0 / s[ 2 ]; F[ 22 ] = -2.0 /  9.0 / s[ 2 ];	F[ 23 ] = -1.0 / 18.0 / s[ 2 ];
				F[ 24 ] = -1.0 / 72.0 / s[ 2 ]; F[ 25 ] = -1.0 / 18.0 / s[ 2 ];	F[ 26 ] = -1.0 / 72.0 / s[ 2 ];
			}
		}
		else if ( WhichF == "FD" )
		{
			if ( ImageDimension == 2 )
			{
				InputVectorValueType sp = s[ 0 ] * s[ 0 ];
				/** Fill the operator. */
				F[ 0 ] = 1.0 / 12.0 / sp;		F[ 1 ] = -1.0 / 6.0 / sp;		F[ 2 ] = 1.0 / 12.0 / sp;
				F[ 3 ] = 1.0 /  3.0 / sp;		F[ 4 ] = -2.0 / 3.0 / sp;		F[ 5 ] = 1.0 /  3.0 / sp;
				F[ 6 ] = 1.0 / 12.0 / sp;		F[ 7 ] = -1.0 / 6.0 / sp;		F[ 8 ] = 1.0 / 12.0 / sp;
			}
			else if ( ImageDimension == 3 )
			{
				InputVectorValueType sp = s[ 0 ] * s[ 0 ];
				/** Fill the operator. First slice. */
				F[ 0 ]  = 1.0 / 72.0 / sp; F[ 1 ]  = -1.0 / 36.0 / sp; F[ 2 ]  = 1.0 / 72.0 / sp;
				F[ 3 ]  = 1.0 / 18.0 / sp; F[ 4 ]  = -1.0 /  9.0 / sp; F[ 5 ]  = 1.0 / 18.0 / sp;
				F[ 6 ]  = 1.0 / 72.0 / sp; F[ 7 ]  = -1.0 / 36.0 / sp; F[ 8 ]  = 1.0 / 72.0 / sp;
				/** Second slice. */
				F[  9 ] = 1.0 / 18.0 / sp; F[ 10 ] = -1.0 / 9.0 / sp;  F[ 11 ] = 1.0 / 18.0 / sp;
				F[ 12 ] = 2.0 /  9.0 / sp; F[ 13 ] = -4.0 / 9.0 / sp;  F[ 14 ] = 2.0 /  9.0 / sp;
				F[ 15 ] = 1.0 / 18.0 / sp; F[ 16 ] = -1.0 / 9.0 / sp;  F[ 17 ] = 1.0 / 18.0 / sp;
				/** Third slice. */
				F[ 18 ] = 1.0 / 72.0 / sp; F[ 19 ] = -1.0 / 36.0 / sp; F[ 20 ] = 1.0 / 72.0 / sp;
				F[ 21 ] = 1.0 / 18.0 / sp; F[ 22 ] = -1.0 /  9.0 / sp; F[ 23 ] = 1.0 / 18.0 / sp;
				F[ 24 ] = 1.0 / 72.0 / sp; F[ 25 ] = -1.0 / 36.0 / sp; F[ 26 ] = 1.0 / 72.0 / sp;
			}
		}
		else if ( WhichF == "FE" )
		{
			if ( ImageDimension == 2 )
			{
				InputVectorValueType sp = s[ 1 ] * s[ 1 ];
				/** Fill the operator. */
				F[ 0 ] = 1.0 / 12.0 / sp;		F[ 1 ] = 1.0 / 3.0 / sp;		F[ 2 ] = 1.0 / 12.0 / sp;
				F[ 3 ] = -1.0 / 6.0 / sp;		F[ 4 ] = -2.0 / 3.0 / sp;		F[ 5 ] = -1.0 / 6.0 / sp;
				F[ 6 ] = 1.0 / 12.0 / sp;		F[ 7 ] = 1.0 / 3.0 / sp;		F[ 8 ] = 1.0 / 12.0 / sp;
			}
			else if ( ImageDimension == 3 )
			{
				InputVectorValueType sp = s[ 1 ] * s[ 1 ];
				/** Fill the operator. First slice. */
				F[ 0 ] =  1.0 / 72.0 / sp;	F[ 1 ] =  1.0 / 18.0 / sp; F[ 2 ] =  1.0 / 72.0 / sp;
				F[ 3 ] = -1.0 / 36.0 / sp;	F[ 4 ] = -1.0 /  9.0 / sp; F[ 5 ] = -1.0 / 36.0 / sp;
				F[ 6 ] =  1.0 / 72.0 / sp;	F[ 7 ] =  1.0 / 18.0 / sp; F[ 8 ] =  1.0 / 72.0 / sp;
				/** Second slice. */
				F[  9 ] =  1.0 / 18.0 / sp;	F[ 10 ] =  2.0 / 9.0 / sp; F[ 11 ] =  1.0 / 18.0 / sp;
				F[ 12 ] = -1.0 /  9.0 / sp;	F[ 13 ] = -4.0 / 9.0 / sp; F[ 14 ] = -1.0 /  9.0 / sp;
				F[ 15 ] =  1.0 / 18.0 / sp;	F[ 16 ] =  2.0 / 9.0 / sp; F[ 17 ] =  1.0 / 18.0 / sp;
				/** Third slice. */
				F[ 18 ] =  1.0 / 72.0 / sp;	F[ 19 ] =  1.0 / 18.0 / sp;	F[ 20 ] =  1.0 / 72.0 / sp;
				F[ 21 ] = -1.0 / 36.0 / sp;	F[ 22 ] = -1.0 /  9.0 / sp;	F[ 23 ] = -1.0 / 36.0 / sp;
				F[ 24 ] =  1.0 / 72.0 / sp;	F[ 25 ] =  1.0 / 18.0 / sp;	F[ 26 ] =  1.0 / 72.0 / sp;
			}
		}
		else if ( WhichF == "FF" )
		{
			if ( ImageDimension == 2 )
			{
				/** Not appropriate. Throw an exception. */
				itkExceptionMacro( << "This type of operator (FF) is not appropriate in 2D." );
			}
			else if ( ImageDimension == 3 )
			{
				InputVectorValueType sp = s[ 2 ] * s[ 2 ];
				/** Fill the operator. First slice. */
				F[ 0 ] = 1.0 / 72.0 / sp;	F[ 1 ] = 1.0 / 18.0 / sp;	F[ 2 ] = 1.0 / 72.0 / sp;
				F[ 3 ] = 1.0 / 18.0 / sp;	F[ 4 ] = 2.0 /  9.0 / sp;	F[ 5 ] = 1.0 / 18.0 / sp;
				F[ 6 ] = 1.0 / 72.0 / sp;	F[ 7 ] = 1.0 / 18.0 / sp;	F[ 8 ] = 1.0 / 72.0 / sp;
				/** Second slice. */
				F[  9 ] = -1.0 / 39.0 / sp; F[ 10 ] = -1.0 / 9.0 / sp;	F[ 11 ] = -1.0 / 36.0 / sp;
				F[ 12 ] = -1.0 /  9.0 / sp; F[ 13 ] = -4.0 / 9.0 / sp;	F[ 14 ] = -1.0 /  9.0 / sp;
				F[ 15 ] = -1.0 / 36.0 / sp; F[ 16 ] = -1.0 / 9.0 / sp;	F[ 17 ] = -1.0 / 36.0 / sp;
				/** Third slice. */
				F[ 18 ] = 1.0 / 72.0 / sp; F[ 19 ] = 1.0 / 18.0 / sp;	F[ 20 ] = 1.0 / 72.0 / sp;
				F[ 21 ] = 1.0 / 18.0 / sp; F[ 22 ] = 2.0 /  9.0 / sp;	F[ 23 ] = 1.0 / 18.0 / sp;
				F[ 24 ] = 1.0 / 72.0 / sp; F[ 25 ] = 1.0 / 18.0 / sp;	F[ 26 ] = 1.0 / 72.0 / sp;
			}
		}
		else if ( WhichF == "FG" )
		{
			if ( ImageDimension == 2 )
			{
				InputVectorValueType sp = s[ 0 ] * s[ 1 ];
				/** Fill the operator. */
				F[ 0 ] =  1.0 / 4.0 / sp;		F[ 1 ] = 0.0;		F[ 2 ] = -1.0 / 4.0 / sp;
				F[ 3 ] =  0.0;							F[ 4 ] = 0.0;		F[ 5 ] =  0.0;
				F[ 6 ] = -1.0 / 4.0 / sp;		F[ 7 ] = 0.0;		F[ 8 ] =  1.0 / 4.0 / sp;
			}
			else if ( ImageDimension == 3 )
			{
				InputVectorValueType sp = s[ 0 ] * s[ 1 ];
				/** Fill the operator. First slice. */
				F[ 0 ] =  1.0 / 24.0 / sp;	F[ 1 ] = 0.0;		F[ 2 ] = -1.0 / 24.0 / sp;
				F[ 3 ] =  0.0;							F[ 4 ] = 0.0;		F[ 5 ] =  0.0;
				F[ 6 ] = -1.0 / 24.0 / sp;	F[ 7 ] = 0.0;		F[ 8 ] =  1.0 / 24.0 / sp;
				/** Second slice. */
				F[  9 ] =  1.0 / 6.0 / sp;	F[ 10 ] = 0.0;	F[ 11 ] = -1.0 / 6.0 / sp;
				F[ 12 ] =  0.0;							F[ 13 ] = 0.0;	F[ 14 ] =  0.0;
				F[ 15 ] = -1.0 / 6.0 / sp;	F[ 16 ] = 0.0;	F[ 17 ] =  1.0 / 6.0 / sp;
				/** Third slice. */
				F[ 18 ] =  1.0 / 24.0 / sp;	F[ 19 ] = 0.0;	F[ 20 ] = -1.0 / 24.0 / sp;
				F[ 21 ] =  0.0;							F[ 22 ] = 0.0;	F[ 23 ] =  0.0;
				F[ 24 ] = -1.0 / 24.0 / sp;	F[ 25 ] = 0.0;	F[ 26 ] =  1.0 / 24.0 / sp;
			}
		}
		else if ( WhichF == "FH" )
		{
			if ( ImageDimension == 2 )
			{
				/** Not appropriate. Throw an exception. */
				itkExceptionMacro( << "This type of operator (FH) is not appropriate in 2D." );
			}
			else if ( ImageDimension == 3 )
			{
				InputVectorValueType sp = s[ 0 ] * s[ 2 ];
				/** Fill the operator. First slice. */
				F[ 0 ] = 1.0 / 24.0 / sp;	F[ 1 ] = 0.0;	F[ 2 ] = -1.0 / 24.0 / sp;
				F[ 3 ] = 1.0 /  6.0 / sp;	F[ 4 ] = 0.0;	F[ 5 ] = -1.0 /  6.0 / sp;
				F[ 6 ] = 1.0 / 24.0 / sp;	F[ 7 ] = 0.0;	F[ 8 ] = -1.0 / 24.0 / sp;
				/** Second slice. */
				F[  9 ] = 0.0;	F[ 10 ] = 0.0; F[ 11 ] = 0.0;
				F[ 12 ] = 0.0;	F[ 13 ] = 0.0; F[ 14 ] = 0.0;
				F[ 15 ] = 0.0;	F[ 16 ] = 0.0; F[ 17 ] = 0.0;
				/** Third slice. */
				F[ 18 ] = -1.0 / 24.0 / sp;	F[ 19 ] = 0.0;	F[ 20 ] = 1.0 / 24.0 / sp;
				F[ 21 ] = -1.0 /  6.0 / sp;	F[ 22 ] = 0.0;	F[ 23 ] = 1.0 /  6.0 / sp;
				F[ 24 ] = -1.0 / 24.0 / sp;	F[ 25 ] = 0.0;	F[ 26 ] = 1.0 / 24.0 / sp;
			}
		}
		else if ( WhichF == "FI" )
		{
			if ( ImageDimension == 2 )
			{
				/** Not appropriate. Throw an exception. */
				itkExceptionMacro( << "This type of operator (FI) is not appropriate in 2D." );
			}
			else if ( ImageDimension == 3 )
			{
				InputVectorValueType sp = s[ 1 ] * s[ 2 ];
				/** Fill the operator. First slice. */
				F[ 0 ] =  1.0 / 24.0 / sp;	F[ 1 ] =  1.0 / 6.0 / sp;	F[ 2 ] =  1.0 / 24.0 / sp;
				F[ 3 ] =  0.0;							F[ 4 ] =  0.0;						F[ 5 ] =  0.0;
				F[ 6 ] = -1.0 / 24.0 / sp;	F[ 7 ] = -1.0 / 6.0 / sp;	F[ 8 ] = -1.0 / 24.0 / sp;
				/** Second slice. */
				F[  9 ] = 0.0;	F[ 10 ] = 0.0; F[ 11 ] = 0.0;
				F[ 12 ] = 0.0;	F[ 13 ] = 0.0; F[ 14 ] = 0.0;
				F[ 15 ] = 0.0;	F[ 16 ] = 0.0; F[ 17 ] = 0.0;
				/** Third slice. */
				F[ 18 ] = -1.0 / 24.0 / sp;	F[ 19 ] = -1.0 / 6.0 / sp;	F[ 20 ] = -1.0 / 24.0 / sp;
				F[ 21 ] =  0.0;							F[ 22 ] =  0.0;							F[ 23 ] =  0.0;
				F[ 24 ] =  1.0 / 24.0 / sp;	F[ 25 ] =  1.0 / 6.0 / sp;	F[ 26 ] =  1.0 / 24.0 / sp;
			}
		}
		else
		{
			/** Throw an exception. */
			itkExceptionMacro( << "Can not create this type of operator." );
		}

	} // end CreateNDOperator


	/**
	 * ************************** FilterSeparable ********************
	 */

	template< class TInputImage, class TOutputImage >
		typename RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::InputScalarImagePointer
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::FilterSeparable( const InputScalarImageType * image, std::vector< NeighborhoodType > Operators )
	{
		/** Sanity check. */
		if ( Operators.size() != ImageDimension )
		{
			itkExceptionMacro( << "ERROR: Number of operators not equal to ImageDimension" );
		}

		/** Create filters, supply them with boundary conditions and operators. */
		std::vector< typename NOIFType::Pointer > filters( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			filters[ i ] = NOIFType::New();
			filters[ i ]->SetOperator( Operators[ i ] );
		}

		/** Create a process accumulator for tracking the progress of this minipipeline. */
		std::vector< typename ProgressAccumulator::Pointer > progresses( ImageDimension );
		for ( unsigned int i = 0; i < ImageDimension; i++ )
		{
			progresses[ i ] = ProgressAccumulator::New();
			progresses[ i ]->SetMiniPipelineFilter( this );
			/** Register the filter with the with progress accumulator using equal weight proportion. */
			progresses[ i ]->RegisterInternalFilter( filters[ i ], 1.0f );
		}

		/** Set up the mini-pipline. */
		filters[ 0 ]->SetInput( image );
		for ( unsigned int i = 1; i < ImageDimension; i++ )
		{
			filters[ i ]->SetInput( filters[ i - 1 ]->GetOutput() );
		}

		/** Execute the mini-pipeline. */
		filters[ ImageDimension - 1 ]->Update();

		/** Return the filtered image. */
		return filters[ ImageDimension - 1 ]->GetOutput();

	} // end FilterSeparable


	/**
	 * ********************** FilterNonSeparable *********************
	 */

	template< class TInputImage, class TOutputImage >
		typename RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::InputScalarImagePointer
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::FilterNonSeparable( const InputScalarImageType * image, NeighborhoodType oper )
	{
		/** Create a filter ... */
		typename NOIFType::Pointer filter = NOIFType::New();

		/** Create a process accumulator for tracking the progress of this minipipeline. */
		typename ProgressAccumulator::Pointer progress = ProgressAccumulator::New();
		progress->SetMiniPipelineFilter( this );

		/** Register the filter with the with progress accumulator
		 * using equal weight proportion. */
		progress->RegisterInternalFilter( filter, 1.0f );

		/** Set up the mini-pipline. */
		filter->SetOperator( oper );
		filter->SetInput( image );

		/** Execute the mini-pipeline. */
		filter->Update();

		/** Return the filtered image. */
		return filter->GetOutput();

	} // end FilterNonSeparable


	/**
	 * ********************** CalculateSubPart *********************
	 */

	template< class TInputImage, class TOutputImage >
		double
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::CalculateSubPart( unsigned int dim, unsigned int part, std::vector<OutputVectorValueType> values )
	{
		/** Initialize the return value. */
		double answer = 0.0;

		/** Extract the correct values from the vector values. */
		OutputVectorValueType mu1_A, mu2_A, mu3_A, mu1_B, mu2_B, mu3_B, mu1_C, mu2_C, mu3_C;
		if ( ImageDimension == 2 )
		{
			mu1_A = values[ 0 ];
			mu2_A = values[ 1 ];
			mu1_B = values[ 2 ];
			mu2_B = values[ 3 ];
		}
		else if ( ImageDimension == 3 )
		{
			mu1_A = values[ 0 ];
			mu2_A = values[ 1 ];
			mu3_A = values[ 2 ];
			mu1_B = values[ 3 ];
			mu2_B = values[ 4 ];
			mu3_B = values[ 5 ];
			mu1_C = values[ 6 ];
			mu2_C = values[ 7 ];
			mu3_C = values[ 8 ];
		}

		/** Calculate the filter response. */
		if ( ImageDimension == 2 )
		{
			if ( dim == 0 && part == 0 )
			{
				/** In this case we calculate the derivative of S^rigid_2D to
				 * the first component of the parameter mu: mu_k,1. This
				 * calcultes only one part of the total derivative to mu_k,1:
				 * i.e. only the part that is later filtered with F_A (again).
				 */
				answer =
					+ 2.0 * ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * ( 1.0 + mu1_A )
					+ ( 1.0 + mu1_A ) * (
						+ 2.0 * mu2_A * mu2_A
						- 2.0
						+ mu1_B * mu1_B
						+ ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						)
					- ( 1.0 + mu2_B );
			}
			else if ( dim == 0 && part == 1 )
			{
				/** In this case we calculate the derivative of S^rigid_2D to
				 * the first component of the parameter mu: mu_k,1. This
				 * calcultes only one part of the total derivative to mu_k,1:
				 * i.e. only the part that is later filtered with F_B (again).
				 */
				answer =
					+ 2.0 * mu1_B * mu1_B * mu1_B
					+ mu1_B * (
						+ 2.0 * ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						- 2.0
						+ ( 1.0 + mu1_A ) * ( 1.0 + mu1_A )
						+ mu2_A * mu2_A
						)
					+ mu2_A;
			}
			else if ( dim == 1 && part == 0 )
			{
				/** In this case we calculate the derivative of S^rigid_2D to
				 * the second component of the parameter mu: mu_k,2. This
				 * calcultes only one part of the total derivative to mu_k,2:
				 * i.e. only the part that is later filtered with F_A (again).
				 */
				answer =
					+ 2.0 * mu2_A * mu2_A * mu2_A
					+ mu2_A * (
						+ 2.0 * ( 1.0 + mu1_A ) * ( 1.0 + mu1_A )
						- 2.0
						+ mu1_B * mu1_B
						+ ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						)
					+ mu1_B;
			}
			else if ( dim == 1 && part == 1 )
			{
				/** In this case we calculate the derivative of S^rigid_2D to
				 * the second component of the parameter mu: mu_k,2. This
				 * calcultes only one part of the total derivative to mu_k,2:
				 * i.e. only the part that is later filtered with F_B (again).
				 */
				answer =
					+ 2.0 * ( 1.0 + mu2_B ) * ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
					+ ( 1.0 + mu2_B ) * (
						+ 2.0 * mu1_B * mu1_B
						- 2.0
						+ ( 1.0 + mu1_A ) * ( 1.0 + mu1_A )
						+ mu2_A * mu2_A
						)
					- ( 1.0 + mu1_A );
			}
			else
				{
				/** Sanity check. */
				itkExceptionMacro( << "ERROR: This combination is not valid." );
			} // end if which subpart
		} // end 2D
		else if ( ImageDimension == 3 )
		{
			if ( dim == 0 && part == 0 )
			{
				/** In this case we calculate the derivative of S^rigid_3D to
				 * the first component of the parameter mu: mu_k,1. This
				 * calcultes only one part of the total derivative to mu_k,1:
				 * i.e. only the part that is later filtered with F_A (again).
				 */
				answer =
					+ 2.0 * ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * ( 1.0 + mu1_A )
					+ ( 1.0 + mu1_A ) * (
						+ 2.0 * mu2_A * mu2_A
						+ 2.0 * mu3_A * mu3_A
						- 2.0
						+ mu1_B * mu1_B
						+ mu1_C * mu1_C
						+ mu2_C * mu2_C * mu3_B * mu3_B
						+ ( 1.0 + mu2_B ) * ( 1.0 + mu2_B ) * ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						- 2.0 * ( 1.0 + mu2_B ) * mu2_C * mu3_B * ( 1.0 + mu3_C )
						)
					+ mu1_B * (
						+ mu2_A * ( 1.0 + mu2_B )
						+ mu3_A * mu3_B
						- mu2_C * mu2_C * mu3_A * mu3_B
						+ ( 1.0 + mu2_B ) * mu2_C * mu3_A * ( 1.0 + mu3_C )
						+ mu2_A * mu2_C * mu3_B * ( 1.0 + mu3_C )
						- mu2_A * ( 1.0 + mu2_B )	* ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
            )
					+ mu1_C * (
						+ mu2_A * mu2_C
						+ mu3_A * ( 1.0 + mu3_C )
						+ ( 1.0 + mu2_B ) * mu2_C * mu3_A * mu3_B
						- ( 1.0 + mu2_B ) * ( 1.0 + mu2_B ) * mu3_A * ( 1.0 + mu3_C )
						- mu2_A * mu2_C * mu3_B * mu3_B
						+ mu2_A * ( 1.0 + mu2_B ) * mu3_B * ( 1.0 + mu3_C )
						)
					+ mu2_C * mu3_B
					- ( 1.0 + mu2_B ) * ( 1.0 + mu3_C );
			}
			else if ( dim == 0 && part == 1 )
			{
				/** In this case we calculate the derivative of S^rigid_3D to
				 * the first component of the parameter mu: mu_k,1. This
				 * calcultes only one part of the total derivative to mu_k,1:
				 * i.e. only the part that is later filtered with F_B (again).
				 */
				answer = 
					( 1.0 + mu1_A ) * (
						+ ( 1.0 + mu1_A ) * mu1_B
						+ mu2_A * ( 1.0 + mu2_B )
						+ mu3_A * mu3_B
						- mu2_C * mu2_C * mu3_A * mu3_B
						+ ( 1.0 + mu2_B ) * mu2_C * mu3_A * ( 1.0 + mu3_C )
						+ mu2_A * mu2_C * mu3_B * ( 1.0 + mu3_C )
						- mu2_A * ( 1.0 + mu2_B ) * ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						)
					+ 2.0 * mu1_B * mu1_B
					+ mu1_B * (
						+ 2.0 * ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						+ 2.0 * mu3_B * mu3_B
						- 2.0
						+ mu1_C * mu1_C
						+ mu2_C * mu2_C * mu3_A * mu3_A
						+ mu2_A * mu2_A * ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						- 2.0 * mu2_A * mu2_C * mu3_A * ( 1.0 + mu3_C )
						)
					+ mu1_C * (
						+ ( 1.0 + mu2_B ) * mu2_C
						+ mu3_B * ( 1.0 + mu3_C )
						- ( 1.0 + mu2_B ) * mu2_C * mu3_A * mu3_A
						+ mu2_A * ( 1.0 + mu2_B ) * mu3_A * ( 1.0 + mu3_C )
						+ mu2_A * mu2_C * mu3_A * mu3_B
						- mu2_A * mu2_A * mu3_B * ( 1.0 + mu3_C )
						)
					- mu2_C * mu3_A
					+ mu2_A * ( 1.0 + mu3_C );
			}
			else if ( dim == 0 && part == 2 )
			{
				/** In this case we calculate the derivative of S^rigid_3D to
				 * the first component of the parameter mu: mu_k,1. This
				 * calcultes only one part of the total derivative to mu_k,1:
				 * i.e. only the part that is later filtered with F_C (again).
				 */
				answer =
					( 1.0 + mu1_A ) * (
						+ ( 1.0 + mu1_A ) * mu1_C
						+ mu2_A * mu2_C
						+ mu3_A * ( 1.0 + mu3_C )
						+ ( 1.0 + mu1_A ) * mu2_C * mu3_B * mu3_B
						+ mu1_C * ( 1.0 + mu2_B ) * mu3_A * mu3_B
						- 2.0 * mu1_B * mu2_C * mu3_A * mu3_B
						+ mu1_B * ( 1.0 + mu2_B ) * mu3_A * ( 1.0 + mu3_C )
						- mu1_C * mu2_A * mu3_B * mu3_B
						+ mu1_B * mu2_A * mu3_B * ( 1.0 + mu3_C )
						- ( 1.0 + mu1_A ) * ( 1.0 + mu2_B ) * mu3_B * ( 1.0 + mu3_C )
						+ mu3_B
						)
					+ mu1_B * (
						+ mu1_B * mu1_C
						+ ( 1.0 + mu2_B ) * mu2_C
						+ mu3_B * ( 1.0 + mu3_C )
						+ mu1_B * mu2_C * mu3_A * mu3_A
						- mu1_C * ( 1.0 + mu2_B ) * mu3_A * mu3_A
						+ mu1_C * mu2_A * mu3_A * mu3_B
						- mu1_B * mu2_A * mu3_A * ( 1.0 + mu3_C )
						- mu3_A
						)
					+ mu1_C * (
						+ 2.0 * mu1_C * mu1_C
						+ 2.0 * mu2_C * mu2_C
						+ 2.0 * ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						- 2.0
						);
			}
			else if ( dim == 1 && part == 0 )
			{
				/** In this case we calculate the derivative of S^rigid_3D to
				 * the second component of the parameter mu: mu_k,2. This
				 * calcultes only one part of the total derivative to mu_k,2:
				 * i.e. only the part that is later filtered with F_A (again).
				 */
				answer =
					+ 2.0 * mu2_A * mu2_A * mu2_A
					+ ( 1.0 + mu1_A ) * (
						+ 2.0 * ( 1.0 + mu1_A ) * mu2_A
						+ mu1_B * ( 1.0 + mu2_B )
						+ mu1_C * mu2_C
						- mu1_C * mu2_C * mu3_B * mu3_B
						+ mu1_C * ( 1.0 + mu2_B ) * mu3_B * ( 1.0 + mu3_C )
						+ mu1_B * mu2_C * mu3_B * ( 1.0 + mu3_C )
						- mu1_B * ( 1.0 + mu2_B ) * ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						)
					+ mu1_B * (
						+ mu1_B * mu2_A * ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						+ mu1_C * ( 1.0 + mu2_B ) * mu3_A * ( 1.0 + mu3_C )
						+ mu1_C * mu2_C * mu3_A * mu3_B
						- mu1_B * mu2_C * mu3_A * ( 1.0 + mu3_C )
						- 2.0 * mu1_C * mu2_A * mu3_B * ( 1.0 + mu3_C )
            + ( 1.0 + mu3_C )
						)
					+ mu1_C * (
						+ mu1_C * mu2_A * mu3_B * mu3_B
						- mu1_C * ( 1.0 + mu2_B ) * mu3_A * mu3_B
						- mu3_B
						)
          + mu2_A * (
						+ 2.0 * mu3_A * mu3_A
						- 2.0
						+ ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						+ mu3_C * mu3_C
						)
					+ ( 1.0 + mu2_B ) * mu3_A * mu3_B
					+ mu2_C * mu3_A * ( 1.0 + mu3_C );
			}
			else if ( dim == 1 && part == 1 )
			{
				/** In this case we calculate the derivative of S^rigid_3D to
				 * the second component of the parameter mu: mu_k,2. This
				 * calcultes only one part of the total derivative to mu_k,2:
				 * i.e. only the part that is later filtered with F_B (again).
				 */
				answer = 
					+ mu2_A * mu2_A * ( 1.0 + mu2_B )
					+ mu2_A * (
						+ ( 1.0 + mu1_A ) * mu1_B
						+ mu3_A * mu3_B
						- mu1_C * mu1_C * mu3_A * mu3_B
						+ mu1_B * mu1_C * mu3_A * ( 1.0 + mu3_C )
						+ ( 1.0 + mu1_A ) * mu1_C * mu3_B * ( 1.0 + mu3_C )
						- ( 1.0 + mu1_A ) * mu1_B * ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						)
					+ ( 1.0 + mu2_B ) * (
						+ ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						+ mu1_B * mu1_B
						+ mu3_B * mu3_B
						- 1.0
						+ mu2_C * mu2_C
						+ mu1_C * mu1_C * mu3_A * mu3_A
						+ ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						- 2.0 * ( 1.0 + mu1_A ) * mu1_C * mu3_A * ( 1.0 + mu3_C )
						)
					+ mu2_C * (
						+ mu1_B * mu1_C
						+ mu3_B * ( 1.0 + mu3_C )
						- mu1_B * mu1_C * mu3_A * mu3_A
						+ ( 1.0 + mu1_A ) * mu1_C * mu3_A * mu3_B
						+ ( 1.0 + mu1_A ) * mu1_B * mu3_A * ( 1.0 + mu3_C )
						- ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * mu3_B * ( 1.0 + mu3_C )
						)
					+ mu1_C * mu3_A
					- ( 1.0 + mu1_A ) * ( 1.0 + mu3_C );
			}
			else if ( dim == 1 && part == 2 )
			{
				/** In this case we calculate the derivative of S^rigid_3D to
				 * the second component of the parameter mu: mu_k,2. This
				 * calcultes only one part of the total derivative to mu_k,2:
				 * i.e. only the part that is later filtered with F_C (again).
				 */
				answer =
					+ mu2_A * mu2_A * mu2_C
					+ mu2_A * (
						+ ( 1.0 + mu1_A ) * mu1_C
						+ mu3_A * ( 1.0 + mu3_C )
						+ mu1_B * mu1_C * mu3_A * mu3_B
						- mu1_B * mu1_B * mu3_A * ( 1.0 + mu3_C )
						- ( 1.0 + mu1_A ) * mu1_C * mu3_B * mu3_B
						+ ( 1.0 + mu1_A ) * mu1_B * mu3_B * ( 1.0 + mu3_C )
						)
					+ mu2_C * (
						+ ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						+ mu2_C * mu2_C
						+ mu1_C * mu1_C
						+ ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						- 1.0
						+ mu1_B * mu1_B * mu3_A * mu3_A
						+ ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * mu3_B * mu3_B
						- 2.0 * ( 1.0 + mu1_A ) * mu1_B * mu3_A * mu3_B
						)
					+ ( 1.0 + mu2_B ) * (
						+ mu3_B * ( 1.0 + mu3_C )
						- mu1_B * mu1_C * mu3_A * mu3_A
						+ ( 1.0 + mu1_A ) * mu1_C * mu3_A * mu3_B
						+ ( 1.0 + mu1_A ) * mu1_B * mu3_A * ( 1.0 + mu3_C )
						- ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * mu3_B * ( 1.0 + mu3_C )
						)
					+ mu1_B * (
						+ mu1_C * mu2_B
						- mu3_A
						)
					+ ( 1.0 + mu1_A ) * mu3_B;
			}
			else if ( dim == 2 && part == 0 )
			{
				/** In this case we calculate the derivative of S^rigid_3D to
				 * the third component of the parameter mu: mu_k,3. This
				 * calcultes only one part of the total derivative to mu_k,3:
				 * i.e. only the part that is later filtered with F_A (again).
				 */
				answer =
					mu3_A * (
						+ 2.0 * mu3_A * mu3_A
						+ 2.0 * ( 1.0 + mu1_A ) * ( 1.0 + mu1_A )
						+ 2.0 * mu2_A * mu2_A
						- 2.0
						+ mu3_B * mu3_B
						+ ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						+ mu1_C * mu1_C * ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						+ mu1_B * mu1_B * mu2_C * mu2_C
						- 2.0 * mu1_B * mu1_C * ( 1.0 + mu2_B ) * mu2_C
						)
					+ ( 1.0 + mu2_B ) * (
						+ mu2_A * mu3_B
						- mu1_C * mu1_C * mu2_A * mu3_B
						+ ( 1.0 + mu1_A ) * mu1_C * mu2_C * mu3_B
						+ mu1_B * mu1_C * mu2_A * ( 1.0 + mu3_C )
						- ( 1.0 + mu1_A ) * mu1_C * ( 1.0 + mu2_B ) * ( 1.0 + mu3_C )
						+ mu1_C
						+ ( 1.0 + mu1_A ) * mu1_B * mu2_C * ( 1.0 + mu3_C )
						)
					+ ( 1.0 + mu1_A ) * (
						+ mu1_B * mu3_B
						+ mu1_C * ( 1.0 + mu3_C )
						- mu1_B * mu2_C * mu2_C * mu3_B
						)
					+ mu2_C * (
						+ mu2_A * ( 1.0 + mu3_C )
						+ mu1_B * mu1_C * mu2_A * mu3_B
						- mu1_B * mu1_B * mu2_A * ( 1.0 + mu3_C )
						+ mu1_B
						);
			}
			else if ( dim == 2 && part == 1 )
			{
				/** In this case we calculate the derivative of S^rigid_3D to
				 * the third component of the parameter mu: mu_k,3. This
				 * calcultes only one part of the total derivative to mu_k,3:
				 * i.e. only the part that is later filtered with F_B (again).
				 */
				answer =
					mu3_A * (
						+ mu3_A * mu3_B
						+ ( 1.0 + mu1_A ) * mu1_B
						+ mu2_A * ( 1.0 + mu2_B )
						- mu1_C * mu1_C * mu2_A * ( 1.0 + mu2_B )
						+ ( 1.0 + mu1_A ) * mu1_C * ( 1.0 + mu2_B ) * mu2_C
						+ mu1_B * mu1_C * mu2_A * mu2_C
						- ( 1.0 + mu1_A ) * mu1_B * mu2_C * mu2_C
						)
					+ mu3_B * (
						+ mu3_B * mu3_B
						+ mu1_B * mu1_B
						+ ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						- 1.0
						+ ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						+ mu1_C * mu1_C * mu2_A * mu2_A
						+ ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * mu2_C * mu2_C
						- 2.0 * ( 1.0 + mu1_A ) * mu1_C * mu2_A * mu2_C
						)
					+ ( 1.0 + mu3_C ) * (
						+ mu1_B * mu1_C
						+ ( 1.0 + mu2_B ) * mu2_C
						- mu1_B * mu1_C * mu2_A * mu2_A
						+ ( 1.0 + mu1_A ) * mu1_C * mu2_A * ( 1.0 + mu2_B )
						+ ( 1.0 + mu1_A ) * mu1_B * mu2_A * mu2_C
						- ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * ( 1.0 + mu2_B ) * mu2_C
						)
					- mu1_C * mu2_A
					+ ( 1.0 + mu1_A ) * mu2_C;
			}
			else if ( dim == 2 && part == 2 )
			{
				/** In this case we calculate the derivative of S^rigid_3D to
				 * the third component of the parameter mu: mu_k,3. This
				 * calcultes only one part of the total derivative to mu_k,3:
				 * i.e. only the part that is later filtered with F_C (again).
				 */
				answer =
					mu3_A * (
						+ mu3_A * ( 1.0 + mu3_C )
						+ ( 1.0 + mu1_A ) * mu1_C
						+ mu2_A * mu2_C
						+ mu1_B * mu1_C * mu2_A * ( 1.0 + mu2_B )
						- ( 1.0 + mu1_A ) * mu1_C * ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						- mu1_B * mu1_B * mu2_A * mu2_C
						+ ( 1.0 + mu1_A ) * mu1_B * ( 1.0 + mu2_B ) * mu2_C
						)
					+ mu3_B * (
						+ mu3_B * ( 1.0 + mu3_C )
						+ mu1_B * mu1_C
						+ ( 1.0 + mu2_B ) * mu2_C
						- mu1_B * mu1_C * mu2_A * mu2_A
						+ ( 1.0 + mu1_A ) * mu1_C * mu2_A * ( 1.0 + mu2_B )
						+ ( 1.0 + mu1_A ) * mu1_B * mu2_A * mu2_C
						+ ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * ( 1.0 + mu2_B ) * mu2_C
						)
					+ ( 1.0 + mu3_C ) * (
						+ ( 1.0 + mu3_C ) * ( 1.0 + mu3_C )
						+ mu1_C * mu1_C
						+ mu2_C * mu2_C
						- 1.0
						+ mu1_B * mu1_B * mu2_A * mu2_A
						+ ( 1.0 + mu1_A ) * ( 1.0 + mu1_A ) * ( 1.0 + mu2_B ) * ( 1.0 + mu2_B )
						- 2.0 * ( 1.0 + mu1_A ) * mu1_B * mu2_A * ( 1.0 + mu2_B )
						)
					+ mu1_B * mu2_A
					- ( 1.0 + mu1_A ) * ( 1.0 + mu2_B );
			}
			else
			{
				/** Sanity check. */
				itkExceptionMacro( << "ERROR: This combination is not valid." );
			} // end if which subpart
		} // end if 3D

		/** Return the answer. */
		return answer;

	} // end CalculateSubPart


	/**
	 * ******************** GetRigidRegulizerValue *******************
	 */
	template< class TInputImage, class TOutputImage >
		typename RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::InputVectorValueType
		RigidRegularizationDerivativeImageFilter< TInputImage, TOutputImage >
		::GetRigidRegulizerValue(void)
	{
		if ( !m_GenerateDataCalled )
		{
			std::cerr << "WARNING: You first have to call GenerateData(), where this is calculated!" << std::endl;
		}

		/** Return a value. */
		return this->m_RigidRegulizerValue;

	} // end GetRigidRegulizerValue


} // end namespace itk

#endif // end #ifndef _itkRigidRegularizationDerivativeImageFilter_txx
