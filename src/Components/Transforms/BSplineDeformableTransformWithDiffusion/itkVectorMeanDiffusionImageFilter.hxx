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
#ifndef _itkVectorMeanDiffusionImageFilter_HXX__
#define _itkVectorMeanDiffusionImageFilter_HXX__

#include "itkVectorMeanDiffusionImageFilter.h"

#include "itkNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "itkProgressReporter.h"

//#include "itkCastImageFilter.h"
//#include "itkImageFileWriter.h"

namespace itk
{
	
	/**
	 * *********************** Constructor **************************
	 */

	template < class TInputImage, class TGrayValueImage >
		VectorMeanDiffusionImageFilter< TInputImage, TGrayValueImage >
		::VectorMeanDiffusionImageFilter()
	{
		m_NumberOfIterations = 0;
		m_Radius.Fill(1);
		m_UseThreshold = false;
		m_Threshold = 0.00001;

		m_RescaleFilter = 0;
		m_GrayValueImage = 0;
		m_Cx = 0;

	} // end Constructor
	

	/**
	 * *************** GenerateInputRequestedRegion *****************
	 */

	template < class TInputImage, class TGrayValueImage >
		void
		VectorMeanDiffusionImageFilter< TInputImage, TGrayValueImage >
		::GenerateInputRequestedRegion() throw (InvalidRequestedRegionError)
	{
		// call the superclass' implementation of this method
		Superclass::GenerateInputRequestedRegion();
		
		// get pointers to the input and output
		typename Superclass::InputImagePointer inputPtr = 
			const_cast< TInputImage * >( this->GetInput() );
		typename Superclass::OutputImagePointer outputPtr = this->GetOutput();
		
		if ( !inputPtr || !outputPtr )
    {
			return;
    }
		
		// get a copy of the input requested region (should equal the output
		// requested region)
		typename TInputImage::RegionType inputRequestedRegion;
		inputRequestedRegion = inputPtr->GetRequestedRegion();
		
		// pad the input requested region by the operator radius
		inputRequestedRegion.PadByRadius( m_Radius );
		
		// crop the input requested region at the input's largest possible region
		if ( inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion()) )
    {
			inputPtr->SetRequestedRegion( inputRequestedRegion );
			return;
    }
		else
    {
			// Couldn't crop the region (requested region is outside the largest
			// possible region).  Throw an exception.
			
			// store what we tried to request (prior to trying to crop)
			inputPtr->SetRequestedRegion( inputRequestedRegion );
			
			// build an exception
			InvalidRequestedRegionError e(__FILE__, __LINE__);
			OStringStream msg;
			msg << static_cast<const char *>(this->GetNameOfClass())
        << "::GenerateInputRequestedRegion()";
			e.SetLocation(msg.str().c_str());
			e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
			e.SetDataObject(inputPtr);
			throw e;
    }

	} // end GenerateInputRequestedRegion
	
	
	/**
	 * ********************** GenerateData **************************
	 */

	template < class TInputImage, class TGrayValueImage >
		void
		VectorMeanDiffusionImageFilter< TInputImage, TGrayValueImage >
		::GenerateData(void)
	{
		/** Create feature image. */
		this->FilterGrayValueImage();

		/** Declare things. */
		unsigned int i, j;
		ZeroFluxNeumannBoundaryCondition< InputImageType >			nbc;
		ZeroFluxNeumannBoundaryCondition< DoubleImageType >			nbc2;
		NeighborhoodIterator< InputImageType >									nit;
		NeighborhoodIterator< DoubleImageType >									nit2;
		VectorRealType sum;
		
		/** Allocate output. */
		typename InputImageType::ConstPointer	input(	this->GetInput() );
		typename InputImageType::Pointer			output( this->GetOutput() );
		typename InputImageType::Pointer			outputtmp = InputImageType::New();

		output->SetRegions( input->GetLargestPossibleRegion() );
		output->Allocate();

		/** Allocate a temporary output image. */
		outputtmp->SetSpacing( input->GetSpacing() );
		outputtmp->SetOrigin( input->GetOrigin() );
		outputtmp->SetRegions( input->GetLargestPossibleRegion() );
		outputtmp->Allocate();
		
		// support progress methods/callbacks
		//ProgressReporter progress( this, threadId, outputRegionForThread.GetNumberOfPixels() );
		
		/** Copy input to output. */
		ImageRegionConstIterator< InputImageType >	in_it(
			input, input->GetLargestPossibleRegion() );
		ImageRegionIterator< InputImageType >				out_it(
			output, input->GetLargestPossibleRegion() );
		in_it.GoToBegin();
		out_it.GoToBegin();
		while ( !in_it.IsAtEnd() )
		{
			out_it.Set( in_it.Get() );
			++in_it;
			++out_it;
		}

		/** Setup neighborhood iterator for the output deformation image. */
		nit = NeighborhoodIterator< InputImageType >( 
			m_Radius, output, output->GetLargestPossibleRegion() );
		unsigned int neighborhoodSize = nit.Size();
		nit.OverrideBoundaryCondition( &nbc );

		/** Setup neighborhood iterator for the "stiffness coefficient" image. */
		nit2 = NeighborhoodIterator< DoubleImageType >( 
			m_Radius, m_Cx, m_Cx->GetLargestPossibleRegion() );
		nit2.OverrideBoundaryCondition( &nbc2 );

		/** Setup iterator over outputtmp. */
		ImageRegionIterator< InputImageType >				oit(
			outputtmp, input->GetLargestPossibleRegion() );

		/** Initialize c and ci. */
		double c = 0.0;
		double ci = 0.0;

		/** Loop over the number of iterations. */
		for ( unsigned int k = 0; k < this->GetNumberOfIterations(); k++ )
		{
			nit.GoToBegin();
			nit2.GoToBegin();
			oit.GoToBegin();

			/** The actual work. */
			while ( !nit.IsAtEnd() )
			{
				/** Initialize the sum to 0. */
				for ( j = 0; j < InputImageDimension; j++ )
				{
					sum[ j ] = NumericTraits< double >::Zero;
				}

				/** Initialize sumc. */
				double sumc = 0.0;
				
				/** Calculate the weighted mean over the neighborhood.
				 * mean = SUM_i{ ci * x_i } / SUM_i{ ci }
				 */
				for ( i = 0; i < neighborhoodSize; ++i )
				{
					/** Get current pixel in this neighborhood. */
					InputPixelType pix = nit.GetPixel( i );

					/** Get ci-value on current index. */
					ci = nit2.GetPixel( i );

					/** Calculate SUM_i{ ci } and SUM_i{ ci * x_i }. */
					sumc += ci;
					for ( j = 0; j < InputImageDimension; j++ )
					{
						sum[ j ] += ci * static_cast< double >( pix[ j ] );
					}
				}
				
				/** Get the mean value by dividing by sumc. */
				InputPixelType mean;
				for ( j = 0; j < InputImageDimension; j++ )
				{
					//if ( sumc == 0.0 ) mean[ j ] = 0.0;
					if ( sumc < 0.001 ) mean[ j ] = 0.0;
					else mean[ j ] = static_cast< ValueType >( sum[ j ] / sumc );
				}
				
				/** Get c. */
				c = nit2.GetCenterPixel();

				/** Set 'y = (1 - c) * x + c * mean' to the temporary output. */
				InputPixelType value = nit.GetCenterPixel() * ( 1.0 -	c ) + mean * c;

				/** Temporary if, so that value don't get to high! *
				for ( j = 0; j < InputImageDimension; j++ )
				{
					if ( value[ j ] > 1000.0 )
					{
						value[ j ] = 1000.0;
					}
				}

				/** Set it. */
				oit.Set( value );
				
				/** Increase all iterators. */
				++nit;
				++nit2;
				++oit;
				//progress.CompletedPixel();

			} // end while

			/** Copy outputtmp to output. */
			if ( this->GetNumberOfIterations() > 0 )
			{
				out_it.GoToBegin();
				oit.GoToBegin();
				while ( !out_it.IsAtEnd() )
				{
					out_it.Set( oit.Get() );
					++out_it;
					++oit;
				}
			} // end if

		} // end for NumberOfIterations
		
	} // end ThreadedGenerateData
	

	/**
	 * ********************* PrintSelf ******************************
	 *
	 * Standard "PrintSelf" method
	 */

	template < class TInputImage, class TGrayValueImage >
		void
		VectorMeanDiffusionImageFilter< TInputImage, TGrayValueImage >
		::PrintSelf( std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf( os, indent );
		os << indent << "Radius: " << m_Radius << std::endl;
		
	} // end PrintSelf

	
	/**
	 * ******************** SetGrayValueImage ***********************
	 */
	
	template< class TInputImage, class TGrayValueImage >
		void
		VectorMeanDiffusionImageFilter< TInputImage, TGrayValueImage >
		::SetGrayValueImage( GrayValueImageType * _arg )
	{
		if ( this->m_GrayValueImage != _arg )
		{
			this->m_GrayValueImage = _arg;
		}

	} // end SetGrayValueImage


	/**
	 * ******************** FilterGrayValueImage ********************
	 *
	 * This function reads an image u(x). This image is rescaled to
	 * intensities between 0.0 and 1.0, giving u~(x). Then we calculate
	 * c(x) = sqrt( u~(x) / m_K ) (or c(x) = exp( u~(x) / m_K ) ). Finally, we
	 * calculate out(x) = c(x) + 1/2 * SUM_i{ d/dx_i c(x) }.
	 */
	
	template< class TInputImage, class TGrayValueImage >
		void
		VectorMeanDiffusionImageFilter< TInputImage, TGrayValueImage >
		::FilterGrayValueImage(void)
	{
		/** Create m_Cx. */
		m_Cx = DoubleImageType::New();
		
		/** Rescale intensity of m_GrayValueImage to values between
		* 0.0 and 1.0.
		*/
		m_RescaleFilter = RescaleImageFilterType::New();
		m_RescaleFilter->SetOutputMinimum( 0.0 );
		m_RescaleFilter->SetOutputMaximum( 1.0 );
		m_RescaleFilter->SetInput( m_GrayValueImage );
		
		/** First set m_Cx = rescaleFilter->GetOutput(). */
		m_Cx = m_RescaleFilter->GetOutput();
		try
		{
			m_Cx->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			xl::xout["error"] << excp << std::endl;
		}
		
		/** Want to write m_Cx? *
		typedef Image< float, InputImageDimension >		FloatImageType;
		typedef CastImageFilter< DoubleImageType, FloatImageType > CastType;
		typedef ImageFileWriter< FloatImageType >		WriterType;
		typename CastType::Pointer caster = CastType::New();
		typename WriterType::Pointer mCxWriter = WriterType::New();
		if ( true )
		{
			caster->SetInput( m_Cx );
			mCxWriter->SetFileName( "Cx1.mhd" );
			mCxWriter->SetInput( caster->GetOutput() );
			mCxWriter->Modified();
			/** Do the writing. *
			try
			{
				mCxWriter->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}
		} // end if*/

		/** Set threshold value T. */
		double T1, T2;
		if ( this->GetUseThreshold() )
		{
			/** To avoid numerical instability, check if the
			 * threshold is between 0.00001 and 0.99999.
			 */
			if ( this->GetThreshold() >= 0.00001 && this->GetThreshold() <= 0.99999 )
			{
				T1 = this->GetThreshold();
				T2 = this->GetThreshold();
			}
			else
			{
				elxout << "WARNING: Threshold should be in the range [ 0.00001, 0.99999 ]." << std::endl;
				// \todo quit program?
			}
		}
		else
		{
			T1 = 0.00001;
			T2 = 0.99999;
		}

		/** Setup iterator. */
		ImageRegionIterator< DoubleImageType > it( m_Cx, m_Cx->GetLargestPossibleRegion() );
		it.GoToBegin();

		/** Then, calculate m_Cx = c(x) = c( rescaled image ). */
		while ( !it.IsAtEnd() )
		{
			/** Threshold or just make sure everything is between 0 and 1. */
			if ( it.Get() < T1 ) it.Set( 0.00001 );
			if ( it.Get() >= T2 ) it.Set( 0.99999 );
			/** Calculate m_Cx = c(x). */
			// float e = 2.7182818284590452353602874713527;
			//it.Set( ::exp( it.Get() * ln(2) ) - 1.0 );
			//it.Set( ::exp( it.Get() ) / ( e - 1.0 ) + 1.0 / ( 1.0 - e ) );
			//it.Set( ::sqrt( it.Get() ) );
			++it;
		}

		/** Want to write m_Cx? *
		if ( true )
		{
			caster->SetInput( m_Cx );
			caster->Modified();
			mCxWriter->SetFileName( "Cx.mhd" );
			mCxWriter->SetInput( caster->GetOutput() );
			mCxWriter->Modified();
			/** Do the writing. *
			try
			{
				mCxWriter->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				xl::xout["error"] << excp << std::endl;
			}
		} // end if*/
		
	} // end FilterGrayValueImage


} // end namespace itk

#endif // end #ifndef _itkVectorMeanDiffusionImageFilter_HXX__

