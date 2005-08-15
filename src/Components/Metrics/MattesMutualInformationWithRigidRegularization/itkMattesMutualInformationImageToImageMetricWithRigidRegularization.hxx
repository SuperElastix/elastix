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
#ifndef _itkMattesMutualInformationImageToImageMetricWithRigidRegularization_HXX__
#define _itkMattesMutualInformationImageToImageMetricWithRigidRegularization_HXX__

#include "itkMattesMutualInformationImageToImageMetricWithRigidRegularization.h"


namespace itk
{
	
	
	/**
	 * ********************* Constructor ****************************
	 */

	template < class TFixedImage, class TMovingImage >
		MattesMutualInformationImageToImageMetricWithRigidRegularization<TFixedImage,TMovingImage>
		::MattesMutualInformationImageToImageMetricWithRigidRegularization()
	{
		/** Initialize members. */
		this->m_RigidPenaltyWeight = NumericTraits<CoordinateRepresentationType>::One;
		this->m_SecondOrderWeight = NumericTraits<CoordinateRepresentationType>::One;
		this->m_DilationRadiusMultiplier = NumericTraits<CoordinateRepresentationType>::One;
		this->m_UseImageSpacing = true;
		this->m_DilateRigidityImages = true;
		this->m_RigidRegulizer = RigidRegulizerMetricType::New();
		this->m_FixedRigidityImage = 0;
		this->m_MovingRigidityImage = 0;
		this->m_RigidityCoefficientImage = RigidityImageType::New();

		this->m_FixedRigidityImageDilation.resize( FixedImageDimension );
		this->m_MovingRigidityImageDilation.resize( MovingImageDimension );
		for ( unsigned int i = 0; i < FixedImageDimension; i++ )
		{
			this->m_FixedRigidityImageDilation[ i ] = DilateFilterType::New();
			this->m_MovingRigidityImageDilation[ i ] = DilateFilterType::New();
		}

		this->m_FixedRigidityImageDilated = 0;
		this->m_MovingRigidityImageDilated = 0;

		/** For printing purposes. */
		this->m_MIValue = this->m_RigidValue = 0.0;
	
	} // end Constructor
	
	
	/**
	 * ********************* PrintSelf ******************************
	 *
	 * Print out internal information about this class.
	 */

	template < class TFixedImage, class TMovingImage  >
		void
		MattesMutualInformationImageToImageMetricWithRigidRegularization<TFixedImage,TMovingImage>
		::PrintSelf( std::ostream& os, Indent indent ) const
	{
		
		this->Superclass::PrintSelf(os, indent);
		
		os << indent << "RigidPenaltyWeight: "
			<< this->m_RigidPenaltyWeight << std::endl;
		os << indent << "SecondOrderWeight: "
			<< this->m_SecondOrderWeight << std::endl;
		os << indent << "UseImageSpacing: ";
		if ( this->m_UseImageSpacing ) os << "true" << std::endl;
		else os << "false" << std::endl;
		os << indent << "DilateRigidityImages: ";
		if ( this->m_DilateRigidityImages ) os << "true" << std::endl;
		else os << "false" << std::endl;
		
	} // end PrintSelf
	
	
	/**
	 * ********************* Initialize *****************************
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		MattesMutualInformationImageToImageMetricWithRigidRegularization<TFixedImage,TMovingImage>
		::Initialize(void) throw ( ExceptionObject )
	{
		/** Call the initialize of the superclass. */
		this->Superclass::Initialize();

		/** Set the B-spline transform to m_RigidRegulizer. */
		if( this->m_TransformIsBSpline )
		{
			this->m_RigidRegulizer->SetBSplineTransform( this->m_BSplineTransform );
		}
		else
		{
			itkExceptionMacro( << "ERROR: this metric expects a BSpline-transform." );
		}

		/** Initialize the rigid regulizer metric. */
		this->m_RigidRegulizer->SetSecondOrderWeight( this->m_SecondOrderWeight );
		this->m_RigidRegulizer->SetUseImageSpacing( this->m_UseImageSpacing );

		/** Allocate the RigidityCoefficientImage, so that it matches the B-spline grid.
		 * Only because the Initialize()-function above is called before,
		 * this code is valid, because there the B-spline transform is set.
		 */
		RigidityImageRegionType region;
		region.SetSize( this->m_BSplineTransform->GetGridRegion().GetSize() );
		region.SetIndex( this->m_BSplineTransform->GetGridRegion().GetIndex() );
		this->m_RigidityCoefficientImage->SetRegions( region );
		this->m_RigidityCoefficientImage->SetSpacing(
			this->m_BSplineTransform->GetGridSpacing() );
		this->m_RigidityCoefficientImage->SetOrigin(
			this->m_BSplineTransform->GetGridOrigin() );
		this->m_RigidityCoefficientImage->Allocate();

		/** Dilate m_FixedRigidityImage and m_MovingRigidityImage. */
		if ( this->m_DilateRigidityImages )
		{
			/** Some declarations. */
			SERadiusType						radius;
			std::vector< StructuringElementType >	structuringElement( FixedImageDimension );

			/** Setup the pipeline. */
			m_FixedRigidityImageDilation[ 0 ]->SetInput( m_FixedRigidityImage );
			m_MovingRigidityImageDilation[ 0 ]->SetInput( m_MovingRigidityImage );

			/** Set stuff for the separate dilation. */
			for ( unsigned int i = 0; i < FixedImageDimension; i++ )
			{
				/** Create the structuring element. */
				radius.Fill( 0 );
				radius.SetElement( i,
					static_cast<unsigned long>(this->m_DilationRadiusMultiplier * this->m_BSplineTransform->GetGridSpacing()[ i ]) );

				structuringElement[ i ].SetRadius( radius );
				structuringElement[ i ].CreateStructuringElement();

				/** Set the kernel into all dilation filters.
				 * The SetKernel() is implemented using a itkSetMacro, so a
				 * this->Modified() is automatically called, which is important,
				 * since this changes every time Initialize() is called (every resolution).
				 */
				this->m_FixedRigidityImageDilation[ i ]->SetKernel( structuringElement[ i ] );
				this->m_MovingRigidityImageDilation[ i ]->SetKernel( structuringElement[ i ] );

				/** Connect the pipelines. */
				if ( i > 0 )
				{
					this->m_FixedRigidityImageDilation[ i ]->SetInput(
						m_FixedRigidityImageDilation[ i - 1 ]->GetOutput() );
					this->m_MovingRigidityImageDilation[ i ]->SetInput(
						m_MovingRigidityImageDilation[ i - 1 ]->GetOutput() );
				}
			}

			/** Do the dilation for m_FixedRigidityImage. */
			try
			{
				this->m_FixedRigidityImageDilation[ FixedImageDimension - 1 ]->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MattesMutualInformationImageToImageMetricWithRigidRegularization - Initialize()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError while dilating m_FixedRigidityImage.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}

			/** Do the dilation for m_MovingRigidityImage. */
			try
			{
				this->m_MovingRigidityImageDilation[ MovingImageDimension - 1 ]->Update();
			}
			catch( itk::ExceptionObject & excp )
			{
				/** Add information to the exception. */
				excp.SetLocation( "MattesMutualInformationImageToImageMetricWithRigidRegularization - Initialize()" );
				std::string err_str = excp.GetDescription();
				err_str += "\nError while dilating m_MovingRigidityImage.\n";
				excp.SetDescription( err_str );
				/** Pass the exception to an higher level. */
				throw excp;
			}

			/** Put the output of the dilation into some dilated images. */
			this->m_FixedRigidityImageDilated =
				this->m_FixedRigidityImageDilation[ FixedImageDimension - 1 ]->GetOutput();
			this->m_MovingRigidityImageDilated =
				this->m_MovingRigidityImageDilation[ MovingImageDimension - 1 ]->GetOutput();

		}
		else
		{
			/** Copy the pointers of the undilated images to the dilated ones
			 * if no dilation is needed.
			 */
			this->m_FixedRigidityImageDilated = this->m_FixedRigidityImage;
			this->m_MovingRigidityImageDilated = this->m_MovingRigidityImage;

		} // end if rigidity images should be dilated
	
	} // end Initialize

	 
	/**
	 * ************************** GetValue **************************
	 *
	 * Get the match Measure.
	 */

	 template < class TFixedImage, class TMovingImage  >
		 typename MattesMutualInformationImageToImageMetricWithRigidRegularization<TFixedImage,TMovingImage>
		 ::MeasureType
		 MattesMutualInformationImageToImageMetricWithRigidRegularization<TFixedImage,TMovingImage>
		 ::GetValue( const ParametersType& parameters ) const
	 {
		 /** Call FillRigidityCoefficientImage. */
		 this->FillRigidityCoefficientImage( parameters );

		 /** Call the superclass to calculate the MI part. */
		 MeasureType MIValue = this->Superclass::GetValue( parameters );

		 /** Calculate the value of the rigid regularization penalty term. */
		 MeasureType RigidValue = this->m_RigidRegulizer->GetValue( parameters );

		 /** Return the value. */
		 return static_cast<MeasureType>( MIValue + this->m_RigidPenaltyWeight * RigidValue );
		 
	} // end GetValue


	 /**
		* ************************ GetDerivative ************************
	  */

	 template < class TFixedImage, class TMovingImage  >
		 void
		 MattesMutualInformationImageToImageMetricWithRigidRegularization<TFixedImage,TMovingImage>
		 ::GetDerivative( const ParametersType& parameters, DerivativeType & derivative ) const
	 {
		 /** Call FillRigidityCoefficientImage. */
		 this->FillRigidityCoefficientImage( parameters );

		 /** Declare stuff. */
		 DerivativeType MIDerivative, RigidDerivative;

		 /** Call the superclass to calculate the MI part. */
		 this->Superclass::GetDerivative( parameters, MIDerivative );

		 /** Calculate the value of the rigid regularization penalty term. */
		 this->m_RigidRegulizer->GetDerivative( parameters, RigidDerivative );

		 /** Calculate the sum of derivatives. */
		 derivative = MIDerivative + this->m_RigidPenaltyWeight * RigidDerivative;
		 
	} // end GetDerivative


	/**
	 * ******************** GetValueAndDerivative *******************
	 *
	 * Get both the Value and the Derivative of the Measure. 
	 * Both are computed on a randomly chosen set of voxels in the
	 * fixed image domain or on all pixels.
	 */

	 template < class TFixedImage, class TMovingImage  >
		 void
		 MattesMutualInformationImageToImageMetricWithRigidRegularization<TFixedImage,TMovingImage>
		 ::GetValueAndDerivative( const ParametersType& parameters,
			MeasureType& value, DerivativeType& derivative ) const
	 {
		 /** Call FillRigidityCoefficientImage. */
		 this->FillRigidityCoefficientImage( parameters );

		 /** Declare stuff. */
		 MeasureType		MIValue, RigidValue;
		 DerivativeType	MIDerivative, RigidDerivative;

		 /** Initialize. */
		 MIValue = RigidValue = NumericTraits<MeasureType>::Zero;
		 MIDerivative.Fill( 0.0 );
		 RigidDerivative.Fill( 0.0 );

		 /** Call the superclass to calculate the MI part. */
		 this->Superclass::GetValueAndDerivative( parameters, MIValue, MIDerivative );
		
		 /** Calculate the rigid regularization penalty term. */
		 this->m_RigidRegulizer->GetValueAndDerivative( parameters, RigidValue, RigidDerivative );
		 
		 /** Calculate the sum of values. */
		 value = static_cast<MeasureType>( MIValue + this->m_RigidPenaltyWeight * RigidValue );

		 /** Calculate the sum of derivatives. */
		 derivative = MIDerivative + this->m_RigidPenaltyWeight * RigidDerivative;

		 /** For printing purposes. */
		 this->m_MIValue = MIValue;
		 this->m_RigidValue = RigidValue;

		 //temp
		 if (0)
		 {
			 std::cout << "MI derivative:" << std::endl;
			 for ( unsigned int i = 0; i < MIDerivative.GetNumberOfElements(); i++ )
			 {
				 std::cout << MIDerivative[ i ] << " ";
			 }
			 std::cout << "\n\nRigid derivative:" << std::endl;
			 for ( unsigned int i = 0; i < RigidDerivative.GetNumberOfElements(); i++ )
			 {
				 std::cout << RigidDerivative[ i ] << " ";
			 }
			 std::cout << "\n\nderivative:" << std::endl;
			 for ( unsigned int i = 0; i < derivative.GetNumberOfElements(); i++ )
			 {
				 std::cout << derivative[ i ] << " ";
			 }
			 std::cout << std::endl;
		 } // end if
		
	} // end GetValueAndDerivative


	 /**
	 * **************** FillRigidityCoefficientImage *****************
	 */

	 template < class TFixedImage, class TMovingImage  >
		 void
		 MattesMutualInformationImageToImageMetricWithRigidRegularization<TFixedImage,TMovingImage>
		 ::FillRigidityCoefficientImage( const ParametersType& parameters ) const
	 {
		 /** Make sure that the transform is up to date. */
		 this->m_BSplineTransform->SetParameters( parameters );

		 /** Create and reset an iterator over m_RigidityCoefficientImage. */
		 RigidityImageIteratorType it( this->m_RigidityCoefficientImage,
			 this->m_RigidityCoefficientImage->GetLargestPossibleRegion() );
		 it.GoToBegin();

		 /** Fill m_RigidityCoefficientImage. */
		 RigidityPixelType fixedValue, movingValue, in;
		 RigidityImagePointType point;
		 RigidityImageIndexType index1, index2;
		 bool isInFixedImage, isInMovingImage;
		 while ( !it.IsAtEnd() )
		 {
			 /** Get current pixel in world coordinates. */
			 this->m_RigidityCoefficientImage
				 ->TransformIndexToPhysicalPoint( it.GetIndex(), point );

			 /** Get the corresponding indices in the fixed and moving RigidityImage's.
				* NOTE: Floating point index results are truncated to integers.
				*/
			 isInFixedImage = this->m_FixedRigidityImageDilated
				 ->TransformPhysicalPointToIndex( point, index1 );
			 isInMovingImage = this->m_MovingRigidityImageDilated
				 ->TransformPhysicalPointToIndex(
				 this->m_BSplineTransform->TransformPoint( point ), index2 );

			 /** Get the values at those positions. */
			 if ( isInFixedImage )
			 {
				 fixedValue = this->m_FixedRigidityImageDilated->GetPixel( index1 );
			 }
			 else
			 {
				 fixedValue = 0.0;
			 }

			 if ( isInMovingImage )
			 {
				 movingValue = this->m_MovingRigidityImageDilated->GetPixel( index2 );
			 }
			 else
			 {
				 movingValue = 0.0;
			 }

			 /** Determine the maximum. */
			 in = ( fixedValue > movingValue ? fixedValue : movingValue );
			
			 /** Set it. */
			 it.Set( in );
			
			 /** Increase iterator. */
			 ++it;
		 }

     /** Set the rigidity coefficients image into the rigid regulizer metric. */
		 this->m_RigidRegulizer->SetRigidityImage( this->m_RigidityCoefficientImage );

		 // tmp
		 if (0)
		 {
			 unsigned int level = 0;
				 //( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
			 std::ostringstream makeFileName("");
			 makeFileName << "RigidityCoefficientImage."
//				 << this->m_Configuration->GetElastixLevel()
				 << ".R" << level
				 << ".mhd";
			 std::string FileName = makeFileName.str();
			 typedef ImageFileWriter<RigidityImageType>	RigidityImageWriterType;
			 typename RigidityImageWriterType::Pointer rigwriter = RigidityImageWriterType::New();
			 rigwriter->SetFileName( FileName.c_str() );
			 rigwriter->SetInput( this->m_RigidityCoefficientImage );
			 rigwriter->Update();
		 }

	 } // end FillRigidityCoefficientImage

	/**
	 * ******************** SetOutputDirectoryName ******************
	 *
	 * This is a copy of the itkSetStringMacro, but with the additional
	 * pass of the OutputDirectoryName to the rigidregulizer metric.
	 */

	 template < class TFixedImage, class TMovingImage  >
		 void
		 MattesMutualInformationImageToImageMetricWithRigidRegularization<TFixedImage,TMovingImage>
		 ::SetOutputDirectoryName( const char * _arg )
	 {
		 /** Set the pointer in this class. */
		 if ( _arg && ( _arg == this->m_OutputDirectoryName ) ) { return; }
		 if ( _arg )
		 {
			 this->m_OutputDirectoryName = _arg;
		 }
		 else
		 {
			 this->m_OutputDirectoryName = "";
		 }
		 this->Modified();

		 /** Set the pointer in the RigidRegulizerMetric class. */
		 this->m_RigidRegulizer->SetOutputDirectoryName( this->m_OutputDirectoryName.c_str() );

	 } // end SetOutputDirectoryName

} // end namespace itk


#endif // end #ifndef _itkMattesMutualInformationImageToImageMetricWithRigidRegularization_HXX__

