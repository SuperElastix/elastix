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
#ifndef _itkMeanSquaresImageToImageMetric2_txx
#define _itkMeanSquaresImageToImageMetric2_txx

#include "itkMeanSquaresImageToImageMetric2.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRandomConstIteratorWithIndex.h"

namespace itk
{

	/**
	* ******************* Constructor *******************
	*/

	template <class TFixedImage, class TMovingImage> 
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::MeanSquaresImageToImageMetric2()
	{
		itkDebugMacro("Constructor");

		this->m_UseAllPixels = true;
		this->m_NumberOfSpatialSamples = 0;
	} // end constructor


	/**
	 * ******************* PrintSelf *******************
	 */

	template < class TFixedImage, class TMovingImage> 
		void
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf( os, indent );
		os << indent << "UseAllPixels: " << this->m_UseAllPixels << std::endl;
		os << indent << "NumberOfSpatialSamples: " << this->m_NumberOfSpatialSamples << std::endl;

	} // end PrintSelf


	/**
	 * ******************* GetValue *******************
	 */

	template <class TFixedImage, class TMovingImage> 
		typename MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>::MeasureType
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValue( const TransformParametersType & parameters ) const
	{
		/** Select which GetValue to use. */
		if ( this->m_UseAllPixels )
		{
			return this->GetValueUsingAllPixels( parameters );
		}
		else
		{
			return this->GetValueUsingSomePixels( parameters );
		}

	} // end GetValue


	/**
	 * ******************* GetValueUsingAllPixels *******************
	 */

	template <class TFixedImage, class TMovingImage> 
		typename MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>::MeasureType
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValueUsingAllPixels( const TransformParametersType & parameters ) const
	{
		itkDebugMacro("GetValue( " << parameters << " ) ");

		/** Some sanity checks. */
		FixedImageConstPointer fixedImage = this->m_FixedImage;
		if ( !fixedImage ) 
		{
			itkExceptionMacro( << "Fixed image has not been assigned" );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

		this->m_NumberOfPixelsCounted = 0;

		/** Some typedefs. */
		typedef ImageRegionConstIteratorWithIndex<FixedImageType> FixedIteratorType;
		typedef typename Superclass::InputPointType			InputPointType;
		typedef typename Superclass::OutputPointType		OutputPointType;

		/** Create iterator over the fixed image. */
		FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

		/** Create variables to store intermediate results. */
		typename FixedImageType::IndexType index;
		InputPointType inputPoint;
		OutputPointType transformedPoint;

		MeasureType measure = NumericTraits< MeasureType >::Zero;

		/** Loop over the fixed image to calculate the mean squares. */
		while ( !ti.IsAtEnd() )
		{
			/** Get the current inputpoint. */
			index = ti.GetIndex();
			fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

			/** Inside the fixed image mask? */
			if ( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
			{
				++ti;
				continue;
			}

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				++ti;
				continue;
			}

			/** In this if-statement the actual calculation of mean squares is done. */
			if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
				const RealType & fixedValue  = ti.Value();

				/** The difference squared. */
				const RealType diff = movingValue - fixedValue; 
				measure += diff * diff;

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

			/** Increase iterator. */
			++ti;
		} // end while loop over fixed image voxels

		/** Calculate the measure value. */
		if ( this->m_NumberOfPixelsCounted > 0 )
		{
			measure /= this->m_NumberOfPixelsCounted;
		}
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
		}

		/** Throw exceptions if necessary. */
		if ( this->m_NumberOfPixelsCounted == 0 )
		{
			itkExceptionMacro( << "All the points mapped outside the moving image" );
		}

		/** Return the mean squares measure value. */
		return measure;

	} // end GetValueUsingAllPixels


	/**
	 * ******************* GetValueUsingSomePixels *******************
	 */

	template <class TFixedImage, class TMovingImage> 
		typename MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>::MeasureType
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValueUsingSomePixels( const TransformParametersType & parameters ) const
	{
		itkDebugMacro("GetValue( " << parameters << " ) ");

		/** Some sanity checks. */
		FixedImageConstPointer fixedImage = this->m_FixedImage;
		if ( !fixedImage ) 
		{
			itkExceptionMacro( << "Fixed image has not been assigned" );
		}

		if ( this->m_NumberOfSpatialSamples == 0 ) 
		{
			itkExceptionMacro( << "NumberOfSpatialSamples has not been set" );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

		this->m_NumberOfPixelsCounted = 0;

		/** Some typedefs. */
		typedef ImageRandomConstIteratorWithIndex<FixedImageType> FixedIteratorType;
		typedef typename Superclass::InputPointType			InputPointType;
		typedef typename Superclass::OutputPointType		OutputPointType;

		/** Create iterator over the fixed image. */
		FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

		/** Set the maximum number of random iterator steps, so that we don't
		 * get stuck in an infinite loop when the two images are not overlapping.
		 */
		ti.SetNumberOfSamples( 10 * this->m_NumberOfSpatialSamples );

		/** Create variables to store intermediate results. */
		typename FixedImageType::IndexType index;
		InputPointType inputPoint;
		OutputPointType transformedPoint;

		MeasureType measure = NumericTraits< MeasureType >::Zero;

		/** Loop over the fixed image to calculate the mean squares. */
		while ( this->m_NumberOfSpatialSamples > this->m_NumberOfPixelsCounted && !ti.IsAtEnd() )
		{
			/** Get the current inputpoint. */
			index = ti.GetIndex();
			fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

			/** Inside the fixed image mask? */
			if ( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
			{
				++ti;
				continue;
			}

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				++ti;
				continue;
			}

			/** In this if-statement the actual calculation of mean squares is done. */
			if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
				const RealType & fixedValue  = ti.Value();

				/** The difference squared. */
				const RealType diff = movingValue - fixedValue; 
				measure += diff * diff;

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

			/** Increase iterator. */
			++ti;
		} // end while loop over fixed image voxels

		/** Calculate the measure value. */
		if ( this->m_NumberOfPixelsCounted > 0 )
		{
			measure /= this->m_NumberOfPixelsCounted;
		}
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
		}

		/** Throw exceptions if necessary. */
		if ( this->m_NumberOfPixelsCounted == 0 )
		{
			itkExceptionMacro( << "All the points mapped outside the moving image" );
		}

		if ( this->m_NumberOfPixelsCounted < this->m_NumberOfSpatialSamples / 4 )
		{
			itkExceptionMacro( "Too many samples map outside the moving image buffer: "
				<< this->m_NumberOfPixelsCounted << " / " << this->m_NumberOfSpatialSamples << std::endl );
		}

		/** Return the mean squares measure value. */
		return measure;

	} // end GetValueUsingSomePixels


	/**
	 * ******************* GetDerivative *******************
	 */

	template < class TFixedImage, class TMovingImage> 
		void
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::GetDerivative( const TransformParametersType & parameters,
		DerivativeType & derivative ) const
	{
		/** When the derivative is calculated, all information for calculating
		 * the metric value is available. It does not cost anything to calculate
		 * the metric value now. Therefore, we have chosen to only implement the
		 * GetValueAndDerivative(), supplying it with a dummy value variable. */
		MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
		this->GetValueAndDerivative( parameters, dummyvalue, derivative );

	} // end GetDerivative


	/**
	 * ******************* GetValueAndDerivative *******************
	 */

	template <class TFixedImage, class TMovingImage>
		void
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValueAndDerivative( const TransformParametersType & parameters, 
		MeasureType & value, DerivativeType & derivative ) const
	{
		/** Select which GetValueAndDerivative to use. */
		if ( this->m_UseAllPixels )
		{
			this->GetValueAndDerivativeUsingAllPixels( parameters, value, derivative );
		}
		else
		{
			this->GetValueAndDerivativeUsingSomePixels( parameters, value, derivative );
		}

	} // end GetValueAndDerivative



	/**
	 * ******************* GetValueAndDerivativeUsingAllPixels *******************
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValueAndDerivativeUsingAllPixels(const TransformParametersType & parameters, 
		MeasureType & value, DerivativeType  & derivative) const
	{
		itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

		/** Some sanity checks. */
		if ( !this->GetGradientImage() )
		{
			itkExceptionMacro( << "The gradient image is null, maybe you forgot to call Initialize()" );
		}

		FixedImageConstPointer fixedImage = this->m_FixedImage;
		if ( !fixedImage ) 
		{
			itkExceptionMacro( << "Fixed image has not been assigned" );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );
		const unsigned int ParametersDimension = this->GetNumberOfParameters();
		
		this->m_NumberOfPixelsCounted = 0;

		/** Some typedefs. */
		typedef ImageRegionConstIteratorWithIndex<FixedImageType> FixedIteratorType;
		typedef typename Superclass::InputPointType			InputPointType;
		typedef typename Superclass::OutputPointType		OutputPointType;
		typedef typename OutputPointType::CoordRepType	CoordRepType;
		typedef ContinuousIndex<CoordRepType,
			MovingImageType::ImageDimension>							MovingImageContinuousIndexType;

		/** Create iterator over the fixed image. */
		FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

		/** Create variables to store intermediate results. */
		typename FixedImageType::IndexType	index;
		InputPointType	inputPoint;
		OutputPointType transformedPoint;
		MovingImageContinuousIndexType			tempIndex;
		typename MovingImageType::IndexType	mappedIndex;

		MeasureType measure = NumericTraits< MeasureType >::Zero;
		
		derivative = DerivativeType( ParametersDimension );
		derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		/** Loop over the fixed image to calculate the mean squares. */
		ti.GoToBegin();
		while ( !ti.IsAtEnd() )
		{
			/** Get the current inputpoint. */
			index = ti.GetIndex();
			fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

			/** Inside the fixed image mask? */
			if ( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
			{
				++ti;
				continue;
			}

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				++ti;
				continue;
			}

			/** In this if-statement the actual calculation of mean squares is done. */
			if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue = this->m_Interpolator->Evaluate( transformedPoint );
				const RealType & fixedValue  = ti.Value();

				/** Get the Jacobian. */
				const TransformJacobianType & jacobian =
					this->m_Transform->GetJacobian( inputPoint ); 

				/** The difference squared. */
				const RealType diff = movingValue - fixedValue; 
				measure += diff * diff;

				/** Get the gradient by NearestNeighboorInterpolation:
				 * which is equivalent to round up the point components.*/
				this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );
				for ( unsigned int j = 0; j < MovingImageDimension; j++ )
				{
					mappedIndex[ j ] = static_cast<long>( vnl_math_rnd( tempIndex[ j ] ) );
				}
				const GradientPixelType gradient = this->GetGradientImage()->GetPixel( mappedIndex );

				/** Calculate the contributions to all parameters. */
				for ( unsigned int par = 0; par < ParametersDimension; par++ )
				{
					RealType sum = NumericTraits< RealType >::Zero;
					for( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
					{
						sum += 2.0 * diff * jacobian( dim, par ) * gradient[ dim ];
					}
					derivative[ par ] += sum;
				}
        
				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

			/** Increase iterator. */
			++ti;
		} // end while loop over fixed image voxels

		/** Calculate the value and the derivative. */
		if ( this->m_NumberOfPixelsCounted > 0 )
		{
			measure /= this->m_NumberOfPixelsCounted;
			for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivative[ i ] /= this->m_NumberOfPixelsCounted;
			}
		}
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
			derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );
		}

		/** Throw exceptions if necessary. */
		if ( this->m_NumberOfPixelsCounted == 0 )
		{
			itkExceptionMacro( << "All the points mapped outside the moving image" );
		}

		/** The return value. */
		value = measure;

	} // end GetValueAndDerivativeUsingAllPixels


	/**
	 * ******************* GetValueAndDerivativeUsingSomePixels *******************
	 */

	template <class TFixedImage, class TMovingImage>
		void
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValueAndDerivativeUsingSomePixels( const TransformParametersType & parameters, 
		MeasureType & value, DerivativeType & derivative ) const
	{
		itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

		/** Some sanity checks. */
		if ( !this->GetGradientImage() )
		{
			itkExceptionMacro( << "The gradient image is null, maybe you forgot to call Initialize()" );
		}

		FixedImageConstPointer fixedImage = this->m_FixedImage;
		if ( !fixedImage ) 
		{
			itkExceptionMacro( << "Fixed image has not been assigned" );
		}

		if ( this->m_NumberOfSpatialSamples == 0 ) 
		{
			itkExceptionMacro( << "NumberOfSpatialSamples has not been set" );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );
		const unsigned int ParametersDimension = this->GetNumberOfParameters();
		
		this->m_NumberOfPixelsCounted = 0;

		/** Some typedefs. */
		typedef ImageRandomConstIteratorWithIndex<FixedImageType>  FixedIteratorType;
		typedef typename Superclass::InputPointType			InputPointType;
		typedef typename Superclass::OutputPointType		OutputPointType;
		typedef typename OutputPointType::CoordRepType	CoordRepType;
		typedef ContinuousIndex<CoordRepType,
			MovingImageType::ImageDimension>							MovingImageContinuousIndexType;

		/** Create iterator over the fixed image. */
		FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

		/** Set the maximum number of random iterator steps, so that we don't
		 * get stuck in an infinite loop when the two images are not overlapping.
		 */
		ti.SetNumberOfSamples( 10 * this->m_NumberOfSpatialSamples );

		/** Create variables to store intermediate results. */
		typename FixedImageType::IndexType	index;
		InputPointType	inputPoint;
		OutputPointType transformedPoint;
		MovingImageContinuousIndexType			tempIndex;	
		typename MovingImageType::IndexType	mappedIndex;

		MeasureType measure = NumericTraits< MeasureType >::Zero;
		
		derivative = DerivativeType( ParametersDimension );
		derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		/** Loop over the fixed image to calculate the mean squares. */
		ti.GoToBegin();
		while ( this->m_NumberOfSpatialSamples > this->m_NumberOfPixelsCounted && !ti.IsAtEnd() )
		{
			/** Get the current inputpoint. */
			index = ti.GetIndex();
			fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

			/** Inside the fixed image mask? */
			if ( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
			{
				++ti;
				continue;
			}

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				++ti;
				continue;
			}

			/** In this if-statement the actual calculation of mean squares is done. */
			if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue = this->m_Interpolator->Evaluate( transformedPoint );
				const RealType & fixedValue  = ti.Value();

				/** Get the Jacobian. */
				const TransformJacobianType & jacobian =
					this->m_Transform->GetJacobian( inputPoint ); 

				/** The difference squared. */
				const RealType diff = movingValue - fixedValue; 
				measure += diff * diff;

				/** Get the gradient by NearestNeighboorInterpolation:
				 * which is equivalent to round up the point components.*/
				this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );
				for ( unsigned int j = 0; j < MovingImageDimension; j++ )
				{
					mappedIndex[ j ] = static_cast<long>( vnl_math_rnd( tempIndex[ j ] ) );
				}
				const GradientPixelType gradient = this->GetGradientImage()->GetPixel( mappedIndex );

				/** Calculate the contributions to all parameters. */
				for ( unsigned int par = 0; par < ParametersDimension; par++ )
				{
					RealType sum = NumericTraits< RealType >::Zero;
					for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
					{
						sum += 2.0 * diff * jacobian( dim, par ) * gradient[ dim ];
					}
					derivative[ par ] += sum;
				}
        
				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

			/** Increase iterator. */
			++ti;
		} // end while loop over fixed image voxels

		/** Calculate the value and the derivative. */
		if ( this->m_NumberOfPixelsCounted > 0 )
		{
			measure /= this->m_NumberOfPixelsCounted;
			for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivative[ i ] /= this->m_NumberOfPixelsCounted;
			}
		}
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
			derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );
		}

		/** Throw exceptions if necessary. */
		if ( this->m_NumberOfPixelsCounted == 0 )
		{
			itkExceptionMacro( << "All the points mapped outside the moving image" );
		}

		if ( this->m_NumberOfPixelsCounted < this->m_NumberOfSpatialSamples / 4 )
		{
			itkExceptionMacro( "Too many samples map outside the moving image buffer: "
				<< this->m_NumberOfPixelsCounted << " / " << this->m_NumberOfSpatialSamples << std::endl );
		}

		/** The return value. */
		value = measure;

	} // end GetValueAndDerivativeUsingSomePixels


} // end namespace itk

#endif // end #ifndef _itkMeanSquaresImageToImageMetric2_txx

