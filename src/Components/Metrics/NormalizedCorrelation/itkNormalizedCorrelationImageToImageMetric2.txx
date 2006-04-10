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
#ifndef _itkNormalizedCorrelationImageToImageMetric2_txx
#define _itkNormalizedCorrelationImageToImageMetric2_txx

#include "itkNormalizedCorrelationImageToImageMetric2.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRandomConstIteratorWithIndex.h"

namespace itk
{

	/**
	 * ******************* Constructor *******************
	 */
	template <class TFixedImage, class TMovingImage> 
		NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>
		::NormalizedCorrelationImageToImageMetric2()
	{
		this->m_SubtractMean = false;
		this->m_UseAllPixels = true;
		this->m_NumberOfSpatialSamples = 0;
	} // end constructor


	/**
	 * ******************* PrintSelf *******************
	 */
	template < class TFixedImage, class TMovingImage> 
		void
		NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>
		::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf( os, indent );
		os << indent << "SubtractMean: " << this->m_SubtractMean << std::endl;
		os << indent << "UseAllPixels: " << this->m_UseAllPixels << std::endl;
		os << indent << "NumberOfSpatialSamples: " << this->m_NumberOfSpatialSamples << std::endl;

	} // end PrintSelf


	/**
	 * ******************* GetValue *******************
	 */
	template <class TFixedImage, class TMovingImage> 
		typename NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>::MeasureType
		NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>
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
		typename NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>::MeasureType
		NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValueUsingAllPixels( const TransformParametersType & parameters ) const
	{
		/** Some sanity checks. */
		FixedImageConstPointer fixedImage = this->m_FixedImage;
		if( !fixedImage ) 
		{
			itkExceptionMacro( << "Fixed image has not been assigned" );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

		this->m_NumberOfPixelsCounted = 0;

		/** Some typedefs. */
		typedef ImageRegionConstIteratorWithIndex<FixedImageType>				FixedIteratorType;
		typedef typename NumericTraits< MeasureType >::AccumulateType		AccumulateType;
		typedef typename Superclass::InputPointType			InputPointType;
		typedef typename Superclass::OutputPointType		OutputPointType;

		/** Create iterator over the fixed image. */
		FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

		/** Create variables to store intermediate results. */
		typename FixedImageType::IndexType index;
		MeasureType measure;
		InputPointType inputPoint;
		OutputPointType transformedPoint;

		AccumulateType sff = NumericTraits< AccumulateType >::Zero;
		AccumulateType smm = NumericTraits< AccumulateType >::Zero;
		AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
		AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
		AccumulateType sm  = NumericTraits< AccumulateType >::Zero;

		/** Loop over the fixed image to calculate the normalized correlation metric NC. */
		while( !ti.IsAtEnd() )
		{
			/** Get the current inputpoint. */
			index = ti.GetIndex();
			fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

			/** Inside the fixed image mask? */
			if( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
			{
				++ti;
				continue;
			}

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				++ti;
				continue;
			}

			/** In this if-statement the actual calculation of NC is done. */
			if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
				const RealType & fixedValue = ti.Value();

				/** Update some sums needed to calculate NC. */
				sff += fixedValue  * fixedValue;
				smm += movingValue * movingValue;
				sfm += fixedValue  * movingValue;
				if ( this->m_SubtractMean ) // faster to get rid of the if?
				{
					sf += fixedValue;
					sm += movingValue;
				}

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

			/** Increase iterator. */
			++ti;
		} // end while loop over fixed image voxels

		/** If SubtractMean, then subtract things from sff, smm and sfm. */
		if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
		{
			sff -= ( sf * sf / this->m_NumberOfPixelsCounted );
			smm -= ( sm * sm / this->m_NumberOfPixelsCounted );
			sfm -= ( sf * sm / this->m_NumberOfPixelsCounted );
		}

		/** The denominator of the NC. */
		const RealType denom = -1.0 * sqrt( sff * smm );

		/** Calculate the measure value. */
		if( this->m_NumberOfPixelsCounted > 0 && denom < 0.00000001 )
		{
			measure = sfm / denom;
		}
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
		}

		/** Return the NC measure value. */
		return measure;

	} // end GetValueUsingAllPixels


	/**
	 * ******************* GetValueUsingSomePixels *******************
	 */
	template <class TFixedImage, class TMovingImage> 
		typename NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>::MeasureType
		NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValueUsingSomePixels( const TransformParametersType & parameters ) const
	{
		/** Some sanity checks. */
		FixedImageConstPointer fixedImage = this->m_FixedImage;
		if( !fixedImage ) 
		{
			itkExceptionMacro( << "Fixed image has not been assigned." );
		}

		if( this->m_NumberOfSpatialSamples == 0 ) 
		{
			itkExceptionMacro( << "NumberOfSpatialSamples has not been set." );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

		this->m_NumberOfPixelsCounted = 0;

		/** Some typedefs. */
		typedef ImageRandomConstIteratorWithIndex<FixedImageType>				FixedIteratorType;
		typedef typename NumericTraits< MeasureType >::AccumulateType		AccumulateType;
		typedef typename Superclass::InputPointType			InputPointType;
		typedef typename Superclass::OutputPointType		OutputPointType;

		/** Create iterator over the fixed image. */
		FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

		/** Create variables to store intermediate results. */
		typename FixedImageType::IndexType index;
		MeasureType measure;
		InputPointType inputPoint;
		OutputPointType transformedPoint;

		AccumulateType sff = NumericTraits< AccumulateType >::Zero;
		AccumulateType smm = NumericTraits< AccumulateType >::Zero;
		AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
		AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
		AccumulateType sm  = NumericTraits< AccumulateType >::Zero;

		/** Loop over the fixed image to calculate the normalized correlation matric NC. */
		while( this->m_NumberOfSpatialSamples > this->m_NumberOfPixelsCounted )
		{
			/** Get the current inputpoint. */
			index = ti.GetIndex();
			fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

			/** Inside the fixed image mask? */
			if( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
			{
				++ti;
				continue;
			}

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				++ti;
				continue;
			}

			/** In this if-statement the actual calculation of NC is done. */
			if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
				const RealType & fixedValue = ti.Value();

				/** Update some sums needed to calculate NC. */
				sff += fixedValue  * fixedValue;
				smm += movingValue * movingValue;
				sfm += fixedValue  * movingValue;
				if ( this->m_SubtractMean ) // faster to get rid of the if?
				{
					sf += fixedValue;
					sm += movingValue;
				}

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

			/** Increase iterator. */
			++ti;
		} // end while loop over fixed image voxels

		/** If SubtractMean, then subtract things from sff, smm and sfm. */
		if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
		{
			sff -= ( sf * sf / this->m_NumberOfPixelsCounted );
			smm -= ( sm * sm / this->m_NumberOfPixelsCounted );
			sfm -= ( sf * sm / this->m_NumberOfPixelsCounted );
		}

		/** The denominator of the NC. */
		const RealType denom = -1.0 * sqrt( sff * smm );

		/** Calculate the measure value. */
		if( this->m_NumberOfPixelsCounted > 0 && denom < 0.00000001 )
		{
			measure = sfm / denom;
		}
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
		}

		/** Return the NC measure value. */
		return measure;

	} // end GetValueUsingSomePixels


	/**
	 * ******************* GetDerivative *******************
	 */
	template < class TFixedImage, class TMovingImage> 
		void
		NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>
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
		NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>
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
	template < class TFixedImage, class TMovingImage> 
		void
		NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValueAndDerivativeUsingAllPixels( const TransformParametersType & parameters, 
		MeasureType & value, DerivativeType & derivative ) const
	{
		/** Some sanity checks. */
		if( !this->GetGradientImage() )
		{
			itkExceptionMacro(<<"The gradient image is null, maybe you forgot to call Initialize()");
		}

		FixedImageConstPointer fixedImage = this->m_FixedImage;
		if( !fixedImage ) 
		{
			itkExceptionMacro( << "Fixed image has not been assigned" );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );
		const unsigned int ParametersDimension = this->GetNumberOfParameters();

		this->m_NumberOfPixelsCounted = 0;
		
		/** Some typedefs. */
		typedef ImageRegionConstIteratorWithIndex<FixedImageType>				FixedIteratorType;
		typedef typename NumericTraits< MeasureType >::AccumulateType		AccumulateType;
		typedef typename Superclass::InputPointType			InputPointType;
		typedef typename Superclass::OutputPointType		OutputPointType;
		typedef typename OutputPointType::CoordRepType	CoordRepType;
		typedef ContinuousIndex<CoordRepType,
			MovingImageDimension>													MovingImageContinuousIndexType;

		/** Create iterator over the fixed image. */
		FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

		/** Create variables to store intermediate results. */
		typename FixedImageType::IndexType index;
		MovingImageContinuousIndexType tempIndex;
		typename MovingImageType::IndexType mappedIndex;
		InputPointType inputPoint;
		OutputPointType transformedPoint;
		
		AccumulateType sff = NumericTraits< AccumulateType >::Zero;
		AccumulateType smm = NumericTraits< AccumulateType >::Zero;
		AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
		AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
		AccumulateType sm  = NumericTraits< AccumulateType >::Zero;

		derivative = DerivativeType( ParametersDimension );
		derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		DerivativeType derivativeF = DerivativeType( ParametersDimension );
		derivativeF.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		DerivativeType derivativeM = DerivativeType( ParametersDimension );
		derivativeM.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		DerivativeType differential = DerivativeType( ParametersDimension );
		differential.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		/** Loop over the fixed image to calculate the normalized correlation metric NC. */
		ti.GoToBegin();
		while( !ti.IsAtEnd() )
		{
			/** Get the current inputpoint. */
			index = ti.GetIndex();
			fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

			/** Inside the fixed image mask? */
			if( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
			{
				++ti;
				continue;
			}

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				++ti;
				continue;
			}

			/** In this if-statement the actual calculation of NC is done. */
			if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue = this->m_Interpolator->Evaluate( transformedPoint );
				const RealType & fixedValue = ti.Value();

				/** Update some sums needed to calculate NC. */
				sff += fixedValue  * fixedValue;
				smm += movingValue * movingValue;
				sfm += fixedValue  * movingValue;
				if ( this->m_SubtractMean ) // is it faster to get rid of the if and calculate sf,sm anyway?
				{
					sf += fixedValue;
					sm += movingValue;
				}

				/** Get the Jacobian. */
				const TransformJacobianType & jacobian = this->m_Transform->GetJacobian( inputPoint ); 

				/** Get the gradient by NearestNeighboorInterpolation:
				 * which is equivalent to round up the point components.*/
				this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );
				for( unsigned int j = 0; j < MovingImageDimension; j++ )
				{
					mappedIndex[ j ] = static_cast<long>( vnl_math_rnd( tempIndex[ j ] ) );
				}
				const GradientPixelType gradient = this->GetGradientImage()->GetPixel( mappedIndex );

				/** Calculate the contributions to all parameters. */
				for( unsigned int par = 0; par < ParametersDimension; par++ )
				{
					RealType sumF = NumericTraits< RealType >::Zero;
					RealType sumM = NumericTraits< RealType >::Zero;
					RealType differentialtmp1 = NumericTraits< RealType >::Zero;
					/** Calculate the inner product of the Jacobian and the gradient.
					 * Then multiply with fixedValue or movingValue. */
					for( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
					{
						const RealType differentialtmp2 = jacobian( dim, par ) * gradient[ dim ];
						differentialtmp1 += differentialtmp2;
						sumF += fixedValue  * differentialtmp2;
						sumM += movingValue * differentialtmp2;
					}
					derivativeF[  par ] += sumF;
					derivativeM[  par ] += sumM;
					differential[ par ] += differentialtmp1;
				} // end for loop over the parameters

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

			/** Increase iterator. */
			++ti;
		} // end while loop over fixed image voxels

		/** If SubtractMean, then subtract things from sff, smm, sfm,
		 * derivativeF and derivativeM. */
		if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
		{
			sff -= ( sf * sf / this->m_NumberOfPixelsCounted );
			smm -= ( sm * sm / this->m_NumberOfPixelsCounted );
			sfm -= ( sf * sm / this->m_NumberOfPixelsCounted );

			for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivativeF[ i ] -= sf * differential[ i ] / this->m_NumberOfPixelsCounted;
				derivativeM[ i ] -= sm * differential[ i ] / this->m_NumberOfPixelsCounted;
			}
		}

		/** The denominator of the value and the derivative. */
		const RealType denom = -1.0 * vcl_sqrt( sff * smm );

		/** Calculate the value and the derivative. */
		if( this->m_NumberOfPixelsCounted > 0 && denom < 0.00000001 )
		{
			value = sfm / denom;
			for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivative[ i ] = ( derivativeF[ i ] - ( sfm / smm ) * derivativeM[ i ] ) / denom;
			}
		}
		else
		{
			value = NumericTraits< MeasureType >::Zero;
			for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivative[ i ] = NumericTraits< MeasureType >::Zero;
			}
		}
	
	} // end GetValueAndDerivativeUsingAllPixels


	/**
	 * ******************* GetValueAndDerivativeUsingSomePixels *******************
	 */
	template <class TFixedImage, class TMovingImage>
		void
		NormalizedCorrelationImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValueAndDerivativeUsingSomePixels( const TransformParametersType & parameters, 
		MeasureType & value, DerivativeType & derivative ) const
	{
		/** Some sanity checks. */
		if( !this->GetGradientImage() )
		{
			itkExceptionMacro(<<"The gradient image is null, maybe you forgot to call Initialize()");
		}

		FixedImageConstPointer fixedImage = this->m_FixedImage;
		if( !fixedImage ) 
		{
			itkExceptionMacro( << "Fixed image has not been assigned" );
		}

		if( this->m_NumberOfSpatialSamples == 0 ) 
		{
			itkExceptionMacro( << "NumberOfSpatialSamples has not been set" );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );
		const unsigned int ParametersDimension = this->GetNumberOfParameters();

		this->m_NumberOfPixelsCounted = 0;
		
		/** Some typedefs. */
		typedef ImageRandomConstIteratorWithIndex<FixedImageType>				FixedIteratorType;
		typedef typename NumericTraits< MeasureType >::AccumulateType		AccumulateType;
		typedef typename Superclass::InputPointType			InputPointType;
		typedef typename Superclass::OutputPointType		OutputPointType;
		typedef typename OutputPointType::CoordRepType	CoordRepType;
		typedef ContinuousIndex<CoordRepType,
			MovingImageDimension>													MovingImageContinuousIndexType;

		/** Create iterator over the fixed image. */
		FixedIteratorType ti( fixedImage, this->GetFixedImageRegion() );

		/** Create variables to store intermediate results. */
		typename FixedImageType::IndexType index;
		MovingImageContinuousIndexType tempIndex;
		typename MovingImageType::IndexType mappedIndex;
		InputPointType inputPoint;
		OutputPointType transformedPoint;
		
		AccumulateType sff = NumericTraits< AccumulateType >::Zero;
		AccumulateType smm = NumericTraits< AccumulateType >::Zero;
		AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
		AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
		AccumulateType sm  = NumericTraits< AccumulateType >::Zero;

		derivative = DerivativeType( ParametersDimension );
		derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		DerivativeType derivativeF = DerivativeType( ParametersDimension );
		derivativeF.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		DerivativeType derivativeM = DerivativeType( ParametersDimension );
		derivativeM.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		DerivativeType differential = DerivativeType( ParametersDimension );
		differential.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		/** Loop over the fixed image to calculate the normalized correlation metric NC. */
		ti.GoToBegin();
		while( this->m_NumberOfSpatialSamples > this->m_NumberOfPixelsCounted )
		{
			/** Get the current inputpoint. */
			index = ti.GetIndex();
			fixedImage->TransformIndexToPhysicalPoint( index, inputPoint );

			/** Inside the fixed image mask? */
			if( this->m_FixedImageMask && !this->m_FixedImageMask->IsInside( inputPoint ) )
			{
				++ti;
				continue;
			}

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				++ti;
				continue;
			}

			/** In this if-statement the actual calculation of NC is done. */
			if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue = this->m_Interpolator->Evaluate( transformedPoint );
				const RealType & fixedValue = ti.Value();

				/** Update some sums needed to calculate NC. */
				sff += fixedValue  * fixedValue;
				smm += movingValue * movingValue;
				sfm += fixedValue  * movingValue;
				if ( this->m_SubtractMean ) // is it faster to get rid of the if and calculate sf,sm anyway?
				{
					sf += fixedValue;
					sm += movingValue;
				}

				/** Get the Jacobian. */
				const TransformJacobianType & jacobian = this->m_Transform->GetJacobian( inputPoint ); 

				/** Get the gradient by NearestNeighboorInterpolation:
				 * which is equivalent to round up the point components.*/
				this->m_MovingImage->TransformPhysicalPointToContinuousIndex( transformedPoint, tempIndex );
				for( unsigned int j = 0; j < MovingImageType::ImageDimension; j++ )
				{
					mappedIndex[ j ] = static_cast<long>( vnl_math_rnd( tempIndex[ j ] ) );
				}
				const GradientPixelType gradient = this->GetGradientImage()->GetPixel( mappedIndex );

				/** Calculate the contributions to all parameters. */
				for( unsigned int par = 0; par < ParametersDimension; par++ )
				{
					RealType sumF = NumericTraits< RealType >::Zero;
					RealType sumM = NumericTraits< RealType >::Zero;
					RealType differentialtmp1 = NumericTraits< RealType >::Zero;
					/** Calculate the inner product of the Jacobian and the gradient.
					 * Then multiply with fixedValue or movingValue. */
					for( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
					{
						const RealType differentialtmp2 = jacobian( dim, par ) * gradient[ dim ];
						differentialtmp1 += differentialtmp2;
						sumF += fixedValue  * differentialtmp2;
						sumM += movingValue * differentialtmp2;
					}
					derivativeF[  par ] += sumF;
					derivativeM[  par ] += sumM;
					differential[ par ] += differentialtmp1;
				} // end for loop over the parameters

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

			/** Increase iterator. */
			++ti;
		} // end while loop over fixed image voxels

		/** If SubtractMean, then subtract things from sff, smm, sfm,
		 * derivativeF and derivativeM. */
		if ( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
		{
			sff -= ( sf * sf / this->m_NumberOfPixelsCounted );
			smm -= ( sm * sm / this->m_NumberOfPixelsCounted );
			sfm -= ( sf * sm / this->m_NumberOfPixelsCounted );

			for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivativeF[ i ] -= sf * differential[ i ] / this->m_NumberOfPixelsCounted;
				derivativeM[ i ] -= sm * differential[ i ] / this->m_NumberOfPixelsCounted;
			}
		}

		/** The denominator of the value and the derivative. */
		const RealType denom = -1.0 * vcl_sqrt( sff * smm );

		/** Calculate the value and the derivative. */
		if( this->m_NumberOfPixelsCounted > 0 && denom < 0.00000001 )
		{
			value = sfm / denom;
			for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivative[ i ] = ( derivativeF[ i ] - ( sfm / smm ) * derivativeM[ i ] ) / denom;
			}
		}
		else
		{
			value = NumericTraits< MeasureType >::Zero;
			for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivative[ i ] = NumericTraits< MeasureType >::Zero;
			}
		}

	} // end GetValueAndDerivativeUsingSomePixels


} // end namespace itk


#endif // end #ifndef _itkNormalizedCorrelationImageToImageMetric2_txx

