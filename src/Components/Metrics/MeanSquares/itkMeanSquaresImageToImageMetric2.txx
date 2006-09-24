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


namespace itk
{

	/**
	* ******************* Constructor *******************
	*/

	template <class TFixedImage, class TMovingImage> 
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::MeanSquaresImageToImageMetric2()
	{
    this->SetUseImageSampler(true);
    this->SetComputeGradient(true);
    
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

	} // end PrintSelf


	/**
	 * ******************* GetValue *******************
	 */

	template <class TFixedImage, class TMovingImage> 
		typename MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>::MeasureType
		MeanSquaresImageToImageMetric2<TFixedImage,TMovingImage>
		::GetValue( const TransformParametersType & parameters ) const
	{
		itkDebugMacro( "GetValue( " << parameters << " ) " );

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

		this->m_NumberOfPixelsCounted = 0;

		/** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator iter;
    typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

		/** Create variables to store intermediate results. */
		InputPointType  inputPoint;
		OutputPointType transformedPoint;

		MeasureType measure = NumericTraits< MeasureType >::Zero;

		/** Loop over the fixed image samples to calculate the mean squares. */
    for ( iter = begin; iter != end; ++iter )
		{
			/** Get the current inputpoint. */
      inputPoint = (*iter).Value().m_ImageCoordinates;

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				continue;
			}

			/** In this if-statement the actual calculation of mean squares is done. */
			if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
        const RealType & fixedValue = (*iter).Value().m_ImageValue;

				/** The difference squared. */
				const RealType diff = movingValue - fixedValue; 
				measure += diff * diff;

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

		} // end for loop over the image sample container

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

	} // end GetValue
	

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
		itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

		/** Some sanity checks. */
		if ( !this->GetGradientImage() )
		{
			itkExceptionMacro( << "The gradient image is null, maybe you forgot to call Initialize()" );
		}

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );
		const unsigned int ParametersDimension = this->GetNumberOfParameters();
		
		this->m_NumberOfPixelsCounted = 0;

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator iter;
    typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

		/** Some typedefs. */
		typedef typename OutputPointType::CoordRepType	CoordRepType;
		typedef ContinuousIndex<CoordRepType,
			MovingImageType::ImageDimension>							MovingImageContinuousIndexType;

		/** Create variables to store intermediate results. */
		InputPointType	                    inputPoint;
		OutputPointType                     transformedPoint;
		MovingImageContinuousIndexType			tempIndex;
		typename MovingImageType::IndexType	mappedIndex;

		MeasureType measure = NumericTraits< MeasureType >::Zero;
		
		derivative = DerivativeType( ParametersDimension );
		derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );

		/** Loop over the fixed image to calculate the mean squares. */
		for ( iter = begin; iter != end; ++iter )
		{
			/** Get the current inputpoint. */
      inputPoint = (*iter).Value().m_ImageCoordinates;

			/** Transform the inputpoint to get the transformed point. */
			transformedPoint = this->m_Transform->TransformPoint( inputPoint );

			/** Inside the moving image mask? */
			if ( this->m_MovingImageMask && !this->m_MovingImageMask->IsInside( transformedPoint ) )
			{
				continue;
			}

			/** In this if-statement the actual calculation of mean squares is done. */
			if ( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
			{
				/** Get the fixedValue = f(x) and the movingValue = m(x+u(x)). */
				const RealType movingValue  = this->m_Interpolator->Evaluate( transformedPoint );
        const RealType & fixedValue = (*iter).Value().m_ImageValue;

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

		} // end for loop over the image sample container

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

	} // end GetValueAndDerivative


} // end namespace itk


#endif // end #ifndef _itkMeanSquaresImageToImageMetric2_txx

