/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkAdvancedKappaStatisticImageToImageMetric_txx
#define _itkAdvancedKappaStatisticImageToImageMetric_txx

#include "itkAdvancedKappaStatisticImageToImageMetric.h"

namespace itk
{

	/**
	* ******************* Constructor *******************
	*/

	template <class TFixedImage, class TMovingImage> 
		AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
		::AdvancedKappaStatisticImageToImageMetric()
	{
    this->SetComputeGradient( true );
    this->SetUseImageSampler( true );
    this->SetUseFixedImageLimiter( false );
    this->SetUseMovingImageLimiter( false );

    this->m_ForegroundValue = 1.0;
    this->m_Complement = false;
    this->m_ForegroundIsNonZero = false;

	} // end constructor


	/**
	 * ******************* PrintSelf *******************
	 */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
		::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf( os, indent );
    os << indent << "Complement: "      << ( this->m_Complement ? "On" : "Off" ) << std::endl; 
    os << indent << "ForegroundValue: " << this->m_ForegroundValue << std::endl;
	} // end PrintSelf()


  /**
	 * *************** EvaluateMovingImageAndTransformJacobianInnerProduct ****************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
    ::EvaluateMovingImageAndTransformJacobianInnerProduct(
		const TransformJacobianType & jacobian, 
		const MovingImageDerivativeType & movingImageDerivative,
    DerivativeType & innerProduct ) const
	{
    typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
    typedef typename DerivativeType::iterator              DerivativeIteratorType;
    JacobianIteratorType jac = jacobian.begin();
    innerProduct.Fill( 0.0 );
    const unsigned int sizeInnerProduct = innerProduct.GetSize();
    for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
    {
      const double imDeriv = movingImageDerivative[ dim ];
      DerivativeIteratorType it = innerProduct.begin();
      for ( unsigned int mu = 0; mu < sizeInnerProduct; mu++ )
      {
        (*it) += (*jac) * imDeriv;
        ++it; ++jac;
      }
    }

	} // end EvaluateMovingImageAndTransformJacobianInnerProduct()


	/**
	 * ******************* GetValue *******************
	 */

	template <class TFixedImage, class TMovingImage> 
		typename AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
		AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
		::GetValue( const TransformParametersType & parameters ) const
	{
		itkDebugMacro( "GetValue( " << parameters << " ) " );
		
    /** Initialize some variables. */
		this->m_NumberOfPixelsCounted = 0;
    MeasureType measure = NumericTraits< MeasureType >::Zero;

    /** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

		/** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    /** Some variables. */
    RealType movingImageValue;
    MovingImagePointType mappedPoint;
    MeasureType fixedForegroundArea  = NumericTraits< MeasureType >::Zero;
    MeasureType movingForegroundArea = NumericTraits< MeasureType >::Zero;
    MeasureType intersection         = NumericTraits< MeasureType >::Zero;

		/** Loop over the fixed image samples to calculate the kappa statistic. */
    for ( fiter = fbegin; fiter != fend; ++fiter )
		{
	    /** Read fixed coordinates and initialize some variables. */
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
      
      /** Transform point and check if it is inside the bspline support region. */
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

      /** Check if point is inside moving mask. */
      if ( sampleOk )
      {
        sampleOk = this->IsInsideMovingMask( mappedPoint );        
      }

      /** Compute the moving image value and check if the point is
       * inside the moving image buffer. */
      if ( sampleOk )
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, 0 );
      }
      
      /** Do the actual calculation of the metric value. */
      if ( sampleOk )
      {
        this->m_NumberOfPixelsCounted++; 

        /** Get the fixed image value. */
        const RealType & fixedImageValue = static_cast<double>( (*fiter).Value().m_ImageValue );

        /** Update the intermediate values. */
        if ( !this->m_ForegroundIsNonZero )
        {
          if ( fixedImageValue == this->m_ForegroundValue ) fixedForegroundArea++;
          if ( movingImageValue == this->m_ForegroundValue ) movingForegroundArea++;
          if ( fixedImageValue == this->m_ForegroundValue
            && movingImageValue == this->m_ForegroundValue ) intersection++;
        }
        else
        {
          if ( vnl_math_abs( fixedImageValue ) > 0.1 ) fixedForegroundArea++;
          if ( vnl_math_abs( movingImageValue ) > 0.1 ) movingForegroundArea++;
          if ( vnl_math_abs( fixedImageValue ) > 0.1
            && vnl_math_abs( movingImageValue ) > 0.1 ) intersection++;
        }
      
			} // end if samplOk

		} // end for loop over the image sample container

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples( sampleContainer->Size(), this->m_NumberOfPixelsCounted);
    
    /** Compute the final metric value. */
    MeasureType areaSum = fixedForegroundArea + movingForegroundArea;
    if ( areaSum < 1e-14 )
    {
      measure = NumericTraits< MeasureType >::Zero;
    }
    else
    {
      measure = 2.0 * intersection / areaSum;
    }
    if ( this->m_Complement )
    { 
      measure = 1.0 - measure;
    }

		/** Return the mean squares measure value. */
		return measure;

	} // end GetValue()
	

	/**
	 * ******************* GetDerivative *******************
	 */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
		::GetDerivative( const TransformParametersType & parameters,
		DerivativeType & derivative ) const
	{
		/** When the derivative is calculated, all information for calculating
		 * the metric value is available. It does not cost anything to calculate
		 * the metric value now. Therefore, we have chosen to only implement the
		 * GetValueAndDerivative(), supplying it with a dummy value variable. */
		MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
		this->GetValueAndDerivative( parameters, dummyvalue, derivative );

	} // end GetDerivative()


	/**
	 * ******************* GetValueAndDerivative *******************
	 */

	template <class TFixedImage, class TMovingImage>
		void
		AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
		::GetValueAndDerivative( const TransformParametersType & parameters, 
		MeasureType & value, DerivativeType & derivative ) const
	{
		itkDebugMacro( "GetValueAndDerivative( " << parameters << " ) " );

    /** Some typedefs. */
    typedef typename DerivativeType::ValueType        DerivativeValueType;
    typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

    /** Initialize some variables. */
    this->m_NumberOfPixelsCounted = 0;
    MeasureType measure = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType( this->m_NumberOfParameters );
		derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

    /** Array that store dM(x)/dmu. */
    DerivativeType imageJacobian( this->m_NonZeroJacobianIndices.GetSize() );
 
		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );
				
    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Some variables. */
    RealType movingImageValue;
    MovingImagePointType mappedPoint;
    MeasureType fixedForegroundArea  = NumericTraits< MeasureType >::Zero;
    MeasureType movingForegroundArea = NumericTraits< MeasureType >::Zero;
    MeasureType intersection         = NumericTraits< MeasureType >::Zero;

    DerivativeType sum1( this->m_NumberOfParameters );
    DerivativeType sum2( this->m_NumberOfParameters );
    sum1.Fill( NumericTraits< DerivativeValueType >::Zero );
    sum2.Fill( NumericTraits< DerivativeValueType >::Zero );

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
		
		/** Loop over the fixed image to calculate the kappa statistic. */
		for ( fiter = fbegin; fiter != fend; ++fiter )
		{
      /** Read fixed coordinates. */
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
 
      /** Transform point and check if it is inside the bspline support region. */
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );
      
      /** Check if point is inside moving mask. */
      if ( sampleOk )
      {
        sampleOk = this->IsInsideMovingMask( mappedPoint );        
      }

      /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
       * the point is inside the moving image buffer. */
      MovingImageDerivativeType movingImageDerivative;
      if ( sampleOk )
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, &movingImageDerivative );
      }
      
      /** Do the actual calculation of the metric value. */
      if ( sampleOk )
      {
        this->m_NumberOfPixelsCounted++; 

        /** Get the fixed image value. */
        const RealType & fixedImageValue = static_cast<RealType>( (*fiter).Value().m_ImageValue );

        /** Get the TransformJacobian dT/dmu. */
        const TransformJacobianType & jacobian = 
          this->EvaluateTransformJacobian( fixedPoint );
        
        /** Compute the innerproducts (dM/dx)^T (dT/dmu). */
        this->EvaluateMovingImageAndTransformJacobianInnerProduct(
          jacobian, movingImageDerivative, imageJacobian );

        /** Compute this pixel's contribution to the measure and derivatives. */
        this->UpdateValueAndDerivativeTerms( 
          fixedImageValue, movingImageValue,
          fixedForegroundArea, movingForegroundArea, intersection,
          imageJacobian,
          sum1, sum2 );

			} // end if sampleOk

		} // end for loop over the image sample container

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples(
      sampleContainer->Size(), this->m_NumberOfPixelsCounted );
       
    /** Compute the final metric value. */
    MeasureType areaSum = fixedForegroundArea + movingForegroundArea;
    if ( areaSum < 1e-14 )
    {
      measure = NumericTraits< MeasureType >::Zero;
    }
    else
    {
      measure = 2.0 * intersection / areaSum;
    }
    if ( this->m_Complement )
    { 
      measure = 1.0 - measure;
    }
    value = measure;

    /** Calculate the derivative. */
    double direction = 1.0;
    if ( this->m_Complement )
    {
      direction = -1.0;
    }
    
    if ( areaSum > 1e-14 )
    {
      for ( unsigned int par = 0; par < this->m_NumberOfParameters; par++ )
      {
        derivative[ par ] = direction * ( areaSum * sum1[ par ] - 2.0 * intersection * sum2[ par ] )
          / ( areaSum * areaSum );
      }
    }

	} // end GetValueAndDerivative()


  /**
	 * *************** UpdateValueAndDerivativeTerms ***************************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
		::UpdateValueAndDerivativeTerms( 
    const RealType fixedImageValue,
    const RealType movingImageValue,
    MeasureType & fixedForegroundArea,
    MeasureType & movingForegroundArea,
    MeasureType & intersection,
    const DerivativeType & imageJacobian,
    DerivativeType & sum1,
    DerivativeType & sum2 ) const
  {
    typedef typename DerivativeType::ValueType        DerivativeValueType;

    /** Update the intermediate values. */
    if ( !this->m_ForegroundIsNonZero )
    {
      if ( fixedImageValue == this->m_ForegroundValue ) fixedForegroundArea++;
      if ( movingImageValue == this->m_ForegroundValue ) movingForegroundArea++;
      if ( fixedImageValue == this->m_ForegroundValue
        && movingImageValue == this->m_ForegroundValue ) intersection++;
    }
    else
    {
      if ( vnl_math_abs( fixedImageValue ) > 0.1 ) fixedForegroundArea++;
      if ( vnl_math_abs( movingImageValue ) > 0.1 ) movingForegroundArea++;
      if ( vnl_math_abs( fixedImageValue ) > 0.1
        && vnl_math_abs( movingImageValue ) > 0.1 ) intersection++;
    }
    
		/** Calculate the contributions to the derivatives with respect to each parameter. */
    if ( this->m_NonZeroJacobianIndices.GetSize() == this->m_NumberOfParameters )
		{
      /** Loop over all jacobians. */
      typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
      typename DerivativeType::iterator sum1it = sum1.begin();
      typename DerivativeType::iterator sum2it = sum2.begin();
      for ( unsigned int mu = 0; mu < this->m_NumberOfParameters; ++mu )
      {
        if ( !this->m_ForegroundIsNonZero )
        {
          if ( fixedImageValue == this->m_ForegroundValue )
          {
            (*sum1it) += 2.0 * (*imjacit);
          }
        }
        else
        {
          if ( vnl_math_abs( fixedImageValue ) > 0.1 )
          {
            (*sum1it) += 2.0 * (*imjacit);
          }
        }
        (*sum2it) += (*imjacit);

        /** Increase iterators. */
        ++imjacit; ++sum1it; ++sum2it;
      }
    }
    else
    {
      /** Only pick the nonzero jacobians. */
      for ( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
      {
        const unsigned int index = this->m_NonZeroJacobianIndices[ i ];
        if ( !this->m_ForegroundIsNonZero )
        {
          if ( fixedImageValue == this->m_ForegroundValue )
          {
            sum1[ index ] += 2.0 * imageJacobian[ i ];
          }
        }
        else
        {
          if ( vnl_math_abs( fixedImageValue ) > 0.1 )
          {
            sum1[ index ] += 2.0 * imageJacobian[ i ];
          }
        }
        sum2[ index ] += imageJacobian[ i ];
      }
    }
  } // end UpdateValueAndDerivativeTerms()


  /**
	 * *************** ComputeGradient ***************************
   *
   * Compute the moving image gradient (dM/dx) and assigns to m_GradientImage.
   * Overrides superclass implementation.
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedKappaStatisticImageToImageMetric<TFixedImage,TMovingImage>
    ::ComputeGradient( void )
  {
    /** Typedefs. */
    typedef itk::ImageRegionIteratorWithIndex< GradientImageType >    GradientIteratorType;
    typedef itk::ImageRegionConstIteratorWithIndex< MovingImageType > MovingIteratorType; 

    /** Create a temporary moving gradient image. */
    typename GradientImageType::Pointer tempGradientImage = GradientImageType::New();
    tempGradientImage->SetRegions( this->m_MovingImage->GetBufferedRegion().GetSize() );
    tempGradientImage->Allocate();

    /** Create and reset iterators. */
    GradientIteratorType git( tempGradientImage, tempGradientImage->GetBufferedRegion() );
    MovingIteratorType mit( this->m_MovingImage, this->m_MovingImage->GetBufferedRegion() );
    git.GoToBegin();
    mit.GoToBegin();

    /** Some temporary variables. */
    typename MovingImageType::IndexType minusIndex, plusIndex, currIndex;
    typename GradientImageType::PixelType tempGradPixel;
    typename MovingImageType::SizeType movingSize
      = this->m_MovingImage->GetBufferedRegion().GetSize();

    /** Loop over the images. */
    while ( !mit.IsAtEnd() )
    {
      /** Get the current index. */
      currIndex = mit.GetIndex();
      minusIndex = currIndex; plusIndex = currIndex;
      for ( unsigned int i = 0; i < MovingImageDimension; i++ )
      {
        /** Check for being on the edge of the moving image. */
        if ( currIndex[ i ] == 0 
          || currIndex[ i ] == movingSize[ i ] - 1 )
        {
          tempGradPixel[ i ] = 0.0;
        }
        else
        {
          /** Get the left, center and right values. */
          minusIndex[ i ] = currIndex[ i ] - 1;
          plusIndex[ i ] = currIndex[ i ] + 1;
          double minusVal = static_cast<double>( this->m_MovingImage->GetPixel( minusIndex ) );
          double val      = static_cast<double>( this->m_MovingImage->GetPixel( currIndex ) );
          double plusVal  = static_cast<double>( this->m_MovingImage->GetPixel( plusIndex ) );

          /** Calculate the gradient. */
          // \todo also use the ForegroundIsNonZero boolean.
          if ( minusVal != this->m_ForegroundValue && plusVal == this->m_ForegroundValue )
          {
            tempGradPixel[ i ] = 1.0;
          }
          else if ( minusVal == this->m_ForegroundValue 
            && plusVal != this->m_ForegroundValue )
          {
            tempGradPixel[ i ] = -1.0;
          }
          else
          {
            tempGradPixel[ i ] = 0.0;
          }
        }

        /** Reset indices. */
        minusIndex = currIndex; plusIndex  = currIndex;

      } // end for loop

      /** Set the gradient value and increase iterators. */
      git.Set( tempGradPixel );
      ++git; ++mit;

    } // end while loop

    this->m_GradientImage = tempGradientImage;

  } // end ComputeGradient()



} // end namespace itk


#endif // end #ifndef _itkAdvancedKappaStatisticImageToImageMetric_txx

