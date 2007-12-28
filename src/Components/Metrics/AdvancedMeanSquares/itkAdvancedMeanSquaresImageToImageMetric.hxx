#ifndef _itkAdvancedMeanSquaresImageToImageMetric_txx
#define _itkAdvancedMeanSquaresImageToImageMetric_txx

#include "itkAdvancedMeanSquaresImageToImageMetric.h"

namespace itk
{

	/**
	* ******************* Constructor *******************
	*/

	template <class TFixedImage, class TMovingImage> 
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::AdvancedMeanSquaresImageToImageMetric()
	{
    this->SetUseImageSampler( true );
    this->SetUseFixedImageLimiter( false );
    this->SetUseMovingImageLimiter( false );

    this->m_UseNormalization = false;
    this->m_NormalizationFactor = 1.0;

	} // end constructor


  /**
	 * ********************* Initialize ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::Initialize(void) throw ( ExceptionObject )
  {
    /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();

    if ( this->GetUseNormalization() )
    {
      /** Try to guess a normalization factor. */
      this->ComputeFixedImageExtrema(
        this->GetFixedImage(),
        this->GetFixedImageRegion() );

      this->ComputeMovingImageExtrema(
        this->GetMovingImage(),
        this->GetMovingImage()->GetBufferedRegion() );

      const double diff1 = this->m_FixedImageTrueMax - this->m_MovingImageTrueMin;
      const double diff2 = this->m_MovingImageTrueMax - this->m_FixedImageTrueMin;
      const double maxdiff = vnl_math_max( diff1, diff2 ); 

      /** We guess that maxdiff/10 is the maximum average difference 
       * that will be observed.
       * \todo We may involve the standard derivation of the image into
       * this estimate.  */
      this->m_NormalizationFactor = 100.0 / maxdiff / maxdiff;
      
    }
    else
    {
      this->m_NormalizationFactor = 1.0;
    }
   
  } // end Initialize


	/**
	 * ******************* PrintSelf *******************
	 */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::PrintSelf(std::ostream& os, Indent indent) const
	{
		Superclass::PrintSelf( os, indent );

	} // end PrintSelf


  /**
	 * *************** EvaluateTransformJacobianInnerProduct ****************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::EvaluateTransformJacobianInnerProduct( 
		const TransformJacobianType & jacobian, 
		const MovingImageDerivativeType & movingImageDerivative,
    DerivativeType & imageJacobian ) const
	{
    typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
    typedef typename DerivativeType::iterator              DerivativeIteratorType;
    JacobianIteratorType jac = jacobian.begin();
    imageJacobian.Fill( 0.0 );
    const unsigned int sizeImageJacobian = imageJacobian.GetSize();
    for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
    {
      const double imDeriv = movingImageDerivative[ dim ];
      DerivativeIteratorType imjac = imageJacobian.begin();
            
      for ( unsigned int mu = 0; mu < sizeImageJacobian; mu++ )
      {
        (*imjac) += (*jac) * imDeriv;
        ++imjac;
        ++jac;
      }
    }
	} // end EvaluateTransformJacobianInnerProduct


	/**
	 * ******************* GetValue *******************
	 */

	template <class TFixedImage, class TMovingImage> 
		typename AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::GetValue( const TransformParametersType & parameters ) const
	{
		itkDebugMacro( "GetValue( " << parameters << " ) " );
		
    /** Initialize some variables */
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

		/** Loop over the fixed image samples to calculate the mean squares. */
    for ( fiter = fbegin; fiter != fend; ++fiter )
		{
	    /** Read fixed coordinates and initialize some variables. */
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
      RealType movingImageValue; 
      MovingImagePointType mappedPoint;
                  
      /** Transform point and check if it is inside the bspline support region. */
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

      /** Check if point is inside mask. */
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
      
      if ( sampleOk )
      {
        this->m_NumberOfPixelsCounted++; 
      
        /** Get the fixed image value. */
        const RealType & fixedImageValue = static_cast<double>( (*fiter).Value().m_ImageValue );

				/** The difference squared. */
				const RealType diff = movingImageValue - fixedImageValue; 
				measure += diff * diff;
        
			} // end if sampleOk

		} // end for loop over the image sample container

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples(
      sampleContainer->Size(), this->m_NumberOfPixelsCounted );
    
    /** Update measure value. */
	  measure *= this->m_NormalizationFactor / 
      static_cast<double>( this->m_NumberOfPixelsCounted );

		/** Return the mean squares measure value. */
		return measure;

	} // end GetValue
	

	/**
	 * ******************* GetDerivative *******************
	 */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
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
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::GetValueAndDerivative( const TransformParametersType & parameters, 
		MeasureType & value, DerivativeType & derivative ) const
	{
		itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

    typedef typename DerivativeType::ValueType        DerivativeValueType;
    typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

    /** Initialize some variables. */
    this->m_NumberOfPixelsCounted = 0;
    MeasureType measure = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType( this->m_NumberOfParameters );
		derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

    /** Arrays that store dM(x)/dmu and dMask(x)/dmu. */
    DerivativeType imageJacobian( this->m_NonZeroJacobianIndices.GetSize() );
 
		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );
				
    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();
		
		/** Loop over the fixed image to calculate the mean squares. */
		for ( fiter = fbegin; fiter != fend; ++fiter )
		{
      /** Read fixed coordinates and initialize some variables. */
      const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
      RealType movingImageValue; 
      MovingImagePointType mappedPoint;
      MovingImageDerivativeType movingImageDerivative;
            
      /** Transform point and check if it is inside the bspline support region. */
      bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint);
      
      /** Check if point is inside mask. */
      if ( sampleOk ) 
      {
        sampleOk = this->IsInsideMovingMask( mappedPoint );        
      }
    
      /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
       * the point is inside the moving image buffer */
      if ( sampleOk )
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative( 
          mappedPoint, movingImageValue, &movingImageDerivative );
      }
            
      if ( sampleOk )
      {
        this->m_NumberOfPixelsCounted++; 
        
        /** Get the fixed image value. */
        const RealType & fixedImageValue = static_cast<RealType>( (*fiter).Value().m_ImageValue );

        /** Get the TransformJacobian dT/dmu. */
        const TransformJacobianType & jacobian = 
          this->EvaluateTransformJacobian( fixedPoint );
        
        /** Compute the innerproducts (dM/dx)^T (dT/dmu) and (dMask/dx)^T (dT/dmu). */
        this->EvaluateTransformJacobianInnerProduct( 
          jacobian, movingImageDerivative, imageJacobian );

        /** Compute this pixel's contribution to the measure and derivatives. */
        this->UpdateValueAndDerivativeTerms( 
          fixedImageValue, movingImageValue,
          imageJacobian,
          measure, derivative );

			} // end if sampleOk

		} // end for loop over the image sample container

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples(
      sampleContainer->Size(), this->m_NumberOfPixelsCounted );
       
    /** Compute the measure value and derivative. */
    const double normal_sum = this->m_NormalizationFactor / 
      static_cast<double>( this->m_NumberOfPixelsCounted );
    measure *= normal_sum;
    derivative *= normal_sum;
   
		/** The return value. */
		value = measure;

	} // end GetValueAndDerivative


  /**
	 * *************** UpdateValueAndDerivativeTerms ***************************
	 */

	template < class TFixedImage, class TMovingImage >
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::UpdateValueAndDerivativeTerms( 
    const RealType fixedImageValue,
    const RealType movingImageValue,
    const DerivativeType & imageJacobian,
    MeasureType & measure,
    DerivativeType & deriv ) const
  {
    typedef typename DerivativeType::ValueType        DerivativeValueType;

    /** The difference squared. */
		const RealType diff = movingImageValue - fixedImageValue; 
    const RealType diffdiff = diff * diff;
		measure += diffdiff;
        	  
		/** Calculate the contributions to the derivatives with respect to each parameter. */
    const RealType diff_2 = diff * 2.0;
    if ( this->m_NonZeroJacobianIndices.GetSize() == this->m_NumberOfParameters )
		{
      /** Loop over all jacobians. */
      typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
      typename DerivativeType::iterator derivit = deriv.begin();
      for ( unsigned int mu = 0; mu < this->m_NumberOfParameters; ++mu )
      {
        (*derivit) += diff_2 * (*imjacit);        
        ++imjacit;
        ++derivit;   
      }
    }
    else
    {
      /** Only pick the nonzero jacobians. */
      for ( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
      {
        const unsigned int index = this->m_NonZeroJacobianIndices[ i ];
        deriv[ index ] += diff_2 * imageJacobian[ i ];       
      }
    }
  } // end UpdateValueAndDerivativeTerms


} // end namespace itk


#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_txx

