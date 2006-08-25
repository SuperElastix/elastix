#ifndef _itkAdvancedMeanSquaresImageToImageMetric_txx
#define _itkAdvancedMeanSquaresImageToImageMetric_txx

#include "itkAdvancedMeanSquaresImageToImageMetric.h"
#include "itkImageRegionExclusionIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h"



namespace itk
{

	/**
	* ******************* Constructor *******************
	*/

	template <class TFixedImage, class TMovingImage> 
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::AdvancedMeanSquaresImageToImageMetric()
	{
    //this->ComputeGradientOff();

    this->m_InternalMovingImageMask = 0;
    this->m_MovingImageMaskInterpolator = 
      MovingImageMaskInterpolatorType::New();
    this->m_MovingImageMaskInterpolator->SetSplineOrder(2);

    this->m_BSplineInterpolator = 0;

    this->m_UseDifferentiableOverlap = true;
    
    
	} // end constructor


  /**
	 * ********************* Initialize ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::Initialize(void) throw ( ExceptionObject )
  {

    /** Check if the interpolator can be cast to a bspline interpolator */
    if ( this->m_Interpolator.IsNotNull() )
    {
      BSplineInterpolatorType * testptr = 
        dynamic_cast< BSplineInterpolatorType * >(
        this->m_Interpolator.GetPointer() );
      if ( testptr )
      {
        this->m_BSplineInterpolator = testptr;
        this->ComputeGradientOff();
      }
      else
      {
        this->m_BSplineInterpolator = 0;
        this->ComputeGradientOn();
      }
    }
    else
    {
      /** An exception will be thrown anyway in the superclass:Initialize()
       * but just to be sure: */
      this->m_BSplineInterpolator = 0;
      this->ComputeGradientOn();
    }

    /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();
    
    /** Initialize the internal moving image mask */
    this->InitializeInternalMasks();
    
  } // end Initialize


  /**
	 * ********************* InitializeInternalMasks *********************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::InitializeInternalMasks(void)
  {
    /** Initialize the internal moving image mask */

    typedef typename MovingImageType::PointType                  MovingOriginType;
    typedef typename MovingImageType::SizeType                   MovingSizeType;
    typedef typename MovingImageType::IndexType                  MovingIndexType;
    typedef typename MovingImageType::RegionType                 MovingRegionType;
    typedef itk::ImageRegionExclusionIteratorWithIndex<
      InternalMovingImageMaskType>                               MovingEdgeIteratorType;
    typedef itk::ImageRegionIteratorWithIndex<
      InternalMovingImageMaskType>                               MovingIteratorType;
    typedef itk::BinaryBallStructuringElement<
      InternalMaskPixelType, MovingImageDimension >              ErosionKernelType;
    typedef itk::BinaryErodeImageFilter<
      InternalMovingImageMaskType,
      InternalMovingImageMaskType, 
      ErosionKernelType >                                        ErodeImageFilterType;
    
    /** Check if the user wants to use a differentiable overlap */
    if ( ! this->m_UseDifferentiableOverlap )
    {
      this->m_InternalMovingImageMask = 0;
      return;
    }

    /** Prepare the internal mask image */
    this->m_InternalMovingImageMask = InternalMovingImageMaskType::New();
    this->m_InternalMovingImageMask->SetRegions( 
      this->GetMovingImage()->GetLargestPossibleRegion() );
    this->m_InternalMovingImageMask->Allocate();
    this->m_InternalMovingImageMask->SetOrigin(
      this->GetMovingImage()->GetOrigin() );
    this->m_InternalMovingImageMask->SetSpacing(
      this->GetMovingImage()->GetSpacing() );

    /** Radius to erode masks */
    const unsigned int radius = 2;

    /** Determine inner region */
    MovingRegionType innerRegion =
      this->m_InternalMovingImageMask->GetLargestPossibleRegion();
    for (unsigned int i=0; i < MovingImageDimension; ++i)
    {
      if ( innerRegion.GetSize()[i] >= 2*radius )
      {
        /** region is large enough to crop; adjust size and index */
        innerRegion.SetSize( i, innerRegion.GetSize()[i] - 2*radius );
        innerRegion.SetIndex( i, innerRegion.GetIndex()[i] + radius );
      }
      else
      {
         innerRegion.SetSize( i, 0);
      }
    }
      
    if ( this->GetMovingImageMask() == 0 )
    {
      /** Fill the internal moving mask with ones */
      this->m_InternalMovingImageMask->FillBuffer(
        itk::NumericTraits<InternalMaskPixelType>::One );
    
      MovingEdgeIteratorType edgeIterator( this->m_InternalMovingImageMask, 
        this->m_InternalMovingImageMask->GetLargestPossibleRegion() );
      edgeIterator.SetExclusionRegion( innerRegion );
      
      /** Set the edges to zero */
      for( edgeIterator.GoToBegin(); ! edgeIterator.IsAtEnd(); ++ edgeIterator )
      {
        edgeIterator.Value() = itk::NumericTraits<InternalMaskPixelType>::Zero;
      }
      
    } // end if no moving mask
    else
    {
      /** Fill the internal moving mask with zeros */
      this->m_InternalMovingImageMask->FillBuffer(
        itk::NumericTraits<InternalMaskPixelType>::Zero );

      MovingIteratorType iterator( this->m_InternalMovingImageMask, innerRegion);
      OutputPointType point;

      /** Set the pixel 1 if inside the mask and to 0 if outside */
      for( iterator.GoToBegin(); ! iterator.IsAtEnd(); ++ iterator )
      {
        const MovingIndexType & index = iterator.GetIndex();
        this->m_InternalMovingImageMask->TransformIndexToPhysicalPoint(index, point);
        iterator.Value() = static_cast<InternalMaskPixelType>(
          this->m_MovingImageMask->IsInside(point) );
      }

      /** Erode it with a radius of 2 */
      typename InternalMovingImageMaskType::Pointer tempImage = 0;
      ErosionKernelType kernel;
      kernel.SetRadius(radius);
      kernel.CreateStructuringElement();
      typename ErodeImageFilterType::Pointer eroder = ErodeImageFilterType::New();
      eroder->SetKernel( kernel );
      eroder->SetForegroundValue( itk::NumericTraits< InternalMaskPixelType >::One  );
	    eroder->SetBackgroundValue( itk::NumericTraits< InternalMaskPixelType >::Zero );
      eroder->SetInput( this->m_InternalMovingImageMask );
      eroder->Update();
      tempImage = eroder->GetOutput();
      tempImage->DisconnectPipeline();
      this->m_InternalMovingImageMask = tempImage;
        
    }
        
    /** Set the internal mask into the interpolator */
    this->m_MovingImageMaskInterpolator->SetInputImage( this->m_InternalMovingImageMask );

    //test: 
    /**OutputPointType midden;
    OutputPointType neterin;
    OutputPointType neteruit;
    OutputPointType neternietuit;
    midden.Fill(128.0);
    neterin.Fill(4.0);
    neteruit.Fill(253.0);
    neternietuit.Fill(251.0);

    double tempje;
    MovingImageMaskDerivativeType deriv;
    this->EvaluateMovingMaskValueAndDerivative(midden, tempje, deriv);
    std::cout << "midden: " << tempje << " " << deriv << std::endl;
    this->EvaluateMovingMaskValueAndDerivative(neterin, tempje, deriv);
    std::cout << "neterin: " << tempje << " " << deriv << std::endl;
    this->EvaluateMovingMaskValueAndDerivative(neteruit, tempje, deriv);
    std::cout << "neteruit: " << tempje << " " << deriv << std::endl;
    this->EvaluateMovingMaskValueAndDerivative(neternietuit, tempje, deriv);
    std::cout << "neternietuit: " << tempje << " " << deriv << std::endl;
*/
 
  } // end InitializeInternalMasks


  /**
	 * **************** EvaluateMovingMaskValue *******************
   * Estimate value of internal moving mask 
   */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::EvaluateMovingMaskValue(
      const OutputPointType & point,
      double & value) const
  {
    if ( this->m_UseDifferentiableOverlap )
    {
      /** NB: a spelling error in the itkImageFunction class! Continous... */
      MovingImageContinuousIndexType cindex;
      this->m_MovingImageMaskInterpolator->ConvertPointToContinousIndex( point, cindex);
  
      /** Compute the value of the mask */
      if ( this->m_MovingImageMaskInterpolator->IsInsideBuffer( cindex ) )
      {
        value = static_cast<double>(
          this->m_MovingImageMaskInterpolator->EvaluateAtContinuousIndex(cindex) );
      }
      else
      {
        value = 0.0;
      }
    }
    else
    {
       /** Use the original mask */
      if ( this->m_MovingImageMask.IsNotNull() )
      {
        value = static_cast<double>(
          static_cast<unsigned char>( this->m_MovingImageMask->IsInside( point ) ) );
      }
      else
      {
        value = 1.0;
      }
    }
  
  } // end EvaluateMovingMaskValue


	/**
	 * **************** EvaluateMovingMaskValueAndDerivative *******************
   * Estimate value and spatial derivative of internal moving mask 
   */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
    ::EvaluateMovingMaskValueAndDerivative(
      const OutputPointType & point,
      double & value,
      MovingImageMaskDerivativeType & derivative) const
  {
    typedef typename MovingImageMaskDerivativeType::ValueType DerivativeValueType;
       
    /** Compute the value and derivative of the mask */

    if ( this->m_UseDifferentiableOverlap )
    {
      /** NB: a spelling error in the itkImageFunction class! Continous... */
      MovingImageContinuousIndexType cindex;
      this->m_MovingImageMaskInterpolator->ConvertPointToContinousIndex( point, cindex);

      if ( this->m_MovingImageMaskInterpolator->IsInsideBuffer( cindex ) )
      {
        value = static_cast<double>(
          this->m_MovingImageMaskInterpolator->EvaluateAtContinuousIndex(cindex) );
        derivative = 
          this->m_MovingImageMaskInterpolator->EvaluateDerivativeAtContinuousIndex(cindex);
      }
      else
      {
        value = 0.0;
        derivative.Fill( itk::NumericTraits<DerivativeValueType>::Zero );
      }
    }
    else
    {
      /** Just ignore the derivative of the mask */
      if ( this->m_MovingImageMask.IsNotNull() )
      {
        value = static_cast<double>(
          static_cast<unsigned char>( this->m_MovingImageMask->IsInside( point ) ) );
      }
      else
      {
        value = 1.0;
      }
      derivative.Fill( itk::NumericTraits<DerivativeValueType>::Zero );
    }
      
   
  } // end EvaluateMovingMaskValueAndDerivative


	/**
	 * ************** EvaluateMovingImageValueAndDerivative *****************
	 */

	template < class TFixedImage, class TMovingImage> 
		bool
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::EvaluateMovingImageValueAndDerivative(
      const OutputPointType & point,
      RealType & value,
      GradientPixelType & derivative) const
  {

    /** NB: a spelling error in the itkImageFunction class! Continous... */
    MovingImageContinuousIndexType cindex;
    this->m_Interpolator->ConvertPointToContinousIndex( point, cindex);
    
    /** Compute the value and derivative of the mask */
    if ( this->m_Interpolator->IsInsideBuffer( cindex ) )
    {
      /** Evaluate the value */
      value  = this->m_Interpolator->EvaluateAtContinuousIndex( cindex );
  
      /** Evaluate the derivative */
      if ( this->m_BSplineInterpolator.IsNotNull() )
      {
        /** Use the BSplineInterpolator */
        derivative = 
          this->m_BSplineInterpolator->EvaluateDerivativeAtContinuousIndex( cindex );
      }
      else
      {
        /** Use the precomputed gradient image */
        /** Get the gradient by NearestNeighboorInterpolation:
			  * which is equivalent to round up the point components.
        * The itk::ImageFunction provides a function to do this.
        */
		    typename MovingImageType::IndexType	mappedIndex;
			  this->m_Interpolator->ConvertContinuousIndexToNearestIndex( cindex, mappedIndex );
			  derivative = this->GetGradientImage()->GetPixel( mappedIndex );
      }

      /** the point was inside the buffer; value and derivative are valid. */
      return true;
    }
    else
    {
      /** do not change the value or derivative; just return false */
      return false;
    }

  } // end EvaluateMovingImageValueAndDerivative


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
	 * ******************* GetValue *******************
	 */

	template <class TFixedImage, class TMovingImage> 
		typename AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
		AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
		::GetValue( const TransformParametersType & parameters ) const
	{
		itkDebugMacro( "GetValue( " << parameters << " ) " );

		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );

		this->m_NumberOfPixelsCounted = 0;
    double normalizationFactor = 0.0;

		/** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator iter;
    typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

		MeasureType measure = NumericTraits< MeasureType >::Zero;

		/** Loop over the fixed image samples to calculate the mean squares. */
    for ( iter = begin; iter != end; ++iter )
		{
			/** Get the current inputpoint. */
      const InputPointType & inputPoint = (*iter).Value().m_ImageCoordinates;

			/** Transform the inputpoint to get the transformed point. */
			const OutputPointType transformedPoint = this->m_Transform->TransformPoint( inputPoint );

      double movingMaskValue = 0.0;
      this->EvaluateMovingMaskValue( transformedPoint, movingMaskValue );
      const double smallNumber1 = 1e-10;

			/** Inside the moving image mask? */
			if ( movingMaskValue < smallNumber1 )
			{
        /** no? then go to next sample */
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
				measure += movingMaskValue * diff * diff;
        normalizationFactor += movingMaskValue;

				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

		} // end for loop over the image sample container

    /** Calculate the value */
    const double smallNumber2 = 1e-10;
		if ( this->m_NumberOfPixelsCounted > 0 &&  normalizationFactor > smallNumber2 )
		{
			measure /= normalizationFactor;
    }
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
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
 
		/** Make sure the transform parameters are up to date. */
		this->SetTransformParameters( parameters );
		const unsigned int ParametersDimension = this->GetNumberOfParameters();
		
		this->m_NumberOfPixelsCounted = 0;
    double normalizationFactor = 0.0;

    /** Update the imageSampler and get a handle to the sample container. */
    this->GetImageSampler()->Update();
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator iter;
    typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

		/** Some typedefs. */
		typedef typename OutputPointType::CoordRepType	CoordRepType;
		
		MeasureType measure = NumericTraits< MeasureType >::Zero;
		
		derivative = DerivativeType( ParametersDimension );
		derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

    DerivativeType ddendmu( ParametersDimension );
    DerivativeType dnumdmu( ParametersDimension );
    ddendmu.Fill( NumericTraits< DerivativeValueType >::Zero );
    dnumdmu.Fill( NumericTraits< DerivativeValueType >::Zero );

		/** Loop over the fixed image to calculate the mean squares. */
		for ( iter = begin; iter != end; ++iter )
		{
			/** Get the current inputpoint. */
      const InputPointType & inputPoint = (*iter).Value().m_ImageCoordinates;

			/** Transform the inputpoint to get the transformed point. */
			const OutputPointType transformedPoint = this->m_Transform->TransformPoint( inputPoint );

      double movingMaskValue = 0.0;
      MovingImageMaskDerivativeType movingMaskDerivative; 
      this->EvaluateMovingMaskValueAndDerivative(
        transformedPoint, movingMaskValue, movingMaskDerivative);
      const double movingMaskDerivativeMagnitude = movingMaskDerivative.GetNorm();
      const double smallNumber1 = 1e-10;

			/** Inside the moving image mask? */
			if ( movingMaskValue < smallNumber1 && movingMaskDerivativeMagnitude < smallNumber1)
			{
				continue;
			}

      RealType movingValue;
      GradientPixelType gradient;

			/** In this if-block the actual calculation of mean squares is done. */
      /** Try to get the movingValue = m(x+u(x)) and the derivative at that point;
       * returns false if the point is not inside the image buffer. */
			if ( this->EvaluateMovingImageValueAndDerivative(
          transformedPoint, movingValue, gradient) )
			{
				/** Get the fixedValue = f(x) */
        const RealType & fixedValue = (*iter).Value().m_ImageValue;
 
				/** Get the Jacobian. */
				const TransformJacobianType & jacobian =
					this->m_Transform->GetJacobian( inputPoint ); 

				/** The difference squared. */
				const RealType diff = movingValue - fixedValue; 
        const RealType diffdiff = diff * diff;
				measure += movingMaskValue * diffdiff;
        normalizationFactor += movingMaskValue;
	      
				/** Calculate the contributions to the derivatives with respect to each parameter. */
        const RealType movmask_diff_2 = movingMaskValue * diff * 2.0;
				for ( unsigned int par = 0; par < ParametersDimension; par++ )
				{
          /** compute inproduct of image gradient and transform jacobian */
					RealType grad_jac = NumericTraits< RealType >::Zero;
          RealType maskderiv_jac = NumericTraits< RealType >::Zero;
          for( unsigned int dim = 0; dim < MovingImageDimension; dim++ )
					{
            const TransformJacobianValueType & jacdimpar = jacobian( dim, par );
						grad_jac += static_cast<RealType>(
              gradient[ dim ] * jacdimpar );
            maskderiv_jac += static_cast<RealType> (
              movingMaskDerivative[ dim ] * jacdimpar );
					}
          dnumdmu[ par ] += movmask_diff_2 * grad_jac + diffdiff * maskderiv_jac;
          ddendmu[ par ] += maskderiv_jac;
				}
        
				/** Update the NumberOfPixelsCounted. */
				this->m_NumberOfPixelsCounted++;

			} // end if IsInsideBuffer()

		} // end for loop over the image sample container

		/** Calculate the value and the derivative. */
    const double smallNumber2 = 1e-10;
		if ( this->m_NumberOfPixelsCounted > 0 &&  normalizationFactor > smallNumber2 )
		{
			measure /= normalizationFactor;
      MeasureType measure_N = measure / normalizationFactor;
      for( unsigned int i = 0; i < ParametersDimension; i++ )
			{
				derivative[ i ] = dnumdmu[i] / normalizationFactor - ddendmu[i] * measure_N ;
			}
		}
		else
		{
			measure = NumericTraits< MeasureType >::Zero;
			derivative.Fill( NumericTraits<ITK_TYPENAME DerivativeType::ValueType>::Zero );
			itkExceptionMacro( << "All the points mapped outside the moving image" );
		}

		/** The return value. */
		value = measure;

	} // end GetValueAndDerivative


} // end namespace itk


#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_txx

