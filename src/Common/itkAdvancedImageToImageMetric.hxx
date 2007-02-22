#ifndef _itkAdvancedImageToImageMetric_txx
#define _itkAdvancedImageToImageMetric_txx

#include "itkAdvancedImageToImageMetric.h"

#include "itkImageRegionExclusionIteratorWithIndex.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h"


namespace itk
{

  /**
	 * ********************* Constructor ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::AdvancedImageToImageMetric()
  {
    this->SetComputeGradient(false); // don't use the default gradient

    this->m_ImageSampler = 0;
    this->m_UseImageSampler = false;
    this->m_RequiredRatioOfValidSamples = 0.25;

    this->m_BSplineInterpolator = 0;
		this->m_InterpolatorIsBSpline = false;
    this->m_ForwardDifferenceFilter = 0;
    
 		this->m_BSplineTransform = 0;
		this->m_BSplineCombinationTransform = 0;
    this->m_NumBSplineParametersPerDim = 0;
		this->m_NumBSplineWeights = 0;
		this->m_NumberOfParameters = 0;
    this->m_TransformIsBSpline = false;
		this->m_TransformIsBSplineCombination = false;
    
    const unsigned int defaultMaskInterpolationOrder = 2;
    this->m_InternalMovingImageMask = 0;
    this->m_MovingImageMaskInterpolator = 
      MovingImageMaskInterpolatorType::New();
    this->m_MovingImageMaskInterpolator->SetSplineOrder( defaultMaskInterpolationOrder );
    this->m_UseDifferentiableOverlap = false;

    this->m_FixedImageLimiter = 0;
    this->m_MovingImageLimiter = 0;
    this->m_UseFixedImageLimiter = false;
    this->m_UseMovingImageLimiter = false;
    this->m_FixedLimitRangeRatio = 0.01;
    this->m_MovingLimitRangeRatio = 0.01;
    this->m_FixedImageTrueMin   = NumericTraits< FixedImagePixelType  >::Zero;
		this->m_FixedImageTrueMax   = NumericTraits< FixedImagePixelType  >::One;
		this->m_MovingImageTrueMin  = NumericTraits< MovingImagePixelType >::Zero;
		this->m_MovingImageTrueMax  = NumericTraits< MovingImagePixelType >::One;
    this->m_FixedImageMinLimit  = NumericTraits< FixedImageLimiterOutputType  >::Zero;
    this->m_FixedImageMaxLimit  = NumericTraits< FixedImageLimiterOutputType  >::One;
    this->m_MovingImageMinLimit = NumericTraits< MovingImageLimiterOutputType >::Zero;
    this->m_MovingImageMaxLimit = NumericTraits< MovingImageLimiterOutputType >::One;
        
  } // end Constructor


  /**
	 * ********************* Initialize ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::Initialize(void) throw ( ExceptionObject )
  {
    /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();

    /** Cache the number of transformation parameters. This line 
     * emphasises that a user has to call Initialize again if the number
     * of parameters is changed */
		this->m_NumberOfParameters = this->m_Transform->GetNumberOfParameters();

    /** Setup the parameters for the gray value limiters */
    this->InitializeLimiters();
	
    /** Connect the image sampler */
    this->InitializeImageSampler();

    /** Check if the interpolator is a bspline interpolator */
    this->CheckForBSplineInterpolator();

    /** Check if the transform is a BSplineTransform or a BSplineCombinationTransform */
    this->CheckForBSplineTransform();
    
    /** Initialize the internal moving image mask */
    this->InitializeInternalMasks();
	
  } // end Initialize

  
  /**
	 * ****************** ComputeFixedImageExtrema ***************************
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
		::ComputeFixedImageExtrema(
    const FixedImageType * image,
    const FixedImageRegionType & region )
  {
 		/** NB: We can't use StatisticsImageFilter to do this because
		 * the filter computes the min/max for the largest possible region. */
		FixedImagePixelType trueMinTemp = NumericTraits<FixedImagePixelType>::max();
		FixedImagePixelType trueMaxTemp = NumericTraits<FixedImagePixelType>::NonpositiveMin();

		typedef ImageRegionConstIterator<FixedImageType> IteratorType;
		IteratorType iterator( image, region );
		for ( iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator )
    {
			const FixedImagePixelType sample = iterator.Get();
      trueMinTemp = vnl_math_min( trueMinTemp, sample );
      trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
    }
    this->m_FixedImageTrueMin = trueMinTemp;
    this->m_FixedImageTrueMax = trueMaxTemp;

    this->m_FixedImageMinLimit = static_cast<FixedImageLimiterOutputType>(
      trueMinTemp - this->m_FixedLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );
    this->m_FixedImageMaxLimit = static_cast<FixedImageLimiterOutputType>(
      trueMaxTemp + this->m_FixedLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );
  } // end ComputeFixedImageExtrema    
		
  
  /**
	 * ****************** ComputeMovingImageExtrema ***************************
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
		::ComputeMovingImageExtrema(
    const MovingImageType * image,
    const MovingImageRegionType & region )
  {
 		/** NB: We can't use StatisticsImageFilter to do this because
		 * the filter computes the min/max for the largest possible region. */
		MovingImagePixelType trueMinTemp = NumericTraits<MovingImagePixelType>::max();
		MovingImagePixelType trueMaxTemp = NumericTraits<MovingImagePixelType>::NonpositiveMin();

		typedef ImageRegionConstIterator<MovingImageType> IteratorType;
		IteratorType iterator( image, region );
		for ( iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator )
    {
			const MovingImagePixelType sample = iterator.Get();
      trueMinTemp = vnl_math_min( trueMinTemp, sample );
      trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
    }
    this->m_MovingImageTrueMin = trueMinTemp;
    this->m_MovingImageTrueMax = trueMaxTemp;

    this->m_MovingImageMinLimit = static_cast<MovingImageLimiterOutputType>(
      trueMinTemp - this->m_MovingLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );
    this->m_MovingImageMaxLimit = static_cast<MovingImageLimiterOutputType>(
      trueMaxTemp + this->m_MovingLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );
  } // end ComputeMovingImageExtrema    
		
	
  /**
   * ****************** InitializeLimiter *****************************
   */

	template <class TFixedImage, class TMovingImage> 
		void
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
		::InitializeLimiters(void)
  {
    /** Set up fixed limiter */
    if ( this->GetUseFixedImageLimiter() )
    {
      if ( this->GetFixedImageLimiter() == 0 )
      {
        itkExceptionMacro(<< "No fixed image limiter has been set!");
      }
      
      this->ComputeFixedImageExtrema(
        this->GetFixedImage(),
        this->GetFixedImageRegion() );

      this->m_FixedImageLimiter->SetLowerThreshold(
        static_cast<RealType>( this->m_FixedImageTrueMin ) );
      this->m_FixedImageLimiter->SetUpperThreshold(
        static_cast<RealType>( this->m_FixedImageTrueMax ) );
      this->m_FixedImageLimiter->SetLowerBound( this->m_FixedImageMinLimit );
      this->m_FixedImageLimiter->SetUpperBound( this->m_FixedImageMaxLimit );
      
      this->m_FixedImageLimiter->Initialize();
    }

    /** Set up moving limiter */
    if ( this->GetUseMovingImageLimiter() )
    {
      if ( this->GetMovingImageLimiter() == 0 )
      {
        itkExceptionMacro(<< "No moving image limiter has been set!");
      }
      
      this->ComputeMovingImageExtrema(
        this->GetMovingImage(),
        this->GetMovingImage()->GetBufferedRegion() );

      this->m_MovingImageLimiter->SetLowerThreshold(
        static_cast<RealType>( this->m_MovingImageTrueMin ) );
      this->m_MovingImageLimiter->SetUpperThreshold(
        static_cast<RealType>( this->m_MovingImageTrueMax ) );
      this->m_MovingImageLimiter->SetLowerBound( this->m_MovingImageMinLimit );
      this->m_MovingImageLimiter->SetUpperBound( this->m_MovingImageMaxLimit );
      
      this->m_MovingImageLimiter->Initialize();
    }
        
  } // end InitializeLimiter


  /**
	 * ********************* InitializeImageSampler ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::InitializeImageSampler(void) throw ( ExceptionObject )
  {
    if ( this->GetUseImageSampler() )
    {
      /** Check if the ImageSampler is set. */
      if( !this->m_ImageSampler )
      {
        itkExceptionMacro( << "ImageSampler is not present" );
      }
      /** Initialize the Image Sampler. */
      this->m_ImageSampler->SetInput( this->m_FixedImage );
      this->m_ImageSampler->SetMask( this->m_FixedImageMask );
      this->m_ImageSampler->SetInputImageRegion( this->GetFixedImageRegion() );
    }
  } // end InitializeImageSampler


  /**
   * ****************** CheckForBSplineInterpolator **********************
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
		::CheckForBSplineInterpolator(void)
  {
    /** Check if the interpolator is of type BSplineInterpolateImageFunction.
		* If so, we can make use of its EvaluateDerivatives method.
		* Otherwise, we use a (forward) finite difference scheme,
    * which is exactly right for linear interpolation. */
		this->m_InterpolatorIsBSpline = false;
		BSplineInterpolatorType * testPtr = 
      dynamic_cast<BSplineInterpolatorType *>( this->m_Interpolator.GetPointer() );
		if ( !testPtr )
    {
      this->m_ForwardDifferenceFilter = ForwardDifferenceFilterType::New();
      this->m_ForwardDifferenceFilter->SetUseImageSpacing(true);
      this->m_ForwardDifferenceFilter->SetInput( this->m_MovingImage );
      this->m_ForwardDifferenceFilter->Update();
      this->m_GradientImage = this->m_ForwardDifferenceFilter->GetOutput();
		
			this->m_BSplineInterpolator = 0;
			itkDebugMacro( "Interpolator is not BSpline" );
    } 
		else
    {
      this->m_ForwardDifferenceFilter = 0;
      this->m_GradientImage = 0;
      this->m_InterpolatorIsBSpline = true;
			this->m_BSplineInterpolator = testPtr;
			itkDebugMacro( "Interpolator is BSpline" );
    }
  } // end CheckForBSplineInterpolator

  
  /**
   * ****************** CheckForBSplineTransform **********************
   * Check if the transform is of type BSplineDeformableTransform.
	 * If so, we can speed up derivative calculations by only inspecting
	 * the parameters in the support region of a point. 
	 */

	template <class TFixedImage, class TMovingImage> 
		void
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
		::CheckForBSplineTransform(void)
  {
		this->m_TransformIsBSpline = false;
		
		BSplineTransformType * testPtr1 = dynamic_cast<BSplineTransformType *>(
			this->m_Transform.GetPointer() );
		if( !testPtr1 )
    {
			this->m_BSplineTransform = 0;
			itkDebugMacro( "Transform is not BSplineDeformable" );
    }
		else
    {
      this->m_TransformIsBSpline = true;
			this->m_BSplineTransform = testPtr1;
			this->m_NumBSplineParametersPerDim = 
        this->m_BSplineTransform->GetNumberOfParametersPerDimension();
			this->m_NumBSplineWeights = this->m_BSplineTransform->GetNumberOfWeights();
			itkDebugMacro( "Transform is BSplineDeformable" );
    }

		/** Check if the transform is of type BSplineCombinationTransform. */
		this->m_TransformIsBSplineCombination = false;
		
		BSplineCombinationTransformType * testPtr2 = 
			dynamic_cast<BSplineCombinationTransformType *>(this->m_Transform.GetPointer() );
		if( !testPtr2 )
    {
			this->m_BSplineCombinationTransform = 0;
			itkDebugMacro( "Transform is not BSplineCombination" );
    }
		else
    {
      this->m_TransformIsBSplineCombination = true;
			this->m_BSplineCombinationTransform = testPtr2;

			/** The current transform in the BSplineCombinationTransform is 
			 * always a BSplineTransform */
			BSplineTransformType * bsplineTransform = 
				dynamic_cast<BSplineTransformType * >(
				this->m_BSplineCombinationTransform->GetCurrentTransform() );

			if (!bsplineTransform)
			{
				itkExceptionMacro(<< "The BSplineCombinationTransform is not properly configured. The CurrentTransform is not set." );
			}
			this->m_NumBSplineParametersPerDim = 
        bsplineTransform->GetNumberOfParametersPerDimension();
			this->m_NumBSplineWeights = bsplineTransform->GetNumberOfWeights();
			itkDebugMacro( "Transform is BSplineCombination" );
    }

    /** Resize the weights and transform index arrays and compute the parameters offset */
		if ( this->m_TransformIsBSpline || this->m_TransformIsBSplineCombination )
    {
			this->m_BSplineTransformWeights =
        BSplineTransformWeightsType( this->m_NumBSplineWeights );
			this->m_BSplineTransformIndices =
        BSplineTransformIndexArrayType( this->m_NumBSplineWeights );
			for ( unsigned int j = 0; j < FixedImageDimension; j++ )
      {
				this->m_BSplineParametersOffset[j] = j * this->m_NumBSplineParametersPerDim; 
      }
      this->m_NonZeroJacobianIndices.SetSize(
        FixedImageDimension * this->m_NumBSplineWeights );
      this->m_InternalTransformJacobian.SetSize( 
        FixedImageDimension, FixedImageDimension * this->m_NumBSplineWeights );
      this->m_InternalTransformJacobian.Fill(0.0);
    }
    else
    {   
      this->m_NonZeroJacobianIndices.SetSize( this->m_NumberOfParameters );
      for ( unsigned int i = 0; i < this->m_NumberOfParameters; ++i)
      {
        this->m_NonZeroJacobianIndices[i] = i;
      }
      m_InternalTransformJacobian.SetSize( 0, 0 );
    }
    		
  } // end CheckForBSplineTransform


  /**
	 * ********************* InitializeInternalMasks *********************
   * Initialize the internal moving image mask
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::InitializeInternalMasks(void)
  {
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
    const unsigned int radius = this->GetMovingImageMaskInterpolationOrder();

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
      MovingImagePointType point;

      /** Set the pixel 1 if inside the mask and to 0 if outside */
      for( iterator.GoToBegin(); ! iterator.IsAtEnd(); ++ iterator )
      {
        const MovingImageIndexType & index = iterator.GetIndex();
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
        
    } // end else (if moving mask)
        
    /** Set the internal mask into the interpolator */
    this->m_MovingImageMaskInterpolator->SetInputImage( this->m_InternalMovingImageMask );
 
  } // end InitializeInternalMasks


  /**
	 * ******************* EvaluateMovingImageValueAndDerivative ******************
	 *
	 * Compute image value and possibly derivative at a transformed point
	 */

  template < class TFixedImage, class TMovingImage >
		bool
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::EvaluateMovingImageValueAndDerivative( 
    const MovingImagePointType & mappedPoint,
    RealType & movingImageValue,
    MovingImageDerivativeType * gradient) const
  {
    /** Check if mapped point inside image buffer */
    MovingImageContinuousIndexType cindex;
    this->m_Interpolator->ConvertPointToContinousIndex( mappedPoint, cindex);
    bool sampleOk = this->m_Interpolator->IsInsideBuffer( cindex );
		if ( sampleOk )
    {
      /** Compute value and possibly derivative */
      movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex( cindex );
      if ( gradient )
      {    
        if( this->m_InterpolatorIsBSpline )
		    {
			    /** Computed moving image gradient using derivative BSpline kernel.*/
	    		(*gradient) = 
            this->m_BSplineInterpolator->EvaluateDerivativeAtContinuousIndex( cindex );
		    }
		    else
		    {
			    /** Get the gradient from the precomputed forward difference image, by 
            * truncating the transformed continuous index */
          MovingImageIndexType index;
				  for ( unsigned int j = 0; j < MovingImageDimension; j++ )
				  {
  					index[ j ] = static_cast<long>( cindex[ j ] );
          }
          (*gradient) = this->m_GradientImage->GetPixel( index );
			  } 
      } // end if gradient
    } // end if sampleOk
    return sampleOk;
	} // end EvaluateMovingImageValueAndDerivative


  /**
	 * ********************** TransformPoint ************************
	 *
	 * Transform a point from FixedImage domain to MovingImage domain.
	 * This function also checks if mapped point is within support region
   * and mask.
	 */

	template < class TFixedImage, class TMovingImage >
		bool
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
		::TransformPoint( 
		const FixedImagePointType & fixedImagePoint, 
		MovingImagePointType & mappedPoint ) const
	{
    bool sampleOk = true;
		if ( !(this->m_TransformIsBSpline) && !(this->m_TransformIsBSplineCombination) )
		{
			mappedPoint = this->m_Transform->TransformPoint( fixedImagePoint );
      sampleOk = true;      
		}
		else
		{
			if (this->m_TransformIsBSpline)
			{
				this->m_BSplineTransform->TransformPoint( 
          fixedImagePoint,
					mappedPoint,
					this->m_BSplineTransformWeights,
					this->m_BSplineTransformIndices,
					sampleOk );
			}
			else if (this->m_TransformIsBSplineCombination)
			{
				this->m_BSplineCombinationTransform->TransformPoint( 
          fixedImagePoint,
				  mappedPoint,
				  this->m_BSplineTransformWeights,
				  this->m_BSplineTransformIndices,
				  sampleOk );
			}
		}
    return sampleOk;
	} // end TransformPoint


  /**
	 * *************** EvaluateTransformJacobian ****************
	 */

	template < class TFixedImage, class TMovingImage >
    const typename AdvancedImageToImageMetric<TFixedImage,TMovingImage>::TransformJacobianType &
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
		::EvaluateTransformJacobian( 
		const FixedImagePointType & fixedImagePoint) const
  {
    if( !(this->m_TransformIsBSpline) && !(this->m_TransformIsBSplineCombination) )
		{
			/** Generic version which works for all transforms. */
			return this->m_Transform->GetJacobian( fixedImagePoint );
		} // end if no bspline transform
		else
		{
			/** If the transform is of type BSplineDeformableTransform or of type
			 * BSplineCombinationTransform, we can obtain a speed up by only 
			 * processing the affected parameters. */
      unsigned int i = 0;
      /** We assume the sizes of the m_InternalTransformJacobian and the
       * m_NonZeroJacobianIndices have already been set; Also we assume
       * that the InternalTransformJacobian is not 'touched' by other
       * functions (some elements always stay zero). */      
			for( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
			{
        for( unsigned int mu = 0; mu < this->m_NumBSplineWeights; mu++ )
				{
				  /* The array weights contains the Jacobian values in a 1-D array 
					 * (because for each parameter the Jacobian is non-zero in only 1 of the
					 * possible dimensions) which is multiplied by the moving image gradient. */
          this->m_InternalTransformJacobian[dim][i] = this->m_BSplineTransformWeights[mu];
				
          /** The parameter number to which this partial derivative corresponds */
					const unsigned int parameterNumber = 
            this->m_BSplineTransformIndices[mu] + this->m_BSplineParametersOffset[dim];
          this->m_NonZeroJacobianIndices[i] = parameterNumber;

          /** Go to next column in m_InternalTransformJacobian */
          ++i;
  			} //end mu for loop
			} //end dim for loop
      return this->m_InternalTransformJacobian;
		} // end if-block transform is BSpline

  } // end EvaluateTransformJacobian

 
  /**
	 * **************** EvaluateMovingMaskValueAndDerivative *******************
   * Estimate value and possibly spatial derivative of internal moving mask 
   */

	template < class TFixedImage, class TMovingImage> 
		void
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::EvaluateMovingMaskValueAndDerivative(
      const MovingImagePointType & point,
      RealType & value,
      MovingImageMaskDerivativeType * derivative) const
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
        value = static_cast<RealType>(
          this->m_MovingImageMaskInterpolator->EvaluateAtContinuousIndex(cindex) );
        if (derivative)
        {
          (*derivative) = this->m_MovingImageMaskInterpolator->
            EvaluateDerivativeAtContinuousIndex(cindex);
        }
      }
      else
      {
        value = 0.0;
        if (derivative)
        {
          derivative->Fill( itk::NumericTraits<DerivativeValueType>::Zero );
        }
      }
    }
    else
    {
      /** Just ignore the derivative of the mask */
      if ( this->m_MovingImageMask.IsNotNull() )
      {
        value = static_cast<RealType>(
          static_cast<unsigned char>( this->m_MovingImageMask->IsInside( point ) ) );
      }
      else
      {
        value = 1.0;
      }
      if (derivative)
      {
        derivative->Fill( itk::NumericTraits<DerivativeValueType>::Zero );
      }
    }
  } // end EvaluateMovingMaskValueAndDerivative


  /**
   * *********************** CheckNumberOfSamples ***********************
   */

  template < class TFixedImage, class TMovingImage >
		void
		AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::CheckNumberOfSamples(
      unsigned long wanted, unsigned long found, double sumOfMaskValues) const
  {
    const double smallNumber2 = 1e-10;
    if( found < wanted * this->GetRequiredRatioOfValidSamples() || sumOfMaskValues < smallNumber2 )
    {
      itkExceptionMacro( "Too many samples map outside moving image buffer: "
        << found << " / " << wanted << std::endl );
    }
    this->m_NumberOfPixelsCounted = found;
  } // end CheckNumberOfSamples


  /**
	 * ********************* PrintSelf ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf( os, indent );
    os << indent << "ImageSampler: " << this->m_ImageSampler.GetPointer() << std::endl;
  } // end PrintSelf


} // end namespace itk


#endif // end #ifndef _itkAdvancedImageToImageMetric_txx

