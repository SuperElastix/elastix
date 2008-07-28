/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkAdvancedImageToImageMetric_txx
#define _itkAdvancedImageToImageMetric_txx

#include "itkAdvancedImageToImageMetric.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

  /**
   * ********************* Constructor ****************************
   */

  template <class TFixedImage, class TMovingImage>
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::AdvancedImageToImageMetric()
  {
    /** don't use the default gradient image as implemented by ITK.
     * It uses a Gaussian derivative, which introduces extra smoothing,
     * which may not always be desired. Also, when the derivatives are 
     * computed using Gaussian filtering, the gray-values should also be
     * blurred, to have a consistent 'image model' */
    this->SetComputeGradient( false ); 

    this->m_ImageSampler = 0;
    this->m_UseImageSampler = false;
    this->m_RequiredRatioOfValidSamples = 0.25;

    this->m_BSplineInterpolator = 0;
    this->m_InterpolatorIsBSpline = false;
    this->m_CentralDifferenceGradientFilter = 0;
    
    this->m_BSplineTransform = 0;
    this->m_BSplineCombinationTransform = 0;
    this->m_NumBSplineParametersPerDim = 0;
    this->m_NumBSplineWeights = 0;
    this->m_NumberOfParameters = 0;
    this->m_TransformIsBSpline = false;
    this->m_TransformIsBSplineCombination = false;
        
    this->m_UseMovingImageDerivativeScales = false;

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
     * of parameters is changed. */
    this->m_NumberOfParameters = this->m_Transform->GetNumberOfParameters();

    /** Setup the parameters for the gray value limiters. */
    this->InitializeLimiters();
  
    /** Connect the image sampler */
    this->InitializeImageSampler();

    /** Check if the interpolator is a bspline interpolator. */
    this->CheckForBSplineInterpolator();

    /** Check if the transform is a BSplineTransform or a BSplineCombinationTransform. */
    this->CheckForBSplineTransform();
  
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
    /** NB: We can't use StatisticsImageFilterWithMask to do this because
     * the filter computes the min/max for the largest possible region.
     * This filter is multi-threaded though.
     */
    FixedImagePixelType trueMinTemp = NumericTraits<FixedImagePixelType>::max();
    FixedImagePixelType trueMaxTemp = NumericTraits<FixedImagePixelType>::NonpositiveMin();
    
    /** If no mask. */
    if ( this->m_FixedImageMask.IsNull() )
    {
      typedef ImageRegionConstIterator<FixedImageType> IteratorType;
      IteratorType it( image, region );
      for ( it.GoToBegin(); !it.IsAtEnd(); ++it )
      {
        const FixedImagePixelType sample = it.Get();
        trueMinTemp = vnl_math_min( trueMinTemp, sample );
        trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
      }
    }
    /** Excluded extrema outside the mask.
     * Because we have to call TransformIndexToPhysicalPoint() and
     * check IsInside() this way is much (!) slower.
     */
    else
    {
      typedef ImageRegionConstIteratorWithIndex<FixedImageType> IteratorType;
      IteratorType it( image, region );

      for ( it.GoToBegin(); !it.IsAtEnd(); ++it )
      {
        OutputPointType point;
        image->TransformIndexToPhysicalPoint( it.GetIndex(), point );
        if ( this->m_FixedImageMask->IsInside( point ) )
        {
          const FixedImagePixelType sample = it.Get();
          trueMinTemp = vnl_math_min( trueMinTemp, sample );
          trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
        }
      }
    }

    /** Update member variables. */
    this->m_FixedImageTrueMin = trueMinTemp;
    this->m_FixedImageTrueMax = trueMaxTemp;

    this->m_FixedImageMinLimit = static_cast<FixedImageLimiterOutputType>(
      trueMinTemp - this->m_FixedLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );
    this->m_FixedImageMaxLimit = static_cast<FixedImageLimiterOutputType>(
      trueMaxTemp + this->m_FixedLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );

  } // end ComputeFixedImageExtrema()
  
  
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

    /** If no mask. */
    if ( this->m_MovingImageMask.IsNull() )
    {
      typedef ImageRegionConstIterator<MovingImageType> IteratorType;
      IteratorType iterator( image, region );
      for ( iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator )
      {
        const MovingImagePixelType sample = iterator.Get();
        trueMinTemp = vnl_math_min( trueMinTemp, sample );
        trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
      }
    }
    /** Excluded extrema outside the mask.
     * Because we have to call TransformIndexToPhysicalPoint() and
     * check IsInside() this way is much (!) slower.
     */
    else
    {
      typedef ImageRegionConstIteratorWithIndex<MovingImageType> IteratorType;
      IteratorType it( image, region );

      for ( it.GoToBegin(); !it.IsAtEnd(); ++it )
      {
        OutputPointType point;
        image->TransformIndexToPhysicalPoint( it.GetIndex(), point );
        if ( this->m_MovingImageMask->IsInside( point ) )
        {
          const MovingImagePixelType sample = it.Get();
          trueMinTemp = vnl_math_min( trueMinTemp, sample );
          trueMaxTemp = vnl_math_max( trueMaxTemp, sample );
        }
      }
    }

    /** Update member variables. */
    this->m_MovingImageTrueMin = trueMinTemp;
    this->m_MovingImageTrueMax = trueMaxTemp;

    this->m_MovingImageMinLimit = static_cast<MovingImageLimiterOutputType>(
      trueMinTemp - this->m_MovingLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );
    this->m_MovingImageMaxLimit = static_cast<MovingImageLimiterOutputType>(
      trueMaxTemp + this->m_MovingLimitRangeRatio * ( trueMaxTemp - trueMinTemp ) );

  } // end ComputeMovingImageExtrema()
  
  
  /**
   * ****************** InitializeLimiter *****************************
   */

  template <class TFixedImage, class TMovingImage> 
    void
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::InitializeLimiters(void)
  {
    /** Set up fixed limiter. */
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
  } // end InitializeImageSampler()


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
     * Otherwise, we precompute the gradients using a central difference scheme,
     * and do evaluate the gradient using nearest neighbour interpolation
     */
    this->m_InterpolatorIsBSpline = false;
    BSplineInterpolatorType * testPtr = 
      dynamic_cast<BSplineInterpolatorType *>( this->m_Interpolator.GetPointer() );
    if ( testPtr )
    {
      this->m_InterpolatorIsBSpline = true;
      this->m_BSplineInterpolator = testPtr;
      itkDebugMacro( "Interpolator is BSpline" );
    }
    else
    {
      this->m_BSplineInterpolator = 0;
      itkDebugMacro( "Interpolator is not BSpline" );
    }

    /** Don't overwrite the gradient image if GetComputeGradient() == true.
     * Otherwise we can use a forward difference derivative, or the derivative
     * provided by the BSpline interpolator.
     */
    if ( !this->GetComputeGradient() )
    {
      if ( !this->m_InterpolatorIsBSpline )
      {
        this->m_CentralDifferenceGradientFilter = CentralDifferenceGradientFilterType::New();
        this->m_CentralDifferenceGradientFilter->SetUseImageSpacing( true );
        this->m_CentralDifferenceGradientFilter->SetInput( this->m_MovingImage );
        this->m_CentralDifferenceGradientFilter->Update();
        this->m_GradientImage = this->m_CentralDifferenceGradientFilter->GetOutput();
      } 
      else
      {
        this->m_CentralDifferenceGradientFilter = 0;
        this->m_GradientImage = 0;
      }
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
    ::CheckForBSplineTransform( void )
  {
    this->m_TransformIsBSpline = false;
    
    BSplineTransformType * testPtr1 = dynamic_cast<BSplineTransformType *>(
      this->m_Transform.GetPointer() );
    if ( !testPtr1 )
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
      dynamic_cast<BSplineCombinationTransformType *>( this->m_Transform.GetPointer() );
    if ( !testPtr2 )
    {
      this->m_BSplineCombinationTransform = 0;
      itkDebugMacro( "Transform is not BSplineCombination" );
    }
    else
    {
      this->m_TransformIsBSplineCombination = true;
      this->m_BSplineCombinationTransform = testPtr2;

      /** The current transform in the BSplineCombinationTransform is 
       * always a BSplineTransform. */
      BSplineTransformType * bsplineTransform = 
        dynamic_cast<BSplineTransformType * >(
        this->m_BSplineCombinationTransform->GetCurrentTransform() );

      if ( !bsplineTransform )
      {
        itkExceptionMacro(<< "The BSplineCombinationTransform is not properly configured. The CurrentTransform is not set." );
      }
      this->m_NumBSplineParametersPerDim = 
        bsplineTransform->GetNumberOfParametersPerDimension();
      this->m_NumBSplineWeights = bsplineTransform->GetNumberOfWeights();
      itkDebugMacro( "Transform is BSplineCombination" );
    }

    /** Resize the weights and transform index arrays and compute the parameters offset. */
    if ( this->m_TransformIsBSpline || this->m_TransformIsBSplineCombination )
    {
      this->m_BSplineTransformWeights =
        BSplineTransformWeightsType( this->m_NumBSplineWeights );
      this->m_BSplineTransformIndices =
        BSplineTransformIndexArrayType( this->m_NumBSplineWeights );
      for ( unsigned int j = 0; j < FixedImageDimension; j++ )
      {
        this->m_BSplineParametersOffset[ j ] = j * this->m_NumBSplineParametersPerDim; 
      }
      this->m_NonZeroJacobianIndices.SetSize(
        FixedImageDimension * this->m_NumBSplineWeights );
      this->m_InternalTransformJacobian.SetSize( 
        FixedImageDimension, FixedImageDimension * this->m_NumBSplineWeights );
      this->m_InternalTransformJacobian.Fill( 0.0 );
    }
    else
    {   
      this->m_NonZeroJacobianIndices.SetSize( this->m_NumberOfParameters );
      for ( unsigned int i = 0; i < this->m_NumberOfParameters; ++i )
      {
        this->m_NonZeroJacobianIndices[ i ] = i;
      }
      this->m_InternalTransformJacobian.SetSize( 0, 0 );
    }
        
  } // end CheckForBSplineTransform


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
    MovingImageDerivativeType * gradient ) const
  {
    /** Check if mapped point inside image buffer. */
    MovingImageContinuousIndexType cindex;
    this->m_Interpolator->ConvertPointToContinuousIndex( mappedPoint, cindex );
    bool sampleOk = this->m_Interpolator->IsInsideBuffer( cindex );
    if ( sampleOk )
    {
      /** Compute value and possibly derivative. */
      movingImageValue = this->m_Interpolator->EvaluateAtContinuousIndex( cindex );
      if ( gradient )
      { 
        if ( this->m_InterpolatorIsBSpline && !this->GetComputeGradient() )
        {
          /** Computed moving image gradient using derivative BSpline kernel. */
          (*gradient) = 
            this->m_BSplineInterpolator->EvaluateDerivativeAtContinuousIndex( cindex );
        }
        else
        {
          /** Get the gradient by NearestNeighboorInterpolation of the gradient image.
           * It is assumed that the gradient image is computed.  */
          MovingImageIndexType index;
          for ( unsigned int j = 0; j < MovingImageDimension; j++ )
          {
            index[ j ] = static_cast<long>( vnl_math_rnd( cindex[ j ] ) );
          }
          (*gradient) = this->m_GradientImage->GetPixel( index );
        }
        if ( this->m_UseMovingImageDerivativeScales )
        {
          for ( unsigned int i = 0; i < MovingImageDimension; ++i )
          {
            (*gradient)[ i ] *= this->m_MovingImageDerivativeScales[ i ];
          }
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
    if ( !this->m_TransformIsBSpline && !this->m_TransformIsBSplineCombination )
    {
      mappedPoint = this->m_Transform->TransformPoint( fixedImagePoint );
      sampleOk = true;      
    }
    else
    {
      if ( this->m_TransformIsBSpline )
      {
        this->m_BSplineTransform->TransformPoint( 
          fixedImagePoint,
          mappedPoint,
          this->m_BSplineTransformWeights,
          this->m_BSplineTransformIndices,
          sampleOk );
      }
      else if ( this->m_TransformIsBSplineCombination )
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
    if ( !this->m_TransformIsBSpline && !this->m_TransformIsBSplineCombination )
    {
      /** Generic version which works for all transforms. */
      return this->m_Transform->GetJacobian( fixedImagePoint );
    } // end if no B-spline transform
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
      for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
      {
        for ( unsigned int mu = 0; mu < this->m_NumBSplineWeights; mu++ )
        {
          /* The array weights contains the Jacobian values in a 1-D array 
           * (because for each parameter the Jacobian is non-zero in only 1 of the
           * possible dimensions) which is multiplied by the moving image gradient. */
          this->m_InternalTransformJacobian[ dim ][ i ] = this->m_BSplineTransformWeights[ mu ];
        
          /** The parameter number to which this partial derivative corresponds */
          const unsigned int parameterNumber = 
            this->m_BSplineTransformIndices[ mu ] + this->m_BSplineParametersOffset[ dim ];
          this->m_NonZeroJacobianIndices[ i ] = parameterNumber;

          /** Go to next column in m_InternalTransformJacobian */
          ++i;
        } //end mu for loop
      } //end dim for loop

      return this->m_InternalTransformJacobian;

    } // end if-block transform is BSpline

  } // end EvaluateTransformJacobian

 
  /**
   * ************************** IsInsideMovingMask *************************
   * Check if point is inside moving mask
   */

  template < class TFixedImage, class TMovingImage> 
    bool
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::IsInsideMovingMask( const MovingImagePointType & point ) const
  {
    /** If a mask has been set: */
    if ( this->m_MovingImageMask.IsNotNull() )
    {
      return this->m_MovingImageMask->IsInside( point );
    }
    
    /** If no mask has been set, just return true. */
    return true;
    
  } // end IsInsideMovingMask


  /**
   * *********************** CheckNumberOfSamples ***********************
   */

  template < class TFixedImage, class TMovingImage >
    void
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::CheckNumberOfSamples(
      unsigned long wanted, unsigned long found ) const
  {
    this->m_NumberOfPixelsCounted = found;
    if ( found < wanted * this->GetRequiredRatioOfValidSamples() )
    {
      itkExceptionMacro( "Too many samples map outside moving image buffer: "
        << found << " / " << wanted << std::endl );
    }
  } // end CheckNumberOfSamples()


  /**
   * ********************* PrintSelf ****************************
   */

  template <class TFixedImage, class TMovingImage>
    void
    AdvancedImageToImageMetric<TFixedImage,TMovingImage>
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf( os, indent );

    /** Variables related to the Sampler */
    os << indent << "Variables related to the Sampler: " << std::endl;
    os << indent.GetNextIndent() << "ImageSampler: " << this->m_ImageSampler.GetPointer() << std::endl;
    os << indent.GetNextIndent() << "UseImageSampler: " << this->m_UseImageSampler << std::endl;

    /** Variables for the Limiters. */
    os << indent << "Variables related to the Limiters: " << std::endl;
    os << indent.GetNextIndent() << "FixedLimitRangeRatio: " << this->m_FixedLimitRangeRatio << std::endl;
    os << indent.GetNextIndent() << "MovingLimitRangeRatio: " << this->m_MovingLimitRangeRatio << std::endl;
    os << indent.GetNextIndent() << "UseFixedImageLimiter: " << this->m_UseFixedImageLimiter << std::endl;
    os << indent.GetNextIndent() << "UseMovingImageLimiter: " << this->m_UseMovingImageLimiter << std::endl;
    os << indent.GetNextIndent() << "FixedImageLimiter: " << this->m_FixedImageLimiter.GetPointer() << std::endl;
    os << indent.GetNextIndent() << "MovingImageLimiter: " << this->m_MovingImageLimiter.GetPointer() << std::endl;
    os << indent.GetNextIndent() << "FixedImageTrueMin: " << this->m_FixedImageTrueMin << std::endl;
    os << indent.GetNextIndent() << "MovingImageTrueMin: " << this->m_MovingImageTrueMin << std::endl;
    os << indent.GetNextIndent() << "FixedImageTrueMax: " << this->m_FixedImageTrueMax << std::endl;
    os << indent.GetNextIndent() << "MovingImageTrueMax: " << this->m_MovingImageTrueMax << std::endl;
    os << indent.GetNextIndent() << "FixedImageMinLimit: " << this->m_FixedImageMinLimit << std::endl;
    os << indent.GetNextIndent() << "MovingImageMinLimit: " << this->m_MovingImageMinLimit << std::endl;
    os << indent.GetNextIndent() << "FixedImageMaxLimit: " << this->m_FixedImageMaxLimit << std::endl;
    os << indent.GetNextIndent() << "MovingImageMaxLimit: " << this->m_MovingImageMaxLimit << std::endl;

    /** Variables related to image derivative computation. */
    os << indent << "Variables related to image derivative computation: " << std::endl;
    os << indent.GetNextIndent() << "InterpolatorIsBSpline: " << this->m_InterpolatorIsBSpline << std::endl;
    os << indent.GetNextIndent() << "BSplineInterpolator: " << this->m_BSplineInterpolator.GetPointer() << std::endl;
    os << indent.GetNextIndent() << "CentralDifferenceGradientFilter: " << this->m_CentralDifferenceGradientFilter.GetPointer() << std::endl;
    
    /** Variables used when the transform is a bspline transform. */
    os << indent << "Variables used when the transform is a B-spline transform: " << std::endl;
    os << indent.GetNextIndent() << "InternalTransformJacobian: " << this->m_InternalTransformJacobian << std::endl;
    os << indent.GetNextIndent() << "NonZeroJacobianIndices: " << this->m_NonZeroJacobianIndices << std::endl;
    os << indent.GetNextIndent() << "TransformIsBSpline: " << this->m_TransformIsBSpline << std::endl;
    os << indent.GetNextIndent() << "TransformIsBSplineCombination: " << this->m_TransformIsBSplineCombination << std::endl;
    os << indent.GetNextIndent() << "BSplineTransform: " << this->m_BSplineTransform.GetPointer() << std::endl;
    os << indent.GetNextIndent() << "BSplineCombinationTransform: " << this->m_BSplineCombinationTransform.GetPointer() << std::endl;
    os << indent.GetNextIndent() << "BSplineTransformWeights: " << this->m_BSplineTransformWeights << std::endl;
    os << indent.GetNextIndent() << "BSplineTransformIndices: " << this->m_BSplineTransformIndices << std::endl;
    os << indent.GetNextIndent() << "BSplineParametersOffset: " << this->m_BSplineParametersOffset << std::endl;
    os << indent.GetNextIndent() << "NumBSplineParametersPerDim: " << this->m_NumBSplineParametersPerDim << std::endl;
    os << indent.GetNextIndent() << "NumBSplineWeights: " << this->m_NumBSplineWeights << std::endl;
       
    /** Other variables. */
    os << indent << "Other variables of the AdvancedImageToImageMetric: " << std::endl;
    os << indent.GetNextIndent() << "NumberOfParameters: " << this->m_NumberOfParameters << std::endl;
    os << indent.GetNextIndent() << "RequiredRatioOfValidSamples: " << this->m_RequiredRatioOfValidSamples << std::endl;
    os << indent.GetNextIndent() << "UseMovingImageDerivativeScales: " << this->m_UseMovingImageDerivativeScales << std::endl;
    os << indent.GetNextIndent() << "MovingImageDerivativeScales: " << this->m_MovingImageDerivativeScales << std::endl;

  } // end PrintSelf


} // end namespace itk


#endif // end #ifndef _itkAdvancedImageToImageMetric_txx

