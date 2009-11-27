/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkMattesMutualInformationImageToImageMetricWithRigidityPenalty_HXX__
#define _itkMattesMutualInformationImageToImageMetricWithRigidityPenalty_HXX__

#include "itkMattesMutualInformationImageToImageMetricWithRigidityPenalty.h"


namespace itk
{
  
  
/**
 * ********************* Constructor ****************************
 */

template < class TFixedImage, class TMovingImage >
MattesMutualInformationImageToImageMetricWithRigidityPenalty<TFixedImage,TMovingImage>
::MattesMutualInformationImageToImageMetricWithRigidityPenalty()
{
  /** Initialize dilation. */
  this->m_DilationRadiusMultiplier = NumericTraits<CoordinateRepresentationType>::One;
  this->m_DilateRigidityImages = true;

  /** Create and set the two metrics. */
  this->m_MattesMutualInformationMetric = MattesMutualInformationMetricType::New();
  this->m_RigidityPenaltyTermMetric     = RigidityPenaltyTermMetricType::New();
  this->SetNumberOfMetrics( 2 );
  this->SetMetric( this->m_MattesMutualInformationMetric, 0 );
  this->SetMetric( this->m_RigidityPenaltyTermMetric, 1 );
  this->SetMetricWeight( 1.0, 0 );
  this->SetMetricWeight( 1.0, 1 );

  /** Initialize rigidity images and their usage. */
  this->m_UseFixedRigidityImage = true;
  this->m_UseMovingRigidityImage = true;
  this->m_FixedRigidityImage = 0;
  this->m_MovingRigidityImage = 0;
  this->m_RigidityCoefficientImage = RigidityImageType::New();

  /** Initialize dilation filter for the rigidity images. */
  this->m_FixedRigidityImageDilation.resize( FixedImageDimension );
  this->m_MovingRigidityImageDilation.resize( MovingImageDimension );
  for ( unsigned int i = 0; i < FixedImageDimension; i++ )
  {
    this->m_FixedRigidityImageDilation[ i ] = 0;
    this->m_MovingRigidityImageDilation[ i ] = 0;
  }

  /** Initialize dilated rigidity images. */
  this->m_FixedRigidityImageDilated = 0;
  this->m_MovingRigidityImageDilated = 0;

  this->m_LocalBSplineTransform = 0;

} // end Constructor()


/**
 * ********************* PrintSelf ******************************
 */

template < class TFixedImage, class TMovingImage  >
void
MattesMutualInformationImageToImageMetricWithRigidityPenalty<TFixedImage,TMovingImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{

  this->Superclass::PrintSelf( os, indent );

  /*os << indent << "RigidPenaltyWeight: "
  << this->m_RigidPenaltyWeight << std::endl;
  os << indent << "SecondOrderWeight: "
  << this->m_SecondOrderWeight << std::endl;
  os << indent << "OrthonormalityWeight: "
  << this->m_OrthonormalityWeight << std::endl;
  os << indent << "PropernessWeight: "
  << this->m_PropernessWeight << std::endl;
  os << indent << "UseImageSpacing: ";
  if ( this->m_UseImageSpacing ) os << "true" << std::endl;
  else os << "false" << std::endl;*/
  os << indent << "DilateRigidityImages: ";
  if ( this->m_DilateRigidityImages ) os << "true" << std::endl;
  else os << "false" << std::endl;
  os << indent << "UseFixedRigidityImage: ";
  if ( this->m_UseFixedRigidityImage ) os << "true" << std::endl;
  else os << "false" << std::endl;
  os << indent << "UseMovingRigidityImage: ";
  if ( this->m_UseMovingRigidityImage ) os << "true" << std::endl;
  else os << "false" << std::endl;

} // end PrintSelf()


/**
 * ********************* Initialize *****************************
 */

template <class TFixedImage, class TMovingImage> 
void
MattesMutualInformationImageToImageMetricWithRigidityPenalty<TFixedImage,TMovingImage>
::Initialize( void ) throw ( ExceptionObject )
{
  /** Call the initialize of the superclass. */
  this->Superclass::Initialize();

  /** Check for an advanced B-spline transform. */
  // This should probably be done in the AdvancedImageToImageMetric
  // SO that we can use m_TransformIsBSpline or something.
  typename BSplineTransformType::Pointer localBSplineTransform = 0;
  BSplineTransformType * testPtr1
    = dynamic_cast<BSplineTransformType *>( this->m_Transform.GetPointer() );
  CombinationTransformType * testPtr2a
    = dynamic_cast<CombinationTransformType *>( this->m_Transform.GetPointer() );
  bool transformSupported = false;
  if ( testPtr1 )
  {
    // is advanced B-spline !
    transformSupported = true;
    localBSplineTransform = testPtr1;
    this->m_RigidityPenaltyTermMetric->SetBSplineTransform( testPtr1 );
  }
  else if ( testPtr2a )
  {
    // is combo transform
    BSplineTransformType * testPtr2b = dynamic_cast<BSplineTransformType *>(
      (testPtr2a->GetCurrentTransform()) );
    if ( testPtr2b )
    {
      transformSupported = true;
      localBSplineTransform = testPtr2b;
      this->m_RigidityPenaltyTermMetric->SetBSplineTransform( testPtr2b );
    }
  }
 

  /** Set the B-spline transform to m_RigidityPenaltyTermMetric. */
  if ( !transformSupported )
  {
    itkExceptionMacro( << "ERROR: this metric expects a B-spline transform." );
  }
  this->m_LocalBSplineTransform = localBSplineTransform;

  /** Allocate the RigidityCoefficientImage, so that it matches the B-spline grid.
   * Only because the Initialize()-function above is called before,
   * this code is valid, because there the B-spline transform is set.
   */
  RigidityImageRegionType region;
  region.SetSize( localBSplineTransform->GetGridRegion().GetSize() );
  region.SetIndex( localBSplineTransform->GetGridRegion().GetIndex() );
  this->m_RigidityCoefficientImage->SetRegions( region );
  this->m_RigidityCoefficientImage->SetSpacing(
    localBSplineTransform->GetGridSpacing() );
  this->m_RigidityCoefficientImage->SetOrigin(
    localBSplineTransform->GetGridOrigin() );
  this->m_RigidityCoefficientImage->Allocate();

  if ( !this->m_UseFixedRigidityImage && !this->m_UseMovingRigidityImage )
  {
    /** Fill the rigidity coefficient image with ones. */
    this->m_RigidityCoefficientImage->FillBuffer( 1.0 );

    /** Set the rigidity coefficients image into the rigid penalty term. */
    this->m_RigidityPenaltyTermMetric->SetRigidityCoefficientImage(
      this->m_RigidityCoefficientImage );
  }
  else
  {
    this->DilateRigidityImages();
  }

} // end Initialize()


/**
 * **************** DilateRigidityImages *****************
 */

template < class TFixedImage, class TMovingImage  >
void
MattesMutualInformationImageToImageMetricWithRigidityPenalty<TFixedImage,TMovingImage>
::DilateRigidityImages( void )
{
  /** Dilate m_FixedRigidityImage and m_MovingRigidityImage. */
  if ( this->m_DilateRigidityImages )
  {
    /** Some declarations. */
    SERadiusType radius;
    std::vector< StructuringElementType >  structuringElement( FixedImageDimension );

    /** Setup the pipeline. */
    if ( this->m_UseFixedRigidityImage )
    {
      /** Create the dilation filters for the fixedRigidityImage. */
      for ( unsigned int i = 0; i < FixedImageDimension; i++ )
      {
        this->m_FixedRigidityImageDilation[ i ] = DilateFilterType::New();
      }
      m_FixedRigidityImageDilation[ 0 ]->SetInput( m_FixedRigidityImage );
    }
    if ( this->m_UseMovingRigidityImage )
    {
      /** Create the dilation filter for the movingRigidityImage. */
      for ( unsigned int i = 0; i < FixedImageDimension; i++ )
      {
        this->m_MovingRigidityImageDilation[ i ] = DilateFilterType::New();
      }
      m_MovingRigidityImageDilation[ 0 ]->SetInput( m_MovingRigidityImage );
    }

    /** Get the B-spline grid spacing. */
    GridSpacingType spacing;
    if ( this->m_LocalBSplineTransform.IsNotNull() )
    {
      spacing = this->m_LocalBSplineTransform->GetGridSpacing();
    }
   
    /** Set stuff for the separate dilation. */
    for ( unsigned int i = 0; i < FixedImageDimension; i++ )
    {
      /** Create the structuring element. */
      radius.Fill( 0 );
      radius.SetElement( i,
        static_cast<unsigned long>(
        this->m_DilationRadiusMultiplier
        * spacing[ i ] ) );

      structuringElement[ i ].SetRadius( radius );
      structuringElement[ i ].CreateStructuringElement();

      /** Set the kernel into all dilation filters.
       * The SetKernel() is implemented using a itkSetMacro, so a
       * this->Modified() is automatically called, which is important,
       * since this changes every time Initialize() is called (every resolution).
       */
      if ( this->m_UseFixedRigidityImage )
      {
        this->m_FixedRigidityImageDilation[ i ]->SetKernel( structuringElement[ i ] );
      }
      if ( this->m_UseMovingRigidityImage )
      {
        this->m_MovingRigidityImageDilation[ i ]->SetKernel( structuringElement[ i ] );
      }

      /** Connect the pipelines. */
      if ( i > 0 )
      {
        if ( this->m_UseFixedRigidityImage )
        {
          this->m_FixedRigidityImageDilation[ i ]->SetInput(
            m_FixedRigidityImageDilation[ i - 1 ]->GetOutput() );
        }
        if ( this->m_UseMovingRigidityImage )
        {
          this->m_MovingRigidityImageDilation[ i ]->SetInput(
            m_MovingRigidityImageDilation[ i - 1 ]->GetOutput() );
        }
      }
    } // end for loop

    /** Do the dilation for m_FixedRigidityImage. */
    if ( this->m_UseFixedRigidityImage )
    {
      try
      {
        this->m_FixedRigidityImageDilation[ FixedImageDimension - 1 ]->Update();
      }
      catch( itk::ExceptionObject & excp )
      {
        /** Add information to the exception. */
        excp.SetLocation( "MattesMutualInformationImageToImageMetricWithRigidityPenalty - Initialize()" );
        std::string err_str = excp.GetDescription();
        err_str += "\nError while dilating m_FixedRigidityImage.\n";
        excp.SetDescription( err_str );
        /** Pass the exception to an higher level. */
        throw excp;
      }
    }

    /** Do the dilation for m_MovingRigidityImage. */
    if ( this->m_UseMovingRigidityImage )
    {
      try
      {
        this->m_MovingRigidityImageDilation[ MovingImageDimension - 1 ]->Update();
      }
      catch( itk::ExceptionObject & excp )
      {
        /** Add information to the exception. */
        excp.SetLocation( "MattesMutualInformationImageToImageMetricWithRigidityPenalty - Initialize()" );
        std::string err_str = excp.GetDescription();
        err_str += "\nError while dilating m_MovingRigidityImage.\n";
        excp.SetDescription( err_str );
        /** Pass the exception to an higher level. */
        throw excp;
      }
    }

    /** Put the output of the dilation into some dilated images. */
    if ( this->m_UseFixedRigidityImage )
    {
      this->m_FixedRigidityImageDilated =
        this->m_FixedRigidityImageDilation[ FixedImageDimension - 1 ]->GetOutput();
    }
    if ( this->m_UseMovingRigidityImage )
    {
      this->m_MovingRigidityImageDilated =
        this->m_MovingRigidityImageDilation[ MovingImageDimension - 1 ]->GetOutput();
    }
  } // end if rigidity images should be dilated
  else
  {
    /** Copy the pointers of the undilated images to the dilated ones
     * if no dilation is needed.
     */
    if ( this->m_UseFixedRigidityImage )
    {
      this->m_FixedRigidityImageDilated = this->m_FixedRigidityImage;
    }
    if ( this->m_UseMovingRigidityImage )
    {
      this->m_MovingRigidityImageDilated = this->m_MovingRigidityImage;
    }

  } // end else if

} // end DilateRigidityImages()


/**
 * **************** FillRigidityCoefficientImage *****************
 */

template < class TFixedImage, class TMovingImage  >
void
MattesMutualInformationImageToImageMetricWithRigidityPenalty<TFixedImage,TMovingImage>
::FillRigidityCoefficientImage( const ParametersType& parameters ) const
{
  if ( !this->m_UseFixedRigidityImage && !this->m_UseMovingRigidityImage )
  {
    return;
  }
  /** Make sure that the transform is up to date. */
  this->m_Transform->SetParameters( parameters );

  /** Create and reset an iterator over m_RigidityCoefficientImage. */
  RigidityImageIteratorType it( this->m_RigidityCoefficientImage,
    this->m_RigidityCoefficientImage->GetLargestPossibleRegion() );
  it.GoToBegin();

  /** Fill m_RigidityCoefficientImage. */
  RigidityPixelType fixedValue, movingValue, in;
  RigidityImagePointType point; point.Fill( 0.0f );
  RigidityImageIndexType index1, index2;
  index1.Fill( 0 ); index2.Fill( 0 );
  fixedValue = NumericTraits<RigidityPixelType>::Zero;
  movingValue = NumericTraits<RigidityPixelType>::Zero;
  in = NumericTraits<RigidityPixelType>::Zero;
  bool isInFixedImage = false;
  bool	isInMovingImage = false;		 
  while ( !it.IsAtEnd() )
  {
    /** Get current pixel in world coordinates. */
    this->m_RigidityCoefficientImage
      ->TransformIndexToPhysicalPoint( it.GetIndex(), point );

    /** Get the corresponding indices in the fixed and moving RigidityImage's.
     * NOTE: Floating point index results are truncated to integers.
     */
    if ( this->m_UseFixedRigidityImage )
    {
      isInFixedImage = this->m_FixedRigidityImageDilated
        ->TransformPhysicalPointToIndex( point, index1 );
    }
    if ( this->m_UseMovingRigidityImage )
    {
      isInMovingImage = this->m_MovingRigidityImageDilated
        ->TransformPhysicalPointToIndex(
        this->m_Transform->TransformPoint( point ), index2 );
    }

    /** Get the values at those positions. */
    if ( this->m_UseFixedRigidityImage )
    {
      if ( isInFixedImage )
      {
        fixedValue = this->m_FixedRigidityImageDilated->GetPixel( index1 );
      }
      else
      {
        fixedValue = 0.0;
      }
    }

    if ( this->m_UseMovingRigidityImage )
    {
      if ( isInMovingImage )
      {
        movingValue = this->m_MovingRigidityImageDilated->GetPixel( index2 );
      }
      else
      {
        movingValue = 0.0;
      }
    }

    /** Determine the maximum. */
    if ( this->m_UseFixedRigidityImage && this->m_UseMovingRigidityImage )
    {
      in = ( fixedValue > movingValue ? fixedValue : movingValue );
    }
    else if ( this->m_UseFixedRigidityImage && !this->m_UseMovingRigidityImage )
    {
      in = fixedValue;
    }
    else if ( !this->m_UseFixedRigidityImage && this->m_UseMovingRigidityImage )
    {
      in = movingValue;
    }
    /** else{} is not happening here, because we assume that one of them is true.
     * In our case we checked that in the derived class: elxMattesMIWRR.
     */

    /** Set it. */
    it.Set( in );

    /** Increase iterator. */
    ++it;
  } // end while loop over rigidity coefficient image

  /** Set the rigidity coefficients image into the rigid regulizer metric. */
  this->m_RigidityPenaltyTermMetric->SetRigidityCoefficientImage(
    this->m_RigidityCoefficientImage );

} // end FillRigidityCoefficientImage()


} // end namespace itk


#endif // end #ifndef _itkMattesMutualInformationImageToImageMetricWithRigidityPenalty_HXX__

