/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkMultiInputImageToImageMetricBase_txx
#define _itkMultiInputImageToImageMetricBase_txx

#include "itkMultiInputImageToImageMetricBase.h"

/** Macros to reduce some copy-paste work.
 * These macros provide the implementation of
 * all Set/GetFixedImage, Set/GetInterpolator etc methods
 *
 * The macros are undef'ed at the end of this file
 */

/** Macro for setting objects. */
#define itkImplementationSetObjectMacro(_name, _type) \
  template <class TFixedImage, class TMovingImage> \
    void \
    MultiInputImageToImageMetricBase<TFixedImage,TMovingImage> \
    ::Set##_name ( _type * _arg, unsigned int pos ) \
  { \
    if ( this->m_##_name##Vector.size() < pos + 1 ) \
    { \
      this->m_##_name##Vector.resize( pos + 1 ); \
      this->m_NumberOf##_name##s = pos + 1; \
    } \
    if ( pos == 0 ) \
    { \
      this->Superclass::Set##_name ( _arg ); \
    } \
    if ( this->m_##_name##Vector[ pos ] != _arg ) \
    { \
      this->m_##_name##Vector[ pos ] = _arg; \
      this->Modified(); \
    } \
  } // comments for allowing ; after calling the macro

/** Macro for setting objects. */
#define itkImplementationSetObjectMacro2(_name, _type) \
  template <class TFixedImage, class TMovingImage> \
    void \
    MultiInputImageToImageMetricBase<TFixedImage,TMovingImage> \
    ::Set##_name ( _type * _arg, unsigned int pos ) \
  { \
    if ( this->m_##_name##Vector.size() < pos + 1 ) \
    { \
      this->m_##_name##Vector.resize( pos + 1 ); \
      this->m_NumberOf##_name##s = pos + 1; \
    } \
    if ( this->m_##_name##Vector[ pos ] != _arg ) \
    { \
      this->m_##_name##Vector[ pos ] = _arg; \
      this->Modified(); \
    } \
  } // comments for allowing ; after calling the macro

/** Macro for getting objects. */
#define itkImplementationGetObjectMacro(_name, _type ) \
  template <class TFixedImage, class TMovingImage> \
  typename MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>:: _type * \
    MultiInputImageToImageMetricBase<TFixedImage,TMovingImage> \
    ::Get##_name ( unsigned int pos ) const \
  { \
    if ( this->m_##_name##Vector.size() < pos + 1 ) \
    { \
      return 0; \
    } \
    return this->m_##_name##Vector [ pos ]; \
  } // comments for allowing ; after calling the macro

/** Macro for getting const objects. */
#define itkImplementationGetConstObjectMacro(_name, _type ) \
  template <class TFixedImage, class TMovingImage> \
  const typename MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>:: _type * \
    MultiInputImageToImageMetricBase<TFixedImage,TMovingImage> \
    ::Get##_name ( unsigned int pos ) const \
  { \
    if ( this->m_##_name##Vector.size() < pos + 1 ) \
    { \
      return 0; \
    } \
    return this->m_##_name##Vector [ pos ]; \
  } // comments for allowing ; after calling the macro


namespace itk
{

  /**
   * ************************ Constructor *************************
   */

  template <class TFixedImage, class TMovingImage>
    MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>
    ::MultiInputImageToImageMetricBase()
  {
    this->m_NumberOfFixedImages = 0;
    this->m_NumberOfFixedImageMasks = 0;
    this->m_NumberOfFixedImageRegions = 0;
    this->m_NumberOfMovingImages = 0;
    this->m_NumberOfMovingImageMasks = 0;
    this->m_NumberOfInterpolators = 0;
    this->m_NumberOfFixedImageInterpolators = 0;

    this->m_InterpolatorsAreBSpline = false;

  } // end Constructor()


  /** Set components. */
  itkImplementationSetObjectMacro( FixedImage, const FixedImageType );
  itkImplementationSetObjectMacro( FixedImageMask, FixedImageMaskType );
  itkImplementationSetObjectMacro( MovingImage, const MovingImageType );
  itkImplementationSetObjectMacro( MovingImageMask, MovingImageMaskType );
  itkImplementationSetObjectMacro( Interpolator, InterpolatorType );
  itkImplementationSetObjectMacro2( FixedImageInterpolator, FixedImageInterpolatorType );

  /** Get components. */
  itkImplementationGetConstObjectMacro( FixedImage, FixedImageType );
  itkImplementationGetObjectMacro( FixedImageMask, FixedImageMaskType );
  itkImplementationGetConstObjectMacro( MovingImage, MovingImageType );
  itkImplementationGetObjectMacro( MovingImageMask, MovingImageMaskType );
  itkImplementationGetObjectMacro( Interpolator, InterpolatorType );
  itkImplementationGetObjectMacro( FixedImageInterpolator, FixedImageInterpolatorType );


  /**
  * ************************ SetFixedImageRegion *************************
  */

  template <class TFixedImage, class TMovingImage>
    void
    MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>
    ::SetFixedImageRegion( const FixedImageRegionType _arg, unsigned int pos )
  {
    if ( this->m_FixedImageRegionVector.size() < pos + 1 )
    {
      this->m_FixedImageRegionVector.resize( pos + 1 );
      this->m_NumberOfFixedImageRegions = pos + 1;
    }
    if ( pos == 0 )
    {
      this->Superclass::SetFixedImageRegion( _arg );
    }
    if ( this->m_FixedImageRegionVector[ pos ] != _arg )
    {
      this->m_FixedImageRegionVector[ pos ] = _arg;
      this->Modified();
    }

  } // end SetFixedImageRegion()


  /**
  * ************************ GetFixedImageRegion *************************
  */

  template <class TFixedImage, class TMovingImage>
    const typename MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>
    ::FixedImageRegionType &
    MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>
    ::GetFixedImageRegion( unsigned int pos ) const
  {
    if ( this->m_FixedImageRegionVector.size() < pos )
    {
      return this->m_DummyFixedImageRegion;
    }
    return this->m_FixedImageRegionVector[ pos ];

  } // end GetFixedImageRegion()


  /**
   * ****************** CheckForBSplineInterpolators **********************
   */

  template <class TFixedImage, class TMovingImage>
  void
  MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>
  ::CheckForBSplineInterpolators( void )
  {
    /** Check if the interpolators are of type BSplineInterpolateImageFunction.
     * If so, we can make use of its EvaluateDerivatives method.
     * Otherwise, an exception is thrown.
     */
    this->m_InterpolatorsAreBSpline = true;
    this->m_BSplineInterpolatorVector.resize( this->m_NumberOfMovingImages );

    for ( unsigned int i = 0; i < this->m_NumberOfMovingImages; ++i )
    {
      BSplineInterpolatorType * testPtr =
        dynamic_cast<BSplineInterpolatorType *>(
        this->m_InterpolatorVector[ i ].GetPointer() );

      if ( testPtr )
      {
        this->m_InterpolatorsAreBSpline &= true;
        this->m_BSplineInterpolatorVector[ i ] = testPtr;
        itkDebugMacro( << "Interpolator " << i << " is B-spline." );
      }
      else
      {
        this->m_InterpolatorsAreBSpline &= false;
        itkDebugMacro( << "Interpolator " << i << " is NOT B-spline." );
        itkExceptionMacro( << "Interpolator " << i << " is NOT B-spline." );
      }
    } // end for-loop

  } // end CheckForBSplineInterpolators()


  /**
   * ****************** Initialize **********************
   */

  template <class TFixedImage, class TMovingImage>
  void
  MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>
  ::Initialize( void ) throw ( ExceptionObject )
  {
    /** Connect the interpolators. */
    for ( unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i )
    {
      this->m_InterpolatorVector[ i ]->SetInputImage( this->m_MovingImageVector[ i ] );
    }

    /** Connect the fixed image interpolators. */
    for ( unsigned int i = 0; i < this->GetNumberOfFixedImageInterpolators(); ++i )
    {
      this->m_FixedImageInterpolatorVector[ i ]->SetInputImage( this->m_FixedImageVector[ i ] );
    }

    /** Check for B-spline interpolators. */
    this->CheckForBSplineInterpolators();

    /** Call the superclass' implementation. */
    this->Superclass::Initialize();

  } // end Initialize()


  /**
   * ********************* InitializeImageSampler ****************************
   */

  template <class TFixedImage, class TMovingImage>
    void
    MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>
    ::InitializeImageSampler( void ) throw ( ExceptionObject )
  {
    if ( this->GetUseImageSampler() )
    {
      /** Check if the ImageSampler is set. */
      if( !this->m_ImageSampler )
      {
        itkExceptionMacro( << "ImageSampler is not present" );
      }

      /** Initialize the Image Sampler: set the fixed images. */
      for ( unsigned int i = 0; i < this->GetNumberOfFixedImages(); ++i )
      {
        this->m_ImageSampler->SetInput( i, this->m_FixedImageVector[ i ] );
      }

      /** Initialize the Image Sampler: set the fixed image masks. */
      for ( unsigned int i = 0; i < this->GetNumberOfFixedImageMasks(); ++i )
      {
        this->m_ImageSampler->SetMask( this->m_FixedImageMaskVector[ i ], i );
      }

      /** Initialize the Image Sampler: set the fixed image regions. */
      for ( unsigned int i = 0; i < this->GetNumberOfFixedImages(); ++i )
      {
        this->m_ImageSampler->SetInputImageRegion( this->m_FixedImageRegionVector[ i ], i );
      }
    }

  } // end InitializeImageSampler()


  /**
   * ******************* EvaluateMovingImageValueAndDerivative ******************
   */

  template <class TFixedImage, class TMovingImage>
  bool
  MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>
  ::EvaluateMovingImageValueAndDerivative(
    const MovingImagePointType & mappedPoint,
    RealType & movingImageValue,
    MovingImageDerivativeType * gradient ) const
  {
    /** Check if the mapped point is inside the moving image buffers of the feature images. */
    bool sampleOk = true;
    for ( unsigned int i = 1; i < this->GetNumberOfInterpolators(); ++i )
    {
      sampleOk &= this->GetInterpolator( i )->IsInsideBuffer( mappedPoint );

      /** If not inside this buffer we can quit. */
      if ( !sampleOk ) return false;
    }

    /** Compute value and possibly derivative of the moving image. */
    return this->Superclass::EvaluateMovingImageValueAndDerivative(
      mappedPoint, movingImageValue, gradient );

  } // end EvaluateMovingImageValueAndDerivative()


  /**
   * ************************ IsInsideMovingMask *************************
   */

  template <class TFixedImage, class TMovingImage>
  bool
  MultiInputImageToImageMetricBase<TFixedImage,TMovingImage>
  ::IsInsideMovingMask( const MovingImagePointType & mappedPoint ) const
  {
    /** If no moving image masks are present 'true' is returned,
     * meaning that this sample is taken into account. Otherwise, the
     * AND of all masks is returned, i.e. the sample should be inside
     * all masks.
     */
    bool inside = true;
    for ( unsigned int i = 0; i < this->GetNumberOfMovingImageMasks(); ++i )
    {
      MovingImageMaskPointer movingImageMask = this->GetMovingImageMask( i );
      if ( movingImageMask.IsNotNull() )
      {
        inside &= movingImageMask->IsInside( mappedPoint );
      }

      /** If the point falls outside one mask, we can skip the rest. */
      if ( !inside )
      {
        return false;
      }
    }
    return inside;

  } // end IsInsideMovingMask()


} // end namespace itk

#undef itkImplementationSetObjectMacro
#undef itkImplementationSetObjectMacro2
#undef itkImplementationGetObjectMacro
#undef itkImplementationGetConstObjectMacro

#endif // end #ifndef _itkMultiInputImageToImageMetricBase_txx

