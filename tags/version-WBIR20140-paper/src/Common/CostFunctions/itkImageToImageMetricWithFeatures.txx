/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkImageToImageMetricWithFeatures_txx
#define _itkImageToImageMetricWithFeatures_txx

#include "itkImageToImageMetricWithFeatures.h"


namespace itk
{

  /**
   * ********************* Constructor ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::ImageToImageMetricWithFeatures()
  {
    this->m_NumberOfFixedFeatureImages = 0;
    this->m_NumberOfMovingFeatureImages = 0;

  } // end Constructor


  /**
   * ********************* Initialize *****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::Initialize(void) throw ( ExceptionObject )
  {
    /** Call the superclass. */
    this->Superclass::Initialize();

    /** Check the fixed stuff. */
    for ( unsigned int i = 0; i < m_NumberOfFixedFeatureImages; i++ )
    {
      /** Check if all the fixed feature images are set. */
      if ( !this->m_FixedFeatureImages[ i ] )
      {
        itkExceptionMacro( << "ERROR: fixed feature image " << i << " is not set." );
      }
      /** Check if all the fixed feature interpolators are set. */
      if ( !this->m_FixedFeatureInterpolators[ i ] )
      {
        itkExceptionMacro( << "ERROR: fixed feature interpolator " << i << " is not set." );
      }
      /** Connect the feature image to the interpolator. */
      this->m_FixedFeatureInterpolators[ i ]->SetInputImage( this->m_FixedFeatureImages[ i ] );
    }

    /** Check the moving stuff. */
    for ( unsigned int i = 0; i < m_NumberOfMovingFeatureImages; i++ )
    {
      /** Check if all the moving feature images are set. */
      if ( !this->m_MovingFeatureImages[ i ] )
      {
        itkExceptionMacro( << "ERROR: moving feature image " << i << " is not set." );
      }
      /** Check if all the moving feature interpolators are set. */
      if ( !this->m_MovingFeatureInterpolators[ i ] )
      {
        itkExceptionMacro( << "ERROR: moving feature interpolator " << i << " is not set." );
      }
      /** Connect the feature image to the interpolator. */
      this->m_MovingFeatureInterpolators[ i ]->SetInputImage( this->m_MovingFeatureImages[ i ] );
    }

    /** Check if the moving feature image interpolators are B-spline interpolators. */
    this->CheckForBSplineFeatureInterpolators();

  } // end Initialize()


  /**
   * ********************* SetNumberOfFixedFeatureImages ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetNumberOfFixedFeatureImages( unsigned int arg )
  {
    if ( this->m_NumberOfFixedFeatureImages != arg )
    {
      this->m_FixedFeatureImages.resize( arg );
      this->m_FixedFeatureInterpolators.resize( arg );
      this->m_NumberOfFixedFeatureImages = arg;
      this->Modified();
    }

  } // end SetNumberOfFixedFeatureImages


  /**
   * ********************* SetFixedFeatureImage ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetFixedFeatureImage( unsigned int i, FixedFeatureImageType * im )
  {
    if ( i + 1 > this->m_NumberOfFixedFeatureImages )
    {
      this->m_FixedFeatureImages.resize( i + 1 );
      this->m_FixedFeatureImages[ i ] = im;
      this->m_NumberOfFixedFeatureImages = i;
      this->Modified();
    }
    else
    {
      if ( this->m_FixedFeatureImages[ i ] != im )
      {
        this->m_FixedFeatureImages[ i ] = im;
        this->Modified();
      }
    }

  } // end SetFixedFeatureImage


  /**
   * ********************* GetFixedFeatureImage ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    const typename ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,
      TFixedFeatureImage,TMovingFeatureImage>::FixedFeatureImageType *
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetFixedFeatureImage( unsigned int i ) const
  {
    return this->m_FixedFeatureImages[ i ].GetPointer();
  } // end GetFixedFeatureImage


  /**
   * ********************* SetFixedFeatureInterpolator ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetFixedFeatureInterpolator( unsigned int i, FixedFeatureInterpolatorType * interpolator )
  {
    if ( i + 1 > this->m_NumberOfFixedFeatureImages )
    {
      this->m_FixedFeatureInterpolators.resize( i + 1 );
      this->m_FixedFeatureInterpolators[ i ] = interpolator;
      this->m_NumberOfFixedFeatureImages = i;
      this->Modified();
    }
    else
    {
      if ( this->m_FixedFeatureInterpolators[ i ] != interpolator )
      {
        this->m_FixedFeatureInterpolators[ i ] = interpolator;
        this->Modified();
      }
    }

  } // end SetFixedFeatureInterpolator


  /**
   * ********************* GetFixedFeatureInterpolator ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    const typename ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,
      TFixedFeatureImage,TMovingFeatureImage>::FixedFeatureInterpolatorType *
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetFixedFeatureInterpolator( unsigned int i ) const
  {
    return this->m_FixedFeatureInterpolators[ i ].GetPointer();
  } // end GetFixedFeatureInterpolator


  /**
   * ********************* SetNumberOfMovingFeatureImages ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetNumberOfMovingFeatureImages( unsigned int arg )
  {
    if ( this->m_NumberOfMovingFeatureImages != arg )
    {
      this->m_MovingFeatureImages.resize( arg );
      this->m_MovingFeatureInterpolators.resize( arg );
      this->m_NumberOfMovingFeatureImages = arg;
      this->Modified();
    }

  } // end SetNumberOfMovingFeatureImages


  /**
   * ********************* SetMovingFeatureImage ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetMovingFeatureImage( unsigned int i, MovingFeatureImageType * im )
  {
    if ( i + 1 > this->m_NumberOfMovingFeatureImages )
    {
      this->m_MovingFeatureImages.resize( i + 1 );
      this->m_MovingFeatureImages[ i ] = im;
      this->m_NumberOfMovingFeatureImages = i;
      this->Modified();
    }
    else
    {
      if ( this->m_MovingFeatureImages[ i ] != im )
      {
        this->m_MovingFeatureImages[ i ] = im;
        this->Modified();
      }
    }

  } // end SetMovingFeatureImage


  /**
   * ********************* GetMovingFeatureImage ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    const typename ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,
      TFixedFeatureImage,TMovingFeatureImage>::MovingFeatureImageType *
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetMovingFeatureImage( unsigned int i ) const
  {
    return this->m_MovingFeatureImages[ i ].GetPointer();
  } // end GetMovingFeatureImage


  /**
   * ********************* SetMovingFeatureInterpolator ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::SetMovingFeatureInterpolator( unsigned int i, MovingFeatureInterpolatorType * interpolator )
  {
    if ( i + 1 > this->m_NumberOfMovingFeatureImages )
    {
      this->m_MovingFeatureInterpolators.resize( i + 1 );
      this->m_MovingFeatureInterpolators[ i ] = interpolator;
      this->m_NumberOfMovingFeatureImages = i;
      this->Modified();
    }
    else
    {
      if ( this->m_MovingFeatureInterpolators[ i ] != interpolator )
      {
        this->m_MovingFeatureInterpolators[ i ] = interpolator;
        this->Modified();
      }
    }

  } // end SetMovingFeatureInterpolator


  /**
   * ********************* GetMovingFeatureInterpolator ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    const typename ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,
      TFixedFeatureImage,TMovingFeatureImage>::MovingFeatureInterpolatorType *
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::GetMovingFeatureInterpolator( unsigned int i ) const
  {
    return this->m_MovingFeatureInterpolators[ i ].GetPointer();
  } // end GetMovingFeatureInterpolator


  /**
   * ****************** CheckForBSplineFeatureInterpolators **********************
   */

  template <class TFixedImage, class TMovingImage,
    class TFixedFeatureImage, class TMovingFeatureImage>
  void
  ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::CheckForBSplineFeatureInterpolators( void )
  {
    /** Check if the interpolators are of type BSplineInterpolateImageFunction.
     * If so, we can make use of its EvaluateDerivatives method.
     * Otherwise, an exception is thrown.
     */
    this->m_FeatureInterpolatorsIsBSpline.resize( this->m_NumberOfMovingFeatureImages, false );
    this->m_FeatureInterpolatorsAreBSpline = true;
    this->m_MovingFeatureBSplineInterpolators.resize( this->m_NumberOfMovingFeatureImages );
    for ( unsigned int i = 0; i < this->m_NumberOfMovingFeatureImages; ++i )
    {
      BSplineInterpolatorType * testPtr =
        dynamic_cast<BSplineInterpolatorType *>(
        this->m_MovingFeatureInterpolators[ i ].GetPointer() );

      if ( testPtr )
      {
        this->m_FeatureInterpolatorsIsBSpline[ i ] = true;
        this->m_FeatureInterpolatorsAreBSpline &= true;
        this->m_MovingFeatureBSplineInterpolators[ i ] = testPtr;
        itkDebugMacro( << "Interpolator " << i << " is B-spline." );
      }
      else
      {
        itkDebugMacro( << "Interpolator " << i << " is NOT B-spline." );
        itkExceptionMacro( << "Interpolator " << i << " is NOT B-spline." );
      }
    } // end for-loop

  } // end CheckForBSplineFeatureInterpolators()


  /**
   * ********************* PrintSelf ****************************
   */

  template <class TFixedImage, class TMovingImage, class TFixedFeatureImage, class TMovingFeatureImage>
    void
    ImageToImageMetricWithFeatures<TFixedImage,TMovingImage,TFixedFeatureImage,TMovingFeatureImage>
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf( os, indent );

    os << indent << "NumberOfFixedFeatureImages: "  << this->m_NumberOfFixedFeatureImages << std::endl;
    os << indent << "NumberOfMovingFeatureImages: " << this->m_NumberOfMovingFeatureImages << std::endl;

    /** Print the feature image pointers. */
    for ( unsigned int i = 0; i < this->m_NumberOfFixedFeatureImages; i++ )
    {
      os << indent << "FixedFeatureImages[" << i << "]: "
        << this->m_FixedFeatureImages[ i ].GetPointer() << std::endl;
    }
    for ( unsigned int i = 0; i < this->m_NumberOfMovingFeatureImages; i++ )
    {
      os << indent << "MovingFeatureImages[" << i << "]: "
        << this->m_MovingFeatureImages[ i ].GetPointer() << std::endl;
    }

    /** Print the feature interpolators pointers. */
    for ( unsigned int i = 0; i < this->m_NumberOfFixedFeatureImages; i++ )
    {
      os << indent << "FixedFeatureInterpolators[" << i << "]: "
        << this->m_FixedFeatureInterpolators[ i ].GetPointer() << std::endl;
    }
    for ( unsigned int i = 0; i < this->m_NumberOfMovingFeatureImages; i++ )
    {
      os << indent << "MovingFeatureInterpolators[" << i << "]: "
        << this->m_MovingFeatureInterpolators[ i ].GetPointer() << std::endl;
    }

  } // end PrintSelf


} // end namespace itk


#endif // end #ifndef _itkImageToImageMetricWithFeatures_txx

