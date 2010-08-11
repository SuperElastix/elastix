/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkMultiInputImageToImageMetricBase_h
#define __itkMultiInputImageToImageMetricBase_h

#include "itkAdvancedImageToImageMetric.h"
#include <vector>

/** Macro for setting the number of objects. */
#define itkSetNumberOfMacro( name ) \
  virtual void SetNumberOf##name##s( const unsigned int _arg ) \
  { \
    if ( this->m_NumberOf##name##s != _arg ) \
    { \
      this->m_##name##Vector.resize( _arg ); \
      this->m_NumberOf##name##s = _arg; \
      this->Modified(); \
    } \
  } // comments for allowing ; after calling the macro


namespace itk
{

/** \class MultiInputImageToImageMetricBase
 *
 * \brief Implements a metric base class that takes multiple inputs.
 *
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedImage, class TMovingImage>
class MultiInputImageToImageMetricBase :
  public AdvancedImageToImageMetric< TFixedImage, TMovingImage >
{
public:

  /** Standard class typedefs. */
  typedef MultiInputImageToImageMetricBase    Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >               Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( MultiInputImageToImageMetricBase, AdvancedImageToImageMetric );

  /** Constants for the image dimensions */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    TMovingImage::ImageDimension );
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    TFixedImage::ImageDimension );

  /** Typedefs from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImagePointer         MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImagePointer          FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::InterpolatorType           InterpolatorType;
  typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::ParametersType             ParametersType;

  typedef InterpolateImageFunction<
    FixedImageType, CoordinateRepresentationType >        FixedImageInterpolatorType;
  typedef typename FixedImageInterpolatorType::Pointer    FixedImageInterpolatorPointer;

  /** Typedef's for storing multiple inputs. */
  typedef std::vector< FixedImageConstPointer >           FixedImageVectorType;
  typedef std::vector< FixedImageMaskPointer >            FixedImageMaskVectorType;
  typedef std::vector< FixedImageRegionType >             FixedImageRegionVectorType;
  typedef std::vector< MovingImageConstPointer >          MovingImageVectorType;
  typedef std::vector< MovingImageMaskPointer >           MovingImageMaskVectorType;
  typedef std::vector< InterpolatorPointer >              InterpolatorVectorType;
  typedef std::vector< FixedImageInterpolatorPointer >    FixedImageInterpolatorVectorType;

  /** ******************** Fixed images ******************** */

  /** Set the fixed images. */
  virtual void SetFixedImage( const FixedImageType *_arg, unsigned int pos );

  /** Set the first fixed image. */
  virtual void SetFixedImage( const FixedImageType *_arg )
  {
    this->SetFixedImage( _arg, 0 );
  };

  /** Get the fixed images. */
  virtual const FixedImageType * GetFixedImage( unsigned int pos ) const;

  /** Get the first fixed image. */
  virtual const FixedImageType * GetFixedImage( void ) const
  {
    return this->GetFixedImage( 0 );
  };

  /** Set the number of fixed images. */
  itkSetNumberOfMacro( FixedImage );

  /** Get the number of fixed images. */
  itkGetConstMacro( NumberOfFixedImages, unsigned int );

  /** ******************** Fixed image masks ******************** */

  /** Set the fixed image masks. */
  virtual void SetFixedImageMask( FixedImageMaskType *_arg, unsigned int pos );

  /** Set the first fixed image mask. */
  virtual void SetFixedImageMask( FixedImageMaskType *_arg )
  {
    this->SetFixedImageMask( _arg, 0 );
  };

  /** Get the fixed image masks. */
  virtual FixedImageMaskType * GetFixedImageMask( unsigned int pos ) const;

  /** Get the first fixed image mask. */
  virtual FixedImageMaskType * GetFixedImageMask( void ) const
  {
    return this->GetFixedImageMask( 0 );
  };

  /** Set the number of fixed image masks. */
  itkSetNumberOfMacro( FixedImageMask );

  /** Get the number of fixed image masks. */
  itkGetConstMacro( NumberOfFixedImageMasks, unsigned int );

  /** ******************** Fixed image regions ******************** */

  /** Set the fixed image regions. */
  virtual void SetFixedImageRegion( const FixedImageRegionType _arg, unsigned int pos );

  /** Set the first fixed image region. */
  virtual void SetFixedImageRegion( const FixedImageRegionType _arg )
  {
    this->SetFixedImageRegion( _arg, 0 );
  };

  /** Get the fixed image regions. */
  virtual const FixedImageRegionType & GetFixedImageRegion( unsigned int pos ) const;

  /** Get the first fixed image region. */
  virtual const FixedImageRegionType & GetFixedImageRegion( void ) const
  {
    return this->GetFixedImageRegion( 0 );
  };

  /** Set the number of fixed image regions. */
  itkSetNumberOfMacro( FixedImageRegion );

  /** Get the number of fixed image regions. */
  itkGetConstMacro( NumberOfFixedImageRegions, unsigned int );

  /** ******************** Moving images ******************** */

  /** Set the moving images. */
  virtual void SetMovingImage( const MovingImageType *_arg, unsigned int pos );

  /** Set the first moving image. */
  virtual void SetMovingImage( const MovingImageType *_arg )
  {
    this->SetMovingImage( _arg, 0 );
  };

  /** Get the moving images. */
  virtual const MovingImageType * GetMovingImage( unsigned int pos ) const;

  /** Get the first moving image. */
  virtual const MovingImageType * GetMovingImage( void ) const
  {
    return this->GetMovingImage( 0 );
  };

  /** Set the number of moving images. */
  itkSetNumberOfMacro( MovingImage );

  /** Get the number of moving images. */
  itkGetConstMacro( NumberOfMovingImages, unsigned int );

  /** ******************** Moving image masks ******************** */

  /** Set the moving image masks. */
  virtual void SetMovingImageMask( MovingImageMaskType *_arg, unsigned int pos );

  /** Set the first moving image mask. */
  virtual void SetMovingImageMask( MovingImageMaskType *_arg )
  {
    this->SetMovingImageMask( _arg, 0 );
  };

  /** Get the moving image masks. */
  virtual MovingImageMaskType * GetMovingImageMask( unsigned int pos ) const;

  /** Get the first moving image mask. */
  virtual MovingImageMaskType * GetMovingImageMask( void ) const
  {
    return this->GetMovingImageMask( 0 );
  };

  /** Set the number of moving image masks. */
  itkSetNumberOfMacro( MovingImageMask );

  /** Get the number of moving image masks. */
  itkGetConstMacro( NumberOfMovingImageMasks, unsigned int );

  /** ******************** Interpolators ********************
   * These interpolators are used for the moving images.
   */

  /** Set the interpolators. */
  virtual void SetInterpolator( InterpolatorType *_arg, unsigned int pos );

  /** Set the first interpolator. */
  virtual void SetInterpolator( InterpolatorType *_arg )
  {
    return this->SetInterpolator( _arg, 0 );
  };

  /** Get the interpolators. */
  virtual InterpolatorType * GetInterpolator( unsigned int pos ) const;

  /** Get the first interpolator. */
  virtual InterpolatorType * GetInterpolator( void ) const
  {
    return this->GetInterpolator( 0 );
  };

  /** Set the number of interpolators. */
  itkSetNumberOfMacro( Interpolator );

  /** Get the number of interpolators. */
  itkGetConstMacro( NumberOfInterpolators, unsigned int );

  /** A function to check if all moving image interpolators are of type B-spline. */
  itkGetConstMacro( InterpolatorsAreBSpline, bool );

  /** ******************** FixedImageInterpolators ********************
   * These interpolators are used for the fixed images.
   */

  /** Set the fixed image interpolators. */
  virtual void SetFixedImageInterpolator( FixedImageInterpolatorType *_arg, unsigned int pos );

  /** Set the first fixed image interpolator. */
  virtual void SetFixedImageInterpolator( FixedImageInterpolatorType *_arg )
  {
    return this->SetFixedImageInterpolator( _arg, 0 );
  };

  /** Get the fixed image interpolators. */
  virtual FixedImageInterpolatorType * GetFixedImageInterpolator( unsigned int pos ) const;

  /** Get the first fixed image interpolator. */
  virtual FixedImageInterpolatorType * GetFixedImageInterpolator( void ) const
  {
    return this->GetFixedImageInterpolator( 0 );
  };

  /** Set the number of fixed image interpolators. */
  itkSetNumberOfMacro( FixedImageInterpolator );

  /** Get the number of fixed image interpolators. */
  itkGetConstMacro( NumberOfFixedImageInterpolators, unsigned int );

  /** ******************** Other public functions ******************** */

  /** Initialisation. */
  virtual void Initialize( void ) throw ( ExceptionObject );

protected:

  /** Constructor. */
  MultiInputImageToImageMetricBase();

  /** Destructor. */
  virtual ~MultiInputImageToImageMetricBase() {};

  /** Typedef's from the Superclass. */
  typedef typename Superclass::MovingImagePointType       MovingImagePointType;
  typedef typename Superclass::MovingImageIndexType       MovingImageIndexType;
  typedef typename Superclass::MovingImageDerivativeType  MovingImageDerivativeType;
  typedef typename Superclass::MovingImageContinuousIndexType MovingImageContinuousIndexType;

  /** Typedef's for the moving image interpolators. */
  typedef typename Superclass::BSplineInterpolatorType    BSplineInterpolatorType;
  typedef typename BSplineInterpolatorType::Pointer       BSplineInterpolatorPointer;
  typedef std::vector<BSplineInterpolatorPointer>         BSplineInterpolatorVectorType;

  /** Initialize variables related to the image sampler; called by Initialize. */
  virtual void InitializeImageSampler( void ) throw ( ExceptionObject );

  /** Check if all interpolators (for the moving image) are of type
   * BSplineInterpolateImageFunction.
   */
  virtual void CheckForBSplineInterpolators( void );

  /** Check if mappedPoint is inside all moving images.
   * If so, the moving image value and possibly derivative are computed.
   */
  virtual bool EvaluateMovingImageValueAndDerivative(
    const MovingImagePointType & mappedPoint,
    RealType & movingImageValue,
    MovingImageDerivativeType * gradient ) const;

  /** IsInsideMovingMask: Returns the AND of all moving image masks. */
  virtual bool IsInsideMovingMask(
    const MovingImagePointType & mappedPoint ) const;

  /** Protected member variables. */
  FixedImageVectorType        m_FixedImageVector;
  FixedImageMaskVectorType    m_FixedImageMaskVector;
  FixedImageRegionVectorType  m_FixedImageRegionVector;
  MovingImageVectorType       m_MovingImageVector;
  MovingImageMaskVectorType   m_MovingImageMaskVector;
  InterpolatorVectorType      m_InterpolatorVector;
  FixedImageInterpolatorVectorType      m_FixedImageInterpolatorVector;

  bool m_InterpolatorsAreBSpline;
  BSplineInterpolatorVectorType  m_BSplineInterpolatorVector;

private:

  MultiInputImageToImageMetricBase(const Self&); // purposely not implemented
  void operator=(const Self&); // purposely not implemented

  /** Private member variables. */
  FixedImageRegionType m_DummyFixedImageRegion;

  unsigned int  m_NumberOfFixedImages;
  unsigned int  m_NumberOfFixedImageMasks;
  unsigned int  m_NumberOfFixedImageRegions;
  unsigned int  m_NumberOfMovingImages;
  unsigned int  m_NumberOfMovingImageMasks;
  unsigned int  m_NumberOfInterpolators;
  unsigned int  m_NumberOfFixedImageInterpolators;

}; // end class MultiInputImageToImageMetricBase

} // end namespace itk

#undef itkSetNumberOfMacro

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiInputImageToImageMetricBase.txx"
#endif

#endif // end #ifndef __itkMultiInputImageToImageMetricBase_h

