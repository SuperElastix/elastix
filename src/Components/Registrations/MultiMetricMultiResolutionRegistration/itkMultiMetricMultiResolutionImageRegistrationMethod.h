/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkMultiMetricMultiResolutionImageRegistrationMethod_h
#define __itkMultiMetricMultiResolutionImageRegistrationMethod_h

#include "itkMultiResolutionImageRegistrationMethod2.h"
#include "itkCombinationImageToImageMetric.h"
#include <vector>

/** defines a method that calls the same method
 * with an extra 0 argument */
#define itkSimpleSetMacro(_name,_type) \
  virtual void Set##_name( _type _arg ) \
  { \
    this->Set##_name ( _arg, 0 ); \
  }

/** defines for example: SetNumberOfInterpolators() */
#define itkSetNumberOfMacro(_name) \
  virtual void SetNumberOf##_name##s(unsigned int _arg) \
  { \
    if ( this->m_##_name##s.size() != _arg ) \
    { \
      this->m_##_name##s.resize( _arg ); \
      this->Modified(); \
    } \
  }

/** defines for example: GetNumberOfInterpolators() */
#define itkGetNumberOfMacro(_name) \
  virtual unsigned int GetNumberOf##_name##s(void) const \
  { \
    return this->m_##_name##s.size(); \
  }


namespace itk
{

/** \class MultiMetricMultiResolutionImageRegistrationMethod
 * \brief Base class for multi-resolution image registration methods
 *
 * This class is an extension of the itk class
 * MultiResolutionImageRegistrationMethod. It allows the use
 * of multiple metrics, which are summed, multiple images,
 * multiple interpolators, and/or multiple image pyramids.
 *
 * Make sure the following is true:\n
 *   nrofmetrics >= nrofinterpolators >= nrofmovingpyramids >= nrofmovingimages\n
 *   nrofmetrics >= nroffixedpyramids >= nroffixedimages\n
 *   nroffixedregions == nroffixedimages\n
 *
 *   nrofinterpolators == nrofmetrics OR nrofinterpolators == 1\n
 *   nroffixedimages == nrofmetrics OR nroffixedimages == 1\n
 *   etc...
 *
 * You may also set an interpolator/fixedimage/etc to NULL, if you
 * happen to know that the corresponding metric is not an
 * ImageToImageMetric, but a regularizer for example (which does
 * not need an image.
 *
 *
 * \sa ImageRegistrationMethod
 * \sa MultiResolutionImageRegistrationMethod
 * \ingroup RegistrationFilters
 */

template <typename TFixedImage, typename TMovingImage>
class MultiMetricMultiResolutionImageRegistrationMethod :
  public MultiResolutionImageRegistrationMethod2<TFixedImage, TMovingImage>
{
public:
  /** Standard class typedefs. */
  typedef MultiMetricMultiResolutionImageRegistrationMethod  Self;
  typedef MultiResolutionImageRegistrationMethod2<
    TFixedImage, TMovingImage>                               Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( MultiMetricMultiResolutionImageRegistrationMethod,
    MultiResolutionImageRegistrationMethod2 );

  /**  Superclass types */
  typedef typename Superclass::FixedImageType           FixedImageType;
  typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType     FixedImageRegionType;
  typedef typename Superclass::MovingImageType          MovingImageType;
  typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;

  typedef typename Superclass::MetricType               MetricType;
  typedef typename Superclass::MetricPointer            MetricPointer;
  typedef typename Superclass::TransformType            TransformType;
  typedef typename Superclass::TransformPointer         TransformPointer;
  typedef typename Superclass::InterpolatorType         InterpolatorType;
  typedef typename Superclass::InterpolatorPointer      InterpolatorPointer;
  typedef typename Superclass::OptimizerType            OptimizerType;
  typedef typename OptimizerType::Pointer               OptimizerPointer;
  typedef typename Superclass::FixedImagePyramidType    FixedImagePyramidType;
  typedef typename Superclass::FixedImagePyramidPointer FixedImagePyramidPointer;
  typedef typename Superclass::MovingImagePyramidType   MovingImagePyramidType;
  typedef typename
    Superclass::MovingImagePyramidPointer               MovingImagePyramidPointer;

  typedef typename Superclass::TransformOutputType      TransformOutputType;
  typedef typename Superclass::TransformOutputPointer   TransformOutputPointer;
  typedef typename
    Superclass::TransformOutputConstPointer             TransformOutputConstPointer;

  typedef typename Superclass::ParametersType           ParametersType;
  typedef typename Superclass::DataObjectPointer        DataObjectPointer;

  /** Extra typedefs */
  typedef CombinationImageToImageMetric<
    FixedImageType, MovingImageType >                   CombinationMetricType;
  typedef typename CombinationMetricType::Pointer       CombinationMetricPointer;

  /** Unfortunately the StopRegistration method is not virtual and
   * the m_Stop member is private in the superclass. That's why
   * we provide the following function to interrupt registration.
   */
  virtual void StopMultiMetricRegistration( void )
  {
    this->m_Stop = true;
  }

  /** Set the Metric. Reimplement this method to check if
   * the metric is a combination metric.
   * GetMetric returns the combination metric.
   * By default, a combination metric is already set on constructing
   * this class.
   */
  virtual void SetMetric( MetricType * _arg );

  /** Get the metric as a pointer to a combination metric type.
   * Use this method to setup the combination metric (set weights,
   * nrofmetrics, submetrics, etc.
   */
  virtual CombinationMetricType * GetCombinationMetric( void ) const
  {
    return this->m_CombinationMetric.GetPointer();
  }

  /** The following methods all have a similar pattern. The
   * SetFixedImage() just calls SetFixedImage(0).
   * SetFixedImage(0) also calls the Superclass::SetFixedImage(). This
   * is defined by the itkSimpleSetMacro.
   * GetFixedImage() just returns GetFixedImage(0) == Superclass::m_FixedImage.
   */

  /** Set/Get the fixed image. */
  virtual void SetFixedImage( const FixedImageType * _arg, unsigned int pos );
  virtual const FixedImageType * GetFixedImage( unsigned int pos ) const;
  virtual const FixedImageType * GetFixedImage( void ) const
  {
    return this->GetFixedImage(0);
  }
  itkSimpleSetMacro( FixedImage, const FixedImageType * );
  itkSetNumberOfMacro( FixedImage );
  itkGetNumberOfMacro( FixedImage );

  /** Set/Get the moving image. */
  virtual void SetMovingImage( const MovingImageType * _arg, unsigned int pos );
  virtual const MovingImageType * GetMovingImage( unsigned int pos ) const;
  virtual const MovingImageType * GetMovingImage( void ) const
  { return this->GetMovingImage(0); }
  itkSimpleSetMacro( MovingImage, const MovingImageType * );
  itkSetNumberOfMacro( MovingImage );
  itkGetNumberOfMacro( MovingImage );

  /** Set/Get the fixed image region. */
  virtual void SetFixedImageRegion( FixedImageRegionType _arg, unsigned int pos );
  virtual const FixedImageRegionType & GetFixedImageRegion( unsigned int pos ) const;
  virtual const FixedImageRegionType & GetFixedImageRegion( void ) const
  { return this->GetFixedImageRegion(0); }
  itkSimpleSetMacro( FixedImageRegion, const FixedImageRegionType );
  itkSetNumberOfMacro( FixedImageRegion );
  itkGetNumberOfMacro( FixedImageRegion );

  /** Set/Get the interpolator. */
  virtual void SetInterpolator( InterpolatorType * _arg, unsigned int pos );
  virtual InterpolatorType * GetInterpolator( unsigned int pos ) const;
  virtual InterpolatorType * GetInterpolator( void )
  { return this->GetInterpolator(0); }
  itkSimpleSetMacro( Interpolator, InterpolatorType * );
  itkSetNumberOfMacro( Interpolator );
  itkGetNumberOfMacro( Interpolator );

  /** Set/Get the FixedImagePyramid. */
  virtual void SetFixedImagePyramid( FixedImagePyramidType * _arg, unsigned int pos );
  virtual FixedImagePyramidType * GetFixedImagePyramid( unsigned int pos ) const;
  virtual FixedImagePyramidType * GetFixedImagePyramid( void )
  { return this->GetFixedImagePyramid(0); }
  itkSimpleSetMacro( FixedImagePyramid, FixedImagePyramidType * );
  itkSetNumberOfMacro( FixedImagePyramid );
  itkGetNumberOfMacro( FixedImagePyramid );

  /** Set/Get the MovingImagePyramid. */
  virtual void SetMovingImagePyramid( MovingImagePyramidType * _arg, unsigned int pos );
  virtual MovingImagePyramidType * GetMovingImagePyramid( unsigned int pos ) const;
  virtual MovingImagePyramidType * GetMovingImagePyramid( void )
  { return this->GetMovingImagePyramid(0); }
  itkSimpleSetMacro( MovingImagePyramid, MovingImagePyramidType * );
  itkSetNumberOfMacro( MovingImagePyramid );
  itkGetNumberOfMacro( MovingImagePyramid );

  /** Method to return the latest modified time of this object or
   * any of its cached ivars.
   */
  unsigned long GetMTime( void ) const;

  /** Get the last transformation parameters visited by
   * the optimizer. Return the member variable declared in this class,
   * and not that of the superclass (which is declared private).
   */
  virtual const ParametersType & GetLastTransformParameters( void ) const
  {
    return this->m_LastTransformParameters;
  }

protected:
  MultiMetricMultiResolutionImageRegistrationMethod();
  virtual ~MultiMetricMultiResolutionImageRegistrationMethod() {};
  void PrintSelf( std::ostream& os, Indent indent ) const;

  typedef std::vector<FixedImageRegionType>  FixedImageRegionPyramidType;

  /** Method invoked by the pipeline in order to trigger the computation of
   * the registration.
   */
  virtual void GenerateData( void );

  /** Initialize by setting the interconnects between the components.
   * This method is executed at every level of the pyramid with the
   * values corresponding to this resolution.
   */
  virtual void Initialize( void ) throw (ExceptionObject);

  /** Compute the size of the fixed region for each level of the pyramid.
   * Actually we would like to override PreparePyramids, but this function
   * is not virtual...
   */
  virtual void PrepareAllPyramids( void );

  /** Function called by PrepareAllPyramids, which checks if the user input
   * regarding the image pyramids is ok.
   */
  virtual void CheckPyramids( void ) throw (ExceptionObject);

  /** Function called by Initialize, which checks if the user input
   * is ok. Called by Initialize().
   */
  virtual void CheckOnInitialize( void ) throw (ExceptionObject);

  /** Variables already defined in the superclass, but as private...  */
  bool m_Stop;
  ParametersType m_LastTransformParameters;

  /** A shortcut to m_Metric of type CombinationMetricPointer. */
  CombinationMetricPointer m_CombinationMetric;

  /** Containers for the pointers supplied by the user. */
  std::vector<FixedImageConstPointer>       m_FixedImages;
  std::vector<MovingImageConstPointer>      m_MovingImages;
  std::vector<FixedImageRegionType>         m_FixedImageRegions;
  std::vector<FixedImagePyramidPointer>     m_FixedImagePyramids;
  std::vector<MovingImagePyramidPointer>    m_MovingImagePyramids;
  std::vector<InterpolatorPointer>          m_Interpolators;

  /** This vector is filled by the PrepareAllPyramids function. */
  std::vector<FixedImageRegionPyramidType>  m_FixedImageRegionPyramids;

  /** Dummy image region. */
  FixedImageRegionType m_NullFixedImageRegion;

private:
  MultiMetricMultiResolutionImageRegistrationMethod(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; // end class


} // end namespace itk

#undef itkSetNumberOfMacro
#undef itkGetNumberOfMacro
#undef itkSimpleSetMacro

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiMetricMultiResolutionImageRegistrationMethod.hxx"
#endif

#endif
