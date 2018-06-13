/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxBSplineInterpolatorFloat_h
#define __elxBSplineInterpolatorFloat_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkBSplineInterpolateImageFunction.h"

namespace elastix
{

/**
 * \class BSplineInterpolatorFloat
 * \brief An interpolator based on the itk::BSplineInterpolateImageFunction.
 *
 * This interpolator interpolates images with an underlying B-spline
 * polynomial.
 *
 * NB: BSplineInterpolation with order 1 is slower than using a LinearInterpolator,
 * but it determines the derivative slightly more accurate at grid points. That's
 * why the registration results can be slightly different.
 *
 * The parameters used in this class are:
 * \parameter Interpolator: Select this interpolator as follows:\n
 *    <tt>(Interpolator "BSplineInterpolatorFloat")</tt>
 * \parameter BSplineInterpolationOrder: the order of the B-spline polynomial. \n
 *    example: <tt>(BSplineInterpolationOrder 3 2 3)</tt> \n
 *    The default order is 1. The parameter can be specified for each resolution.\n
 *    If only given for one resolution, that value is used for the other resolutions as well.
 *
 * \ingroup Interpolators
 */

template< class TElastix >
class BSplineInterpolatorFloat :
  public
  itk::BSplineInterpolateImageFunction<
  typename InterpolatorBase< TElastix >::InputImageType,
  typename InterpolatorBase< TElastix >::CoordRepType,
  float >,        //CoefficientType
  public
  InterpolatorBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef BSplineInterpolatorFloat Self;
  typedef itk::BSplineInterpolateImageFunction<
    typename InterpolatorBase< TElastix >::InputImageType,
    typename InterpolatorBase< TElastix >::CoordRepType,
    float >                                   Superclass1;
  typedef InterpolatorBase< TElastix >    Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( BSplineInterpolatorFloat, BSplineInterpolateImageFunction );

  /** Name of this class.
   * Use this name in the parameter file to select this specific interpolator. \n
   * example: <tt>(Interpolator "BSplineInterpolatorFloat")</tt>\n
   */
  elxClassNameMacro( "BSplineInterpolatorFloat" );

  /** Get the ImageDimension. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::OutputType               OutputType;
  typedef typename Superclass1::InputImageType           InputImageType;
  typedef typename Superclass1::IndexType                IndexType;
  typedef typename Superclass1::ContinuousIndexType      ContinuousIndexType;
  typedef typename Superclass1::PointType                PointType;
  typedef typename Superclass1::Iterator                 Iterator;
  typedef typename Superclass1::CoefficientDataType      CoefficientDataType;
  typedef typename Superclass1::CoefficientImageType     CoefficientImageType;
  typedef typename Superclass1::CoefficientFilter        CoefficientFilter;
  typedef typename Superclass1::CoefficientFilterPointer CoefficientFilterPointer;
  typedef typename Superclass1::CovariantVectorType      CovariantVectorType;

  /** Typedefs inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Execute stuff before each new pyramid resolution:
   * \li Set the spline order.
   */
  virtual void BeforeEachResolution( void );

protected:

  /** The constructor. */
  BSplineInterpolatorFloat() {}
  /** The destructor. */
  virtual ~BSplineInterpolatorFloat() {}

private:

  /** The private constructor. */
  BSplineInterpolatorFloat( const Self & );   // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );             // purposely not implemented

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxBSplineInterpolatorFloat.hxx"
#endif

#endif // end #ifndef __elxBSplineInterpolatorFloat_h
