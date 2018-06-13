/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxLinearInterpolator_h
#define __elxLinearInterpolator_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkLinearInterpolateImageFunction.h"

namespace elastix
{

/**
 * \class LinearInterpolator
 * \brief An interpolator based on the itk::AdvancedLinearInterpolateImageFunction.
 *
 * This interpolator interpolates images using linear interpolation.
 * In principle, this is the same as using the BSplineInterpolator with
 * the setting (BSplineInterpolationOrder 1). However, the LinearInterpolator
 * is significantly faster.
 *
 * The parameters used in this class are:
 * \parameter Interpolator: Select this interpolator as follows:\n
 *    <tt>(Interpolator "LinearInterpolator")</tt>
 *
 * \ingroup Interpolators
 */

template< class TElastix >
class LinearInterpolator :
  public itk::AdvancedLinearInterpolateImageFunction<
  typename InterpolatorBase< TElastix >::InputImageType,
  typename InterpolatorBase< TElastix >::CoordRepType >,
  public InterpolatorBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef LinearInterpolator Self;
  typedef itk::AdvancedLinearInterpolateImageFunction<
    typename InterpolatorBase< TElastix >::InputImageType,
    typename InterpolatorBase< TElastix >::CoordRepType >
    Superclass1;
  typedef InterpolatorBase< TElastix >    Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( LinearInterpolator, itk::AdvancedLinearInterpolateImageFunction );

  /** Name of this class.
   * Use this name in the parameter file to select this specific interpolator. \n
   * example: <tt>(Interpolator "LinearInterpolator")</tt>\n
   */
  elxClassNameMacro( "LinearInterpolator" );

  /** Get the ImageDimension. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::OutputType          OutputType;
  typedef typename Superclass1::InputImageType      InputImageType;
  typedef typename Superclass1::IndexType           IndexType;
  typedef typename Superclass1::ContinuousIndexType ContinuousIndexType;
  typedef typename Superclass1::PointType           PointType;

  /** Typedefs inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

protected:

  /** The constructor. */
  LinearInterpolator() {}
  /** The destructor. */
  virtual ~LinearInterpolator() {}

private:

  /** The private constructor. */
  LinearInterpolator( const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );      // purposely not implemented

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxLinearInterpolator.hxx"
#endif

#endif // end #ifndef __elxLinearInterpolator_h
