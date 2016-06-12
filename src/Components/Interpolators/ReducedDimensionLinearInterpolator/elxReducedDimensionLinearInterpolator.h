/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __elxReducedDimensionLinearInterpolator_h
#define __elxReducedDimensionLinearInterpolator_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkReducedDimensionLinearInterpolateImageFunction.h"

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
class ReducedDimensionLinearInterpolator :
  public itk::ReducedDimensionLinearInterpolateImageFunction<
  typename InterpolatorBase< TElastix >::InputImageType,
  typename InterpolatorBase< TElastix >::CoordRepType >,
  public InterpolatorBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef ReducedDimensionLinearInterpolator Self;
  typedef itk::ReducedDimensionLinearInterpolateImageFunction<
    typename InterpolatorBase< TElastix >::InputImageType,
    typename InterpolatorBase< TElastix >::CoordRepType >
    Superclass1;
  typedef InterpolatorBase< TElastix >    Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ReducedDimensionLinearInterpolator, itk::ReducedDimensionLinearInterpolateImageFunction );

  /** Name of this class.
   * Use this name in the parameter file to select this specific interpolator. \n
   * example: <tt>(Interpolator "LinearInterpolator")</tt>\n
   */
  elxClassNameMacro( "ReducedDimensionLinearInterpolator" );

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
  ReducedDimensionLinearInterpolator() {}
  /** The destructor. */
  virtual ~ReducedDimensionLinearInterpolator() {}

private:

  /** The private constructor. */
  ReducedDimensionLinearInterpolator( const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );      // purposely not implemented

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxReducedDimensionLinearInterpolator.hxx"
#endif

#endif // end #ifndef __elxReducedDimensionLinearInterpolator_h
