/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxBSplineResampleInterpolator_h
#define __elxBSplineResampleInterpolator_h

#include "itkBSplineInterpolateImageFunction.h"
#include "elxIncludes.h"

namespace elastix
{

  /**
  * \class BSplineResampleInterpolator
  * \brief A resample-interpolator based on B-splines.
  *
  * The parameters used in this class are:
  * \parameter ResampleInterpolator: Select this resample interpolator as follows:\n
  *   <tt>(ResampleInterpolator "FinalBSplineInterpolator")</tt>
  * \parameter FinalBSplineInterpolationOrder: the order of the B-spline used to resample
  *    the deformed moving image; possible values: (0-5) \n
  *    example: <tt>(FinalBSplineInterpolationOrder 3) </tt> \n
  *    Default: 3.
  *
  * The transform parameters necessary for transformix, additionally defined by this class, are:
  * \transformparameter FinalBSplineInterpolationOrder: the order of the B-spline used to resample
  *    the deformed moving image; possible values: (0-5) \n
  *    example: <tt>(FinalBSplineInterpolationOrder 3) </tt> \n
  *    Default: 3.
  *
  * With very large images, memory problems may be avoided by using the BSplineResampleInterpolatorFloat.
  * The differences of the result are generally negligible.
  * If you are really in memory problems, you may use the LinearResampleInterpolator,
  * or the NearestNeighborResampleInterpolator.
  *
  * \ingroup ResampleInterpolators
  * \sa BSplineResampleInterpolatorFloat
  */

  template < class TElastix >
  class BSplineResampleInterpolator :
    public
    itk::BSplineInterpolateImageFunction<
    typename ResampleInterpolatorBase<TElastix>::InputImageType,
    typename ResampleInterpolatorBase<TElastix>::CoordRepType,
    double >, //CoefficientType
    public ResampleInterpolatorBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef BSplineResampleInterpolator           Self;
    typedef itk::BSplineInterpolateImageFunction<
      typename ResampleInterpolatorBase<TElastix>::InputImageType,
      typename ResampleInterpolatorBase<TElastix>::CoordRepType,
      double >                                    Superclass1;
    typedef ResampleInterpolatorBase<TElastix>    Superclass2;
    typedef itk::SmartPointer<Self>               Pointer;
    typedef itk::SmartPointer<const Self>         ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( BSplineResampleInterpolator, itk::BSplineInterpolateImageFunction );

    /** Name of this class.
    * Use this name in the parameter file to select this specific resample interpolator. \n
    * example: <tt>(ResampleInterpolator "FinalBSplineInterpolator")</tt>\n
    */
    elxClassNameMacro( "FinalBSplineInterpolator" );

    /** Dimension of the image. */
    itkStaticConstMacro( ImageDimension, unsigned int,Superclass1::ImageDimension );

    /** Typedef's inherited from the superclass. */
    typedef typename Superclass1::OutputType                OutputType;
    typedef typename Superclass1::InputImageType            InputImageType;
    typedef typename Superclass1::IndexType                 IndexType;
    typedef typename Superclass1::ContinuousIndexType       ContinuousIndexType;
    typedef typename Superclass1::PointType                 PointType;
    typedef typename Superclass1::Iterator                  Iterator;
    typedef typename Superclass1::CoefficientDataType       CoefficientDataType;
    typedef typename Superclass1::CoefficientImageType      CoefficientImageType;
    typedef typename Superclass1::CoefficientFilter         CoefficientFilter;
    typedef typename Superclass1::CoefficientFilterPointer  CoefficientFilterPointer;
    typedef typename Superclass1::CovariantVectorType       CovariantVectorType;

    /** Typedef's from ResampleInterpolatorBase. */
    typedef typename Superclass2::ElastixType               ElastixType;
    typedef typename Superclass2::ElastixPointer            ElastixPointer;
    typedef typename Superclass2::ConfigurationType         ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer      ConfigurationPointer;
    typedef typename Superclass2::RegistrationType          RegistrationType;
    typedef typename Superclass2::RegistrationPointer       RegistrationPointer;
    typedef typename Superclass2::ITKBaseType               ITKBaseType;

    /** Typedef that is used in the elastix dll version */
    typedef typename Superclass2::ParameterMapType          ParameterMapType;

    /** Execute stuff before the actual registration:
    * \li Set the spline order.
    */
    virtual void BeforeRegistration( void );

    /** Function to read transform-parameters from a file. */
    virtual void ReadFromFile( void );

    /** Function to write transform-parameters to a file. */
    virtual void WriteToFile( void ) const;

    /** Function to create transform parameters map. */
    virtual void CreateTransformParametersMap( ParameterMapType *paramsMap ) const;

  protected:

    /** The constructor. */
    BSplineResampleInterpolator() {}
    /** The destructor. */
    virtual ~BSplineResampleInterpolator() {}

  private:

    /** The private constructor. */
    BSplineResampleInterpolator( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );              // purposely not implemented

  }; // end class BSplineResampleInterpolator


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxBSplineResampleInterpolator.hxx"
#endif

#endif // end __elxBSplineResampleInterpolator_h

