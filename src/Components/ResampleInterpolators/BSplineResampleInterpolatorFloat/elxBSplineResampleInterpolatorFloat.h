/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxBSplineResampleInterpolatorFloat_h
#define __elxBSplineResampleInterpolatorFloat_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkBSplineInterpolateImageFunction.h"

namespace elastix
{

  /**
  * \class BSplineResampleInterpolatorFloat
  * \brief A resample-interpolator based on B-splines.
  *
  * Compared to the BSplineResampleInterpolator this class uses
  * a float CoefficientType, instead of double. You can select
  * this resample interpolator if memory burden is an issue.
  *
  * The parameters used in this class are:
  * \parameter ResampleInterpolator: Select this resample interpolator as follows:\n
  *   <tt>(ResampleInterpolator "FinalBSplineInterpolatorFloat")</tt>
  * \parameter FinalBSplineInterpolationOrder: the order of the B-spline used to resample
  *    the deformed moving image; possible values: (0-5) \n
  *    example: <tt>(FinalBSplineInterpolationOrder 3 ) </tt> \n
  *    Default: 3.
  *
  * The transform parameters necessary for transformix, additionally defined by this class, are:
  * \transformparameter FinalBSplineInterpolationOrder: the order of the B-spline used to resample
  *    the deformed moving image; possible values: (0-5) \n
  *    example: <tt>(FinalBSplineInterpolationOrder 3) </tt> \n
  *    Default: 3.
  *
  * \ingroup ResampleInterpolators
  * \sa BSplineResampleInterpolator
  */

  template < class TElastix >
  class BSplineResampleInterpolatorFloat :
    public
    itk::BSplineInterpolateImageFunction<
    typename ResampleInterpolatorBase<TElastix>::InputImageType,
    typename ResampleInterpolatorBase<TElastix>::CoordRepType,
    float >, //CoefficientType
    public ResampleInterpolatorBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef BSplineResampleInterpolatorFloat      Self;
    typedef itk::BSplineInterpolateImageFunction<
      typename ResampleInterpolatorBase<TElastix>::InputImageType,
      typename ResampleInterpolatorBase<TElastix>::CoordRepType,
      float >                                     Superclass1;
    typedef ResampleInterpolatorBase<TElastix>    Superclass2;
    typedef itk::SmartPointer<Self>               Pointer;
    typedef itk::SmartPointer<const Self>         ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( BSplineResampleInterpolatorFloat, BSplineInterpolateImageFunction );

    /** Name of this class.
    * Use this name in the parameter file to select this specific resample interpolator. \n
    * example: <tt>(ResampleInterpolator "FinalBSplineInterpolatorFloat")</tt>\n
    */
    elxClassNameMacro( "FinalBSplineInterpolatorFloat" );

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

    /** Execute stuff before the actual registration:
    * \li Set the spline order.
    */
    virtual void BeforeRegistration( void );

    /** Function to read transform-parameters from a file. */
    virtual void ReadFromFile( void );

    /** Function to write transform-parameters to a file. */
    virtual void WriteToFile( void ) const;

  protected:

    /** The constructor. */
    BSplineResampleInterpolatorFloat() {}
    /** The destructor. */
    virtual ~BSplineResampleInterpolatorFloat() {}

  private:

    /** The private constructor. */
    BSplineResampleInterpolatorFloat( const Self& );  // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );              // purposely not implemented

  }; // end class BSplineResampleInterpolatorFloat


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxBSplineResampleInterpolatorFloat.hxx"
#endif

#endif // end __elxBSplineResampleInterpolatorFloat_h

