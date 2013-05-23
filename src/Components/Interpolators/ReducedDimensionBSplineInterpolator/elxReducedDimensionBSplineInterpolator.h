/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxReducedDimensionBSplineInterpolator_h
#define __elxReducedDimensionBSplineInterpolator_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkReducedDimensionBSplineInterpolateImageFunction.h"

namespace elastix
{


  /**
   * \class ReducedDimensionBSplineInterpolator
   * \brief An interpolator based on the itkReducedDimensionBSplineInterpolateImageFunction.
   *
   * This interpolator interpolates images with an underlying B-spline
   * polynomial. It only interpolates in the InputImageDimension - 1 dimensions
   * of the image.
   *
   * The parameters used in this class are:
   * \parameter Interpolator: Select this interpolator as follows:\n
   *    <tt>(Interpolator "ReducedDimensionBSplineInterpolator")</tt>
   * \parameter BSplineInterpolationOrder: the order of the B-spline polynomial. \n
   *    example: <tt>(BSplineInterpolationOrder 1 1 1)</tt> \n
   *    The default order is 1. The parameter can be specified for each resolution.\n
   *    If only given for one resolution, that value is used for the other resolutions as well. \n
   *    Currently only first order B-spline interpolation is supported.
   *
   * \ingroup Interpolators
   */

  template < class TElastix >
    class ReducedDimensionBSplineInterpolator :
    public
      itk::ReducedDimensionBSplineInterpolateImageFunction<
        typename InterpolatorBase<TElastix>::InputImageType,
        typename InterpolatorBase<TElastix>::CoordRepType,
        double > , //CoefficientType
    public
      InterpolatorBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef ReducedDimensionBSplineInterpolator  Self;
    typedef itk::ReducedDimensionBSplineInterpolateImageFunction<
      typename InterpolatorBase<TElastix>::InputImageType,
      typename InterpolatorBase<TElastix>::CoordRepType,
      double >                                  Superclass1;
    typedef InterpolatorBase<TElastix>          Superclass2;
    typedef itk::SmartPointer<Self>             Pointer;
    typedef itk::SmartPointer<const Self>       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( ReducedDimensionBSplineInterpolator, ReducedDimensionBSplineInterpolateImageFunction );

    /** Name of this class.
     * Use this name in the parameter file to select this specific interpolator. \n
     * example: <tt>(Interpolator "ReducedDimensionBSplineInterpolator")</tt>\n
     */
    elxClassNameMacro( "ReducedDimensionBSplineInterpolator" );

    /** Get the ImageDimension. */
    itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );

    /** Typedefs inherited from the superclass. */
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

    /** Typedefs inherited from Elastix. */
    typedef typename Superclass2::ElastixType               ElastixType;
    typedef typename Superclass2::ElastixPointer            ElastixPointer;
    typedef typename Superclass2::ConfigurationType         ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer      ConfigurationPointer;
    typedef typename Superclass2::RegistrationType          RegistrationType;
    typedef typename Superclass2::RegistrationPointer       RegistrationPointer;
    typedef typename Superclass2::ITKBaseType               ITKBaseType;

    /** Execute stuff before each new pyramid resolution:
     * \li Set the spline order.
     */
    virtual void BeforeEachResolution(void);

  protected:

    /** The constructor. */
    ReducedDimensionBSplineInterpolator() {}
    /** The destructor. */
    virtual ~ReducedDimensionBSplineInterpolator() {}

  private:

    /** The private constructor. */
    ReducedDimensionBSplineInterpolator( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );      // purposely not implemented

  }; // end class ReducedDimensionBSplineInterpolator


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxReducedDimensionBSplineInterpolator.hxx"
#endif

#endif // end #ifndef __elxReducedDimensionBSplineInterpolator_h

