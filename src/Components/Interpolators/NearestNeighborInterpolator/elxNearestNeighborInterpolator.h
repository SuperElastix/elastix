/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxNearestNeighborInterpolator_h
#define __elxNearestNeighborInterpolator_h

#include "itkNearestNeighborInterpolateImageFunction.h"
#include "elxIncludes.h"

namespace elastix
{

using namespace itk;

  /**
   * \class NearestNeighborInterpolator
   * \brief An interpolator based on the itkNearestNeighborInterpolateImageFunction.
   *
   * This interpolator interpolates images using nearest neighbour interpolation.
   * The image derivatives are computed using a central difference scheme.
   *
   * The parameters used in this class are:
   * \parameter Interpolator: Select this interpolator as follows:\n
   *    <tt>(Interpolator "NearestNeighborInterpolator")</tt>
   *
   * \ingroup Interpolators
   */

  template < class TElastix >
    class NearestNeighborInterpolator :
    public
      NearestNeighborInterpolateImageFunction<
        ITK_TYPENAME InterpolatorBase<TElastix>::InputImageType,
        ITK_TYPENAME InterpolatorBase<TElastix>::CoordRepType >,
    public
      InterpolatorBase<TElastix>
  {
  public:

    /** Standard ITK-stuff. */
    typedef NearestNeighborInterpolator                 Self;
    typedef NearestNeighborInterpolateImageFunction<
      typename InterpolatorBase<TElastix>::InputImageType,
      typename InterpolatorBase<TElastix>::CoordRepType > Superclass1;
    typedef InterpolatorBase<TElastix>          Superclass2;
    typedef SmartPointer<Self>                  Pointer;
    typedef SmartPointer<const Self>            ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro( NearestNeighborInterpolator, NearestNeighborInterpolateImageFunction );

    /** Name of this class.
     * Use this name in the parameter file to select this specific interpolator. \n
     * example: <tt>(Interpolator "NearestNeighborInterpolator")</tt>\n
     */
    elxClassNameMacro( "NearestNeighborInterpolator" );

    /** Get the ImageDimension. */
    itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );

    /** Typedefs inherited from the superclass. */
    typedef typename Superclass1::OutputType                OutputType;
    typedef typename Superclass1::InputImageType            InputImageType;
    typedef typename Superclass1::IndexType                 IndexType;
    typedef typename Superclass1::ContinuousIndexType       ContinuousIndexType;
    typedef typename Superclass1::PointType                 PointType;

    /** Typedefs inherited from Elastix. */
    typedef typename Superclass2::ElastixType               ElastixType;
    typedef typename Superclass2::ElastixPointer            ElastixPointer;
    typedef typename Superclass2::ConfigurationType         ConfigurationType;
    typedef typename Superclass2::ConfigurationPointer      ConfigurationPointer;
    typedef typename Superclass2::RegistrationType          RegistrationType;
    typedef typename Superclass2::RegistrationPointer       RegistrationPointer;
    typedef typename Superclass2::ITKBaseType               ITKBaseType;

  protected:

    /** The constructor. */
    NearestNeighborInterpolator() {}
    /** The destructor. */
    virtual ~NearestNeighborInterpolator() {}

  private:

    /** The private constructor. */
    NearestNeighborInterpolator( const Self& ); // purposely not implemented
    /** The private copy constructor. */
    void operator=( const Self& );      // purposely not implemented

  }; // end class NearestNeighborInterpolator


} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxNearestNeighborInterpolator.hxx"
#endif

#endif // end #ifndef __elxNearestNeighborInterpolator_h

