/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxRayCastInterpolator_h
#define __elxRayCastInterpolator_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedRayCastInterpolateImageFunction.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedTransform.h"
#include "../Transforms/EulerTransform/itkEulerTransform.h"

namespace elastix
{

/**
 * \class RayCastInterpolator
 * \brief An interpolator based on the itkAdvancedRayCastInterpolateImageFunction.
 *
 *
 *
 * The parameters used in this class are:
 * \parameter Interpolator: Select this interpolator as follows:\n
 *    <tt>(Interpolator "RayCastInterpolator")</tt>
 *
 * \ingroup Interpolators
 */

template< class TElastix >
class RayCastInterpolator :
  public
  itk::AdvancedRayCastInterpolateImageFunction<
  typename InterpolatorBase< TElastix >::InputImageType,
  typename InterpolatorBase< TElastix >::CoordRepType >,
  public
  InterpolatorBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef RayCastInterpolator Self;
  typedef itk::AdvancedRayCastInterpolateImageFunction<
    typename InterpolatorBase< TElastix >::InputImageType,
    typename InterpolatorBase< TElastix >::CoordRepType > Superclass1;
  typedef InterpolatorBase< TElastix >    Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( RayCastInterpolator, AdvancedRayCastInterpolateImageFunction );

  /** Name of this class.
   * Use this name in the parameter file to select this specific interpolator. \n
   * example: <tt>(Interpolator "RayCastInterpolator")</tt>\n
   */
  elxClassNameMacro( "RayCastInterpolator" );

  /** Get the ImageDimension. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::OutputType          OutputType;
  typedef typename Superclass1::InputImageType      InputImageType;
  typedef typename Superclass1::IndexType           IndexType;
  typedef typename Superclass1::ContinuousIndexType ContinuousIndexType;
  typedef typename Superclass1::PointType           PointType;
  typedef typename Superclass1::SizeType            SizeType;
  typedef typename InputImageType::SpacingType      SpacingType;

  /** Typedefs inherited from Elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Typedef's for CombinationTransform */
  typedef typename itk::EulerTransform<
    typename InterpolatorBase< TElastix >::CoordRepType,
    ImageDimension >                                      EulerTransformType;
  typedef typename EulerTransformType::ParametersType TransformParametersType;
  typedef typename EulerTransformType::Pointer        EulerTransformPointer;
  typedef typename itk::AdvancedTransform<
    typename InterpolatorBase< TElastix >::CoordRepType,
    itkGetStaticConstMacro( ImageDimension ),
    itkGetStaticConstMacro( ImageDimension ) >            AdvancedTransformType;
  typedef typename AdvancedTransformType::Pointer AdvancedTransformPointer;
  typedef typename itk::AdvancedCombinationTransform<
    typename InterpolatorBase< TElastix >::CoordRepType,
    itkGetStaticConstMacro( ImageDimension ) >            CombinationTransformType;
  typedef typename CombinationTransformType::Pointer CombinationTransformPointer;

protected:

  /** The constructor. */
  RayCastInterpolator() {}

  /** The destructor. */
  virtual ~RayCastInterpolator() {}

  virtual int BeforeAll( void );

  virtual void BeforeRegistration( void );

  virtual void BeforeEachResolution( void );

private:

  /** The private constructor. */
  RayCastInterpolator( const Self & );  // purposely not implemented

  /** The private copy constructor. */
  void operator=( const Self & );      // purposely not implemented

  EulerTransformPointer       m_PreTransform;
  TransformParametersType     m_PreParameters;
  CombinationTransformPointer m_CombinationTransform;

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxRayCastInterpolator.hxx"
#endif

#endif // end #ifndef __elxRayCastInterpolator_h
