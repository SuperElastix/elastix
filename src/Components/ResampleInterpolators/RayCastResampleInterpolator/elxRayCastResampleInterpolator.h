/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxRayCastResampleInterpolator_h
#define __elxRayCastResampleInterpolator_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkAdvancedRayCastInterpolateImageFunction.h"
#include "itkAdvancedCombinationTransform.h"
#include "itkAdvancedTransform.h"
#include "itkEulerTransform.h"

namespace elastix
{

/**
 * \class RayCastResampleInterpolator
 * \brief An interpolator based on ...
 *
 * \ingroup Interpolators
 */

template< class TElastix >
class RayCastResampleInterpolator :
  public
  itk::AdvancedRayCastInterpolateImageFunction<
  typename ResampleInterpolatorBase< TElastix >::InputImageType,
  typename ResampleInterpolatorBase< TElastix >::CoordRepType >,
  public ResampleInterpolatorBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef RayCastResampleInterpolator Self;
  typedef itk::AdvancedRayCastInterpolateImageFunction<
    typename ResampleInterpolatorBase< TElastix >::InputImageType,
    typename ResampleInterpolatorBase< TElastix >::CoordRepType >    Superclass1;
  typedef ResampleInterpolatorBase< TElastix > Superclass2;
  typedef itk::SmartPointer< Self >            Pointer;
  typedef itk::SmartPointer< const Self >      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( RayCastResampleInterpolator, AdvancedRayCastInterpolateImageFunction );

  /** Name of this class.
  * Use this name in the parameter file to select this specific resample interpolator. \n
  * example: <tt>(ResampleInterpolator "FinalRayCastInterpolator")</tt>\n
  */
  elxClassNameMacro( "FinalRayCastInterpolator" );

  /** Dimension of the image. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass1::ImageDimension );

  /** Typedef's inherited from the superclass. */
  typedef typename Superclass1::OutputType          OutputType;
  typedef typename Superclass1::InputImageType      InputImageType;
  typedef typename Superclass1::IndexType           IndexType;
  typedef typename Superclass1::ContinuousIndexType ContinuousIndexType;
  typedef typename Superclass1::PointType           PointType;
  typedef typename Superclass1::SizeType            SizeType;
  typedef typename InputImageType::SpacingType      SpacingType;

  /** Typedef's from ResampleInterpolatorBase. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;

  /** Typedef's for CombinationTransform */
  typedef typename itk::EulerTransform<
    typename InterpolatorBase< TElastix >::CoordRepType, ImageDimension >
    EulerTransformType;
  typedef typename EulerTransformType::ParametersType TransformParametersType;
  typedef typename EulerTransformType::Pointer        EulerTransformPointer;
  typedef typename itk::AdvancedTransform<
    typename InterpolatorBase< TElastix >::CoordRepType,
    itkGetStaticConstMacro( ImageDimension ),
    itkGetStaticConstMacro( ImageDimension ) >             AdvancedTransformType;
  typedef typename AdvancedTransformType::Pointer AdvancedTransformPointer;
  typedef typename itk::AdvancedCombinationTransform<
    typename InterpolatorBase< TElastix >::CoordRepType,
    itkGetStaticConstMacro( ImageDimension ) >            CombinationTransformType;
  typedef typename CombinationTransformType::Pointer CombinationTransformPointer;

  virtual int BeforeAll( void );

  virtual void BeforeRegistration( void );

  /** Function to read transform-parameters from a file. */
  virtual void ReadFromFile( void );

  /** Function to write transform-parameters to a file. */
  virtual void WriteToFile( void ) const;

protected:

  /** The constructor. */
  RayCastResampleInterpolator() {}

  /** The destructor. */
  virtual ~RayCastResampleInterpolator() {}

  /** Helper function to initialize the combination transform
   * with a pre-transform.
   */
  void InitializeRayCastInterpolator( void );

private:

  /** The private constructor. */
  RayCastResampleInterpolator( const Self & ); // purposely not implemented

  /** The private copy constructor. */
  void operator=( const Self & );              // purposely not implemented

  EulerTransformPointer       m_PreTransform;
  TransformParametersType     m_PreParameters;
  CombinationTransformPointer m_CombinationTransform;

};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxRayCastResampleInterpolator.hxx"
#endif

#endif // end __elxRayCastResampleInterpolator_h
