/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

  If you use the StatisticalShapePenalty anywhere we would appreciate if you cite the following article:
  F.F. Berendsen et al., Free-form image registration regularized by a statistical shape model: application to organ segmentation in cervical MR, Comput. Vis. Image Understand. (2013), http://dx.doi.org/10.1016/j.cviu.2012.12.006

======================================================================*/

#ifndef __elxStatisticalShapePenalty_H__
#define __elxStatisticalShapePenalty_H__

#include "elxIncludes.h"
#include "itkStatisticalShapePointPenalty.h"

#include "elxTimer.h"
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <vcl_iostream.h>

namespace elastix
{
using namespace itk;

  /**
   * \class StatisticalShapePenalty
   * \brief An metric based on the itk::StatisticalShapePointPenalty.
   *
   * The parameters used in this class are:
   * \parameter Metric: Select this metric as follows:\n
   *    <tt>(Metric "StatisticalShapePenalty")</tt>
   * \parameter ShrinkageIntensity: The mixing ratio ($\beta$) of the provided covariance matrix and an identity matrix. $\Sigma' = (1-\beta)\Sigma + \beta \sigma_0^2 I$ Can be defined for each resolution\n
   *    example: <tt>(ShrinkageIntensity 0.2)</tt>
   * \parameter BaseVariance: The width ($\sigma_0^2$) of the non-informative prior. Can be defined for each resolution\n
   *    example: <tt>(BaseVariance 1000.0)</tt>
   *
   * \ingroup Metrics
   *
   */

template <class TElastix >
class StatisticalShapePenalty
  : public
  StatisticalShapePointPenalty<
    typename MetricBase<TElastix>::FixedPointSetType,
    typename MetricBase<TElastix>::MovingPointSetType >,
  public MetricBase<TElastix>
{
public:

  /** Standard ITK-stuff. */
  typedef StatisticalShapePenalty    Self;
  typedef StatisticalShapePointPenalty<
    typename MetricBase<TElastix>::FixedPointSetType,
    typename MetricBase<TElastix>::MovingPointSetType > Superclass1;
  typedef MetricBase<TElastix>                          Superclass2;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( StatisticalShapePenalty,
    StatisticalShapePointPenalty );

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "StatisticalShapePenalty")</tt>\n
   */
  elxClassNameMacro( "StatisticalShapePenalty" );

  /** Typedefs from the superclass. */
  typedef typename Superclass1::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass1::FixedPointSetType          FixedPointSetType;
  typedef typename Superclass1::FixedPointSetConstPointer  FixedPointSetConstPointer;
  typedef typename Superclass1::MovingPointSetType         MovingPointSetType;
  typedef typename Superclass1::MovingPointSetConstPointer MovingPointSetConstPointer;

//  typedef typename Superclass1::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass1::TransformType              TransformType;
  typedef typename Superclass1::TransformPointer           TransformPointer;
  typedef typename Superclass1::InputPointType             InputPointType;
  typedef typename Superclass1::OutputPointType            OutputPointType;
  typedef typename Superclass1::TransformParametersType    TransformParametersType;
  typedef typename Superclass1::TransformJacobianType      TransformJacobianType;
//  typedef typename Superclass1::RealType                   RealType;
  typedef typename Superclass1::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass1::MeasureType                MeasureType;
  typedef typename Superclass1::DerivativeType             DerivativeType;
  typedef typename Superclass1::ParametersType             ParametersType;

  typedef typename OutputPointType::CoordRepType          CoordRepType;
  typedef vnl_vector<CoordRepType>                        VnlVectorType;

  /** Other typedef's. */
  typedef itk::Object                                 ObjectType;
  /*typedef itk::AdvancedTransform<
    CoordRepType,
    itkGetStaticConstMacro( FixedImageDimension ),
    itkGetStaticConstMacro( MovingImageDimension ) >  ITKBaseType;
    */
  typedef itk::AdvancedCombinationTransform<CoordRepType,
    itkGetStaticConstMacro( FixedImageDimension ) >   CombinationTransformType;
  typedef typename
    CombinationTransformType::InitialTransformType    InitialTransformType;


  /** Typedefs inherited from elastix. */
  typedef typename Superclass2::ElastixType               ElastixType;
  typedef typename Superclass2::ElastixPointer            ElastixPointer;
  typedef typename Superclass2::ConfigurationType         ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer      ConfigurationPointer;
  typedef typename Superclass2::RegistrationType          RegistrationType;
  typedef typename Superclass2::RegistrationPointer       RegistrationPointer;
  typedef typename Superclass2::ITKBaseType               ITKBaseType;
  typedef typename Superclass2::FixedImageType            FixedImageType;
  typedef typename Superclass2::MovingImageType           MovingImageType;


	/** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  /** Assuming fixed and moving pointsets are of equal type, which implicitly
   * assumes that the fixed and moving image are of the same type.
   */
  typedef FixedPointSetType   PointSetType;
  typedef FixedImageType      ImageType;

  /** Typedef for timer. */
  typedef tmr::Timer          TimerType;
  typedef TimerType::Pointer  TimerPointer;

  /** Sets up a timer to measure the initialization time and calls the
   * Superclass' implementation.
   */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /**
   * Do some things before registration:
   * \li Load and set the pointsets.
   */
  virtual void BeforeRegistration( void );

  virtual void BeforeEachResolution( void );

  /** Function to read the corresponding points. */
  unsigned int ReadLandmarks(
    const std::string & landmarkFileName,
    typename PointSetType::Pointer & pointSet,
    const typename ImageType::ConstPointer image );

  unsigned int ReadShape(
    const std::string & ShapeFileName,
    typename PointSetType::Pointer & pointSet,
    const typename ImageType::ConstPointer image );

  /** Overwrite to silence warning. */
  virtual void SelectNewSamples( void ){ };

protected:

  /** The constructor. */
  StatisticalShapePenalty(){};
  /** The destructor. */
  virtual ~StatisticalShapePenalty() {}

private:

  /** The private constructor. */
  StatisticalShapePenalty( const Self& ); // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self& );              // purposely not implemented

}; // end class StatisticalShapePenalty


} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#include "elxStatisticalShapePenalty.hxx"
#endif

#endif // end #ifndef __elxStatisticalShapePenalty_H__

