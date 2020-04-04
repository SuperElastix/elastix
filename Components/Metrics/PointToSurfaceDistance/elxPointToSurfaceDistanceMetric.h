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
#ifndef __elxPointToSurfaceDistanceMetric_H__
#define __elxPointToSurfaceDistanceMetric_H__

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkPointToSurfaceDistanceMetric.h"

namespace elastix
{

/**
 *
 * For more information check the paper:\n
 * Gunay, G. , Luu, M. H., Moelker, A. , Walsum, T. and Klein, S. (2017), 
 * "Semiautomated registration of pre‐ and intraoperative CT for image‐guided percutaneous 
 * liver tumor ablation interventions," Med. Phys., 44: 3718-3725.
 *
 * The parameters used in this class are:\n
 * \parameter Metric: Select this metric as follows:
 *    </tt>(Metric "PointToSurfaceDistance")</tt>
 *
 * \parameter PointToSurfaceDistanceAverage: A parameter to get average of the points by dividing the weight of the penalty with the number of the points included. 
 *    example: </tt>(PointToSurfaceDistanceAverage "true")</tt> \n
 *    Default is "true".
 *
 * \Command line parameters
 * \parameter -fp    "file" : Elastix pointset file. 
 *  example: 

 * \parameter -dt    "file" : Distance transform (file) of segmentation of the organ under interest in the moving image.
 * \parameter -seg   "file" : Segmentation (file) of the organ under interest in the moving image. With this parameter the algorithm internally computes the distance transform.
 * \parameter -dtout "file" : If the user wants to get the distance transform from organ segmentation, this parameter is called with the name of the distance transform image file to write .
 *
 *
 * \ingroup Metrics
 *
 */

template< class TElastix >
class PointToSurfaceDistanceMetric :
  public
  itk::PointToSurfaceDistanceMetric <
  typename MetricBase< TElastix >::FixedPointSetType,
  typename MetricBase< TElastix >::MovingPointSetType >,
  public MetricBase< TElastix >
{
public:

  using Self = PointToSurfaceDistanceMetric;

  /** Standard ITK-stuff. */
  using Superclass1 = itk::PointToSurfaceDistanceMetric<typename MetricBase< TElastix >::FixedPointSetType, typename MetricBase< TElastix >::MovingPointSetType >;
  using Superclass2 = MetricBase< TElastix >;
  using Pointer = itk::SmartPointer< Self >;
  using ConstPointer = itk::SmartPointer< const Self >;

  ITK_DISALLOW_COPY_AND_ASSIGN(PointToSurfaceDistanceMetric);
  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( PointToSurfaceDistanceMetric, itk::PointToSurfaceDistanceMetric );

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "PointToSurfaceDistance")</tt>\n
   */
  elxClassNameMacro( "PointToSurfaceDistance" );

  /** Typedefs from the superclass. */
  using CoordinateRepresentationType = typename Superclass1::CoordinateRepresentationType;
  using FixedPointSetType = typename Superclass1::FixedPointSetType;
  using FixedPointSetConstPointer = typename Superclass1::FixedPointSetConstPointer;
  using MovingPointSetType = typename Superclass1::MovingPointSetType;
  using MovingPointSetConstPointer = typename Superclass1::MovingPointSetConstPointer;
  using TransformType = typename Superclass1::TransformType;
  using TransformPointer = typename Superclass1::TransformPointer;
  using InputPointType = typename Superclass1::InputPointType;
  using OutputPointType = typename Superclass1::OutputPointType;
  using TransformParametersType = typename Superclass1::TransformParametersType;
  using TransformJacobianType = typename Superclass1::TransformJacobianType;
  using FixedImageMaskType = typename Superclass1::FixedImageMaskType;
  using FixedImageMaskPointer = typename Superclass1::FixedImageMaskPointer;
  using MovingImageMaskType = typename Superclass1::MovingImageMaskType;
  using MovingImageMaskPointer = typename Superclass1::MovingImageMaskPointer;
  using MeasureType = typename Superclass1::MeasureType;
  using DerivativeType = typename Superclass1::DerivativeType;
  using ParametersType = typename Superclass1::ParametersType;

   /** Typedefs inherited from elastix. */
  using ElastixType = typename Superclass2::ElastixType;
  using ElastixPointer = typename Superclass2::ElastixPointer;
  using ConfigurationType = typename Superclass2::ConfigurationType;
  using ConfigurationPointer = typename Superclass2::ConfigurationPointer;
  using RegistrationType = typename Superclass2::RegistrationType;
  using RegistrationPointer = typename Superclass2::RegistrationPointer;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using FixedImageType = typename Superclass2::FixedImageType;
  using MovingImageType = typename Superclass2::MovingImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,  FixedImageType::ImageDimension );

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

  using PointSetType = FixedPointSetType;
  using ImageType = typename Superclass1::ImageType;


  /** Sets up a timer to measure the initialisation time and
   * calls the Superclass' implementation.
   */
  virtual void Initialize();

  /**
   * Do some things before all:
   * \li Check and print the command line arguments fp and mp.
   *   This should be done in BeforeAllBase and not BeforeAll.
   */
  virtual int BeforeAllBase();

  /**
   * Do some things before registration:
   * \li Load and set the pointsets.
   */
  virtual void BeforeRegistration();

  using ImageConstPointer = typename Superclass1::ImageType::ConstPointer;

  unsigned int ReadLandmarks( const std::string & landmarkFileName, typename PointSetType::Pointer & pointSet, typename FixedImageType::ConstPointer image );

  /** Overwrite to silence warning. */
  virtual void SelectNewSamples( void ){}

protected:

  /** The constructor. */
  PointToSurfaceDistanceMetric() = default;
  /** The destructor. */
  virtual ~PointToSurfaceDistanceMetric() = default;

};



} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxPointToSurfaceDistanceMetric.hxx"
#endif

#endif
