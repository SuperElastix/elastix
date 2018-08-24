/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxPointToSurfaceDistanceMetric_H__
#define __elxPointToSurfaceDistanceMetric_H__

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkPointToSurfaceDistanceMetric.h"

namespace elastix
{

/**
 * \class TransformRigidityPenalty
 * \brief A penalty term based on non-rigidity.
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

  /** Standard ITK-stuff. */
  typedef PointToSurfaceDistanceMetric Self;
  typedef itk::PointToSurfaceDistanceMetric<
  typename MetricBase< TElastix >::FixedPointSetType,
  typename MetricBase< TElastix >::MovingPointSetType >    Superclass1;
  typedef MetricBase< TElastix >          Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

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
  typedef typename Superclass1::CoordinateRepresentationType    CoordinateRepresentationType;
  typedef typename Superclass1::FixedPointSetType               FixedPointSetType;
  typedef typename Superclass1::FixedPointSetConstPointer       FixedPointSetConstPointer;
  typedef typename Superclass1::MovingPointSetType              MovingPointSetType;
  typedef typename Superclass1::MovingPointSetConstPointer      MovingPointSetConstPointer;
  typedef typename Superclass1::TransformType                   TransformType;
  typedef typename Superclass1::TransformPointer                TransformPointer;
  typedef typename Superclass1::InputPointType                  InputPointType;
  typedef typename Superclass1::OutputPointType                 OutputPointType;
  typedef typename Superclass1::TransformParametersType         TransformParametersType;
  typedef typename Superclass1::TransformJacobianType           TransformJacobianType;
  typedef typename Superclass1::FixedImageMaskType              FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer           FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType             MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer          MovingImageMaskPointer;
  typedef typename Superclass1::MeasureType                     MeasureType;
  typedef typename Superclass1::DerivativeType                  DerivativeType;
  typedef typename Superclass1::ParametersType                  ParametersType;

   /** Typedefs inherited from elastix. */
  typedef typename Superclass2::ElastixType                     ElastixType;
  typedef typename Superclass2::ElastixPointer                  ElastixPointer;
  typedef typename Superclass2::ConfigurationType               ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer            ConfigurationPointer;
  typedef typename Superclass2::RegistrationType                RegistrationType;
  typedef typename Superclass2::RegistrationPointer             RegistrationPointer;
  typedef typename Superclass2::ITKBaseType                     ITKBaseType;
  typedef typename Superclass2::FixedImageType                  FixedImageType;
  typedef typename Superclass2::MovingImageType                 MovingImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,  FixedImageType::ImageDimension );

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int, MovingImageType::ImageDimension );

  typedef FixedPointSetType PointSetType;
  typedef typename Superclass1::ImageType    ImageType;


  /** Sets up a timer to measure the initialisation time and
   * calls the Superclass' implementation.
   */
  virtual void Initialize( void ) throw ( itk::ExceptionObject );

  /**
   * Do some things before all:
   * \li Check and print the command line arguments fp and mp.
   *   This should be done in BeforeAllBase and not BeforeAll.
   */
  virtual int BeforeAllBase( void );

  /**
   * Do some things before each resolution:
   * \li Set CheckNumberOfSamples setting
   * \li Set UseNormalization setting
   */
  //virtual void BeforeEachResolution( void );

  /**
   * Do some things before registration:
   * \li Load and set the pointsets.
   */
  virtual void BeforeRegistration( void );

  /** Function to read the corresponding points. */
/** Function to read the corresponding points. */
  typedef typename Superclass1::ImageType::ConstPointer ImageConstPointer;

  unsigned int ReadLandmarks( const std::string & landmarkFileName, typename PointSetType::Pointer & pointSet, typename FixedImageType::ConstPointer image );

  /** Overwrite to silence warning. */
  virtual void SelectNewSamples( void ){}

protected:

  /** The constructor. */
  PointToSurfaceDistanceMetric(){}
  /** The destructor. */
  virtual ~PointToSurfaceDistanceMetric() {}

private:

  /** The private constructor. */
  PointToSurfaceDistanceMetric( const Self & ); // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );              // purposely not implemented

};



} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxPointToSurfaceDistanceMetric.hxx"
#endif

#endif
