/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxActiveRegistrationModelShapeMetric_H__
#define __elxActiveRegistrationModelShapeMetric_H__

#include "elxIncludes.h"
#include "itkActiveRegistrationModelShapeMetric.h"

#include "itkDirectory.h"
#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"

#include "itkStatisticalModel.h"
#include "itkStandardMeshRepresenter.h"
#include "itkPCAModelBuilder.h"
#include "itkReducedVarianceModelBuilder.h"
#include "itkStatismoIO.h"

namespace elastix
{
/**
 * \class ActiveRegistrationModelShapeMetric
 * \brief A dummy metric to generate transformed meshes at each iteration.
 * This metric does not contribute to the cost function, but provides the
 * options to read vtk polydata meshes from the command-line and write the
 * transformed meshes to disk each iteration or resolution level.
 * The command-line options for input meshes is: -fmesh<[A-Z]><MetricNumber>.
 * This metric can be used as a base for other mesh-based penalties.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "ActiveRegistrationModelShapeMetric")</tt>
 * \parameter
 *    <tt>(WriteResultMeshAfterEachIteration "True")</tt>
 * \parameter
 *    <tt>(WriteResultMeshAfterEachResolution "True")</tt>
 * \ingroup Metrics
 *
 */

//TODO: define a base class templated on meshes in stead of 2 pointsets.
template< class TElastix >
class ActiveRegistrationModelShapeMetric :
  public
  itk::ActiveRegistrationModelShapeMetric<
  typename MetricBase< TElastix >::FixedPointSetType,
  typename MetricBase< TElastix >::MovingPointSetType >,
  public MetricBase< TElastix >
{
public:

  /** Standard ITK-stuff. */
  typedef ActiveRegistrationModelShapeMetric Self;
  typedef itk::ActiveRegistrationModelShapeMetric<
    typename MetricBase< TElastix >::FixedPointSetType,
    typename MetricBase< TElastix >::MovingPointSetType > Superclass1;
  typedef MetricBase< TElastix >          Superclass2;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro(ActiveRegistrationModelShapeMetric, itk::ActiveRegistrationModelShapeMetric );

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "ActiveRegistrationModelShapeMetric")</tt>\n
   */
  elxClassNameMacro( "ActiveRegistrationModelShapeMetric" );

  /** Typedefs from the superclass. */

  typedef typename Superclass1::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass1::FixedPointSetType            FixedPointSetType;
  typedef typename Superclass1::MovingPointSetType           MovingPointSetType;

  typedef typename Superclass1::TransformType           TransformType;
  typedef typename Superclass1::TransformPointer        TransformPointer;
  typedef typename Superclass1::InputPointType          InputPointType;
  typedef typename Superclass1::OutputPointType         OutputPointType;
  typedef typename Superclass1::TransformParametersType TransformParametersType;
  typedef typename Superclass1::TransformJacobianType   TransformJacobianType;
  typedef typename Superclass1::FixedImageMaskType      FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer   FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType     MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer  MovingImageMaskPointer;
  
  typedef typename Superclass1::MeasureType             MeasureType;
  typedef typename Superclass1::DerivativeType          DerivativeType;
  typedef typename Superclass1::ParametersType          ParametersType;

  typedef typename OutputPointType::CoordRepType CoordRepType;
  
  /** Other typedef's. */
  typedef itk::Object ObjectType;

  typedef itk::AdvancedCombinationTransform< CoordRepType,
    itkGetStaticConstMacro( FixedImageDimension ) >   CombinationTransformType;
  typedef typename
    CombinationTransformType::InitialTransformType InitialTransformType;

  /** Typedefs inherited from elastix. */
  typedef typename Superclass2::ElastixType          ElastixType;
  typedef typename Superclass2::ElastixPointer       ElastixPointer;
  typedef typename Superclass2::ConfigurationType    ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass2::RegistrationType     RegistrationType;
  typedef typename Superclass2::RegistrationPointer  RegistrationPointer;
  typedef typename Superclass2::ITKBaseType          ITKBaseType;
  typedef typename Superclass2::FixedImageType       FixedImageType;
  typedef typename Superclass2::MovingImageType      MovingImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  typedef FixedImageType ImageType;

  typedef typename Superclass1::StatisticalModelVectorType                    StatisticalModelVectorType;
  typedef vector< std::string >                                               StatisticalModelPathVectorType;
  
  /** ActiveRegistrationModel types */
  typedef typename Superclass1::StatisticalModelMeshType                      StatisticalModelMeshType;
  typedef typename Superclass1::StatisticalModelMeshPointer                   StatisticalModelMeshPointer;
  
  typedef typename Superclass1::MeshReaderType                                MeshReaderType;
  typedef typename Superclass1::MeshReaderPointer                             MeshReaderPointer;
  
  typedef typename Superclass1::StatisticalModelRepresenterType               StatisticalModelRepresenterType;
  typedef typename Superclass1::StatisticalModelRepresenterPointer            StatisticalModelRepresenterPointer;

  typedef typename Superclass1::ModelBuilderType                              ModelBuilderType;
  typedef typename Superclass1::ModelBuilderPointer                           ModelBuilderPointer;
  
  typedef typename Superclass1::ReducedVarianceModelBuilderType               ReducedVarianceModelBuilderType;
  typedef typename Superclass1::ReducedVarianceModelBuilderPointer            ReducedVarianceModelBuilderPointer;

  typedef typename Superclass1::StatisticalModelIdType                        StatisticalModelIdType;
  typedef typename Superclass1::StatisticalModelPointer                       StatisticalModelPointer;

  typedef typename Superclass1::StatisticalModelMatrixContainerType           StatisticalModelMatrixContainerType;
  typedef typename Superclass1::StatisticalModelMatrixContainerPointer        StatisticalModelMatrixContainerPointer;

  typedef typename Superclass1::StatisticalModelVectorContainerType           StatisticalModelVectorContainerType;
  typedef typename Superclass1::StatisticalModelVectorContainerPointer        StatisticalModelVectorContainerPointer;

  typedef typename Superclass1::StatisticalModelScalarContainerType           StatisticalModelScalarContainerType;
  typedef typename Superclass1::StatisticalModelScalarContainerPointer        StatisticalModelScalarContainerPointer;
  
  typedef typename Superclass1::StatisticalModelDataManagerType               StatisticalModelDataManagerType;
  typedef typename Superclass1::StatisticalModelDataManagerPointer            StatisticalModelDataManagerPointer;
  
  itkSetMacro( MetricNumber, unsigned long );
  itkGetMacro( MetricNumber, unsigned long );

  StatisticalModelDataManagerPointer ReadMeshesFromDirectory( std::string shapeDataDirectory,
                                                              std::string fixedPointSetFilename );
  
  unsigned long ReadMesh( const std::string & meshFilename, StatisticalModelMeshPointer& mesh );

  typedef itk::MeshFileWriter< StatisticalModelMeshType >  MeshFileWriterType;
  typedef typename MeshFileWriterType::Pointer             MeshFileWriterPointer;
  void WriteMesh( const char * filename, StatisticalModelMeshType mesh );
  
  StatisticalModelPathVectorType ReadPath( std::string parameter );

  StatisticalModelVectorType ReadNoiseVariance();

  StatisticalModelVectorType ReadTotalVariance();

  /** Sets up a timer to measure the initialization time and calls the
   * Superclass' implementation.
   */
  virtual void Initialize( void );

  /**
   * Do some things before registration:
   * \li Load and set the pointsets.
   */
  virtual int BeforeAllBase( void );

  virtual void BeforeRegistration( void );

  virtual void AfterRegistration( void );

  /** Overwrite to silence warning. */
  virtual void SelectNewSamples( void ){}

protected:
  
  /** The constructor. */
  ActiveRegistrationModelShapeMetric(){}
  /** The destructor. */
  virtual ~ActiveRegistrationModelShapeMetric() {}

private:

  /** The private constructor. */
  ActiveRegistrationModelShapeMetric(const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );        // purposely not implemented
  
  unsigned long m_MetricNumber;

  StatisticalModelPathVectorType m_LoadShapeModelFileNames;
  StatisticalModelPathVectorType m_SaveShapeModelFileNames;
  StatisticalModelPathVectorType m_ShapeDirectories;
  StatisticalModelPathVectorType m_ReferenceFilenames;
  
}; // end class ActiveRegistrationModel

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxActiveRegistrationModelShapeMetric.hxx"
#endif

#endif // end #ifndef __elxActiveRegistrationModelShapeMetric_h__

