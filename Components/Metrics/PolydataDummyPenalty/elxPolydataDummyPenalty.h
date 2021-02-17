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
#ifndef elxPolydataDummyPenalty_h
#define elxPolydataDummyPenalty_h

#include "elxIncludes.h"
#include "itkPolydataDummyPenalty.h"

//#include "elxMetricBase.h"

#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"

namespace elastix
{
/**
 * \class PolydataDummyPenalty
 * \brief A dummy metric to generate transformed meshes at each iteration.
 * This metric does not contribute to the cost function, but provides the
 * options to read vtk polydata meshes from the command-line and write the
 * transformed meshes to disk each iteration or resolution level.
 * The command-line options for input meshes is: -fmesh<[A-Z]><MetricNumber>.
 * This metric can be used as a base for other mesh-based penalties.
 *
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "PolydataDummyPenalty")</tt>
 * \parameter
 *    <tt>(WriteResultMeshAfterEachIteration "True")</tt>
 * \parameter
 *    <tt>(WriteResultMeshAfterEachResolution "True")</tt>
 * \ingroup Metrics
 *
 */

// TODO: define a base class templated on meshes in stead of 2 pointsets.
// typedef unsigned char DummyPixelType;
// typedef unsigned char BinaryPixelType;
// typedef itk::Mesh<BinaryPixelType,FixedImageDimension> FixedMeshType;
// typedef itk::Mesh <DummyPixelType, MetricBase<TElastix>::FixedImageDimension>  FixedMeshType; //pixeltype is unused,
// but necessary for the declaration, so a type with the smallest memory footprint is used.
//  template <class TElastix >
// class PolydataDummyPenalty
//  : public
//  itk::MeshPenalty < itk::Mesh<DummyPixelType, MetricBase <TElastix>::FixedImageDimension > >,
//  public MetricBase<TElastix>
//
template <class TElastix>
class ITK_TEMPLATE_EXPORT PolydataDummyPenalty
  : public itk::MeshPenalty<typename MetricBase<TElastix>::FixedPointSetType,
                            typename MetricBase<TElastix>::MovingPointSetType>
  , public MetricBase<TElastix>
{
public:
  /** Standard ITK-stuff. */
  typedef PolydataDummyPenalty Self;
  typedef itk::MeshPenalty<typename MetricBase<TElastix>::FixedPointSetType,
                           typename MetricBase<TElastix>::MovingPointSetType>
                                        Superclass1;
  typedef MetricBase<TElastix>          Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(PolydataDummyPenalty, itk::MeshPenalty);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "PolydataDummyPenalty")</tt>\n
   */
  elxClassNameMacro("PolydataDummyPenalty");

  /** Typedefs from the superclass. */
  typedef typename Superclass1::FixedMeshType         FixedMeshType;
  typedef typename Superclass1::FixedMeshPointer      FixedMeshPointer;
  typedef typename Superclass1::FixedMeshConstPointer FixedMeshConstPointer;

  typedef typename Superclass1::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass1::FixedPointSetType            FixedPointSetType;
  typedef typename Superclass1::FixedPointSetConstPointer    FixedPointSetConstPointer;
  typedef typename Superclass1::FixedMeshContainerType       FixedMeshContainerType;
  typedef typename Superclass1::FixedMeshContainerPointer    FixedMeshContainerPointer;
  typedef typename Superclass1::MappedMeshContainerType      MappedMeshContainerType;
  typedef typename Superclass1::MappedMeshContainerPointer   MappedMeshContainerPointer;
  typedef typename Superclass1::MovingPointSetType           MovingPointSetType;
  typedef typename Superclass1::MovingPointSetConstPointer   MovingPointSetConstPointer;
  typedef typename Superclass1::CellInterfaceType            CellInterfaceType;

  //  typedef typename Superclass1::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass1::TransformType           TransformType;
  typedef typename Superclass1::TransformPointer        TransformPointer;
  typedef typename Superclass1::InputPointType          InputPointType;
  typedef typename Superclass1::OutputPointType         OutputPointType;
  typedef typename Superclass1::TransformParametersType TransformParametersType;
  typedef typename Superclass1::TransformJacobianType   TransformJacobianType;
  //  typedef typename Superclass1::RealType                   RealType;
  typedef typename Superclass1::FixedImageMaskType     FixedImageMaskType;
  typedef typename Superclass1::FixedImageMaskPointer  FixedImageMaskPointer;
  typedef typename Superclass1::MovingImageMaskType    MovingImageMaskType;
  typedef typename Superclass1::MovingImageMaskPointer MovingImageMaskPointer;
  typedef typename Superclass1::MeasureType            MeasureType;
  typedef typename Superclass1::DerivativeType         DerivativeType;
  typedef typename Superclass1::ParametersType         ParametersType;

  typedef typename OutputPointType::CoordRepType CoordRepType;

  typedef typename Superclass1::MeshIdType MeshIdType;
  /** Other typedef's. */
  typedef itk::Object ObjectType;
  /*typedef itk::AdvancedTransform<
  CoordRepType,
  itkGetStaticConstMacro( FixedImageDimension ),
  itkGetStaticConstMacro( MovingImageDimension ) >  ITKBaseType;
  */
  typedef itk::AdvancedCombinationTransform<CoordRepType, itkGetStaticConstMacro(FixedImageDimension)>
                                                                  CombinationTransformType;
  typedef typename CombinationTransformType::InitialTransformType InitialTransformType;

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
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Assuming fixed and moving pointsets are of equal type, which implicitly
   * assumes that the fixed and moving image are of the same type.
   */
  typedef FixedPointSetType PointSetType;
  typedef FixedMeshType     MeshType;
  typedef FixedImageType    ImageType;

  /** Typedef for timer. */
  // typedef tmr::Timer          TimerType;
  // typedef TimerType::Pointer  TimerPointer;

  /** Sets up a timer to measure the initialization time and calls the
   * Superclass' implementation.
   */
  void
  Initialize(void) override;

  /**
   * Do some things before registration:
   * \li Load and set the pointsets.
   */
  int
  BeforeAllBase(void) override;

  void
  BeforeRegistration(void) override;

  void
  AfterEachIteration(void) override;

  void
  AfterEachResolution(void) override;

  /** Function to read the corresponding points. */
  unsigned int
  ReadMesh(const std::string & meshFileName, typename FixedMeshType::Pointer & mesh);

  void
  WriteResultMesh(const char * filename, MeshIdType meshId);

  unsigned int
  ReadTransformixPoints(const std::string & filename, typename MeshType::Pointer & mesh);

  /** Overwrite to silence warning. */
  void
  SelectNewSamples(void) override
  {}

protected:
  /** The constructor. */
  PolydataDummyPenalty();
  /** The destructor. */
  ~PolydataDummyPenalty() override = default;

private:
  elxOverrideGetSelfMacro;

  /** The deleted copy constructor. */
  PolydataDummyPenalty(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

  unsigned int m_NumberOfMeshes;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxPolydataDummyPenalty.hxx"
#endif

#endif // end #ifndef elxPolydataDummyPenalty_h
