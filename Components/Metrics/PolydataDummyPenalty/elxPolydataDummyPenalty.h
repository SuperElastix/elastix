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
  ITK_DISALLOW_COPY_AND_MOVE(PolydataDummyPenalty);

  /** Standard ITK-stuff. */
  using Self = PolydataDummyPenalty;
  using Superclass1 = itk::MeshPenalty<typename MetricBase<TElastix>::FixedPointSetType,
                                       typename MetricBase<TElastix>::MovingPointSetType>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

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
  using typename Superclass1::FixedMeshType;
  using typename Superclass1::FixedMeshPointer;
  using typename Superclass1::FixedMeshConstPointer;

  using typename Superclass1::CoordinateRepresentationType;
  using typename Superclass1::FixedPointSetType;
  using typename Superclass1::FixedPointSetConstPointer;
  using typename Superclass1::FixedMeshContainerType;
  using typename Superclass1::FixedMeshContainerPointer;
  using typename Superclass1::MappedMeshContainerType;
  using typename Superclass1::MappedMeshContainerPointer;
  using typename Superclass1::MovingPointSetType;
  using typename Superclass1::MovingPointSetConstPointer;
  using typename Superclass1::CellInterfaceType;

  //  using typename Superclass1::FixedImageRegionType;
  using typename Superclass1::TransformType;
  using typename Superclass1::TransformPointer;
  using typename Superclass1::InputPointType;
  using typename Superclass1::OutputPointType;
  using typename Superclass1::TransformParametersType;
  using typename Superclass1::TransformJacobianType;
  //  using typename Superclass1::RealType;
  using typename Superclass1::FixedImageMaskType;
  using typename Superclass1::FixedImageMaskPointer;
  using typename Superclass1::MovingImageMaskType;
  using typename Superclass1::MovingImageMaskPointer;
  using typename Superclass1::MeasureType;
  using typename Superclass1::DerivativeType;
  using typename Superclass1::ParametersType;

  using CoordRepType = typename OutputPointType::CoordRepType;

  using typename Superclass1::MeshIdType;
  /** Other typedef's. */
  /*typedef itk::AdvancedTransform<
  CoordRepType,
  itkGetStaticConstMacro( FixedImageDimension ),
  itkGetStaticConstMacro( MovingImageDimension ) >  ITKBaseType;
  */
  using CombinationTransformType = itk::AdvancedCombinationTransform<CoordRepType, Self::FixedImageDimension>;
  using InitialTransformType = typename CombinationTransformType::InitialTransformType;

  /** Typedefs inherited from elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;
  using typename Superclass2::FixedImageType;
  using typename Superclass2::MovingImageType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Assuming fixed and moving pointsets are of equal type, which implicitly
   * assumes that the fixed and moving image are of the same type.
   */
  using PointSetType = FixedPointSetType;
  using MeshType = FixedMeshType;
  using ImageType = FixedImageType;

  /** Typedef for timer. */
  // typedef tmr::Timer          TimerType;
  // typedef TimerType::Pointer  TimerPointer;

  /** Sets up a timer to measure the initialization time and calls the
   * Superclass' implementation.
   */
  void
  Initialize() override;

  /**
   * Do some things before registration:
   * \li Load and set the pointsets.
   */
  int
  BeforeAllBase() override;

  void
  BeforeRegistration() override;

  void
  AfterEachIteration() override;

  void
  AfterEachResolution() override;

  /** Function to read the corresponding points. */
  unsigned int
  ReadMesh(const std::string & meshFileName, typename FixedMeshType::Pointer & mesh);

  void
  WriteResultMesh(const char * filename, MeshIdType meshId);

  unsigned int
  ReadTransformixPoints(const std::string & filename, typename MeshType::Pointer & mesh);

  /** Overwrite to silence warning. */
  void
  SelectNewSamples() override
  {}

protected:
  /** The constructor. */
  PolydataDummyPenalty();
  /** The destructor. */
  ~PolydataDummyPenalty() override = default;

private:
  elxOverrideGetSelfMacro;

  unsigned int m_NumberOfMeshes;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxPolydataDummyPenalty.hxx"
#endif

#endif // end #ifndef elxPolydataDummyPenalty_h
