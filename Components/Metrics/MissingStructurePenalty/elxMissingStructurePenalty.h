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
#ifndef elxMissingStructurePenalty_h
#define elxMissingStructurePenalty_h

#include "elxIncludes.h"
#include "itkMissingStructurePenalty.h"

#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"

namespace elastix
{

/**
 * \class MissingStructurePenalty
 * \brief .
 *
 * \brief Computes the (pseudo) volume of the transformed surface mesh of a structure.\n
 * A metric based on the itk::MissingStructurePenalty.\n
 * \author F.F. Berendsen, Image Sciences Institute, UMC Utrecht, The Netherlands
 * \note If you use the MissingStructurePenalty anywhere we would appreciate if you cite the following article:\n
 * F.F. Berendsen, A.N.T.J. Kotte, A.A.C. de Leeuw, I.M. Juergenliemk-Schulz,\n
 * M.A. Viergever and J.P.W. Pluim "Registration of structurally dissimilar \n
 * images in MRI-based brachytherapy ", Phys. Med. Biol. 59 (2014) 4033-4045.\n
 * http://stacks.iop.org/0031-9155/59/4033
 * The parameters used in this class are:
 * \parameter Metric: Select this metric as follows:\n
 *    <tt>(Metric "MissingStructurePenalty")</tt>
 * \parameter
 *    <tt>(WriteResultMeshAfterEachIteration "True")</tt>
 * \parameter
 *    <tt>(WriteResultMeshAfterEachResolution "True")</tt>
 * The command-line options for input meshes is: -fmesh<[A-Z]><MetricNumber>.
 * \ingroup RegistrationMetrics
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT MissingStructurePenalty
  : public itk::MissingVolumeMeshPenalty<typename MetricBase<TElastix>::FixedPointSetType,
                                         typename MetricBase<TElastix>::MovingPointSetType>
  , public MetricBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MissingStructurePenalty);

  /** Standard ITK-stuff. */
  using Self = MissingStructurePenalty;
  using Superclass1 = itk::MissingVolumeMeshPenalty<typename MetricBase<TElastix>::FixedPointSetType,
                                                    typename MetricBase<TElastix>::MovingPointSetType>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MissingStructurePenalty, itk::MissingVolumeMeshPenalty);

  /** Name of this class.
   * Use this name in the parameter file to select this specific metric. \n
   * example: <tt>(Metric "MissingStructurePenalty")</tt>\n
   */
  elxClassNameMacro("MissingStructurePenalty");

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
  MissingStructurePenalty();
  /** The destructor. */
  ~MissingStructurePenalty() override = default;

private:
  elxOverrideGetSelfMacro;

  unsigned int m_NumberOfMeshes;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxMissingStructurePenalty.hxx"
#endif

#endif // end #ifndef elxMissingStructurePenalty_h
