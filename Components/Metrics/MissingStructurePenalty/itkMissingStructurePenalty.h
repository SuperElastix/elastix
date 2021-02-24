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
#ifndef itkMissingStructurePenalty_h
#define itkMissingStructurePenalty_h

#include "itkSingleValuedPointSetToPointSetMetric.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"
#include "itkMesh.h"
#include "itkVectorContainer.h"
#include "vnl_adjugate_fixed.h"

namespace itk
{

/** \class MissingVolumeMeshPenalty
 * \brief Computes the (pseudo) volume of the transformed surface mesh of a structure.\n
 *
 * \author F.F. Berendsen, Image Sciences Institute, UMC Utrecht, The Netherlands
 * \note If you use the MissingStructurePenalty anywhere we would appreciate if you cite the following article:\n
 * F.F. Berendsen, A.N.T.J. Kotte, A.A.C. de Leeuw, I.M. Juergenliemk-Schulz,\n
 * M.A. Viergever and J.P.W. Pluim "Registration of structurally dissimilar \n
 * images in MRI-based brachytherapy ", Phys. Med. Biol. 59 (2014) 4033-4045.\n
 * http://stacks.iop.org/0031-9155/59/4033
 * \ingroup RegistrationMetrics
 */
template <class TFixedPointSet, class TMovingPointSet>
class ITK_TEMPLATE_EXPORT MissingVolumeMeshPenalty
  : public SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>
{
public:
  /** Standard class typedefs. */
  typedef MissingVolumeMeshPenalty                                              Self;
  typedef SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet> Superclass;
  typedef SmartPointer<Self>                                                    Pointer;
  typedef SmartPointer<const Self>                                              ConstPointer;

  /** Type used for representing point components  */

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MissingVolumeMeshPenalty, SingleValuedPointSetToPointSetMetric);

  /** Types transferred from the base class */
  typedef typename Superclass::TransformType           TransformType;
  typedef typename Superclass::TransformPointer        TransformPointer;
  typedef typename Superclass::TransformParametersType TransformParametersType;
  typedef typename Superclass::TransformJacobianType   TransformJacobianType;

  typedef typename Superclass::MeasureType         MeasureType;
  typedef typename Superclass::DerivativeType      DerivativeType;
  typedef typename Superclass::DerivativeValueType DerivativeValueType;

  typedef typename Superclass::InputPointType    InputPointType;
  typedef typename Superclass::OutputPointType   OutputPointType;
  typedef typename OutputPointType::CoordRepType CoordRepType;
  typedef vnl_vector<CoordRepType>               VnlVectorType;

  typedef typename Superclass::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  /** Constants for the pointset dimensions. */
  itkStaticConstMacro(FixedPointSetDimension, unsigned int, Superclass::FixedPointSetDimension);
  itkStaticConstMacro(MovingPointSetDimension, unsigned int, Superclass::MovingPointSetDimension);

  /** Typedefs. */
  typedef unsigned char DummyMeshPixelType;
  typedef DefaultStaticMeshTraits<DummyMeshPixelType, FixedPointSetDimension, FixedPointSetDimension, CoordRepType>
                                                                           MeshTraitsType;
  typedef Mesh<DummyMeshPixelType, FixedPointSetDimension, MeshTraitsType> FixedMeshType;
  typedef typename FixedMeshType::PointIdentifier                          FixedMeshPointIdentifier;

  typedef typename FixedMeshType::ConstPointer FixedMeshConstPointer;
  typedef typename FixedMeshType::Pointer      FixedMeshPointer;
  typedef typename MeshTraitsType::CellType    CellInterfaceType;

  typedef typename FixedMeshType::PointType                                       MeshPointType;
  typedef typename FixedMeshType::PointType::VectorType                           VectorType;
  typedef typename VectorType::const_pointer                                      VectorConstPointer;
  typedef itk::Vector<typename VectorType::ValueType, FixedPointSetDimension - 1> SubVectorType;

  typedef typename FixedMeshType::PointsContainer              MeshPointsContainerType;
  typedef typename MeshPointsContainerType::Pointer            MeshPointsContainerPointer;
  typedef typename MeshPointsContainerType::ConstPointer       MeshPointsContainerConstPointer;
  typedef typename FixedMeshType::PointsContainerConstIterator MeshPointsContainerConstIteratorType;
  typedef typename FixedMeshType::PointsContainerIterator      MeshPointsContainerIteratorType;

  typedef typename FixedMeshType::PointDataContainer             MeshPointDataContainerType;
  typedef typename FixedMeshType::PointDataContainerConstPointer MeshPointDataContainerConstPointer;
  typedef typename FixedMeshType::PointDataContainerPointer      MeshPointDataContainerPointer;
  // typedef typename FixedMeshType::PointDataContainerConstIterator     MeshPointDataContainerConstIteratorType;
  typedef typename FixedMeshType::PointDataContainerIterator MeshPointDataContainerConstIteratorType;
  typedef typename MeshPointDataContainerType::Iterator      MeshPointDataContainerIteratorType;

  typedef unsigned int                                       MeshIdType;
  typedef VectorContainer<MeshIdType, FixedMeshConstPointer> FixedMeshContainerType;
  typedef typename FixedMeshContainerType::Pointer           FixedMeshContainerPointer;
  typedef typename FixedMeshContainerType::ConstPointer      FixedMeshContainerConstPointer;
  typedef typename FixedMeshContainerType::ElementIdentifier FixedMeshContainerElementIdentifier;

  typedef VectorContainer<MeshIdType, FixedMeshPointer>  MappedMeshContainerType;
  typedef typename MappedMeshContainerType::Pointer      MappedMeshContainerPointer;
  typedef typename MappedMeshContainerType::ConstPointer MappedMeshContainerConstPointer;

  typedef Array<DerivativeValueType> MeshPointsDerivativeValueType;

  itkSetConstObjectMacro(FixedMeshContainer, FixedMeshContainerType);
  itkGetConstObjectMacro(FixedMeshContainer, FixedMeshContainerType);

  itkSetObjectMacro(MappedMeshContainer, MappedMeshContainerType);
  itkGetModifiableObjectMacro(MappedMeshContainer, MappedMeshContainerType);

  /** Initialize the Metric by making sure that all the components are
   *  present and plugged together correctly.
   */
  void
  Initialize(void) override;

  /** Set the fixed mask. */
  // \todo: currently not used
  // itkSetConstObjectMacro( FixedImageMask, FixedImageMaskType );

  /** Get the fixed mask. */
  // itkGetConstObjectMacro( FixedImageMask, FixedImageMaskType );

  /**  Get the value for single valued optimizers. */
  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & Derivative) const override;

  /**  Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const override;

protected:
  MissingVolumeMeshPenalty();
  ~MissingVolumeMeshPenalty() override;

  /** PrintSelf. */
  // void PrintSelf(std::ostream& os, Indent indent) const;

  /** Member variables. */
  FixedMeshConstPointer m_FixedMesh;

  mutable FixedMeshContainerConstPointer m_FixedMeshContainer;
  mutable MappedMeshContainerPointer     m_MappedMeshContainer;

private:
  void
  SubVector(const VectorType & fullVector, SubVectorType & subVector, const unsigned int leaveOutIndex) const;

  MissingVolumeMeshPenalty(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMissingStructurePenalty.hxx"
#endif

#endif
